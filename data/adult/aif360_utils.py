import numpy as np
import pandas as pd
from aif360.datasets import GermanDataset, AdultDataset, BankDataset, StandardDataset
from src.data.bank.scalers import LogisticRegression

import FaX_methods
import shap
from src.data.bank.scalers import train_test_split
from aif360.sklearn.datasets import fetch_compas
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction

class FaXAIF():
    """
    Layer on top of FaX AI methods for using dataframes.
    Layer is not explicitly needed but shows general way on how to convert
    dataframes from common ML-fairness libraries to the expected numpy arrays.

    This allows for AIF data to be used with FaX AI methods.

    Args:
        X_df: dataframe of training samples
        Y_df: dataframe of training outcomes
        prot_attr: name of protected attribute
        model_type: (MIM/optimization) which FaX AI algorithm to use
        influence: (shap/mde) which influence measure to use for OPT approach
        params: a dictionary of the parameters to be used for OPT approach
    """
    def __init__(self, X_df, Y_df, prot_attr, model, model_type='MIM', influence='shap', params=None):

        self.X = X_df.features
        self.Z = X_df.convert_to_dataframe()[0][prot_attr].to_numpy().reshape(-1, 1)
        self.Y = Y_df

        # self.X = X_df[X_df.columns.drop(prot_attr)].values
        # self.Z = X_df[prot_attr].values.reshape(-1, 1)
        # self.Y = Y_df.values
        # print(self.Z)
        self.prot_attr = prot_attr

        # print(type(self.X))
        if model_type == 'MIM':
            self.model = FaX_methods.MIM(self.X, self.Z, self.Y, model)
        elif model_type in ['optimization','OPT']:
            self.model = FaX_methods.Optimization(self.X, self.Z, self.Y, influence=influence, params=params)

    def predict_proba(self, X_df):
        X = X_df.features
        return self.model.predict_proba(X)

    def predict(self, X_df):
        X = X_df.features
        return self.model.predict(X)

def load_compas(prot_attr = 'race',subsample=0,train_size=0.7):
    """
    Loads and preprocesses the COMPAS dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    Args:
        prot_attr: name of protected attribute
        subsample: number of positive outcomes to remove for the disadvantaged group
        train_size: percentage of data to be used for training
    """
    X, y = fetch_compas(binary_race=True)
    X = X.drop(columns=['age_cat', 'c_charge_desc'])

    X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
    y = 1 - pd.Series(y.factorize(sort=True)[0], index=y.index)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=train_size, random_state=1234567)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_train = X_train.rename(columns={"race_Caucasian": "race", "sex_Female": "sex"})
    X_test = X_test.rename(columns={"race_Caucasian": "race", "sex_Female": "sex"})
    X_train = X_train[ [prot_attr] + [ col for col in X_train.columns if col not in [prot_attr,"c_charge_degree_M","race_Caucasian","race_African-American","sex_Male"] ] ]
    X_test = X_test[ [prot_attr] + [ col for col in X_test.columns if col not in [prot_attr,"c_charge_degree_M","race_Caucasian","race_African-American","sex_Male"] ] ]

    if subsample:
        rep_ind = y_train.iloc[y_train.index.get_level_values(prot_attr) == 0].loc[lambda x : x==1][:subsample].index
        y_train = y_train.drop(rep_ind)
        X_train = X_train.drop(rep_ind)

    return X_train, X_test, y_train, y_test



def load_titanic(prot_attr = 'sex',train_size=0.7):
    """
    Loads and preprocesses the Titanic dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    Args:
        prot_attr: name of protected attribute
        train_size: percentage of data to be used for training
    """
    def custom_preprocessing(df):
        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        return df
    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test.loc[:, 'Survived'] = 0

    from src.data.bank.scalers import TransformerMixin
    from src.data.bank.scalers import Pipeline, FeatureUnion
    from typing import List

    class SelectCols(TransformerMixin):
        """Select columns from a DataFrame."""

        def __init__(self, cols: List[str]) -> None:
            self.cols = cols

        def fit(self, x: None) -> "SelectCols":
            """Nothing to do."""
            return self

        def transform(self, x: pd.DataFrame) -> pd.DataFrame:
            """Return just selected columns."""
            return x[self.cols]


    sc = SelectCols(cols=['Sex', 'Survived'])
    sc.transform(train.sample(5))


    class LabelEncoder(TransformerMixin):
        """Convert non-numeric columns to numeric using label encoding.
        Handles unseen data on transform."""

        def fit(self, x: pd.DataFrame) -> "LabelEncoder":
            """Learn encoder for each column."""
            encoders = {}
            for c in x:
                v, k = zip(pd.factorize(x[c].unique()))
                encoders[c] = dict(zip(k[0], v[0]))

            self.encoders_ = encoders

            return self

        def transform(self, x) -> pd.DataFrame:
            """For columns in x that have learned encoders, apply encoding."""
            x = x.copy()
            for c in x:
                # Ignore new, unseen values
                x.loc[~x[c].isin(self.encoders_[c]), c] = np.nan
                # Map learned labels
                x.loc[:, c] = x[c].map(self.encoders_[c])

            # Return without nans
            return x.fillna(-2).astype(int)


    le = LabelEncoder()
    le.fit_transform(train[['Pclass', 'Sex']].sample(5))


    class NumericEncoder(TransformerMixin):
        """Remove invalid values from numerical columns, replace with median."""

        def fit(self, x: pd.DataFrame) -> "NumericEncoder":
            """Learn median for every column in x."""
            self.encoders_ = {
                c: pd.to_numeric(x[c],
                                 errors='coerce').median(skipna=True) for c in x}

            return self

        def transform(self, x: pd.DataFrame) -> pd.DataFrame:
            # Create a list of new DataFrames, each with 2 columns
            output_dfs = []
            for c in x:
                new_cols = pd.DataFrame()
                # Find invalid values that aren't nans (-inf, inf, string)
                invalid_idx = pd.to_numeric(x[c].replace([-np.inf, np.inf],
                                                         np.nan),
                                            errors='coerce').isnull()

                # Copy to new df for this column
                new_cols.loc[:, c] = x[c].copy()
                # Replace the invalid values with learned median
                new_cols.loc[invalid_idx, c] = self.encoders_[c]
                # Mark these replacement in a new column called
                # "[column_name]_invalid_flag"
                new_cols.loc[:, f"{c}_invalid_flag"] = invalid_idx.astype(np.int8)

                output_dfs.append(new_cols)

            # Concat list of output_dfs to single df
            df = pd.concat(output_dfs,
                           axis=1)

            return df.fillna(0)


    ne = NumericEncoder()
    ne.fit_transform(train[['Age', 'Fare']].sample(5))

    # LabelEncoding fork: Select object columns -> label encode
    pp_object_cols = Pipeline([('select', SelectCols(cols=['Sex', 'Survived',
                                                           'Cabin', 'Ticket',
                                                           'SibSp', 'Embarked',
                                                           'Parch', 'Pclass',
                                                           'Name'])),
                               ('process', LabelEncoder())])

    # NumericEncoding fork: Select numeric columns -> numeric encode
    pp_numeric_cols = Pipeline([('select', SelectCols(cols=['Age',
                                                            'Fare'])),
                                ('process', NumericEncoder())])

    # We won't use the next part, but typically the pipeline would continue to
    # the model (after dropping 'Survived' from the training data, of course).
    # For example:
    pp_pipeline = FeatureUnion([('object_cols', pp_object_cols),
                                ('numeric_cols', pp_numeric_cols)])

    model_pipeline = Pipeline([('pp', pp_pipeline),
                               ('mod', LogisticRegression())])
    train_ = train

    # .fit_transform on train
    train_pp = pd.concat((pp_numeric_cols.fit_transform(train_),
                          pp_object_cols.fit_transform(train_)),
                         axis=1)

    # .transform on test
    test_pp = pd.concat((pp_numeric_cols.transform(test),
                         pp_object_cols.transform(test)),
                        axis=1)
    test_pp.sample(5)

    target = 'Survived'
    x_columns = [c for c in train_pp if c != target]
    x_train, y_train = train_pp[x_columns], train_pp[target]
    x_test = test_pp[x_columns]

    df = pd.concat((x_train, y_train), axis=1)

    dataset_orig = StandardDataset(df,
                                   label_name='Survived',
                                   protected_attribute_names=['Sex'],
                                   favorable_classes=[1],
                                   privileged_classes=[[1]])

    privileged_groups = [{'Sex': 1}]
    unprivileged_groups = [{'Sex': 0}]

    data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)

    X_train = data_orig_train.features
    y_train = data_orig_train.labels.ravel()

    X_test = data_orig_test.features
    y_test = data_orig_test.labels.ravel()
    return data_orig_train, data_orig_test, X_train, X_test, y_train, y_test
def load_bank(prot_attr = 'sex',train_size=0.7):
    """
    Loads and preprocesses the Bank Marketing Income dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    Args:
        prot_attr: name of protected attribute
        train_size: percentage of data to be used for training
    """

    def custom_preprocessing(df):
        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        return df


    dataset_orig = BankDataset(protected_attribute_names=['age'],
                               features_to_keep=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                                                 'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate',
                                                 'cons.price.idx',
                                                 'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays',
                                                 'previous', 'poutcome'])
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]

    data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)

    X_train = data_orig_train.features
    y_train = data_orig_train.labels.ravel()

    X_test = data_orig_test.features
    y_test = data_orig_test.labels.ravel()

    return data_orig_train, data_orig_test, X_train, X_test, y_train, y_test

def load_census(prot_attr = 'sex',train_size=0.7):
    """
    Loads and preprocesses the Adult Census Income dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    Args:
        prot_attr: name of protected attribute
        train_size: percentage of data to be used for training
    """
    # X, y, sample_weight = fetch_adult()
    #
    # X = X.drop(columns=['native-country'])
    #
    # X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    # y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
    #
    # y = pd.Series(y.factorize(sort=True)[0], index=y.index)
    # # there is one unused category ('Never-worked') that was dropped during dropna
    # X.workclass.cat.remove_unused_categories(inplace=True)
    # (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=train_size, random_state=1234567)
    # print(X_train.columns)
    #
    # X_train = pd.get_dummies(X_train)
    # print(X_train.columns.values)
    # X_test = pd.get_dummies(X_test)
    #
    # X_train = X_train.rename(columns={"race_White": "race", "sex_Male": "sex"})
    # X_test = X_test.rename(columns={"race_White": "race", "sex_Male": "sex"})
    #
    # X_train = X_train[ [prot_attr] + [ col for col in X_train.columns if col not in [prot_attr,"race_White","race_Non-white","sex_Female","sex_Male"] ] ]
    # X_test =X_test[ [prot_attr] + [ col for col in X_test.columns if col not in [prot_attr,"race_White","race_Non-white","sex_Female","sex_Male"] ] ]
    def custom_preprocessing(df):
        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        return df


    dataset_orig = AdultDataset(protected_attribute_names=['race'],
                                privileged_classes=[[1]],
                                categorical_features=['workclass', 'education', 'marital-status', 'occupation',
                                                      'relationship', 'native-country'],
                                features_to_drop=['income', 'native-country', 'hours-per-week'],
                                custom_preprocessing=custom_preprocessing)
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]

    data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)
    X_train = data_orig_train.features

    y_train = data_orig_train.labels.ravel()
    X_test = data_orig_test.features
    y_test = data_orig_test.labels.ravel()
    # print(X_train)

    return data_orig_train, data_orig_test, X_train, X_test, y_train, y_test

def load_german(prot_attr = 'sex',train_size=0.7):
    """
    Loads and preprocesses the German Credit dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    Args:
        prot_attr: name of protected attribute
        train_size: percentage of data to be used for training
    """
    # X, y = fetch_german(numeric_only=True)
    #
    # X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    # y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
    #
    # y = pd.Series(y.factorize(sort=True)[0], index=y.index)
    #
    # (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=train_size, random_state=1234567)
    # X_train = X_train[[prot_attr] + [ col for col in X_train.columns if col not in [prot_attr] ] ]
    # X_test =X_test[ [prot_attr] + [ col for col in X_test.columns if col not in [prot_attr] ] ]
    def custom_preprocessing(df):
        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'

        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                      'A92': 0.0, 'A95': 0.0}
        df['sex'] = df['personal_status'].replace(status_map)

        # group credit history, savings, and employment
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['status'] = df['status'].apply(lambda x: group_status(x))
        df['credit'] = df['credit'].replace({2: 0.0, 1: 1.0})
        return df


    dataset_orig = GermanDataset(protected_attribute_names=['sex'],
                                 privileged_classes=[[1]],
                                 features_to_keep=['age', 'sex', 'employment', 'housing', 'savings', 'credit_amount',
                                                   'month', 'purpose'],
                                 custom_preprocessing=custom_preprocessing)
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)
    # print(data_orig_train)
    # print(data_orig_train.convert_to_dataframe()[0]['sex'].to_numpy().reshape(-1, 1))
    X_train = data_orig_train.features

    y_train = data_orig_train.labels.ravel()
    X_test = data_orig_test.features
    y_test = data_orig_test.labels.ravel()
    # print(X_train)


    return data_orig_train, data_orig_test, X_train, X_test, y_train, y_test

def shap_aif(model, train_data, test_data, explainer_samples=100):
    """
    Middle layer to make AIF360 formated dataframes and methods work with the
    SHAP library.

    Args:
        model: trained AIF360 or sklearn compatible model
        train_data: data for integrating out features to generate explainer
        test_data: data to calculate the SHAP values for
        explainer_samples: number of samples to use when generating explainer
    """
    def transform_shap(x,Xt):
        df= pd.DataFrame(x, columns=Xt.columns)
        index_names = list(Xt.index.names)
        if len(index_names)==1:
            df = df.set_index(index_names[0], drop=False)
        else:
            df.index = pd.MultiIndex.from_arrays(df[index_names[1:]].values.T, names=index_names[1:])
        return df
    f = lambda x, Xt= test_data: model.predict_proba(transform_shap(x, Xt))[:,1]
    explainer = shap.KernelExplainer(f, shap.kmeans(train_data, 100))
    expected_value = explainer.expected_value
    shap_values = explainer.shap_values(test_data)
    return shap_values, expected_value, explainer

class ExponentiatedGradientReductionFix(ExponentiatedGradientReduction):
    """
    Fix for exponentiated gradient AIF implementation to work with SHAP.
    Original implementation errors on data following SHAP integrating out features.
    """
    def predict(self, X):
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        return self.classes_[self.model.predict(X)]


    def predict_proba(self, X):
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        return self.model._pmf_predict(X)
