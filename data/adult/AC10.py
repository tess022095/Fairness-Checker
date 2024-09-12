#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

from fairness_permutation_importance import permutation_importance

sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
import time
start_time = time.time()
dir = 'res/adult10-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']

# In[2]:


train_path = '../../../data/adult/adult.data'
test_path = '../../../data/adult/adult.test'

column_names = ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']
na_values=['?']

train = pd.read_csv(train_path, header=None, names=column_names, 
                    skipinitialspace=True, na_values=na_values)
test = pd.read_csv(test_path, header=0, names=column_names,
                   skipinitialspace=True, na_values=na_values)

df_impute = pd.concat([test, train], ignore_index=True)


# In[3]:


##### Drop na values
# dropped = df.dropna()
# count = df.shape[0] - dropped.shape[0]
# print("Missing Data: {} rows removed.".format(count))
# df = dropped

df_fillna = df_impute.copy()
from scalers import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df_impute['workclass'] = imputer.fit_transform(df_impute[['workclass']]).ravel()
df_impute['occupation'] = imputer.fit_transform(df_impute[['occupation']]).ravel()
df_impute['native-country'] = imputer.fit_transform(df_impute[['native-country']]).ravel()


df_fillna["workclass"] = df_fillna["workclass"].fillna("X")
df_fillna["occupation"] = df_fillna["occupation"].fillna("X")
df_fillna["native-country"] = df_fillna["native-country"].fillna("x")


# nested_categorical_feature_transformation = Pipeline(steps=[
#         ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
# #         ('encode', OneHotEncoder(handle_unknown='ignore'))
#     ])


# In[4]:


# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
df_impute = pd.get_dummies(df_impute, columns=cat_feat, prefix_sep='=')
df_fillna = pd.get_dummies(df_fillna, columns=cat_feat, prefix_sep='=')

# for feature in cat_feat:
#     le = LabelEncoder()
#     y2_df[feature] = le.fit_transform(y2_df[feature])
    
# for feature in cat_feat:
#     le = LabelEncoder()
#     y1_df[feature] = le.fit_transform(y1_df[feature])


# In[5]:




pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

# data_impute, X_impute, y_impute = load_adult_data(df_impute, pro_att_name, priv_class, reamining_cat_feat)
# data_fillna, X_fillna, y_fillna = load_adult_data(df_fillna, pro_att_name, priv_class, reamining_cat_feat)

# data_impute, att = data_impute.convert_to_dataframe()
# data_impute = data_impute.astype(int)
# data_impute.to_csv("ac10_impute_" + str(len(data_impute.columns)), index=False)
#
# data_fillna, att = data_fillna.convert_to_dataframe()
# data_fillna = data_fillna.astype(int)
# data_fillna.to_csv("ac10_fillna_" + str(len(data_fillna.columns)), index=False)

seed = 87
data_train_1, data_test_1 = train_test_split(df_impute, test_size = 0.3, random_state = seed) # stratify=df['race']
data_train_2, data_test_2 = train_test_split(df_fillna, test_size = 0.3, random_state = seed) #

data_orig_train_1, X_train_1, y_train_1 = load_adult_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_adult_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_adult_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_adult_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)


model_1 = DecisionTreeClassifier()
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = DecisionTreeClassifier()
model_2 = model_2.fit(X_train_2, y_train_2)
print("intial time",  start_time - time.time())
y_pred_1 = model_1.predict(X_test_1)
y_pred_2 = model_2.predict(X_test_2)

pro_att = 'race'
from predefine import disparate_impact, statistical_parity_difference, equal_opportunity_difference, average_odds_difference
import sklearn

print("Accuracy: ", sklearn.metrics.accuracy_score(y_test_1, y_pred_1))
print("DI: ", disparate_impact(data_orig_test_1, y_pred_1, pro_att))
print("SPD: ", statistical_parity_difference(data_orig_test_1, y_pred_1, pro_att))
print("EOD: ", equal_opportunity_difference(data_orig_test_1, y_pred_1, y_test_1, pro_att))
print("AOD: ", average_odds_difference(data_orig_test_1, y_pred_1, y_test_1, pro_att))

print("Accuracy: ", sklearn.metrics.accuracy_score(y_test_2, y_pred_2))
print("DI: ", disparate_impact(data_orig_test_2, y_pred_2, pro_att))
print("SPD: ", statistical_parity_difference(data_orig_test_2, y_pred_2, pro_att))
print("EOD: ", equal_opportunity_difference(data_orig_test_2, y_pred_2, y_test_2, pro_att))
print("AOD: ", average_odds_difference(data_orig_test_2, y_pred_2, y_test_2, pro_att))



print(permutation_importance("DT", data_orig_train_1, data_orig_test_1, "race", "DID"))
print(permutation_importance("DT", data_orig_train_2, data_orig_test_2, "race", "DID"))
import sklearn
from sklearn.inspection import permutation_importance
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, \
    equalized_odds_difference, equalized_odds_ratio
def DIScorer(y_true, y_pred):
    df, feature = data_orig_train_1.convert_to_dataframe()
    attr = df["race"]
    return abs(demographic_parity_difference(y_true, y_pred, sensitive_features=attr))
#sklearn make_scorer for the churnScorer
di = sklearn.metrics.make_scorer(DIScorer, greater_is_better = False)

def EODScorer(y_true, y_pred):
    df, feature = data_orig_train_1.convert_to_dataframe()
    attr = df["race"]
    return abs(equalized_odds_difference(y_true, y_pred, sensitive_features=attr))
#sklearn make_scorer for the churnScorer
eod = sklearn.metrics.make_scorer(EODScorer, greater_is_better = False)


res = {}
start_time = time.time()
result = permutation_importance(model_1, X_train_1, y_train_1, scoring=eod, n_repeats=10,
                                random_state=0) #'nr.employed', 'emp.var.rate', 'duration'
print("---Num3---", (time.time() - start_time))
start_time = time.time()
result = permutation_importance(model_2, X_train_2, y_train_2, scoring=eod, n_repeats=10,
                                random_state=0)
print("---Num4---", (time.time() - start_time))