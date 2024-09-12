import math
import time

import sklearn
from aif360.algorithms.preprocessing import DisparateImpactRemover
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, \
    equalized_odds_difference, equalized_odds_ratio
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from predefine import disparate_impact, statistical_parity_difference, equal_opportunity_difference, average_odds_difference


def permutation_importance(model_type, data_train, data_test, attr, metric):
    df_test, features = data_test.convert_to_dataframe()
    df_train, features = data_train.convert_to_dataframe()
    diff = {}

    X_train = data_train.features
    y_train = data_train.labels.ravel()
    X_test = data_test.features
    y_test = data_test.labels.ravel()
    if model_type == "NN":
        model = KNeighborsClassifier()
    if model_type == "SVC":
        model = LinearSVC()
    if model_type == "DT":
        model = DecisionTreeClassifier()
    if model_type == "bagging":
        model = RandomForestClassifier()
    if model_type == "boosting":
        model = GradientBoostingClassifier()
    if model_type == "lrg":
        model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    old_score = abs(equalized_odds_difference(y_test, y_pred, sensitive_features=df_test[attr]))
    start_time = time.time()
    DIR = DisparateImpactRemover()
    data_train_transf = DIR.fit_transform(data_train)
    df_train = df_train.drop(columns=['income-per-year'])
    df_train_copy = df_train.copy()
    df_train_transf, features = data_train_transf.convert_to_dataframe()
    df_train_transf = df_train_transf.drop(columns=['income-per-year'])

    for col in data_train.feature_names:
        if col != attr:
            df_train_copy[col] = df_train_transf[col]
            X_train_transf = df_train_copy.values
            model.fit(X_train_transf, y_train)
            y_pred_transf = model.predict(X_test)
            if metric == "DIR":
                new_score = abs(math.log(demographic_parity_ratio(y_test, y_pred_transf, sensitive_features=df_test[attr])))
            if metric == "DID":
                new_score = abs(demographic_parity_difference(y_test, y_pred_transf, sensitive_features=df_test[attr]))
            if metric == "EOD":
                new_score = abs(equalized_odds_difference(y_test, y_pred_transf, sensitive_features=df_test[attr]))
            if metric == "EOR":
                new_score = abs(math.log(equalized_odds_ratio(y_test, y_pred_transf, sensitive_features=df_test[attr])))
            diff[col] = new_score - old_score   # old_score < new_score
            df_train_copy[col] = df_train[col]
    print("---Num---", (time.time() - start_time))
    return dict(sorted(diff.items(), key=lambda item: item[1]))



# import sys
# sys.path.append('../../../')
# from utils.packages import *
# from utils.ml_fairness import *
# from utils.standard_data import *
# import time
# start_time = time.time()
# dir = 'res/adult1-'
#
# d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# diff_file = dir + 'fairness' + '.csv'
# if (not os.path.isfile(diff_file)):
#     with open(diff_file, 'a') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(d_fields)
#
# train_path = '../../../data/adult/adult.data'
# test_path = '../../../data/adult/adult.test'
#
# column_names = ['age', 'workclass', 'fnlwgt', 'education',
#                 'education-num', 'marital-status', 'occupation', 'relationship',
#                 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
#                 'native-country', 'income-per-year']
# na_values = ['?']
#
# train = pd.read_csv(train_path, header=None, names=column_names,
#                     skipinitialspace=True, na_values=na_values)
# test = pd.read_csv(test_path, header=0, names=column_names,
#                    skipinitialspace=True, na_values=na_values)
#
# df_good = pd.concat([test, train], ignore_index=True)
#
# ##### Drop na values
# dropped = df_good.dropna()
# count = df_good.shape[0] - dropped.shape[0]
# print("Missing Data: {} rows removed.".format(count))
# df_good = dropped
# print("--- %s drop: ---" % (time.time() - start_time))
# df_bad = df_good.copy()
# # Create a one-hot encoding of the categorical variables.
# cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
# df_bad = pd.get_dummies(df_bad, columns=cat_feat, prefix_sep='=')
#
# ## Implement label encoder instead of one-hot encoder
# for feature in cat_feat:
#     le = LabelEncoder()
#     df_good[feature] = le.fit_transform(df_good[feature])
# seed = 30
# df_good = df_good.drop(columns=["hours-per-week"])
# # df_good = df_good.drop(columns=["native-country"])
# # df_good = df_good.drop(columns=["fnlwgt"])
# # df_good = df_good.drop(columns=["age"])
# # df_good = df_good.drop(columns=["occupation"])
# good_train, good_test = train_test_split(df_good, test_size=0.3, random_state=77)  #
# print("--- %s split ---" % (time.time() - start_time))
#
# pro_att_name = ['race']  # ['race', 'sex']
# priv_class = ['White']  # ['White', 'Male']
# reamining_cat_feat = []
# # df_good = df_good.drop(columns=[ "hours-per-week"])
# # df_good = df_good.drop(columns=["capital-loss", "hours-per-week", "age"])
# df_train, df_test = train_test_split(df_good, test_size=0.3, random_state=seed)  #
#
# data_train, X_train, y_train = load_adult_data(df_train, pro_att_name, priv_class, reamining_cat_feat)
# data_test, X_test, y_test = load_adult_data(df_test, pro_att_name, priv_class, reamining_cat_feat)
#
# sc = RobustScaler()
#
# trained = sc.fit(X_train)
# X_train = trained.transform(X_train)
# X_test = trained.transform(X_test)
#
# data_train.features = X_train
# data_test.features = X_test
#
# model = LogisticRegression()
# model = model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# pro_att = "race"
#
# print("Accuracy: ", sklearn.metrics.accuracy_score(y_test, y_pred))
# print("DI: ", disparate_impact(data_test, y_pred, pro_att))
# print("SPD: ", statistical_parity_difference(data_test, y_pred, pro_att))
# print("EOD: ", equal_opportunity_difference(data_test, y_pred, y_test, pro_att))
# print("AOD: ", average_odds_difference(data_test, y_pred, y_test, pro_att))
# # df, feature = data.convert_to_dataframe()
# print(permutation_importance(LogisticRegression(), data_train, data_test, "race"))