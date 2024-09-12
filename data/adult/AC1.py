#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import sklearn

from fairness_permutation_importance import permutation_importance

sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
import time
start_time = time.time()
dir = 'res/adult1-'

d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']


train_path = '../../../data/adult/adult.data'
test_path = '../../../data/adult/adult.test'

column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income-per-year']
na_values = ['?']

train = pd.read_csv(train_path, header=None, names=column_names,
                    skipinitialspace=True, na_values=na_values)
test = pd.read_csv(test_path, header=0, names=column_names,
                   skipinitialspace=True, na_values=na_values)

df_good = pd.concat([test, train], ignore_index=True)

##### Drop na values
dropped = df_good.dropna()
count = df_good.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df_good = dropped
print("--- %s drop: ---" % (time.time() - start_time))
df_bad = df_good.copy()
# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
df_bad = pd.get_dummies(df_bad, columns=cat_feat, prefix_sep='=')
df_bad = df_bad.drop(columns=['age', 'fnlwgt', 'capital-loss'])
## Implement label encoder instead of one-hot encoder
for feature in cat_feat:
    le = LabelEncoder()
    df_good[feature] = le.fit_transform(df_good[feature])
seed = randrange(100)
# df_bad = df_bad.drop(columns=["fnlwgt", "capital-gain"])
good_train, good_test = train_test_split(df_good, test_size=0.3, random_state=73)  #
bad_train, bad_test = train_test_split(df_bad, test_size=0.3, random_state=73)  # stratify=df['race']
print("--- %s split ---" % (time.time() - start_time))

pro_att_name = ['race']  # ['race', 'sex']
priv_class = ['White']  # ['White', 'Male']
reamining_cat_feat = []

good_data_orig_train, good_X_train, good_y_train = load_adult_data(good_train, pro_att_name, priv_class, reamining_cat_feat)
good_data_orig_test, good_X_test, good_y_test = load_adult_data(good_test, pro_att_name, priv_class, reamining_cat_feat)

bad_data_orig_train, bad_X_train, bad_y_train = load_adult_data(bad_train, pro_att_name, priv_class, reamining_cat_feat)
bad_data_orig_test, bad_X_test, bad_y_test = load_adult_data(bad_test, pro_att_name, priv_class, reamining_cat_feat)

sc = StandardScaler()

trained = sc.fit(good_X_train)
good_X_train = trained.transform(good_X_train)
good_X_test = trained.transform(good_X_test)

good_data_orig_train.features = good_X_train
good_data_orig_test.features = good_X_test
# trained_1 = sc.fit(bad_X_train)
# bad_X_train = trained_1.transform(bad_X_train)
# bad_X_test = trained_1.transform(bad_X_test)
#
bad_data_orig_train.features = bad_X_train
bad_data_orig_test.features = bad_X_test

# In[5]:
good_model = LogisticRegression()
good_model = good_model.fit(good_X_train, good_y_train)

bad_model = LogisticRegression()
bad_model = bad_model.fit(bad_X_train, bad_y_train)
print("intial time",  start_time - time.time())


good_y_pred = good_model.predict(good_X_test)
bad_y_pred = bad_model.predict(bad_X_test)

from predefine import disparate_impact, statistical_parity_difference, equal_opportunity_difference, average_odds_difference

pro_att = 'race'

print("Accuracy: ", sklearn.metrics.accuracy_score(good_y_test, good_y_pred))
print("DI: ", disparate_impact(good_data_orig_test, good_y_pred, pro_att))
print("SPD: ", statistical_parity_difference(good_data_orig_test, good_y_pred, pro_att))
print("EOD: ", equal_opportunity_difference(good_data_orig_test, good_y_pred, good_y_test, pro_att))
print("AOD: ", average_odds_difference(good_data_orig_test, good_y_pred, good_y_test, pro_att))

print("Accuracy: ", sklearn.metrics.accuracy_score(bad_y_test, bad_y_pred))
print("DI: ", disparate_impact(bad_data_orig_test, bad_y_pred, pro_att))
print("SPD: ", statistical_parity_difference(bad_data_orig_test, bad_y_pred, pro_att))
print("EOD: ", equal_opportunity_difference(bad_data_orig_test, bad_y_pred, bad_y_test, pro_att))
print("AOD: ", average_odds_difference(bad_data_orig_test, bad_y_pred, bad_y_test, pro_att))

print(permutation_importance("lrg", good_data_orig_train, good_data_orig_test, "race", "DIR"))

print(permutation_importance("lrg", bad_data_orig_train, bad_data_orig_test, "race", "DIR"))


import sklearn
from sklearn.inspection import permutation_importance
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, \
    equalized_odds_difference, equalized_odds_ratio
def DIScorer(y_true, y_pred):
    df, feature = data_orig_train_1.convert_to_dataframe()
    attr = df["age"]
    return abs(demographic_parity_difference(y_true, y_pred, sensitive_features=attr))
#sklearn make_scorer for the churnScorer
di = sklearn.metrics.make_scorer(DIScorer, greater_is_better = False)

def EODScorer(y_true, y_pred):
    df, feature = data_orig_train_1.convert_to_dataframe()
    attr = df["age"]
    return abs(equalized_odds_difference(y_true, y_pred, sensitive_features=attr))
#sklearn make_scorer for the churnScorer
eod = sklearn.metrics.make_scorer(EODScorer, greater_is_better = False)

result = permutation_importance(good_model, X_train_1, y_train_1, scoring=di, n_repeats=10,
                                random_state=0)
res = {}
print(result.importances_mean)
print(len(result.importances_mean), df.columns)
for i in range(len(result.importances_mean)):
    res[df.columns[i]] = result.importances_mean[i]
print(dict(sorted(res.items(), key=lambda item: item[1])))

result = permutation_importance(good_model, X_train_1, y_train_1, scoring=eod, n_repeats=10,
                                random_state=0) #'nr.employed', 'emp.var.rate', 'duration'
res = {}

for i in range(len(result.importances_mean)):
    res[df.columns[i]] = result.importances_mean[i]
print(dict(sorted(res.items(), key=lambda item: item[1])))
result = permutation_importance(bad_model, X_train_2, y_train_2, scoring=di, n_repeats=10,
                                random_state=0)
res = {}
print(result.importances_mean)
print(len(result.importances_mean), df.columns)
for i in range(len(result.importances_mean)):
    res[df.columns[i]] = result.importances_mean[i]
print(dict(sorted(res.items(), key=lambda item: item[1])))
result = permutation_importance(bad_model, X_train_2, y_train_2, scoring=eod, n_repeats=10,
                                random_state=0)
res = {}

for i in range(len(result.importances_mean)):
    res[df.columns[i]] = result.importances_mean[i]
print(dict(sorted(res.items(), key=lambda item: item[1])))