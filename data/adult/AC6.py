#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

from fairness_permutation_importance import permutation_importance
import sklearn

sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
import time
start_time = time.time()
dir = 'res/adult6-'
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

df = pd.concat([test, train], ignore_index=True)

##### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped


# In[3]:


y1_df = df.copy()

df["marital-status"] = df["marital-status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
df["marital-status"] = df["marital-status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
df["marital-status"] = df["marital-status"].map({"Married":0, "Single":1})
df["marital-status"] = df["marital-status"]

excluded_feat = ["workclass","education","occupation","relationship","native-country"]

df.drop(labels=excluded_feat, axis=1, inplace=True)

y1_df.drop(labels=excluded_feat, axis=1, inplace=True)


# In[4]:


# Create a one-hot encoding of the categorical variables.
cat_feat = ['age', 'hours-per-week', 'sex']
ccat_feat = ['age', 'hours-per-week', 'sex', 'marital-status']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
y1_df = pd.get_dummies(y1_df, columns=ccat_feat, prefix_sep='=')

# y2_cat_feat = ["workclass","education","occupation","relationship","native-country", 'age', 'hours-per-week', 'sex']
# cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


# In[5]:



pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

# data1, X1, y1 = load_adult_data(df, pro_att_name, priv_class, reamining_cat_feat)
# data2, X2, y2 = load_adult_data(y1_df, pro_att_name, priv_class, reamining_cat_feat)



# data1, att = data1.convert_to_dataframe()
# data1 = data1.astype(int)
# data1.to_csv("ac6_1_" + str(len(data1.columns)), index=False)
# #
# data2, att = data2.convert_to_dataframe()
# data2 = data2.astype(int)
# data2.to_csv("ac6_2_" + str(len(data2.columns)), index=False)

# data_orig_train_2, X_train_2, y_train_1 = load_adult_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
# data_orig_test_1, X_test_1, y_test_2 = load_adult_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)

seed = 7
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
data_train_2, data_test_2 = train_test_split(y1_df, test_size = 0.3, random_state = seed) #

data_orig_train_1, X_train_1, y_train_1 = load_adult_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_adult_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_adult_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_adult_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)

from xgboost import XGBClassifier
# n_estimators=1600,learning_rate=0.05
model_1 = XGBClassifier()
model_1 = model_1.fit(X_train_1, y_train_1, verbose=False)

model_2 = XGBClassifier()
model_2 = model_2.fit(X_test_2, y_test_2, verbose=False)

y_pred_1 = model_1.predict(X_test_1)
y_pred_2 = model_2.predict(X_test_2)
print("intial time",  start_time - time.time())
pro_att = 'race'
from predefine import disparate_impact, statistical_parity_difference, equal_opportunity_difference, average_odds_difference

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


print(permutation_importance("boosting", data_orig_train_1, data_orig_test_1, "race", "DID"))

print(permutation_importance("boosting", data_orig_train_2, data_orig_test_2, "race", "DID"))
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