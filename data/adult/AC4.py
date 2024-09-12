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
dir = 'res/adult4-'
import time
start_time = time.time()
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

# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

## Implement label encoder instead of one-hot encoder
for feature in cat_feat:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])


# In[3]:


#
# data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
# data_train_2, data_test_2 = train_test_split(df, test_size = 0.3, random_state = seed) #

pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

# data, X, y= load_adult_data(df, pro_att_name, priv_class, reamining_cat_feat)
# print(data)
# data, att = data.convert_to_dataframe()
# data = data.astype(int)
# data.to_csv("ac4_" + str(len(data.columns)), index=False)

# In[4]:
seed = 13
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
data_train_2, data_test_2 = train_test_split(df, test_size = 0.3, random_state = seed) #

data_orig_train_1, X_train_1, y_train_1 = load_adult_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_adult_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_adult_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_adult_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)

sc = StandardScaler()
trained = sc.fit(X_train_1)
X_train_1 = trained.transform(X_train_1)
X_test_1 = trained.transform(X_test_1)

X_train_2 = trained.transform(X_train_2)
X_test_2 = trained.transform(X_test_2)

pca = PCA(n_components=14)
trained = pca.fit(X_train_1)
X_train_1 = trained.transform(X_train_1)
X_test_1 = trained.transform(X_test_1)

X_train_2 = trained.transform(X_train_2)
X_test_2 = trained.transform(X_test_2)

data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1

data_orig_train_2.features = X_train_2
data_orig_test_2.features = X_test_2

from xgboost import XGBClassifier
model_1 = XGBClassifier()
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = XGBClassifier()
model_2 = model_2.fit(X_train_2, y_train_2)
print("intial time",  start_time - time.time())
y_pred_1 = model_1.predict(X_test_1)
y_pred_2 = model_2.predict(X_test_2)

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