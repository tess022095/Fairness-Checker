#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import sklearn.metrics

sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
import time
start_time = time.time()
dir = 'res/bank1-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']


# In[2]:


file_path = '../../../data/bank/bank-additional-full.csv'

column_names = []
na_values=['unknown']

df = pd.read_csv(file_path, sep=';', na_values=na_values)

#### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped

df['age'] = df['age'].apply(lambda x: float(x >= 25))

# Create a one-hot encoding of the categorical variables.
# cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

labelencoder_X = LabelEncoder()
df['job']      = labelencoder_X.fit_transform(df['job'])
df['marital']  = labelencoder_X.fit_transform(df['marital'])
df['education']= labelencoder_X.fit_transform(df['education'])
df['default']  = labelencoder_X.fit_transform(df['default'])
df['housing']  = labelencoder_X.fit_transform(df['housing'])
df['loan']     = labelencoder_X.fit_transform(df['loan'])
df['contact']     = labelencoder_X.fit_transform(df['contact'])
df['month']       = labelencoder_X.fit_transform(df['month'])
df['day_of_week'] = labelencoder_X.fit_transform(df['day_of_week'])
df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1, 2, 3], inplace  = True)

def duration(data):
    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5

    return data

# y2_df = df.copy()
df = duration(df)

pro_att_name = ['age']
priv_class = [1]
reamining_cat_feat = []
data, X, y = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
data, att = data.convert_to_dataframe()
data = data.astype(int)
seed = 67
df = df.drop(columns=['euribor3m', 'nr.employed', 'emp.var.rate'])
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed)
data_train_2, data_test_2 = train_test_split(df, test_size = 0.3, random_state = seed)
print(data_train_1)
data_orig_train_1, X_train_1, y_train_1 = load_bank_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_bank_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_bank_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_bank_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

model_1 = XGBClassifier()
model_1 = model_1.fit(X_train_1, y_train_1)


sc = StandardScaler()
X_train_2 = sc.fit_transform(X_train_2)
X_test_2 = sc.fit_transform(X_test_2)

model_2 = XGBClassifier()
model_2 = model_2.fit(X_train_2, y_train_2)

y_pred_1 = model_1.predict(X_test_1)
y_pred_2 = model_2.predict(X_test_2)
data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1

data_orig_train_2.features = X_train_2
data_orig_test_2.features = X_test_2
print("intial time",  start_time - time.time())
pro_att = 'age'
fair = get_fair_metrics_and_plot(data_orig_test_1, model_1)
print("model1", fair)

fair = get_fair_metrics_and_plot(data_orig_test_2, model_2)
print("model2", fair)
from fairness_permutation_importance import permutation_importance
import time

print(permutation_importance("boosting", data_orig_train_1, data_orig_test_1, "age", "DIR"))

print(permutation_importance("boosting", data_orig_train_2, data_orig_test_2, "age", "DIR"))

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

import time
start_time = time.time()
result = permutation_importance(model_1, X_train_1, y_train_1, scoring=eod, n_repeats=10,
                                random_state=0) #'euribor3m', 'nr.employed', 'emp.var.rate'
print("---Num3---", (time.time() - start_time))

import time
start_time = time.time()
result = permutation_importance(model_2, X_train_2, y_train_2, scoring=di, n_repeats=10,
                                random_state=0)
print("---Num4---", (time.time() - start_time))