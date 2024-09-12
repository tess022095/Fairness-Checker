#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
import time
start_time = time.time()
dir = 'res/bank5-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']


# In[2]:


file_path = '../../../data/bank/bank-additional-full.csv'

column_names = []
na_values=['unknown']

df = pd.read_csv(file_path, sep=';', na_values=na_values)

### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped

df['age'] = df['age'].apply(lambda x: float(x >= 25))

# y1_df = df.copy()

# df['poutcome'] = df['poutcome'].map({'failure': -1,'nonexistent': 0,'success': 1})
# df['default'] = df['default'].map({'yes': -1,'unknown': 0,'no': 1})
# df['housing'] = df['housing'].map({'yes': -1,'unknown': 0,'no': 1})
# df['loan'] = df['loan'].map({'yes': -1,'unknown': 0,'no': 1})

# nominal = ['job','marital','education','contact','month','day_of_week']
# df = pd.get_dummies(df, columns=nominal)

y1_nominal = ['job','marital','education','contact','month','day_of_week', 'poutcome', 'default', 'housing', 'loan']
df = pd.get_dummies(df, columns=y1_nominal)


# In[3]:


# data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = 0) #, stratify=df['race']
# data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = 0)


# In[ ]:



pro_att_name = ['age'] # ['race', 'sex']
priv_class = [1] # ['White', 'Male']
reamining_cat_feat = []

data1, X1, y1 = load_bank_data(df, pro_att_name, priv_class, reamining_cat_feat)
data1, att = data1.convert_to_dataframe()
data1 = data1.astype(int)


seed = 37
# df = df.drop(columns=['cons.price.idx', 'cons.conf.idx'])
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed)
data_train_2, data_test_2 = train_test_split(df, test_size = 0.3, random_state = seed)

data_orig_train_1, X_train_1, y_train_1 = load_bank_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_bank_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_bank_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_bank_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)

from sklearn.svm import SVC
model_1 = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
model_1 = model_1.fit(X_train_1, y_train_1)

sc = StandardScaler()
y2_X_train = sc.fit_transform(X_train_1)
y2_X_test = sc.fit_transform(X_test_1)

model_2 = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
model_2 = model_2.fit(X_train_2, y_train_2)

y_pred_1 = model_1.predict(X_test_1)
y_pred_2 = model_2.predict(X_test_2)
data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1

data_orig_train_2.features = X_train_2
data_orig_test_2.features = X_test_2
pro_att = 'age'
print("intial time",  start_time - time.time())
fair = get_fair_metrics_and_plot(data_orig_test_1, model_1)
print("model1", fair)

fair = get_fair_metrics_and_plot(data_orig_test_2, model_2)
print("model2", fair)

from fairness_permutation_importance import permutation_importance

import time
import sklearn
print(permutation_importance("SVC", data_orig_train_1, data_orig_test_1, "age", "DIR"))
print(permutation_importance("SVC", data_orig_train_2, data_orig_test_2, "age", "DIR"))

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