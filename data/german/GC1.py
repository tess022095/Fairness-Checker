#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/german1-'
import time
start_time = time.time()
# Path(dir).mkdir(parents=True, exist_ok=True)

d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']


# In[2]:


filepath = '../../../data/german/german.data'
column_names = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit']
na_values=[]
df = pd.read_csv(filepath, sep=' ', header=None, names=column_names,na_values=na_values)
# df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
df = german_custom_preprocessing(df)
feat_to_drop = ['personal_status']
df = df.drop(feat_to_drop, axis=1)

cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

df = df.drop(columns=['purpose=A45'])
seed = 15
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
data_train_2, data_test_2 = train_test_split(df, test_size = 0.3, random_state = seed) #

pro_att_name = ['sex'] # ['sex', 'age']
priv_class = [1]
reamining_cat_feat = []

data_orig_train_1, X_train_1, y_train_1 = load_german_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_german_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_german_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_german_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)

y1_test_df = data_orig_test_2.copy()
## Feature engineering
# pca = PCA(n_components=3)
#
# trained = pca.fit(X_train_1)
# X_train_1 = trained.transform(X_train_1)
# X_test_1 = trained.transform(X_test_1)
#
# X_train_2 = trained.transform(X_train_2)
# X_test_2 = trained.transform(X_test_2)
#
# data_orig_train_1.features = X_train_1
# data_orig_test_1.features = X_test_1
#
# data_orig_train_2.features = X_train_2
# data_orig_test_2.features = X_test_2
#
# trained_2 = pca.fit(X_train_2)
# X_train_2 = trained_2.transform(X_train_2)
# X_test_2 = trained_2.transform(X_test_2)
# data_orig_train_2.features = X_train_2
# data_orig_test_2.features = X_test_2

# from sklearn.feature_selection import SelectKBest
# skb = SelectKBest(k=2)
#
# trained_1 = skb.fit(X_train_1, y_train_1)
# X_train_1 = trained_1.transform(X_train_1)
# X_test_1 = trained_1.transform(X_test_1)
# data_orig_train_1.features = X_train_1
# data_orig_test_1.features = X_test_1


#Seting the Hyper Parameters
param_grid = {"max_depth": [3,5, 7, 10,None],
              "n_estimators":[3,5,10,25,50,150],
              "max_features": [2,4,7,15,20]}

model_1 = GridSearchCV(RandomForestClassifier(), param_grid={}, cv=5, scoring='recall', verbose=0)
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = GridSearchCV(RandomForestClassifier(), param_grid={}, cv=5, scoring='recall', verbose=0)
model_2 = model_2.fit(X_train_2, y_train_2)

y_pred_1 = model_1.predict(X_test_1)
y_pred_2 = model_2.predict(X_test_2)
data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1

data_orig_train_2.features = X_train_2
data_orig_test_2.features = X_test_2
pro_att = 'sex'
print("intial time",  start_time - time.time())
fair = get_fair_metrics_and_plot(data_orig_test_1, model_1)
print("model1", fair)

fair = get_fair_metrics_and_plot(data_orig_test_2, model_2)
print("model2", fair)
from fairness_permutation_importance import permutation_importance

import time
print(permutation_importance("bagging", data_orig_train_1, data_orig_test_1, "sex", "DIR"))

print(permutation_importance("bagging", data_orig_train_2, data_orig_test_2, "sex", "DIR"))

import sklearn
from sklearn.inspection import permutation_importance
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, \
    equalized_odds_difference, equalized_odds_ratio
def DIScorer(y_true, y_pred):
    df, feature = data_orig_train_1.convert_to_dataframe()
    attr = df["sex"]
    return abs(demographic_parity_difference(y_true, y_pred, sensitive_features=attr))
#sklearn make_scorer for the churnScorer
di = sklearn.metrics.make_scorer(DIScorer, greater_is_better = False)

def EODScorer(y_true, y_pred):
    df, feature = data_orig_train_1.convert_to_dataframe()
    attr = df["sex"]
    return abs(equalized_odds_difference(y_true, y_pred, sensitive_features=attr))
#sklearn make_scorer for the churnScorer
eod = sklearn.metrics.make_scorer(EODScorer, greater_is_better = False)
start_time = time.time()
result = permutation_importance(model_1, X_train_1, y_train_1, scoring=di, n_repeats=10,
                                random_state=0)
print("---Num3---", (time.time() - start_time))


start_time = time.time()
result = permutation_importance(model_2, X_train_2, y_train_2, scoring=eod, n_repeats=10,
                                random_state=0)#'purpose=A45'
print("---Num4---", (time.time() - start_time))