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
dir = 'res/titanic4-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']



# In[2]:


# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'


# In[3]:


# Load data
train = pd.read_csv('../../../data/titanic/train.csv')
test = pd.read_csv('../../../data/titanic/test.csv')
df = train


# In[4]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

## Imputation
df[ 'Age' ] = df.Age.fillna( df.Age.mean() )
df[ 'Fare' ] = df.Fare.fillna( df.Fare.mean() )
## filna(-1)

    
## Custom(feature)
title = pd.DataFrame()
title[ 'Title' ] = df[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
title[ 'Title' ] = title.Title.map( Title_Dictionary )
df[ 'Title' ] = title[ 'Title' ]
df[ 'Ticket' ] = df[ 'Ticket' ].map( cleanTicket )
df[ 'Cabin' ] = df.Cabin.fillna( 'U' )
df[ 'FamilySize' ] = df[ 'Parch' ] + df[ 'SibSp' ] + 1
df[ 'Family_Single' ] = df[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
df[ 'Family_Small' ]  = df[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
df[ 'Family_Large' ]  = df[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )


# Basic
# One-hot encoder
cat_feat = ['Title', 'Ticket', 'Cabin'] #   'Ticket', 'Embarked'
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

drop_column = ['Embarked', 'PassengerId', 'Name']
df.drop(drop_column, axis=1, inplace = True)

# Basic
# One-hot encoder
# cat_feat = ['Ticket', 'Cabin'] #   'Ticket', 'Embarked'
# y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')

# drop_column = ['Embarked', 'PassengerId', 'Name']
# y1_df.drop(drop_column, axis=1, inplace = True)


# In[5]:

print(df)
# df = df.drop(columns=['Fare'])
seed = randrange(100)
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']
data_train_2, data_test_2 = train_test_split(df, test_size = 0.3, random_state = seed) #


pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

data_orig_train_1, X_train_1, y_train_1 = load_titanic_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_titanic_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_titanic_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_titanic_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)


# FeatureSelection
from sklearn.feature_selection import RFECV
model = LogisticRegression()
rfecv = RFECV( estimator = model , step = 1 , cv = 2 , scoring = 'accuracy' )
trained_rfecv = rfecv.fit(X_train_1, y_train_1)
X_train_1 = trained_rfecv.transform(X_train_1)
X_test_1 = trained_rfecv.transform(X_test_1)
data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1



model_1 = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                             intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                             penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                             verbose=0, warm_start=False)

model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                             intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                             penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                             verbose=0, warm_start=False)

model_2 = model_2.fit(X_train_2, y_train_2)
data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1

data_orig_train_2.features = X_train_2
data_orig_test_2.features = X_test_2
pro_att = 'Sex'
print("intial time",  start_time - time.time())
import sklearn

# print("Accuracy: ", sklearn.metrics.accuracy_score(y_test_1, y_pred_1))
# print("DI: ", disparate_impact(data_orig_test_1, y_pred_1, pro_att))
# print("SPD: ", statistical_parity_difference(data_orig_test_1, y_pred_1, pro_att))
# print("EOD: ", equal_opportunity_difference(data_orig_test_1, y_pred_1, y_test_1, pro_att))
# print("AOD: ", average_odds_difference(data_orig_test_1, y_pred_1, y_test_1, pro_att))
#
# print("Accuracy: ", sklearn.metrics.accuracy_score(y_test_2, y_pred_2))
# print("DI: ", disparate_impact(data_orig_test_2, y_pred_2, pro_att))
# print("SPD: ", statistical_parity_difference(data_orig_test_2, y_pred_2, pro_att))
# print("EOD: ", equal_opportunity_difference(data_orig_test_2, y_pred_2, y_test_2, pro_att))
# print("AOD: ", average_odds_difference(data_orig_test_2, y_pred_2, y_test_2, pro_att))
fair = get_fair_metrics_and_plot(data_orig_test_1, model_1)
print("model1", fair)

fair = get_fair_metrics_and_plot(data_orig_test_2, model_2)
print("model2", fair)
from fairness_permutation_importance import permutation_importance

from fairness_permutation_importance import permutation_importance
import time
print(permutation_importance("lrg", data_orig_train_1, data_orig_test_1, "Sex", "DIR"))

print(permutation_importance("lrg", data_orig_train_2, data_orig_test_2, "Sex", "DIR"))

import sklearn
from sklearn.inspection import permutation_importance
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, \
    equalized_odds_difference, equalized_odds_ratio
def DIScorer(y_true, y_pred):
    df, feature = data_orig_train_1.convert_to_dataframe()
    attr = df["Sex"]
    return abs(demographic_parity_difference(y_true, y_pred, sensitive_features=attr))
#sklearn make_scorer for the churnScorer
di = sklearn.metrics.make_scorer(DIScorer, greater_is_better = False)

def EODScorer(y_true, y_pred):
    df, feature = data_orig_train_1.convert_to_dataframe()
    attr = df["Sex"]
    return abs(equalized_odds_difference(y_true, y_pred, sensitive_features=attr))
#sklearn make_scorer for the churnScorer
eod = sklearn.metrics.make_scorer(EODScorer, greater_is_better = False)

result = permutation_importance(model_1, X_train_1, y_train_1, scoring=di, n_repeats=10,
                                random_state=0)#'Title=Mme', 'Pclass', 'Title=Mr'
start_time = time.time()
result = permutation_importance(model_1, X_train_1, y_train_1, scoring=eod, n_repeats=10,
                                random_state=0) #'Title=Mr', 'Deck=D'
print("---Num3---", (time.time() - start_time))

start_time = time.time()
result = permutation_importance(model_2, X_train_2, y_train_2, scoring=eod, n_repeats=10,
                                random_state=0)
print("---Num4---", (time.time() - start_time))