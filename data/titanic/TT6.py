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
dir = 'res/titanic6-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']



# In[2]:


# Load data
train = pd.read_csv('../../../data/titanic/train.csv')
test = pd.read_csv('../../../data/titanic/test.csv')
df = train


# In[3]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

y1_df = df.copy()

df['Child'] = df['Age']<=10
df['Cabin_known'] = df['Cabin'].isnull() == False
df['Age_known'] = df['Age'].isnull() == False
df['Family'] = df['SibSp'] + df['Parch']
df['Alone']  = (df['SibSp'] + df['Parch']) == 0
df['Large_Family'] = (df['SibSp']>2) | (df['Parch']>3)
df['Deck'] = df['Cabin'].str[0]
df['Deck'] = df['Deck'].fillna(value='U')
df['Ttype'] = df['Ticket'].str[0]
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
df['Fare_cat'] = pd.DataFrame(np.floor(np.log10(df['Fare'] + 1))).astype('int')
df['Bad_ticket'] = df['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])
df['Young'] = (df['Age']<=30) | (df['Title'].isin(['Master','Miss','Mlle']))
df['Shared_ticket'] = np.where(df.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)
df['Ticket_group'] = df.groupby('Ticket')['Name'].transform('count')
df['Fare_eff'] = df['Fare']/df['Ticket_group']
df['Fare_eff_cat'] = np.where(df['Fare_eff']>16.0, 2, 1)
df['Fare_eff_cat'] = np.where(df['Fare_eff']<8.5,0,df['Fare_eff_cat'])
df['Age'].fillna(df['Age'].median(), inplace = True)
df = df.fillna({"Embarked": "S"})

cat_feat = ['Embarked', 'Deck', 'Title', 'Ttype',]
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


drop_column = ['Name', 'Ticket', 'Fare', 'Cabin']
df.drop(drop_column, axis=1, inplace = True)



y1_df['Age'].fillna(y1_df['Age'].median(), inplace = True)
y1_df = y1_df.fillna({"Embarked": "S"})
y1_df['Cabin'].fillna(y1_df['Cabin'].mode(), inplace = True)

# One-hot encoder
cat_feat = ['Cabin', 'Ticket', 'Embarked']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')

y1_df = y1_df.drop(['PassengerId'], axis = 1)
y1_df = y1_df.drop(['Name'], axis = 1)


# In[4]:


df = df.drop(columns=['Pclass'])
y1_df = y1_df.drop(columns=['Pclass'])
seed = randrange(100)
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']
data_train_2, data_test_2 = train_test_split(y1_df, test_size = 0.3, random_state = seed) #

pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

data_orig_train_1, X_train_1, y_train_1 = load_titanic_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_titanic_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_titanic_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_titanic_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)

import xgboost as xgb
model_1 = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1)

model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1)

model_2 = model_2.fit(X_train_2, y_train_2)

y_pred_1 = model_1.predict(X_test_1)
y_pred_2 = model_2.predict(X_test_2)
data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1

data_orig_train_2.features = X_train_2
data_orig_test_2.features = X_test_2
pro_att = 'Sex'
print("intial time",  start_time - time.time())
fair = get_fair_metrics_and_plot(data_orig_test_1, model_1)
print("model1", fair)

fair = get_fair_metrics_and_plot(data_orig_test_2, model_2)
print("model2", fair)


from fairness_permutation_importance import permutation_importance
import time
print(permutation_importance("boosting", data_orig_train_1, data_orig_test_1, "Sex", "DIR"))

print(permutation_importance("boosting", data_orig_train_2, data_orig_test_2, "Sex", "DIR"))

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
