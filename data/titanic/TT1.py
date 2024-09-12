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
dir = 'res/titanic1-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']



# In[2]:


# Load data
train = pd.read_csv('../../../data/titanic/train.csv')
test = pd.read_csv('../../../data/titanic/test.csv')
df = train


# In[3]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

### Missing value
df['Age'].fillna(df['Age'].median(), inplace = True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
df['Fare'].fillna(df['Fare'].median(), inplace = True)
df = df.dropna()

### Feature engineering
## Drop feature

drop_column = ['PassengerId', 'Cabin', 'Ticket', 'Name', 'Parch']
## generation
df['FamilySize'] = df ['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 1
df['IsAlone'].loc[df['FamilySize'] > 1] = 0
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (df['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
df['Title'] = df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

df.drop(drop_column, axis=1, inplace = True)

### Encoding
y1_df = df.copy()
## Binning
df['Fare'] = pd.qcut(df['Fare'], 4)
df['Age'] = pd.cut(df['Age'].astype(int), 5)

## LabelEncoder
label = LabelEncoder()
df['Embarked'] = label.fit_transform(df['Embarked'])
df['Title'] = label.fit_transform(df['Title'])
df['Age'] = label.fit_transform(df['Age'])
df['Fare'] = label.fit_transform(df['Fare'])

## One-hot encoder
cat_feat = ['Title', 'Embarked']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')


# In[4]:


df.drop(columns=['Pclass'])
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




from xgboost import XGBClassifier

model_1 = XGBClassifier(learning_rate= 0.01, max_depth= 4, n_estimators= 300, seed= 0)
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = XGBClassifier(learning_rate= 0.01, max_depth= 4, n_estimators= 300, seed= 0)
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