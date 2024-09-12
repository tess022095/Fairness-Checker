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
dir = 'res/titanic5-'
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
## Custom feature
df["Age"] = df["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)

df["CabinBool"] = (df["Cabin"].notnull().astype('int'))

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
df['Title'] = df['Title'].map(title_mapping)
df['Title'] = df['Title'].fillna(0)

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
for x in range(len(train["AgeGroup"])):
    if df["AgeGroup"][x] == "Unknown":
        df["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
df['AgeGroup'] = df['AgeGroup'].map(age_mapping)

df = df.fillna({"Embarked": "S"})
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
df['Embarked'] = df['Embarked'].map(embarked_mapping)
df['FareBand'] = pd.qcut(df['Fare'], 4, labels = [1, 2, 3, 4])


df = df.drop(['Fare'], axis = 1)
df = df.drop(['Cabin'], axis = 1)
df = df.drop(['Ticket'], axis = 1)
df = df.drop(['PassengerId'], axis = 1)
df = df.drop(['Name'], axis = 1)


y1_df['Age'].fillna(y1_df['Age'].median(), inplace = True)
y1_df = y1_df.fillna({"Embarked": "S"})
y1_df['Embarked'] = y1_df['Embarked'].map(embarked_mapping)
y1_df['Cabin'].fillna(y1_df['Cabin'].mode(), inplace = True)

# One-hot encoder
cat_feat = ['Cabin', 'Ticket', 'Embarked']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')


y1_df = y1_df.drop(['PassengerId'], axis = 1)
y1_df = y1_df.drop(['Name'], axis = 1)


# In[4]:


df = df.drop(columns=['CabinBool', 'Pclass'])
y1_df = y1_df.drop(columns=['Cabin=E25', 'Cabin=F38', 'Ticket=110564'])
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




from sklearn.ensemble import GradientBoostingClassifier
model_1 = GradientBoostingClassifier()
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = GradientBoostingClassifier()
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