#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
from xgboost import XGBClassifier
import time
start_time = time.time()
dir = 'res/titanic2-'
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
## Feature Engineering
df['Family'] =  df["Parch"] + df["SibSp"]
df['Family'].loc[df['Family'] > 0] = 1
df['Family'].loc[df['Family'] == 0] = 0
df = df.drop(['SibSp','Parch'], axis=1)
df = df.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

y1_df = y1_df.drop(['PassengerId','Name'], axis=1)


# In[4]:


# Missing value
average_age_titanic   = y1_df["Age"].mean()
std_age_titanic       = y1_df["Age"].std()
count_nan_age_titanic = y1_df["Age"].isnull().sum()
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
y1_df["Age"][np.isnan(y1_df["Age"])] = rand_1

y1_df["Embarked"] = y1_df["Embarked"].fillna("S")
y1_df["Fare"].fillna(y1_df["Fare"].median(), inplace=True)

y1_df['Fare'] = y1_df['Fare'].astype(int)
y1_df[ 'Cabin' ] = y1_df.Cabin.fillna( 'U' )
y1_df[ 'Ticket' ] = y1_df.Ticket.fillna( 'X' )
y1_df = y1_df.dropna()
# One-hot encoder
cat_feat = ['Embarked', 'Ticket', 'Cabin']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')


average_age_titanic   = df["Age"].mean()
std_age_titanic       = df["Age"].std()
count_nan_age_titanic = df["Age"].isnull().sum()
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
df["Age"][np.isnan(df["Age"])] = rand_1

df["Embarked"] = df["Embarked"].fillna("S")
df["Fare"].fillna(df["Fare"].median(), inplace=True)

df['Fare'] = df['Fare'].astype(int)
df = df.dropna()
# One-hot encoder
cat_feat = ['Embarked']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


# In[5]:


df = df.drop(columns=['Pclass'])
y1_df = y1_df.drop(columns=['Ticket=31418', 'Ticket=237671', 'Ticket=11753'])
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

model_1 = RandomForestClassifier(n_estimators=100)
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = RandomForestClassifier(n_estimators=100)
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
print(permutation_importance("bagging", data_orig_train_1, data_orig_test_1, "Sex", "DIR"))

print(permutation_importance("bagging", data_orig_train_2, data_orig_test_2, "Sex", "DIR"))
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