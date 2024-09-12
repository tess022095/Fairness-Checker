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
dir = 'res/titanic3-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']



# In[2]:


# Load data
train = pd.read_csv('../../../data/titanic/train.csv')
test = pd.read_csv('../../../data/titanic/test.csv')
df = train


# In[3]:


def name_converted(feature):
    result = ''
    if feature in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col', 'Rev', 'Dona', 'Dr']:
        result = 'rare'
    elif feature in ['Ms', 'Mlle']:
        result = 'Miss'
    elif feature == 'Mme':
        result = 'Mrs'
    else:
        result = feature
    return result
def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a


# In[4]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

### Feature
y1_df = df.copy()

## Custom(feature)
df['title'] = [i.split('.')[0].split(',')[1].strip() for i in df.Name]
df.title = df.title.map(name_converted)

## Family_size seems like a good feature to create
df['family_size'] = df.SibSp + df.Parch+1
df['is_alone'] = [1 if i<2 else 0 for i in df.family_size]
df['family_group'] = df['family_size'].map(family_group)

df['calculated_fare'] = df.Fare/df.family_size
df['fare_group'] = df['calculated_fare'].map(fare_group)

df.drop(['Ticket'], axis=1, inplace=True)
df.drop(['PassengerId'], axis=1, inplace=True)
df.drop(['Name'], axis=1, inplace=True)


## Imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(missing_values=np.NaN)
df['Age'] = imputer.fit_transform(df[['Age']]).ravel()

df = pd.get_dummies(df, columns=['title',"Pclass", 'Cabin','Embarked', 'family_group', 'fare_group'], drop_first=False)
# y1_df = pd.get_dummies(y1_df, columns=['title',"Pclass", 'Cabin','Embarked', 'family_group', 'fare_group'], drop_first=False)

y1_df.drop(['Ticket'], axis=1, inplace=True)
y1_df.drop(['PassengerId'], axis=1, inplace=True)
y1_df.drop(['Name'], axis=1, inplace=True)

imputer = KNNImputer(missing_values=np.NaN)
y1_df['Age'] = imputer.fit_transform(y1_df[['Age']]).ravel()

y1_df = pd.get_dummies(y1_df, columns=["Pclass", 'Cabin','Embarked'], drop_first=False)
# y1_df = pd.get_dummies(y1_df, columns=['title',"Pclass", 'Cabin','Embarked', 'family_group', 'fare_group'], drop_first=False)


# In[5]:


y1_df = y1_df.drop(columns=['Cabin_E63', 'Cabin_C111'])
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




from sklearn.ensemble import BaggingClassifier
model_1 = BaggingClassifier(estimator=None, bootstrap=True, bootstrap_features=False,
                            max_features=1.0, max_samples=1.0, n_estimators=180,
                            n_jobs=None, oob_score=False, random_state=None, verbose=0,
                            warm_start=False)
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = BaggingClassifier(estimator=None, bootstrap=True, bootstrap_features=False,
                            max_features=1.0, max_samples=1.0, n_estimators=180,
                            n_jobs=None, oob_score=False, random_state=None, verbose=0,
                            warm_start=False)
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