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
dir = 'res/titanic7-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']



# In[2]:


import string
def extract_surname(data):    
    
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families


# In[3]:


# Load data
train = pd.read_csv('../../../data/titanic/train.csv')
test = pd.read_csv('../../../data/titanic/test.csv')
df = train


# In[4]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

y1_df = df.copy()

## Custom(feature)
df['Embarked'] = df['Embarked'].fillna('S')
med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df['Fare'] = df['Fare'].fillna(med_fare)
df['Fare'] = pd.qcut(df['Fare'], 13, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
idx = df[df['Deck'] == 'T'].index
df.loc[idx, 'Deck'] = 'A'
df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')
df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')


df['Age'].fillna(df['Age'].median(), inplace = True)
df['Age'] = pd.cut(df['Age'].astype(int), 10)
df['Age'].fillna(df['Age'].mode(), inplace = True)
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df['Family_Size_Grouped'] = df['Family_Size'].map(family_map)
df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
df['Title'] = df['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df['Title'] = df['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')
df['Family'] = extract_surname(df['Name'])


# ### Encoding
# ## LabelEncoder

non_numeric_features = ['Age', 'Embarked', 'Deck', 'Title', 'Family', 'Family_Size_Grouped']
for feature in non_numeric_features:        
    df[feature] = LabelEncoder().fit_transform(df[feature])

# df.drop(['Cabin'], inplace=True, axis=1)
drop_column = ['PassengerId', 'Cabin', 'Name', 'Ticket']
df.drop(drop_column, axis=1, inplace = True)


# cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


y1_df[ 'Cabin' ] = y1_df.Cabin.fillna( 'U' )
y1_df['Age'].fillna(y1_df['Age'].median(), inplace = True)
y1_df['Embarked'] = y1_df['Embarked'].fillna('S')
non_numeric_features = ['Embarked', 'Cabin']

for feature in non_numeric_features:        
    y1_df[feature] = LabelEncoder().fit_transform(y1_df[feature])
    
    
drop_column = ['PassengerId', 'Name', 'Ticket']
y1_df.drop(drop_column, axis=1, inplace = True)


# In[5]:



seed = randrange(100)
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']
data_train_2, data_test_2 = train_test_split(y1_df, test_size = 0.3, random_state = seed) #

pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

data_orig_train_1, X_train_1, y_train_1 = load_titanic_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_titanic_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)
y2_test_orig = data_orig_test_1.copy()

data_orig_train_2, X_train_2, y_train_2 = load_titanic_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_titanic_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)



sc = StandardScaler()
X_train_1 = sc.fit_transform(X_train_1)
X_test_1 = sc.fit_transform(X_test_1)
data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1


X_train_2 = sc.fit_transform(X_train_2)
X_test_2 = sc.fit_transform(X_test_2)
data_orig_train_2.features = X_train_2
data_orig_test_2.features = X_test_2

model_1 = RandomForestClassifier(criterion='gini',
                                 n_estimators=1750,
                                 max_depth=7,
                                 min_samples_split=6,
                                 min_samples_leaf=6,
                                 oob_score=True,
                                 random_state=42,
                                 n_jobs=-1,
                                 verbose=1)
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = RandomForestClassifier(criterion='gini',
                                 n_estimators=1750,
                                 max_depth=7,
                                 min_samples_split=6,
                                 min_samples_leaf=6,
                                 oob_score=True,
                                 random_state=42,
                                 n_jobs=-1,
                                 verbose=1)
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
