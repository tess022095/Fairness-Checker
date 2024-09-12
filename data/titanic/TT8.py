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
dir = 'res/titanic8-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']


# In[2]:


import re 
from sklearn.ensemble import RandomForestRegressor
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',
                 'TicketNumber', 'Title','Pclass','FamilySize',
                 'FsizeD','NameLength',"NlengthD",'Deck']]
    # Split sets into train and test
    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model_1
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model_1 to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df


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
df["Embarked"] = df["Embarked"].fillna('C')
def fill_missing_fare(df):
    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df
df=fill_missing_fare(df)

df["Deck"]=df.Cabin.str[0]
df.Deck.fillna('Z', inplace=True)

df["FamilySize"] = df["SibSp"] + df["Parch"]+1

df.loc[df["FamilySize"] == 1, "FsizeD"] = 'singleton'
df.loc[(df["FamilySize"] > 1)  &  (df["FamilySize"] < 5) , "FsizeD"] = 'small'
df.loc[df["FamilySize"] >4, "FsizeD"] = 'large'

df["NameLength"] = df["Name"].apply(lambda x: len(x))
bins = [0, 20, 40, 57, 85]
group_names = ['short', 'okay', 'good', 'long']
df['NlengthD'] = pd.cut(df['NameLength'], bins, labels=group_names)

df["TicketNumber"] = df["Ticket"].str.extract('(\d{2,})', expand=True)
df["TicketNumber"] = df["TicketNumber"].apply(pd.to_numeric)
df.TicketNumber.fillna(df["TicketNumber"].median(), inplace=True)

titles = df["Name"].apply(get_title)
df["Title"] = titles
# Titles with very low cell counts to be combined to "rare" level
rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
df.loc[df["Title"] == "Mlle", "Title"] = 'Miss'
df.loc[df["Title"] == "Ms", "Title"] = 'Miss'
df.loc[df["Title"] == "Mme", "Title"] = 'Mrs'
df.loc[df["Title"] == "Dona", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Lady", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Countess", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Capt", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Col", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Don", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Major", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Rev", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Sir", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Jonkheer", "Title"] = 'Rare Title'
df.loc[df["Title"] == "Dr", "Title"] = 'Rare Title'

labelEnc=LabelEncoder()
cat_vars=['Embarked','Sex',"Title","FsizeD","NlengthD",'Deck']
for col in cat_vars:
    df[col]=labelEnc.fit_transform(df[col])

df=fill_missing_age(df)

drop_column = ['PassengerId', 'Cabin', 'Ticket', 'Name', 'Parch']
df.drop(drop_column, axis=1, inplace = True)
    

std_scale = StandardScaler().fit(df[['Age', 'Fare']])
df[['Age', 'Fare']] = std_scale.transform(df[['Age', 'Fare']])


y1_df['Age'].fillna(y1_df['Age'].median(), inplace = True)
y1_df = y1_df.fillna({"Embarked": "S"})
y1_df['Cabin'].fillna(y1_df['Cabin'].mode(), inplace = True)

# One-hot encoder
cat_feat = ['Cabin', 'Ticket', 'Embarked']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')

y1_df = y1_df.drop(['PassengerId'], axis = 1)
y1_df = y1_df.drop(['Name'], axis = 1)


# In[5]:



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




model_1 = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9, min_samples_split=6, min_samples_leaf=4)
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9, min_samples_split=6, min_samples_leaf=4)
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