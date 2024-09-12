#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
dir = 'res/german6-'
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
df['age'] = df['age'].apply(lambda x: float(x >= 26))
df = german_custom_preprocessing(df)
feat_to_drop = ['personal_status']
df = df.drop(feat_to_drop, axis=1)

cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
# num_feat = ['residence_since', 'age', 'investment_as_income_percentage', 'credit_amount', 'number_of_credits', 'people_liable_for', 'month']


# In[3]:


##### Pipeline
# Labelencoder
# y2_df = df.copy()
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')
y1_df = df.copy()
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')
for f in cat_feat:
    label = LabelEncoder()
    df[f] = label.fit_transform(df[f])


# In[4]:


features = ['age', 'sex', 'employment', 'housing', 'savings',
       'number_of_credits', 'credit_amount', 'month', 'purpose', 'credit']

df = df[features]


# In[5]:


seed = 47
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
data_train_2, data_test_2 = train_test_split(y1_df, test_size = 0.3, random_state = seed) #

pro_att_name = ['age'] # ['sex', 'age']
priv_class = [1]
reamining_cat_feat = []

data_orig_train_1, X_train_1, y_train_1 = load_german_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_german_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_german_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_german_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)

y1_test_df = data_orig_test_1.copy()


# In[6]:


sc = StandardScaler()

trained_1 = sc.fit(X_train_1)
X_train_1 = trained_1.transform(X_train_1)
X_test_1 = trained_1.transform(X_test_1)

data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1


trained_2 = sc.fit(X_train_2)
X_train_2 = trained_2.transform(X_train_2)
X_test_2 = trained_2.transform(X_test_2)




# In[7]:


pca = PCA(n_components=None)

trained_1 = pca.fit(X_train_1)
X_train_1 = trained_1.transform(X_train_1)
X_test_1 = trained_1.transform(X_test_1)

trained_2 = pca.fit(X_train_2)
X_train_2 = trained_2.transform(X_train_2)
X_test_2 = trained_2.transform(X_test_2)

data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1

data_orig_train_2.features = X_train_2
data_orig_test_2.features = X_test_2
# In[8]:


model_1 = RandomForestClassifier()
model_2 = RandomForestClassifier()


model_1 = model_1.fit(X_train_1, y_train_1)
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

# print(permutation_importance("bagging", data_orig_train_1, data_orig_test_1, "sex", "DIR"))
# print(permutation_importance("bagging", data_orig_train_1, data_orig_test_1, "sex", "DID"))
# print(permutation_importance("bagging", data_orig_train_1, data_orig_test_1, "sex", "EOD"))
# print(permutation_importance("bagging", data_orig_train_1, data_orig_test_1, "sex", "EOR"))
#
# print(permutation_importance("bagging", data_orig_train_2, data_orig_test_2, "sex", "DIR"))
# print(permutation_importance("bagging", data_orig_train_2, data_orig_test_2, "sex", "DID"))
# print(permutation_importance("bagging", data_orig_train_2, data_orig_test_2, "sex", "EOD"))
# print(permutation_importance("bagging", data_orig_train_2, data_orig_test_2, "sex", "EOR"))

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