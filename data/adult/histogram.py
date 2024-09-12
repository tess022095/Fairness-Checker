import sys

from aif360.algorithms.preprocessing import DisparateImpactRemover

sys.path.append('../../../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
import time
start_time = time.time()
dir = 'res/adult1-'

d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'

train_path = '../../../data/adult/adult.data'
test_path = '../../../data/adult/adult.test'

column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income-per-year']
na_values = ['?']

train = pd.read_csv(train_path, header=None, names=column_names,
                    skipinitialspace=True, na_values=na_values)
test = pd.read_csv(test_path, header=0, names=column_names,
                   skipinitialspace=True, na_values=na_values)

df_good = pd.concat([test, train], ignore_index=True)

##### Drop na values
dropped = df_good.dropna()
count = df_good.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df_good = dropped
print("--- %s drop: ---" % (time.time() - start_time))
df_bad = df_good.copy()
# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
df_bad = pd.get_dummies(df_bad, columns=cat_feat, prefix_sep='=')

## Implement label encoder instead of one-hot encoder
for feature in cat_feat:
    le = LabelEncoder()
    df_good[feature] = le.fit_transform(df_good[feature])
seed = randrange(100)

# df_good['income-per-year'] = df_good['income-per-year'].map({'>50K': 1, '>50K.': 1, '<=50K': 0, '<=50K.': 0})
#
# feature = 'age'
# label = 'income-per-year'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 3: Plot the histogram
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df_good, x=feature, hue=label, multiple="stack", palette="viridis", kde=True)
# plt.title(f'Histogram of {feature} grouped by {label}')
# plt.xlabel(feature)
# plt.ylabel('Frequency')
# plt.show()
pro_att_name = ['race']  # ['race', 'sex']
priv_class = ['White']  # ['White', 'Male']
reamining_cat_feat = []
good_data_orig, good_X, good_y = load_adult_data(df_good, pro_att_name, priv_class, reamining_cat_feat)
df, attr = good_data_orig.convert_to_dataframe()
print(good_data_orig.feature_names)
from scipy.stats import chi2_contingency

# Example: Education Level
# contingency_table = pd.crosstab(df['race'], df['workclass'])
# chi2, p_value, _, _ = chi2_contingency(contingency_table)
# print(f"Chi-Square Statistic: {chi2}, P-Value: {p_value}")
# print(df)
# feature = 'age'
# label = 'income-per-year'
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x=feature, hue=label, multiple="stack", palette="viridis", kde=True)
# plt.title(f'Histogram of {feature} grouped by {label}')
# plt.xlabel(feature)
# plt.ylabel('Frequency')
# plt.show()
# plt.figure(figsize=(10, 6))
#
# for column in df.columns:
#     plt.hist(df[column], bins=5, alpha=0.5, label=column)
#
# plt.legend(loc='upper right')
# plt.title('Histograms of All Features')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
#
# plt.show()
# model = LogisticRegression()
# start_time = time.time()
DIR = DisparateImpactRemover()
data_transf_train = DIR.fit_transform(good_data_orig)
# Train and save the model_1
# rf_transf = model.fit(data_transf_train.features,
#                      data_transf_train.labels.ravel())

# fair = get_fair_metrics_and_plot(data_orig_test, rf_transf)
# print("DI", fair, start_time - time.time())


# sc = RobustScaler()
# trained = sc.fit(good_X)
# good_X_train = trained.transform(good_X)
# good_data_orig.features = good_X_train

df1, attr = data_transf_train.convert_to_dataframe()
contingency_table = pd.crosstab(df1['race'], df1['workclass'])
chi2, p_value, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}, P-Value: {p_value}")
feature = 'age'
label = 'income-per-year'
plt.figure(figsize=(10, 6))
sns.histplot(data=df1, x=feature, hue=label, multiple="stack", palette="viridis", kde=True)
plt.title(f'Histogram of {feature} grouped by {label}')
plt.xlabel(feature)
plt.ylabel('Frequency')
plt.show()
plt.savefig('foo.png')

# num_features = len(df.columns)
#
# # Create subplots
# fig, axes = plt.subplots(num_features, 1, figsize=(10, 5 * num_features))
#
# # Generate histograms for each feature
# for i, column in enumerate(df.columns):
#     axes[i].hist(df[column].dropna(), bins=30, edgecolor='k', alpha=0.7)
#     axes[i].set_title(f'Histogram of {column}')
#     axes[i].set_xlabel(column)
#     axes[i].set_ylabel('Frequency')
#
# # Adjust layout
# plt.tight_layout()
# plt.show()