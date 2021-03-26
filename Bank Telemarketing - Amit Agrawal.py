# =============================================================================
# Import all necessary libraries
# =============================================================================

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# Import the raw data
# =============================================================================

raw_data = pd.read_excel(r'C:\Users\Amit Agrawal\Dropbox\QL Case Study\bank-additional-full.xlsx')
raw_data.info()

# Rename fields

raw_data.rename(columns = {'cons.price.idx':'cons_price_idx', 'cons.conf.idx':'cons_conf_idx', 'emp.var.rate':'emp_var_rate', 'nr.employed':'nr_employed'}, inplace = True)

# =============================================================================
# Scoping
# =============================================================================

# We will be building the model as a decision making tool before campaign starts, hence not use the featues related to current campaign

raw_data = raw_data.drop(['contact', 'month', 'day_of_week', 'duration', 'campaign'], axis = 1)
raw_data.dtypes

# =============================================================================
# Exploratory Data Analysis
# =============================================================================

# Univariate distribution of dependent variable

print(raw_data['y'].value_counts())
raw_data['y'].value_counts().plot.bar()

# Distribution of all numeric variables

raw_data[['age', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m']].describe()

# Distribution of all categorical variables

print(raw_data['job'].value_counts())
raw_data['job'].value_counts().plot.bar()

print(raw_data['marital'].value_counts())
raw_data['marital'].value_counts().plot.bar()

print(raw_data['education'].value_counts())
raw_data['education'].value_counts().plot.bar()

print(raw_data['default'].value_counts())
raw_data['default'].value_counts().plot.bar()

print(raw_data['housing'].value_counts())
raw_data['housing'].value_counts().plot.bar()

print(raw_data['loan'].value_counts())
raw_data['loan'].value_counts().plot.bar()

print(raw_data['pdays'].value_counts())
raw_data['pdays'].value_counts().plot.bar()

print(raw_data['previous'].value_counts())
raw_data['previous'].value_counts().plot.bar()

print(raw_data['poutcome'].value_counts())
raw_data['poutcome'].value_counts().plot.bar()

# Outlier treatment

print(raw_data['age'].plot.hist())
age_p95 = raw_data['age'].quantile(0.95)
print(age_p95)
raw_data.age.loc[raw_data.age > age_p95]  = age_p95
print(raw_data['age'].plot.hist())

print(raw_data['cons_price_idx'].plot.hist())

print(raw_data['cons_conf_idx'].plot.hist())

print(raw_data['euribor3m'].plot.hist())
euribor3m_p05 = raw_data['euribor3m'].quantile(0.05)
print(euribor3m_p05)
raw_data.euribor3m.loc[raw_data.euribor3m < euribor3m_p05]  = euribor3m_p05
print(raw_data['euribor3m'].plot.hist())

# Correlation check

corrMatrix = raw_data.corr()
print (corrMatrix)
sn.heatmap(corrMatrix, annot=True)

# =============================================================================
# Data Preprocessing
# =============================================================================

# Create classifier

#Create the classifier
df_one = pd.get_dummies(raw_data["y"]) 

# Binary Data is Concatenated into Dataframe 
df_two = pd.concat((df_one, raw_data), axis=1) 

# We want no =0 and yes =1 So we drop no colunm here 
df_two = df_two.drop(["no"], axis=1) 

# Rename the Column 
df_classifier = df_two.rename(columns={"yes": "classifier"}) 
df_classifier
df_classifier['classifier'] = df_classifier['classifier'].astype(str)
df_classifier.dtypes

# Create previous flag

df_classifier['previous_flag'] = np.where(df_classifier['previous'] > 0, 1, 0 )
df_classifier['previous_flag'] = df_classifier['previous_flag'].astype(str)

# Feature engineering

df_classifier.loc[df_classifier['job'] == 'housemaid', 'job_seg'] = 'employed'
df_classifier.loc[df_classifier['job'] == 'services', 'job_seg'] = 'employed'
df_classifier.loc[df_classifier['job'] == 'admin.', 'job_seg'] = 'employed'
df_classifier.loc[df_classifier['job'] == 'blue-collar', 'job_seg'] = 'employed'
df_classifier.loc[df_classifier['job'] == 'technician', 'job_seg'] = 'employed'
df_classifier.loc[df_classifier['job'] == 'management', 'job_seg'] = 'employed'
df_classifier.loc[df_classifier['job'] == 'self-employed', 'job_seg'] = 'employed'
df_classifier.loc[df_classifier['job'] == 'entrepreneur', 'job_seg'] = 'employed'
df_classifier.loc[df_classifier['job'] == 'retired', 'job_seg'] = 'retired'
df_classifier.loc[df_classifier['job'] == 'unemployed', 'job_seg'] = 'unemployed'
df_classifier.loc[df_classifier['job'] == 'unknown', 'job_seg'] = 'unknown'
df_classifier.loc[df_classifier['job'] == 'student', 'job_seg'] = 'student'

df_classifier.loc[df_classifier['education'] == 'basic.4y', 'edu_seg'] = 'basic'
df_classifier.loc[df_classifier['education'] == 'basic.6y', 'edu_seg'] = 'basic'
df_classifier.loc[df_classifier['education'] == 'basic.9y', 'edu_seg'] = 'basic'
df_classifier.loc[df_classifier['education'] == 'high.school', 'edu_seg'] = 'high_school'
df_classifier.loc[df_classifier['education'] == 'professional.course', 'edu_seg'] = 'higher_studies'
df_classifier.loc[df_classifier['education'] == 'university.degree', 'edu_seg'] = 'higher_studies'
df_classifier.loc[df_classifier['education'] == 'unknown', 'edu_seg'] = 'unknown'
df_classifier.loc[df_classifier['education'] == 'illiterate', 'edu_seg'] = 'illiterate'

# Create the input data for logistics regression modelling

model_df = df_classifier[['age', 'job_seg', 'marital', 'edu_seg', 'default', 'housing', 'loan', 'poutcome', 'classifier', 'previous_flag', 'cons_price_idx', 'cons_conf_idx', 'euribor3m']]

# Create dummy variables

model_df = pd.get_dummies(model_df, columns =['job_seg', 'marital', 'edu_seg', 'default', 'housing', 'loan', 'poutcome'])
model_df = model_df.apply(pd.to_numeric)
model_df.dtypes

# =============================================================================
# Logistic Forecast Model
# =============================================================================

# Run Logistics Regression to identify the significant predictors using smf
# Drop all unknown columns from all categorical fields
lr_model= smf.logit(formula="classifier~ age + previous_flag + cons_price_idx + cons_conf_idx + euribor3m + C(job_seg_employed) + C(job_seg_retired) + C(job_seg_student) + C(job_seg_unemployed) + C(edu_seg_basic) + C(edu_seg_high_school) + C(edu_seg_higher_studies) + C(edu_seg_illiterate) + C(marital_divorced) + C(marital_married) + C(marital_single) + C(default_no) + C(default_yes) + C(housing_no) + C(housing_yes) + C(loan_no) + C(loan_yes) + C(poutcome_failure) + C(poutcome_success)", data = model_df).fit()
lr_model.summary()

# Create model using sklearn

# Keep only significant factors based on the p values (<=0.05) of the above results and split the data into x and y datasets

y_data = model_df['classifier']
x_data = model_df[['job_seg_retired', 'job_seg_student', 'edu_seg_basic' , 'default_no', 'cons_price_idx', 'cons_conf_idx', 'euribor3m']]

# Split the data into training and test

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2)

# Create the Logistics Model
# There is class imbalance problem, hence using class_weight=balanced

model = LogisticRegression(class_weight='balanced')
model.fit(x_training_data, y_training_data)

model.coef_

# Train the model and create predictions

predictions = model.predict(x_test_data)

# Measure performance

print(classification_report(y_test_data, predictions))
print(confusion_matrix(y_test_data, predictions))
print(confusion_matrix)
accuracy_score(y_test_data, predictions)
balanced_accuracy_score(y_test_data, predictions)
precision_score(y_test_data, predictions)
recall_score(y_test_data, predictions)
roc_auc_score(y_test_data, predictions)
f1_score(y_test_data, predictions)

fpr, tpr, _ = metrics.roc_curve(y_test_data,  predictions)
auc = metrics.roc_auc_score(y_test_data, predictions)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# Check for p values of the selected variables

logit_model=sm.Logit(y_training_data,x_training_data)
result=logit_model.fit()
print(result.summary())

# =============================================================================
# Random Forest Model
# =============================================================================

# Split the data into x and y datasets

y_data = model_df['classifier']
x_data = model_df.drop('classifier', axis = 1)

# Split the data into training and test

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2)

# Create the Random Forest Classifier Model
# There is class imbalance problem, hence using class_weight=balanced

clf = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
            max_depth=3, max_features='auto', max_leaf_nodes=3,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=20, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=999, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
clf.fit(x_training_data,y_training_data)

# Train the model and create predictions

y_pred=clf.predict(x_test_data)

# Measure performance

confusion_matrix = pd.crosstab(y_test_data, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

print('Accuracy: ',metrics.accuracy_score(y_test_data, y_pred))

fpr, tpr, _ = metrics.roc_curve(y_test_data,  y_pred)
auc = metrics.roc_auc_score(y_test_data, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# Identify variables that have highest impact on the outcome

featureImportances = pd.Series(clf.feature_importances_).sort_values(ascending=False)
print(featureImportances)

sn.barplot(x=round(featureImportances,4), y=featureImportances)
plt.xlabel('Features Importance')
plt.show()

# Recreate the model and measure performance with only the top 6 most important factors

y_data = model_df['classifier']
x_data = model_df[['euribor3m', 'poutcome_success', 'cons_conf_idx' , 'cons_price_idx', 'previous_flag', 'poutcome_nonexistent']]

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2)

clf = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
            max_depth=4, max_features='auto', max_leaf_nodes=4,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=20, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=999, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
clf.fit(x_training_data,y_training_data)
y_pred=clf.predict(x_test_data)

confusion_matrix = pd.crosstab(y_test_data, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

print('Accuracy: ',metrics.accuracy_score(y_test_data, y_pred))
plt.show()

fpr, tpr, _ = metrics.roc_curve(y_test_data,  y_pred)
auc = metrics.roc_auc_score(y_test_data, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

featureImportances = pd.Series(clf.feature_importances_).sort_values(ascending=False)
print(featureImportances)

sn.barplot(x=round(featureImportances,4), y=featureImportances)
plt.xlabel('Features Importance')
plt.show()

# measure performance

accuracy_score(y_test_data, y_pred)
balanced_accuracy_score(y_test_data, y_pred)
precision_score(y_test_data, y_pred)
recall_score(y_test_data, y_pred)
f1_score(y_test_data, y_pred)
roc_auc_score(y_test_data, y_pred)