#%% [markdown]

# DATS 6103, Intro to Data Mining, Fall 2021
# Team 3 Final Project Code and Analysis
# Julia Jin, Eric Newman, Alice Yang

#%% [markdown]

# Sections:
# 1. Data import and cleaning (line 22)
# 2. EDA (line 91)
#   a. Basic EDA - plots (line 94)
#   b. PCA, unsupervised learning (line 127)
# 3. Feature selection (line 353)
#   a. Lasso regression (line )
#   b. Logistic regression (line 518)
# 4. Classification models: logistic regression, KNN, SVM
# 5. Model comparison
# 6. Results and conclusion

#%% [markdown]
# Section 1: Data Import and Cleaning

#%%
# import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
# ARRAY based
# X = sm.add_constant(X)
# model = sm.glm(y, X, family)

from statsmodels.formula.api import glm
# FORMULA based (we had been using this for ols)
# model = glm(formula, data, family)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# read in file
bank_df = pd.read_csv('bank-additional-full.csv', sep = ';', engine = 'python')

# basic data checks
dfChkBasics(bank_df)

# %%
# data dictionary
# data source: https://www.kaggle.com/henriqueyamahata/bank-marketing

## bank client data:
# age: age of individual
# job: job
# marital: marital status (note: divorced = divorced or widowed)
# education: highest level of education
# default: has credit in default?
# housing: has housing loan
# loan: has personal loan

## related to last contact of current campaign:
# contact: contact communication type
# month: last contact month of year
# day_of_week: day of week (mon, tues, wed, etc.) last contact was made
# duration: last contact duration in seconds (note: highly affects output target; if duration = 0 then y = no)

## other attributes:
# campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# previous: number of contacts performed before this campaign and for this client (numeric)
# poutcome: outcome of the previous marketing campaign

## social and economic context attributes:
# emp.var.rate: employment variation rate - quarterly indicator
# cons.price.idx: consumer price index - monthly indicator
# euribor3m: euribor 3 month rate
# nr.employed: number of employees - quarterly indicator

## output variable - desired target
# y: has the client subscribed a term deposit? (binary: 'yes', 'no')

#%% 
# transform response variable to numeric (0 = "no", 1 = "yes")
bank_df['y'] = bank_df['y'].replace(to_replace=['no', 'yes'], value=[0, 1])
bank_df.head()

#%% [markdown]
# Section 2: EDA

#%%
# basic eda around demographics
# looking for any obvious differences in target output by demographic variable

# stacked histogram by age
yes = bank_df[bank_df['y'] == 'yes']
no = bank_df[bank_df['y'] == 'no']

fig, ax = plt.subplots()
ax.hist(yes['age'], label = 'yes', histtype = 'step', color = 'blue')
ax.hist(no['age'], label = 'no', histtype = 'step', color = 'red')

ax.set_xlabel('Age')
ax.set_ylabel('Number of People')
ax.legend()
plt.title('Histogram of Ages by Campaign Result')

plt.show()
# majority of people did not say yes to deposit
# maybe yes results tend to be younger? need to show as proportions - we can use box plots for this

#%%
ax = sns.boxplot(x = 'y', y = 'age', data = bank_df)
# doesn't appear to be huge difference, but those who said yes did have larger IQR of age

#%%
# examine job
print(bank_df['job'].value_counts())


# %%
sns.pairplot(bank_df)

#%% [markdown]
# Principal Components Analysis

# %%
# use only numeric variables - keep y for coloring
bankdf_num = bank_df[['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']]


# %%
# scale data
from sklearn.preprocessing import StandardScaler

features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

x = bankdf_num.loc[:, features].values
y = bankdf_num.loc[:, ['y']].values

x = StandardScaler().fit_transform(x)

# %%
# pca projection
from sklearn.decomposition import PCA
import pandas as pd

pca = PCA()

pcomp = pca.fit_transform(x)

pdf = pd.DataFrame(data = pcomp, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10'])

# %%
# concat in the target y variable for plotting
pdf_final = pd.concat([pdf, bankdf_num[['y']]], axis = 1)


# %%
# want to plot out cumulative proportion of variance explained by each principal component so we know how many we need
# use scree plot
import numpy as np
import matplotlib.pyplot as plt

varpropcuml = np.cumsum(pca.explained_variance_ratio_)

# %%
# scree plot
pc_values = np.arange(pca.n_components_) + 1

plt.plot(pc_values, varpropcuml, 'mo-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.show()

print(varpropcuml)

# slow increase in variance explained by principal component until either pc6 or pc7
# pcs 1-6 explain 91.5% of the variance - could maybe even get away with 4 or 5, but need to at least plot 1, 2, 3, and 4

#%%
# display the loadings for reference to interpret the PC plots below
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'], index=features)
loadings

# %%
# pc plots
plt.rcParams['legend.title_fontsize'] = 14

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 2', fontsize = 15)
ax.set_title('PC2 vs. PC1', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc1']
               , pdf_final.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)

# PC1 has strong loadings from euribor3m, emp.var.rate, and nr.employed
# PC2 has strong loadings from pdays, previous, and cons.conf.idx
# PC1 contains variance in 3 big economic indicators, PC2 has strong loadings from 2 previous campaign indicators & consumer confidence index
# Significant overlap, but some separation when PC1 economic variation more extreme - fewer people sign up under poor economic conditions
# Small cluster of nos when economic conditions ok but campaign numbers higher - too many calls?


# %%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 3', fontsize = 15)
ax.set_title('PC3 vs. PC1', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc1']
               , pdf_final.loc[indicesToKeep, 'pc3']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)

# Again, a lot of overlap
# Distinction when PC1 low and PC3 high
# PC3 has strongest loadings for age and consumer confidence index
# So when economic conditions poor and age and consumer confidence index high, more nos - focus on young people when economic conditions poor?

# %%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 4', fontsize = 15)
ax.set_title('PC4 vs. PC1', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc1']
               , pdf_final.loc[indicesToKeep, 'pc4']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)

# PC4 strong loads of duration of the call and # contacts of the campaign
# When economic conditions poor, if duration of call low and # contacts in campaign low, almost no chance of a yes

# %%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 5', fontsize = 15)
ax.set_title('PC5 vs. PC1', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc1']
               , pdf_final.loc[indicesToKeep, 'pc5']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)

# Tough to discern a trend here

#%%
# PC 2 vs others
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 2', fontsize = 15)
ax.set_ylabel('PC 3', fontsize = 15)
ax.set_title('PC3 vs. PC2', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc2']
               , pdf_final.loc[indicesToKeep, 'pc3']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)

# When previous days since last campaign, # contacts before campaign are low but age high, a lot of nos
# Could be that older people need more touch points

#%%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 2', fontsize = 15)
ax.set_ylabel('PC 4', fontsize = 15)
ax.set_title('PC4 vs. PC2', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc2']
               , pdf_final.loc[indicesToKeep, 'pc4']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)

# ok - here we see the most stark clustering yet
# When PC2 and PC4 low, almost certain a no
# # days since last campaign, # previous contacts, and consumer confidence index low
# duration & # of contacts within this campaign low
# no chance - basically customer doesn't know you and doesn't take any time on the phone

# %%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 2', fontsize = 15)
ax.set_ylabel('PC 5', fontsize = 15)
ax.set_title('PC5 vs. PC2', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc2']
               , pdf_final.loc[indicesToKeep, 'pc5']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)
# not much discernable trend


# %%
# PC3
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 3', fontsize = 15)
ax.set_ylabel('PC 4', fontsize = 15)
ax.set_title('PC4 vs. PC3', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc3']
               , pdf_final.loc[indicesToKeep, 'pc4']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)

# Really strong discernable clustering
# PC3 high and PC4 low = no
# Old age and low duration and low # of contacts = no

# %%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 3', fontsize = 15)
ax.set_ylabel('PC 5', fontsize = 15)
ax.set_title('PC5 vs. PC3', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc3']
               , pdf_final.loc[indicesToKeep, 'pc5']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)
# tough to discern

# %%
# PC4
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 4', fontsize = 15)
ax.set_ylabel('PC 5', fontsize = 15)
ax.set_title('PC5 vs. PC4', fontsize = 20)
targets = ['no', 'yes']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pdf_final['y'] == target
    ax.scatter(pdf_final.loc[indicesToKeep, 'pc4']
               , pdf_final.loc[indicesToKeep, 'pc5']
               , c = color
               , s = 50
               , alpha = 0.3)
ax.legend(targets, title = 'Response', fontsize = 14)

# Pretty clear cluster
# Low PC4 but high PC5 = no
# Low duration of call and high number of contacts, higher age, higher duration? tough to interpret

#%% [markdown]
# Section 3: Feature Selection

#%%
# Lasso regression
# transform dataset

# make response binary
bank_df['y'] = bank_df['y'].replace(to_replace=['no', 'yes'], value=[0, 1])

# change column names
bank_df.columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']
bank_df.head()

#%%
# create data df
data = bank_df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']].copy()
data.head()

#%%
# numerically encode categorical varaibles
colstr = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for col in colstr:
  print(data[col].value_counts())

# map one by one (dict of dicts didn't work)
job = {'admin.' : 0, 'blue-collar': 1, 'technician': 2, 'services': 3, 'management': 4, 'retired': 5, 'entrepreneur': 6, 'self-employed': 7, 'housemaid': 8, 'unemployed': 9, 'student': 10, 'unknown': 11}
data['job'] = data['job'].map(job)

marital = {'married': 0, 'single': 1, 'divorced': 2, 'unknown': 3}
data['marital'] = data['marital'].map(marital)

education = {'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': 7}
data['education'] = data['education'].map(education)

default = {'no': 0, 'yes': 1, 'unknown': 2}
data['default'] = data['default'].map(default)

housing = {'no': 0, 'yes': 1, 'unknown': 2}
data['housing'] = data['housing'].map(housing)

loan = {'no': 0, 'yes': 1, 'unknown': 2}
data['loan'] = data['loan'].map(loan)

contact = {'telephone': 0, 'cellular': 1}
data['contact'] = data['contact'].map(contact)

month = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
data['month'] = data['month'].map(month)

day_of_week = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
data['day_of_week'] = data['day_of_week'].map(day_of_week)

poutcome = {'failure': 0, 'success': 1, 'nonexistent': 2}
data['poutcome'] = data['poutcome'].map(poutcome)

# define dictinary of dictionaries
#coldict = {'job': job, 'marital': marital, 'education': education, 'default': default, 'housing': housing, 'loan': loan, 'contact': contact, 'month': month, 'day_of_week': day_of_week, 'poutcome': poutcome}

# map numeric values by dictionary
#for key, value in coldict:
#  data[key] = data[key].map(value)

data.head()

# %%
# set X and y

X = data.copy()
y = bank_df['y']

print(X.shape)
print(y.shape)

#%%
# scale X
scaler = StandardScaler().fit(X) 

X = scaler.transform(X)

# %%
# select features present under different regularization strengths (C = inverse of regularization strength; lower C value means stronger regularization of features)

#cs = [0.001, 0.01, 0.1, 1, 10]

sel = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.0002, solver = 'liblinear'))
sel.fit(X, y)
a = sel.get_support()

sel = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.001, solver = 'liblinear'))
sel.fit(X, y)
b = sel.get_support()

sel = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.0025, solver = 'liblinear'))
sel.fit(X, y)
c = sel.get_support()

sel = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.005, solver = 'liblinear'))
sel.fit(X, y)
d = sel.get_support()

sel = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.0075, solver = 'liblinear'))
sel.fit(X, y)
e = sel.get_support()

# %%
# put together pandas dataframe

array = np.array([a, b, c, d, e])
array

index_values = [0.0002, 0.001, 0.01, 0.1, 1]
cols = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']

lassodf = pd.DataFrame(data = array, index = index_values, columns = cols)

lassdf_final = lassodf.transpose()
# print out above and put in slide deck

#%%
# Logistic regression

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
# ARRAY based
# X = sm.add_constant(X)
# model = sm.glm(y, X, family)

from statsmodels.formula.api import glm
# FORMULA based (we had been using this for ols)
# model = glm(formula, data, family)

bank_df = pd.read_csv('bank-additional-full.csv')
exec(open("import.py").read())
bank_df = bank_df

bank_df = bank_df.replace(to_replace=['no', 'yes'], value=[0, 1])
bank_df.columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']

#%%
# Model construction
# Start with some features
modelBank = glm(formula='y~C(default)+C(housing)+C(loan)+C(contact)+duration+campaign+pdays+previous+C(poutcome)', data=bank_df, family=sm.families.Binomial())
modelBankFit = modelBank.fit()
print(modelBankFit.summary())

# %%
# Drop features with p-value > 0.05
# Now add more features
modelBank = glm(formula='y~age+C(marital)+C(education)+duration+campaign', data=bank_df, family=sm.families.Binomial())
modelBankFit = modelBank.fit()
print(modelBankFit.summary())
# %%
# Drop features that have less effect on y
modelBank = glm(formula='y~age+duration+campaign', data=bank_df, family=sm.families.Binomial())
modelBankFit = modelBank.fit()
print(modelBankFit.summary())

#%%
from sklearn import linear_model

x_cols = ['age', 'duration', 'campaign']
x_bank = bank_df[x_cols]
y_bank = bank_df['y']
fullfit = linear_model.LogisticRegression()
fullfit.fit(x_bank , y_bank)
print('score:', fullfit.score(x_bank , y_bank))
print('intercept:', fullfit.intercept_)
print('coef_:', fullfit.coef_)

#%%
# Let us also do the model evaluation using train and test sets
from sklearn.model_selection import train_test_split

(X_train1, X_test1, y_train1, y_test1) = train_test_split(x_bank, y_bank, test_size = 0.2, random_state=1)
# training size is 0.8
full_split1 = linear_model.LogisticRegression()
full_split1.fit(X_train1, y_train1)
print('Accuracy (with the test set):', full_split1.score(X_test1, y_test1))
print('Accuracy (with the train set):', full_split1.score(X_train1, y_train1))
print('intercept:', full_split1.intercept_)
print('coef_:', full_split1.coef_)

print("\nReady to continue.")

#%%
print(full_split1.predict(X_test1))

print("\nReady to continue.")

test = full_split1.predict_proba(X_test1)

#%%
# Precision-Recall vs Threshold

y_pred=full_split1.predict(X_test1)

y_pred_probs=full_split1.predict_proba(X_test1) 
  # probs_y is a 2-D array of probability of being labeled as 0 (first 
  # column of array) vs 1 (2nd column in array)

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test1, y_pred_probs[:, 1]) 
   #retrieve probability of being 1(in second column of probs_y)
pr_auc = metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])

print("\nReady to continue.")

# Our graph suggests that the optimal cutoff is at the intercept of precision and recall
# So the cutoff is 0.2.

#%%
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test1))]
# predict probabilities
lr_probs = full_split1.predict_proba(X_test1)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test1, ns_probs)
lr_auc = roc_auc_score(y_test1, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test1, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test1, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

#%%
# Classification Report
#
from sklearn.metrics import classification_report
y_true, y_pred = y_test1, full_split1.predict(X_test1[x_cols])
print(classification_report(y_true, y_pred))

#                         predicted 
#                   0                  1
# Actual 0   True Negative  TN      False Positive FP
# Actual 1   False Negative FN      True Positive  TP
# 
# Accuracy    = (TP + TN) / Total
# Precision   = TP / (TP + FP)
# Recall rate = TP / (TP + FN) = Sensitivity
# Specificity = TN / (TN + FP)
# F1_score is the "harmonic mean" of precision and recall
#          F1 = 2 (precision)(recall)/(precision + recall)

print("\nReady to continue.")

# %%
# Find Optimal cutoff with the function we defined
####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################

from sklearn.metrics import auc

# Add prediction probability to dataframe
X_test1['pred'] = modelBankFit.predict(X_test1[x_cols])

# Find optimal probability threshold
fpr, tpr, threshold = roc_curve(y_test1, X_test1['pred'])
i = np.arange(len(tpr)) 
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

print("Opitmal cutoff : %f" % roc_t['threshold'])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

#%%
#
# Change the proba to 0 and 1 base on the cutoff value.

cut_off = 0.095390
predictions = (X_test1['pred']>cut_off).astype(int)
print(predictions)

print("\nReady to continue.")

# %%
# Classification Report
#
from sklearn.metrics import classification_report
y_true, y_pred = y_test1, predictions
print(classification_report(y_true, y_pred))

#                         predicted 
#                   0                  1
# Actual 0   True Negative  TN      False Positive FP
# Actual 1   False Negative FN      True Positive  TP
# 
# Accuracy    = (TP + TN) / Total
# Precision   = TP / (TP + FP)
# Recall rate = TP / (TP + FN) = Sensitivity
# Specificity = TN / (TN + FP)
# F1_score is the "harmonic mean" of precision and recall
#          F1 = 2 (precision)(recall)/(precision + recall)

print("\nReady to continue.")

# %%
# # Deviance
# Formula
# D = −2LL(β)
# * Measure of error
# * Lower deviance → better model fit

print('Null deviance: ' + str(modelBankFit.null_deviance))
print('Residual deviance: ' + str(-2*modelBankFit.llf))

# %%
# Feature importance
from matplotlib import pyplot

importance = full_split1.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print(x_cols[i]+' score: %.5f' % (v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
# Recall this is a classification problem with classes 0 and 1. 
# Notice that the coefficients are both positive and negative. 
# The positive scores indicate a feature that predicts class 1, 
# whereas the negative scores indicate a feature that predicts class 0.



#%% [markdown]
# Section 4: Classification Models


#%% [markdown]
# Section 5: Model Comparison

#%% [markdown]
# Section 6: Results and Conclusions