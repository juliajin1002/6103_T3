#%% [markdown]

# DATS 6103, Intro to Data Mining, Fall 2021
# Team 3 Final Project Code and Analysis
# Julia Jin, Eric Newman, Alice Yang

#%% [markdown]

# Sections:
# 1. Data import and cleaning
# 2. Principal components analysis
# 3. Feature selection
#   a. Lasso regression
#   b. Logistic regression
# 4. Classification models: logistic regression, KNN, SVM
# 5. Model comparison
# 6. Results and conclusion

#%% [markdown]
# Section 1: Data Import and Cleaning

#%%
# import libraries and read in file

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

# read in file
bank_df = pd.read_csv('bank-additional-full.csv', sep = ';', engine = 'python')

#%%
# Standard quick checks
def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  # print(f'\n{cnt}: dtypes: ')
  # cnt+=1
  # print(dframe.dtypes)

  # try:
  #   print(f'\n{cnt}: columns: ')
  #   cnt+=1
  #   print(dframe.columns)
  # except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

bank_df = bank_df

print('bank_df is now loaded into the environment.')

# %%
# for reference/data dictionary
dfChkBasics(bank_df)

#%% [markdown]
# Section 2: Principal Components Analysis

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
# feature selection

# attempt to do feature selection with lasso regression (L1) penalty in logistic regression model
# if that doesn't work, default to feature importance by graphic coefficients
# or can do feature importance/selection by testing ANOVA and chi-sq of each variable with response

#%%
# import libraries and dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

exec(open("import.py").read())
bank_df = bank_df
bank_df['y'] = bank_df['y'].replace(to_replace=['no', 'yes'], value=[0, 1])

# %%
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
lassdf_final

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

#%%
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 


#%%
bank_df=pd.read_csv('bank-additional-full.csv')
exec(open("import.py").read())

#%%[markdown]
#Processing the data
#
#there are unknown values in the columns job, marital, education, default, housing, and loan.
#I droped the the unknowns in job, marital and education since they have many levels and it's 
#hard to decide how we're going to categorize the unknown while transforming these columns
# into numeric. And since our data has over 40k rows, and the largest number of unknown rows
#1731, I decided to drop them to prevent misleading model and results.
#
#For all the binary columns, since unknown could be either 'yes' or 'no', I transfered all the unknowns into 0.5,
# as a average value.
#%%
bank=bank_df.copy()
bank.replace({'no':0,'yes':1},inplace=True)
bank.replace({'job':'unknown'},value=np.nan,inplace=True)
bank.replace({'marital':'unknown'},value=np.nan,inplace=True)
bank.replace({'education':'unknown'},value=np.nan,inplace=True)
bank.replace({'default':'unknown'},value=0.5,inplace=True)
bank.replace({'housing':'unknown'},value=0.5,inplace=True)
bank.replace({'loan':'unknown'},value=0.5,inplace=True)
bank.dropna(inplace=True)

#%%
#tranforming job column
bank['management']=[1 if r.job=='management' else 0 for i,r in bank.iterrows() ]
bank['admin.']=[1 if r.job=='admin.' else 0 for i,r in bank.iterrows() ]
bank['blue-collar']=[1 if r.job=='blue-collar' else 0 for i,r in bank.iterrows() ]
bank['technician']=[1 if r.job=='technician' else 0 for i,r in bank.iterrows() ]
bank['housemaid']=[1 if r.job=='housemaid' else 0 for i,r in bank.iterrows() ]
bank['retired']=[1 if r.job=='retired' else 0 for i,r in bank.iterrows() ]
bank['unemployed']=[1 if r.job=='unemployed' else 0 for i,r in bank.iterrows() ]
bank['entrepreneur']=[1 if r.job=='entrepreneur' else 0 for i,r in bank.iterrows() ]
bank['student']=[1 if r.job=='student' else 0 for i,r in bank.iterrows() ]
bank['self-employed']=[1 if r.job=='self-employed' else 0 for i,r in bank.iterrows() ]
bank['services']=[1 if r.job=='services' else 0 for i,r in bank.iterrows() ]
bank.drop(columns=['job'],inplace=True)



#%%
#transforming marital column
bank['single']=[1 if r.marital=='single' else 0 for i,r in bank.iterrows() ]
bank['married']=[1 if r.marital=='married' else 0 for i,r in bank.iterrows() ]
bank['divorced']=[1 if r.marital=='divorced' else 0 for i,r in bank.iterrows() ]
bank=bank.drop(columns=['marital'])
#%%
#education
bank.replace({'basic.4y':1,'basic.6y':2,'basic.9y':3,'high.school':4,'illiterate':5,'professional.course':6,'university.degree':7},inplace=True)

#%%
#Month
bank['jan']=[1 if r.month=='jan' else 0 for i,r in bank.iterrows() ]
bank['feb']=[1 if r.month=='feb' else 0 for i,r in bank.iterrows() ]
bank['mar']=[1 if r.month=='mar' else 0 for i,r in bank.iterrows() ]
bank['apr']=[1 if r.month=='apr' else 0 for i,r in bank.iterrows() ]
bank['may']=[1 if r.month=='may' else 0 for i,r in bank.iterrows() ]
bank['jun']=[1 if r.month=='jun' else 0 for i,r in bank.iterrows() ]
bank['jul']=[1 if r.month=='jul' else 0 for i,r in bank.iterrows() ]
bank['aug']=[1 if r.month=='aug' else 0 for i,r in bank.iterrows() ]
bank['sep']=[1 if r.month=='sep' else 0 for i,r in bank.iterrows() ]
bank['oct']=[1 if r.month=='oct' else 0 for i,r in bank.iterrows() ]
bank['nov']=[1 if r.month=='nov' else 0 for i,r in bank.iterrows() ]
bank['dec']=[1 if r.month=='dec' else 0 for i,r in bank.iterrows() ]
bank.drop(columns=['month'],inplace=True)

#%%
#day_of_week
bank['mon']=[1 if r.day_of_week=='mon' else 0 for i,r in bank.iterrows() ]
bank['tue']=[1 if r.day_of_week=='tue' else 0 for i,r in bank.iterrows() ]
bank['wed']=[1 if r.day_of_week=='wed' else 0 for i,r in bank.iterrows() ]
bank['thu']=[1 if r.day_of_week=='thu' else 0 for i,r in bank.iterrows() ]
bank['fri']=[1 if r.day_of_week=='fri' else 0 for i,r in bank.iterrows() ]
bank.drop(columns=['day_of_week'],inplace=True)

#%%
#contact
bank.replace({'contact':{'cellular':0,'telephone':1}},inplace=True)


#%%
#poutcome
bank.replace({'poutcome':{'failure':0,'nonexistent':1,'success':2,}},inplace=True)
#%%
#Standardizing numeric values:
from sklearn.preprocessing import StandardScaler

num_col=['age','education','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
scaler=StandardScaler()
scaler.fit(bank[num_col])
num_trans=scaler.transform(bank[num_col])
bank[num_col]=num_trans
#%% Logistic Regression with all 47 variables
bank_x=bank.drop(columns=['y'])
bank_y=bank['y']
x_train,x_test,y_train,y_test=train_test_split(bank_x,bank_y,train_size=0.85,random_state=1)
lr=LogisticRegression(max_iter=10000)
lr.fit(x_train,y_train)
print('Test accuracy',lr.score(x_train,y_train))
print('The accuracy score for LogisticRegression model is', lr.score(x_test,y_test))
pd.DataFrame(lr.coef_.reshape(47,1),index=np.array(bank_x.columns),columns=['coefficients'])

# %%[markdown]
#Grouping the columns based on coefficients:
#
#Since we have so many columns in our data, the models we build might result
#in overfitting. In this case, we would like to combine some bianry columns with 
# similar impact (by looking at their coefficients) into groups to reduce dimension.
#
#For all the jobs we have:
#Retired, Student have strong positive effect, so we group them into job_sp column
#
#Management,Admin, technician, unemployed have weak positive effect, grouped into job_wp
#
#Self-employed and services have weak nagative effect (as job_wn)
#
#Blue-collar, housemaid, entrepreneur are grouped as job_sn
#%%
bank['job_sp']=bank[['retired','student']].sum(axis=1)
bank['job_w']=bank[['management','admin.','technician','unemployed']].sum(axis=1)
bank['job_wn']=bank[['self-employed','services']].sum(axis=1)
bank['job_sn']=bank[['blue-collar','housemaid','entrepreneur']].sum(axis=1)
bank.drop(columns=['management','admin.','blue-collar','technician','housemaid','retired','unemployed','entrepreneur','student','self-employed','services'],inplace=True)

#%%[markdown]
#For all the months we have:
#March and August have a strong positive impact, so we will group them into m_sp
#
#September, October and December have a weaker positive impact (as m_wp)
#
#April, May, Jun, and November seems to have a strong negative effect on y (as n_sn)
#
#January, Febuary, July have a weaker nigative impact/ no impact on y (as m_wn)

#%%
bank['m_sp']=bank[['mar','aug']].sum(axis=1)
bank['m_wp']=bank[['sep','oct','dec']].sum(axis=1)
bank['m_sn']=bank[['apr','may','jun','nov']].sum(axis=1)
bank['m_wn']=bank[['jan','feb','jul']].sum(axis=1)
bank.drop(columns=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],inplace=True)

#%%[markdown]
#For all day_of_week we have:
#Wednesday has strong positive effect (as d_sp)
#
#Tuesday has weaker positive effect (as d_wp)
#
#Monday has strong negative effect (as d_sn)
#
#Thursday and Friday has weak negative effect (as d_wn)
#%%
bank['d_sp']=bank['wed']
bank['d_wp']=bank['tue']
bank['d_sn']=bank['mon']
bank['d_wn']=bank[['thu','fri']].sum(axis=1)
bank.drop(columns=['mon','tue','wed','thu','fri'],inplace=True)

#%% split data into x and y
bank_x=bank.drop(columns=['y'])
bank_y=bank.y

#%%
#Linear SVC model
x_train,x_test,y_train,y_test=train_test_split(bank_x,bank_y,train_size=0.7,random_state=1)
svc_linear=SVC(kernel='linear')
svc_linear.fit(x_train,y_train)
#Confusin Matrix
print(confusion_matrix(y_test, svc_linear.predict(x_test)))
#Coefficients
pd.DataFrame(svc_linear.coef_.reshape(31,1),index=np.array(bank_x.columns),columns=['coefficients'])

#%%
#LogisticRegression model
lr=LogisticRegression(max_iter=10000)
lr.fit(x_train,y_train)
#Confusion Matrix
print(confusion_matrix(y_test, lr.predict(x_test)))
#Coefficients
pd.DataFrame(lr.coef_.reshape(31,1),index=np.array(bank_x.columns),columns=['coefficients'])

#%%
#KNN model
#Tuning for the optimal k
for n in np.arange(1,452,50):
    knn=KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train,y_train)
    print(f'The train accuracy for {n}th nearest neighbor is: {knn.score(x_test,y_test)}')
#%%
#The best k-th nearest neighbor is 101, choose to use n=100
knn=KNeighborsClassifier(n_neighbors=100)
knn.fit(x_train,y_train)
#Confusion Matrix
print(confusion_matrix(y_test, knn.predict(x_test)))


#%% [markdown]
# Section 5: Model Comparison
#%%
#Accuracy Comparison:
#Linear SVC
print('Train accuracy score for Linear SVC model is',svc_linear.score(x_train,y_train))
print('The test accuracy score for Linear SVC model is', svc_linear.score(x_test,y_test))
#Logistic Regression
print('Test accuracy score for LogisticRegression model is',lr.score(x_train,y_train))
print('The accuracy score for LogisticRegression model is', lr.score(x_test,y_test))
#KNN
print('Train accuracy score for KNN with k=100 is',knn.score(x_train,y_train))
print('The test accuracy score for KNN with k=100 is', knn.score(x_test,y_test))
#According to the test accuracy scores, Logistic Regression has the highest test accuracy 
#among all.

#%%
#Precision Comparison:
#Linear SVC
print(classification_report(y_test, svc_linear.predict(x_test)))
#Logistic Regression
print(classification_report(y_test, lr.predict(x_test)))
#KNN
print(classification_report(y_test, knn.predict(x_test)))
#According to the classification report for the three models, knn has the highest
#precision among all models.

#%%[markdown]
#For this business problem, we want to choose the best model which maximize both Accuracy and 
#precision for prediction. Since the two models with the highest precision (Logistic Regression and KNN)
# have a precision that is really closed to each other, we may conclude that Logistic Regression is the 
#best model among all, with a relavantly high precision and the highest accuracy.

#%% [markdown]
# Section 6: Results and Conclusions
#
#Unsupervised learning revealed clusters of interest for economic conditions, customer demographics (age), 
# and marketing techniques.
#
# Feature selection revealed most important explanatory factors.
#
# Of the predictive models, logistic regression performed the best on the test data.
#
#As a conclusion, These insights can lead banking institutions to fine-tune their marketing targets (older), 
# techniques (established relationships vs. no chance), and timing (in more stable employment markets)

# EOF
