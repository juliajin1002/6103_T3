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


#%% [markdown]
# Section 4: Classification Models


#%% [markdown]
# Section 5: Model Comparison

#%% [markdown]
# Section 6: Results and Conclusions