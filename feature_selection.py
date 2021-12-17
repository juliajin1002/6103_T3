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

# print out above and put in slide deck

# %%
