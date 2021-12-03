#%%
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

bank_df = pd.read_csv('bank2.csv')
#%%
# Model construction
# Start with some features about the call recepient
modelBank = glm(formula='y~age+C(marital)+C(education)+C(default)+balance+C(housing)+C(loan)+C(contact)', data=bank_df, family=sm.families.Binomial())
modelBankFit = modelBank.fit()
print(modelBankFit.summary())

# %%
# Drop features with p-value > 0.05
modelBank = glm(formula='y~C(housing)+C(loan)', data=bank_df, family=sm.families.Binomial())
modelBankFit = modelBank.fit()
print(modelBankFit.summary())
# %%
# Now add more features
modelBank = glm(formula='y~C(housing)+C(loan)+duration+campaign+pdays+previous+C(poutcome)', data=bank_df, family=sm.families.Binomial())
modelBankFit = modelBank.fit()
print(modelBankFit.summary())
# %%
# Drop features that have less effect on y
modelBank = glm(formula='y~C(housing)+C(loan)+duration+campaign', data=bank_df, family=sm.families.Binomial())
modelBankFit = modelBank.fit()
print(modelBankFit.summary())
# %%
