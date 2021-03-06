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