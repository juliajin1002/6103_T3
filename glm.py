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

#%%
from sklearn import linear_model

x_cols = ['housing', 'loan', 'duration', 'campaign']
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
y_pred1 = full_split1.predict(X_test1)
full_split1.score(X_test1, y_test1)

print('score (train):', full_split1.score(X_train1, y_train1)) 
print('score (test):', full_split1.score(X_test1, y_test1)) 
print('intercept:', full_split1.intercept_)
print('coef_:', full_split1.coef_)

print("\nReady to continue.")
#%%

test = full_split1.predict_proba(X_test1)
print(test)
print("\nReady to continue.")


# %%
# # Deviance
# Formula
# D = −2LL(β)
# * Measure of error
# * Lower deviance → better model fit
print(-2*modelBankFit.llf)
# Compare to the null deviance
print(modelBankFit.null_deviance)
# 499.98  # df = 399 
# %%
# # Interpretation
np.exp(modelBankFit.params)
np.exp(modelBankFit.conf_int())

# %%
from sklearn.metrics import roc_curve, auc

# Add prediction to dataframe
bank_df['pred'] = full_split1.predict(bank_df[x_cols])

fpr, tpr, thresholds =roc_curve(bank_df['y'], bank_df['pred'])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
print("Opitmal cutoff : %f" % roc_t['thresholds'])

#%%
#
# Write a function to change the proba to 0 and 1 base on a cut off value.

cut_off = 0.108161
predictions = (full_split1.predict_proba(X_test1)[:,1]>cut_off).astype(int)
print(predictions)

print("\nReady to continue.")

#%%
def predictcutoff(arr, cutoff):
  arrbool = arr[:,1]>cutoff
  arr= arr[:,1]*arrbool/arr[:,1]
  # arr= arr[:,1]*arrbool
  return arr.astype(int)

test = full_split1.predict_proba(X_test1)
p = predictcutoff(test, cut_off)
print(p)

print("\nReady to continue.")

#%%
print(predictcutoff(test, 0.5))

print("\nReady to continue.")
# %%
# Classification Report
#
from sklearn.metrics import classification_report
y_true, y_pred = y_test1, full_split1.predict(X_test1)
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

# %%
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

# %%
