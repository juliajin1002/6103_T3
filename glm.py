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

