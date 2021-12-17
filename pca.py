# PCA for Smart Question 2

#%%
# run import and functions scripts
exec(open("import.py").read())
exec(open("functions.py").read())

bank_df = bank_df

print('bank_df is now loaded into the environment.')

# %%
# for reference/data dictionary
dfChkBasics(bank_df)

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

# %%
