#%%
# data cleaning and eda

#%%
# run import and functions scripts
exec(open("import.py").read())
exec(open("functions.py").read())

bank_df = bank_df

print('bank_df is now loaded into the environment.')

# %%
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

# %%
