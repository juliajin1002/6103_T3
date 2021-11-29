#%%
# data cleaning and eda

#%%
# run import and functions scripts
exec(open("import.py").read())
exec(open("functions.py").read())

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
# day: last contact day
# month: last contact month of year
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