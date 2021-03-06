import mysql.connector
from mysql.connector import Error
import pandas as pd
#%%
def dbCon_dsLand(tbname, ind_col_name = ""):
  """ 
  connect to datasci.land database for this class and pull all rows from table
  :param str tbname: table name that exist on the server 
  :param str ind_col_name: optional, name of index column 
  :return: pandas.Dataframe
  """

  df = None # set a global variable to store the dataframe
  hostname = 'pysql.datasci.land'
  dbname = 'datascilanddb0'
  username = '6103_fa21'
  pwd = 'ke49jWxv7'
  query = 'SELECT * FROM `'+ dbname +'`.`'+ tbname + '`'

  try:
    connection = mysql.connector.connect(host=hostname, database=dbname, user=username, password=pwd)
    if connection.is_connected():
      # optional output
      db_Info = connection.get_server_info()
      print(f'Connected to MySQL Server version {db_Info}')
      cursor = connection.cursor()
      cursor.execute("select database();")
      record = cursor.fetchone()
      print(f"You're connected to database: {record}")
      # read query into dataframe df
      df = pd.read_sql(query, connection, index_col= ind_col_name) if (ind_col_name) else pd.read_sql(query, connection) # tables often have unique Id field
      print(f'Dataframe is loaded.')
      cursor.close()
      connection.close()
      print("MySQL connection is closed")

  except Error as e:
    print(f'Error while connecting to MySQL {e}')
      
  return df

# print("\nFunction dbCon_dsLand loaded. Ready to continue.")

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

# examples:
# dfChkBasics(df, True)


