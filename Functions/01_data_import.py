# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: Jun 14, 2019
#
# PURPOSE: The script takes into a path of original data folder, read the data there,
#          manipulating the date includes removing duplication, dropping NaN values,
#          updating the data types, and adding the columns names. Besides of these, the
#          function below also reverses the sign of the return for those files have the sell
#          signal types
#
# INPUT:
#     - folder_path (String): the path of the data sources
#     - stock_list (List): a list of previous loaded stock names
#     - type (String): identify the input dataset type; either 'train' or 'test'
#
# OUTPUT:
#     - df_comb (DataFrame): contains all cleaned data together
#     - stock_list (List): a list of loaded stock names (previous + new)

# Packages Required
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def data_import(folder_path, stock_list, type = 'train'):

    # check whether the input folder_path is valid
    if not os.path.isdir(folder_path):
        raise NotADirectoryError('the given folder path is invalid')

    # check whether the input stock_list is in correct data type
    if not isinstance(stock_list, list):
        raise TypeError('the stock list needs to be a list')

    # check whether the input type is valid
    if type not in ['train', 'test']:
        raise ValueError('the dataset type value is invalid')

    # start loading the data
    df_comb = pd.DataFrame()
    print()
    print("Loading data:")

    for filename in os.listdir(folder_path):
        # check with file is in txt format and with Buy/Sell in the filename
        if filename.endswith(".txt") and (("Buy" in filename) or ("Sell" in filename)):
          # define signal type based on the filename
          if "Buy" in filename:
              signal_type = 'buy'
          else:
              signal_type = 'sell'
          # capture the stock name; stock name is before Buy/Sell
          pos = max(filename.find('Sell'), filename.find('Buy'))
          if pos == -1:
            pos = 45 #if cannot find Buy/Sell, then returns the whole string as filename
          stock = str(filename[0:pos]).replace('1','')
          print(stock, " ", end = '')

          # read the data file
          try:
            temp_df = pd.read_csv(os.path.join(folder_path, filename), delimiter= '\s+', header = None)
          except:
            print('The following file cannot be read: ' + folder_path + filename)
            print('Please check the file path')

          # check the data format for train and test
          if type == 'train':
              if temp_df.shape[1] != 124:
                  raise Exception('Please check the input format; train data has to include 124 columns')
          else:
              if temp_df.shape[1] != 123:
                  raise Exception('Please check the input format; test data has to include 123 columns')

          # delete the first 21 rows - suggested by the partner as the first 21 rows maybe bad data
          temp_df = temp_df.iloc[21:,:]
          temp_df.rename(columns={123:'rtn'}, inplace=True) #only for the train data, test data will ignore this line automatically
          temp_df = temp_df.reset_index(drop=True)

          # assign signal types
          if signal_type == 'buy':
            temp_df = pd.concat([pd.Series([1] * len(temp_df), name = 'signal_type'), temp_df], axis=1)
          else:
            temp_df = pd.concat([pd.Series([0] * len(temp_df), name = 'signal_type'), temp_df], axis=1)
            if type == 'train':
                temp_df['rtn'] = -temp_df['rtn']

          # add stock name to the DataFrame
          if stock not in stock_list:
            stock_list.append(stock)
          temp_df = pd.concat([pd.Series([stock_list.index(stock)] * temp_df.shape[0], name = 'stock_name'), temp_df], axis=1)

          # Combine dataframes
          df_comb = pd.concat([df_comb, temp_df])

        else:
            continue

    # Remove Duplication in DataFrame
    n1 = len(df_comb)
    print('Original Data Size:', n1)
    df_comb.drop_duplicates(subset = df_comb.columns[2:], inplace=True)
    n2 = len(df_comb)
    print('Removed', n1-n2, 'duplicated values')

    # Drop NaN values in DataFrame
    df_comb = df_comb.dropna()
    n3 = len(df_comb)
    print('Removed', (n2-n3), 'NaN values')

    # Set all columns except stock names to numerics
    for col in df_comb.columns:
      if col != 'stock_name':
        df_comb[col] = pd.to_numeric(df_comb[col],errors='coerce')

    # Rename the columns
    osc_headers = []
    stk_headers = []
    macd_headers = []
    for i in range(41):
        osc_headers.append('osc' + str(i)) #a list of osc names
        stk_headers.append('stk' + str(i)) #a list of stk names
        macd_headers.append('macd' + str(i)) ##a list of macd names
    if type == 'train':
        df_comb.columns = ['stock_name', 'signal_type'] + osc_headers + stk_headers + macd_headers + ['rtn']
    else:
        df_comb.columns = ['stock_name', 'signal_type'] + osc_headers + stk_headers + macd_headers

    # Reset index
    df_comb = df_comb.reset_index(drop=True)

    return df_comb, stock_list
