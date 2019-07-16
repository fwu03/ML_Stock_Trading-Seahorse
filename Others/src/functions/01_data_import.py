# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: May 10, 2019
#
# PURPOSE: The script takes in a folder which includes all oscillator datasets,
#          and generate a clean dataframe that includes all data.
#
# INPUT:
#     - A dataset folder
#
# OUTPUT:
#     - A dataframe includes all data

# Packages Required
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def data_import(folder_path):
    df_gp = pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            stock = filename[0:4]
            #print("Loading stock data:", stock, ",")
            try:
                temp_df = pd.read_csv(os.path.join(folder_path, filename), delimiter= '\s+', header = None)
                temp_df.rename(columns={123:'rtn'}, inplace=True)
                temp_df = pd.concat([pd.Series([stock] * temp_df.shape[0], name = 'stock'), temp_df], axis=1)
                temp_df = temp_df.iloc[21:,:]
                df_gp = pd.concat([df_gp, temp_df])
                continue
            except:
                print('The following file cannot be read: ' + folder_path + filename)
        else:
            continue

    # Drop NaN values
    df_gp = df_gp.dropna()

    # Set all columns except stock names to numerics
    for col in df_gp.columns:
        if col != 'stock':
            df_gp[col] = pd.to_numeric(df_gp[col],errors='coerce')

    # Rename the columns
    osc_headers = []
    stk_headers = []
    macd_headers = []
    for i in range(41):
        osc_headers.append('osc' + str(i))
        stk_headers.append('stk' + str(i))
        macd_headers.append('macd' + str(i))
    df_gp.columns = ['stock'] + osc_headers + stk_headers + macd_headers + ['rtn']

    # Reset index
    df_gp = df_gp.reset_index(drop=True)

    return df_gp
