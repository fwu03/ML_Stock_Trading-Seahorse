# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: Jun 14, 2019
#
# PURPOSE: The script takes into a cleaned dataframe (from script 01_data_import.py)
#          and splits them into different groups
#
# INPUT:
#     - df_gp (DataFrame): a dataframe with either 126 columns for train set or 125
#       columns for test set
#     - type (String): either 'train' for training dataset or 'test' for testing dataset
#
# OUTPUT:
#     - stock_gp (Series): a series of stock names in string format
#     - signal_gp (Series): a series of 1 or 0 where 1 indicates the signal comes from buy
#                           dataset; 0 indicates the signal comes from sell dataset
#     - osc_gp (DataFrame): a 41 columns dataframe includes all oscillator time series
#     - stk_gp (DataFrame): a 41 columns dataframe includes all stock price time series
#     - macd_gp (DataFrame): a 41 columns dataframe includes all macd time series
#     - rtn_gp (Series): a series of actual returns for type equals 'train'; or a series of
#                        0 for type  equals 'test'
#     - label_gp (Series): same as the rtn_gp

# Packages Required
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def data_organize(df_gp, type='train'):

    # check whether the input dataframe is a pandas DataFrame
    if isinstance(df_gp, pd.DataFrame) == 0:
        raise TypeError('the first input parameter needs to be a pandas DataFrame')

    # Check the input dataframe has a correct format (shape)
    if type == 'train':
        if df_gp.shape[1] != 126:
            raise ValueError('the input dataframe has a wrong format')
    else:
        if df_gp.shape[1] != 125:
            raise ValueError('the input dataframe has a wrong format')

    # check whether the input type has a valid value
    if type not in ['train', 'test']:
        raise ValueError('the dataset type value is invalid')

    # Split the dataframe into multiple groups
    stock_gp = df_gp.iloc[:, 0]
    signal_gp = df_gp.iloc[:, 1]
    osc_gp = df_gp.iloc[:, 2:43]
    stk_gp = df_gp.iloc[:, 43:84]
    macd_gp = df_gp.iloc[:, 84:125]

    # Return 0 return if it is test data
    if type == 'train':
        rtn_gp = df_gp.iloc[:, 125]
    else:
        rtn_gp = pd.Series([0] * len(stock_gp), name = 'return')

    return stock_gp, signal_gp, osc_gp, stk_gp, macd_gp, rtn_gp
