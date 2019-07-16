# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: June 14, 2019
#
# PURPOSE: The script takes into a dataframe where each row is a signal time serie, and
#          then calculate the volatility level for each signal by taking its variance;
# INPUT:
#     - data (DataFrame): a dataframe contains bunch of signal time series
#     - name (String): column name for the output dataframe
# OUTPUT:
#     - smooth (DataFrame): a single column dataframe includes the volatility level
#                           of the provided signal time series

# Packages Required
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def volatility(data, name = 'stk'):
    # Input Check
    # 1. check if the input is a dataframe
    if isinstance(data, pd.DataFrame) == 0:
        raise Exception('time series should be in dataframe')
    # 2. check if the dataframe is empty
    if data.empty:
        raise Exception('data is empty')
    # 3. check if the dataframe contains Null values
    if data.isnull().values.any():
        raise Exception('data contains NaN values')

    # Calculation variance = volatility
    vol = []
    for i in range(data.shape[0]):
        vol.append(np.var(data.iloc[i,:]))
    vol = pd.DataFrame(vol, columns=["vol" + name])
    return vol
