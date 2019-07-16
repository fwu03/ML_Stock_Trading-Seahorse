# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: Jun 14, 2019
#
# PURPOSE: The script takes into a dataframe where each row is a signal time serie, and
#          then calculate the smoothness level for each signal by using the differencing method;
#          the differencing method takes the variance of the first derivatives of the signal
#          time series, and then divided by the mean of the series
# INPUT:
#     - data (DataFrame): a dataframe contains bunch of signal time series
#     - name (String): column name for the output dataframe
# OUTPUT:
#     - smooth (DataFrame): a single column dataframe includes the smoothness level
#                           of the provided signal time series
#                           The smaller this value the smoother the time series is.

# Packages Required
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def smooth_generator(data, name = "osc"):

    # Input check
    # 1. check if the input is a dataframe
    if isinstance(data, pd.DataFrame) == 0:
        raise Exception('time series should be in dataframe')
    # 2. check if the dataframe is empty
    if data.empty:
        raise Exception('data is empty')
    # 3. check if the dataframe contains Null values
    if data.isnull().values.any():
        raise Exception('data contains NaN values')

    # Calculation
    smooth_list = []
    for i in range(data.shape[0]):
        amp_avg = np.mean(abs(data.iloc[i,:]))
        var_diff = np.var(np.diff(np.diff(data.iloc[i,:])))
        # If the curve is a horizontal line then return 0
        if (amp_avg == 0) | (var_diff == 0):
            # print("we detect zero")
            smooth_list.append(0)
        else:
            smooth_list.append(var_diff/amp_avg)
    smooth = pd.DataFrame(smooth_list, columns=["smooth_" + name])

    return smooth
