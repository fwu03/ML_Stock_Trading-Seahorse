# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# Date: Jun 17, 2019
#
# PURPOSE: The script takes into a dataframe where each row is a signal time series
#          (in our special case, this time series is defined as the oscillator data specially),
#          abd then fit the data to a polynomial curve with degree of freedem equals to 5. After
#          fitting, the sum of residual is calculated and recorded as our output in to a column
#          of a newly created dataframe.
# INPUT:
#      - data (Dataframe): a dataframe contains bunch of signal time Series
#      - name (string): column name for the output DataFrame
# OUTPUT:
#      - smooth (DataFrame): a single column dataframe include the smoothness value of the provided
#                            time series. The smaller this value the smoother the time series is

import numpy as np
import pandas as pd


def smooth_fit_based_generator(data, name = "osc"):
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


    # calculate the smoothness
    smooth_fit = []
    list_x = range(41)
    for i in range(data.shape[0]):
        list_y = data.iloc[i,:]
        poly = np.polyfit(list_x,list_y,5, full = True)
        # Append the residuals of the fit
        smooth_fit.append(poly[1][0])

    df_smoo_diff = pd.DataFrame(smooth_fit, columns=["fit_smooth_" + name])

    return df_smoo_diff
