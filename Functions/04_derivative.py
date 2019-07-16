# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: Jun 14, 2019
#
# PURPOSE: The script takes into a dataframe where each row is a signal time series, and
#          calculate its first derivative values; If the relative is True, the function below
#          will calculate its relative first derivative values;
# INPUT:
#     - data (DataFrame): a dataframe where each row is a signal time series
#     - name (String): column name for the output dataframe; default at 'd'
#     - relative (Boolean): indicator to decide whether to calculate relative derivative or not
#     - method (String): derivative calculation method; either 'mean' or 'value' method
# OUTPUT:
#     - deriv_df (DataFrame): a dataframe includes the first derivatives values of the given dataframe

# Packages Required
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def derivative(data, name = "d", relative = False, method = 'mean'):
    # check whether the input dataframe is a pandas DataFrame
    if isinstance(data, pd.DataFrame) == 0:
        raise TypeError('the first input parameter needs to be a pandas DataFrame')

    # check whether the input dataframe contains NaN values
    if data.isnull().values.any():
        raise ValueError('data contains NaN values')

    # check whether the input 'relative' has a valid value
    if relative not in [True, False]:
        raise ValueError('the relative value is invalid')

    # check whether the input 'method' has a valid value
    if method not in ['value', 'mean']:
        raise ValueError('the method value is invalid')

    # Calculation
    dy = []
    # Replace all the zeros in the dataset with 0.00001 to avoid Inf when doing division
    data = data.replace(0, 0.00001)
    for i in range(data.shape[0]):
        y = pd.Series(data.iloc[i,:])
        # Relative change
        if relative == True:
            # Method mean calculate the change in relative to the average absolute value
            if method == 'mean':
                temp_dy = list(np.diff(y/(np.mean(np.abs(y)))))
            # Method value calculate the change in relative to the previous value
            elif method == 'value':
                temp_dy = list(np.diff(y)/y[:-1])
            else:
                print('Please enter the correct method name: mean or value')
        else:
            temp_dy = list(np.diff(y))
        dy.append(temp_dy)

    # Assign column names
    col_name = []
    for i in range(len(temp_dy)):
        col_name.append(name + str(i))

    deriv_df = pd.DataFrame(dy, columns=col_name)

    return deriv_df
