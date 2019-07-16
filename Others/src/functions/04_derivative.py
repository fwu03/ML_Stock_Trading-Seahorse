# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: May 10, 2019
#
# PURPOSE: The script takes in a dataframe and generate its first derivative values as a dataframe.
#
# INPUT:
#     - data: dataframe
#     - space: int, default at 1
#     - name: str, default at 'macd'
#
# OUTPUT:
#     - A dataframe includes the first derivatives of the given dataframe

# Packages Required
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def derivative(data, space = 1, name = "macd"):
    dy = []
    for i in range(data.shape[0]):
        y = pd.Series(data.iloc[i,:])
        temp_dy = list(np.gradient(y, space))
        dy.append(temp_dy)

    col_name = []
    for i in range(data.shape[1]):
        col_name.append(name + "deriv"+ str(i))

    deriv_df = pd.DataFrame(dy, columns=col_name)

    return deriv_df
