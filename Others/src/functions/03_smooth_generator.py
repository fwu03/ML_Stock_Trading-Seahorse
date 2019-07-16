# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: May 10, 2019
#
# PURPOSE: The script takes in a dataframe and calculate its smoothness level based on differencing method
#
# INPUT:
#     - A dataframe
#
# OUTPUT:
#     - A single column dataframe includes smoothness level

# Packages Required
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def smooth_generator(data):
    smooth_list = []
    for i in range(data.shape[0]):
        smooth_list.append(np.var(np.diff(data.iloc[i,:])))
    smooth = pd.DataFrame(smooth_list, columns=["smooth"])

    return smooth
