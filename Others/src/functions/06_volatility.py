# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: May 10, 2019
#
# PURPOSE: The script takes in a dataframe and calculate its volatility values
#
# INPUT:
#     - A dataframe
#
# OUTPUT:
#     - A volatility dataframe

# Packages Required
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def volatility(data):
    vol = []
    for i in range(data.shape[0]):
        vol.append(np.var(data.iloc[i,:]))
    vol = pd.DataFrame(vol, columns=["volatility"])
    return vol
