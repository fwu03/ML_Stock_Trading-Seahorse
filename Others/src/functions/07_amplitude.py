# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: May 10, 2019
#
# PURPOSE: The script takes in a dataframe and calculate its amplitude values
#
# INPUT:
#     - A dataframe
#
# OUTPUT:
#     - A amplitude dataframe

# Packages Required
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def amplitude(data):
    amp = []
    for i in range(data.shape[0]):
        amp.append(np.var(np.diff(data.iloc[i,:]))/(np.mean(abs(data.iloc[i,:]))))
    amp = pd.DataFrame(amp, columns=["amplitude"])
    return amp
