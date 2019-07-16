# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: May 10, 2019
#
# PURPOSE: The script takes in a dataframe and split them into different groups
#
# INPUT:
#     - A dataframe
#
# OUTPUT:
#     - Multiple dataframes

# Packages Required
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def data_organize(df_gp):
    try:
        name_gp = df_gp.iloc[:, 0]
        osc_gp = df_gp.iloc[:, 1:42]
        stk_gp = df_gp.iloc[:, 42:83]
        macd_gp = df_gp.iloc[:, 83:124]
        rtn_gp = df_gp.iloc[:, 124]
        label_gp = np.sign(rtn_gp).map({1: 1, -1: 0, 0:0})
        results_gp = label_gp.map({1: 'EARN', 0: 'LOSS'})
    except:
        print('Please check the dataframe index')

    return name_gp, osc_gp, stk_gp, macd_gp, rtn_gp, label_gp, results_gp
