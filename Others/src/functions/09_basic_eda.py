# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: May 10, 2019
#
# PURPOSE: The script takes in X and y values and run Random Forest Classifier on them,
#          and generates a summary table with its defined trade class
#
# INPUT:
#     - X_w_rtn: a dataframe indicates all features with return values
#     - y: a dataframe indicate label (i.e. 0 or 1)
#     - test_size: float, default at 0.2
#
# OUTPUT:
#     - A summary dataframe in trade class format

# Packages Required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def EDA(index, osc_gp, stk_gp, macd_gp, style = 'bmh', figsize=(7,7)):

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        num_layout = 3
        layout = (num_layout, 1)

        osc_ax = plt.subplot2grid(layout, (0, 0))
        osc_ax.plot(range(41), osc_gp.iloc[index,:])
        osc_ax.invert_xaxis()
        osc_ax.set_title('Oscillator 3-Day Time Series')
        osc_ax.set_xlabel('Time')
        osc_ax.set_ylabel('Oscillator')

        stk_ax = plt.subplot2grid(layout, (1, 0))
        stk_ax.plot(range(41), stk_gp.iloc[index,:])
        stk_ax.invert_xaxis()
        stk_ax.set_title('Stock Price 3-Day Time Series')
        stk_ax.set_xlabel('Time')
        stk_ax.set_ylabel('Stock Price')

        macd_ax = plt.subplot2grid(layout, (2, 0))
        macd_ax.plot(range(41), macd_gp.iloc[index,:])
        macd_ax.invert_xaxis()
        macd_ax.set_title('MACD 3-Day Time Series')
        macd_ax.set_xlabel('Time')
        macd_ax.set_ylabel('MACD')


        plt.tight_layout()
