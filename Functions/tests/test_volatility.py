# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# Date: Jun 18, 2019

# PURPOSE: This script is designed to test the script for the volitility calculation. There are 4 main unit test applied to both
#          function.
#
# Functions:
#    - test_random_line(): testing if both functions are able to process any random time series
#    - test_NA_line(): testing if both functions are able to do the exceptional handling for NA values
#    - test_non_df_input(): testing if the input is not a DataFrame
#    - test_empty_df(): testing if the input is an empty dataframe




import numpy as np
import pandas as pd
import os
import pytest
import sys

import random

sys.path.append('../')
volatility_module = __import__('05_volatility')

# 1. Create a random time series
random_line = [random.randrange(1, 20, 1) for _ in range(41)]
random_line_df = pd.DataFrame(random_line).transpose()

# 2. Create a time series contains NA values
NA_line = [random.randrange(1, 20, 1) for _ in range(41)]
NA_line[2] = np.nan
NA_line_df = pd.DataFrame(NA_line).transpose()

# 3. create an empty time series
columns = list(range(20))
index = range(10)
empty_df = pd.DataFrame(index = index, columns=columns)

# define the sigma for float handling
sigma = 0.0001


# 1. test if the function is able to achieve its original goal
def test_random_line():
    line_var = np.var(random_line_df.iloc[0,:])
    assert abs(volatility_module.volatility(random_line_df).iloc[0,0] - line_var) < sigma, "volitility function went wrong for random time series"


# 2. test if the function is able to handling time series with NA values
def test_NA_line():
    with pytest.raises(Exception):
        volatility_module.volatility(NA_line_df)

# 3. test if the function is able to handle time series with non-dataframe input
def test_non_df_input():
    with pytest.raises(Exception):
        volatility_module.volatility(random_line)

# 4. test if the function is able to handle empty dataframe as input
def test_empty_df():
    with pytest.raises(Exception):
        volatility_module.volatility(empty_df)
