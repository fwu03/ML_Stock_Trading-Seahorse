# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# Date: Jun 19, 2019

# PURPOSE: This script is designed to test the script for ratio calculation. There are 5 main unit test applied to both
#          function.
#
# Functions:
#    - test_zero_lines(): testing if both functions are able to process the horozontal line pass the (0,0) point
#    - test_h_lines(): testing if both functions are able to process any horozontal line;
#    - test_random_lines(): testing if both functions are able to process any random time series
#    - test_wrong_inputs(): testing if both functions are able to do the exceptional handling for wrong inputs
#    - test_non_df_input(): testing if the input is not a DataFrame




import numpy as np
import pandas as pd
import os
import pytest
import sys

import random

sys.path.append('../')
ratio_module = __import__('06_ratio')

# 1. Create a random time series
random_line_1 = [random.randrange(1, 20, 1) for _ in range(41)]
random_line_1_df = pd.DataFrame(random_line_1).transpose()

# 2. create another random time series with different dimension
random_line_2 = [random.randrange(1, 20, 1) for _ in range(45)]
random_line_2_df = pd.DataFrame(random_line_2).transpose()



# 3. Create a line/time series with all 0
zero_line = np.zeros(shape = (41,1))
zero_line_df = pd.DataFrame(zero_line).transpose()

# 4. Create a horizontal line with non-zero values
h_line = np.ones(shape = (41,1))
h_line_df = pd.DataFrame(h_line).transpose()

# 5. Create a time series contains NA values
NA_line = [random.randrange(1, 20, 1) for _ in range(41)]
NA_line[2] = np.nan
NA_line_df = pd.DataFrame(NA_line).transpose()

# 6. create an empty time series
columns = list(range(20))
index = range(10)
empty_df = pd.DataFrame(index = index, columns=columns)

# define the sigma for float handling
sigma = 0.0001


# 1. test if the function is able to achieve its original goal using two random line
# these two lines are completely parallel, so the sum should be the column number
def test_random_lines():
    assert sum(ratio_module.int_ratio(random_line_1_df, random_line_1_df).iloc[0,:]) == 5, "Ratio function went wrong from random time series"
    assert sum(ratio_module.int_ratio(random_line_1_df, random_line_1_df, num_col=10).iloc[0,:]) == 10, "Ratio function went wrong from random time series"
    assert sum(ratio_module.int_ratio(random_line_1_df, random_line_1_df, num_col = 15).iloc[0,:]) == 15, "Ratio function went wrong from random time series"

# 2. test if the function is able to handle time serires with all zero inputs
# these two lines are completely parallel, so the sum should be the column number
def test_zero_lines():
    assert sum(ratio_module.int_ratio(zero_line_df, zero_line_df).iloc[0,:]) == 5, "Ratio function went wrong from random time series"
    assert sum(ratio_module.int_ratio(zero_line_df, zero_line_df, num_col=10).iloc[0,:]) == 10, "Ratio function went wrong from random time series"
    assert sum(ratio_module.int_ratio(zero_line_df, zero_line_df, num_col = 15).iloc[0,:]) == 15, "Ratio function went wrong from random time series"

# 3. test if the function is able to handle two horizontal lines
# these two lines are completely parallel, so the sum should be the column number
def test_h_lines():
    assert sum(ratio_module.int_ratio(h_line_df, h_line_df).iloc[0,:]) == 5, "Ratio function went wrong from random time series"
    assert sum(ratio_module.int_ratio(h_line_df, h_line_df, num_col=10).iloc[0,:]) == 10, "Ratio function went wrong from random time series"
    assert sum(ratio_module.int_ratio(h_line_df, h_line_df, num_col = 15).iloc[0,:]) == 15, "Ratio function went wrong from random time series"


# Exceptional handling tests
# 1. test if the function is able to handle non dataframe input
def test_nan_df_inputs():
    with pytest.raises(Exception):
        ratio_module.int_ratio(h_line, h_line)
    with pytest.raises(Exception):
        ratio_module.int_ratio(h_line_df, h_line)
    with pytest.raises(Exception):
        ratio_module.int_ratio(h_line, h_line_df)

# 2. test if the function is able to handle wrong inputs
def test_wrong_inputs():
        # uneven dimension for the inputs
        with pytest.raises(Exception):
            ratio_module.int_ratio(random_line_1_df, random_line_2_df)
        # wrong column number input
        with pytest.raises(Exception):
            ratio_module.int_ratio(random_line_1_df, random_line_1_df, num_col= 50)
        # empty input
        with pytest.raises(Exception):
            ratio_module.int_ratio(empty_df, empty_df)
        # input has NA values
        with pytest.raises(Exception):
            ratio_module.int_ratio(NA_line_df, NA_line_df)
