# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# Date: Jun 17, 2019

# PURPOSE: This script is designed to test the script for both the smoothness using differencing and
#          the smoothness calculated by polynomial fitting. There are 6 main unit test applied to both
#          function.
#
# Functions:
#    - test_zero_line(): testing if both functions are able to process the horozontal line pass the (0,0) point
#    - test_h_line(): testing if both functions are able to process any horozontal line;
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
smooth_generator_module = __import__('03-1_smooth_generator')
smooth_fit_based_generator_module = __import__('03-2_smooth_fit_based_generator')

# 1. Create a line/time series with all 0
zero_line = np.zeros(shape = (41,1))
zero_line_df = pd.DataFrame(zero_line).transpose()

# 2. Create a horizontal line with non-zero values
h_line = np.ones(shape = (41,1))
h_line_df = pd.DataFrame(h_line).transpose()

# 3. Create a random time series
random_line = [random.randrange(1, 20, 1) for _ in range(41)]
random_line_df = pd.DataFrame(random_line).transpose()

# 4. Create a time series contains NA values
NA_line = [random.randrange(1, 20, 1) for _ in range(41)]
NA_line[2] = np.nan
NA_line_df = pd.DataFrame(NA_line).transpose()
# test how the function handling a completely zero value line

# 5. create an empty time series
columns = list(range(20))
index = range(10)
empty_df = pd.DataFrame(index = index, columns=columns)

# define the sigma for float handling
sigma = 0.0001
# unit test for smooth generator function

# 1. check time series with all zero values
def test_zero_line():
    assert smooth_generator_module.smooth_generator(zero_line_df).iloc[0,0] == 0, "smooth generator function went wrong for all zero line"
    assert smooth_fit_based_generator_module.smooth_fit_based_generator(zero_line_df).iloc[0,0] == 0, "Fitting based smooth generating function went wrong for all zero line"
# 2. check time series as a horizontal line
def test_h_line():
    assert smooth_generator_module.smooth_generator(h_line_df).iloc[0,0] == 0, "smooth generator function when wrong for horizontal line"
    assert smooth_fit_based_generator_module.smooth_fit_based_generator(zero_line_df).iloc[0,0] == 0, "Fitting based smooth generating function went wrong for horizontal line"

# 3. check time series as a random line (regular time series)
def test_random_line():
    line_avg = np.mean(abs(random_line_df.iloc[0,:]))
    line_var = np.var(np.diff(np.diff(random_line_df.iloc[0,:])))
    assert abs(smooth_generator_module.smooth_generator(random_line_df).iloc[0,0] - line_var/line_avg) < sigma, "smooth generator function went wrong for random line"

    list_x = range(41)
    list_y = random_line_df.iloc[0,:]
    poly = np.polyfit(list_x, list_y, 5, full = True)
    assert abs(smooth_fit_based_generator_module.smooth_fit_based_generator(random_line_df).iloc[0,0] - poly[1][0]) < sigma, "Fitting based smooth generating function went wrong for random line"

# Exceptional handling check
# 1. check a line with NA values
def test_NA_line():
    with pytest.raises(Exception):
        smooth_generator_module.smooth_generator(NA_line_df)
        smooth_fit_based_generator_module.smooth_fit_based_generator(NA_line_df)

# 2. check the non-dataframe input
def test_non_df_input():
    with pytest.raises(Exception):
        smooth_generator_module.smooth_generator(random_line)
        smooth_fit_based_generator_module.smooth_fit_based_generator(random_line)

# 3. check the empty dataframe input
def test_empty_df():
    with pytest.raises(Exception):
        smooth_generator_module.smooth_generator(empty_df)
        smooth_fit_based_generator_module.smooth_fit_based_generator(empty_df)
