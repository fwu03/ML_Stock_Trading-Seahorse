# coding: utf-8
# This script is to test whether the derivative() function works properly. 
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# Date: Jun 19, 2019

import pandas as pd
import numpy as np
import pytest
import os
import sys

sys.path.append('../')
dr = __import__('04_derivative')

# prepare a dataframe with valid values 
input_df = pd.concat([pd.Series(range(0, 10)), pd.Series(range(20, 40, 2))], axis=1).T
input_df = input_df.replace(0, 0.00001)
# prepare a dataframe with a NaN value
input_df_with_NaN = input_df.copy()
input_df_with_NaN.iloc[0, 2] = np.nan
# set tolerance level 
tol = 1e-6

# test whether the function handles properly when the first parameter (pandas dataframe) is in wrong datatype 
def test_df_datatype_wrong():
    with pytest.raises(TypeError):
        dr.derivative('wrong data type')

# test whether the function handles properly when there are NaN values in the dataframe 
def test_NaN_values():
    with pytest.raises(ValueError): 
        dr.derivative(input_df_with_NaN)

# test whether the function handles properly when the third parameter has a wrong datatype
def test_wrong_relative_value():
    with pytest.raises(ValueError):
        dr.derivative(input_df, relative=123)

# test whether the function works properly when all inputs are valid 
def test_derivative(): 
    
    # firstly, test whether the function can calculate absolute derivatives correctly
    dy = []
    dy.append(list(np.diff(input_df.iloc[0, :])))
    dy.append(list(np.diff(input_df.iloc[1, :])))
    
    col_names = [] 
    for i in range(len(dy[0])):
        col_names.append('d' + str(i))
        
    expected_output = pd.DataFrame(dy, columns = col_names)
    actual_output = dr.derivative(input_df, name='d', relative=False)
    assert actual_output.equals(expected_output), "Function derivative() didn't work properly."
    
    # secondly, test whether the function can calculate relative derivatives correctly
    exp_prc_mvmt = []
    exp_prc_mvmt.append(list(np.diff(input_df.iloc[0, :])/input_df.iloc[0, :-1]))
    exp_prc_mvmt.append(list(np.diff(input_df.iloc[1, :])/input_df.iloc[1, :-1]))
    
    actual_output = dr.derivative(input_df, name='d', relative=True, method='value')
    assert np.allclose(np.array(actual_output), np.array(exp_prc_mvmt), tol), "Function derivative() didn't work properly."
    