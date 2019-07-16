# coding: utf-8
# This script is to test whether the data_organize() function works properly. 
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# Date: Jun 19, 2019


import pandas as pd
import numpy as np
import pytest
import os
import sys

sys.path.append('../')
do = __import__('02_data_organize')

input_df = pd.read_csv('unit_test_data/do_input_df.csv', header=0, index_col=0) 

# test whether the function handles properly when the first input (pandas dataframe) is in wrong datatype
def test_df_datatype_wrong(): 
    with pytest.raises(TypeError): 
        do.data_organize('wrong first input datatype', 'train')
        
# test whether the function handles properly when the first input (pandas dataframe) has a wrong shape
def test_df_format_wrong(): 
    with pytest.raises(ValueError): 
        do.data_organize(input_df.iloc[:, :20], 'train')
    
# test whether the function handles properly when the second input has a wrong value
def test_type_value_wrong(): 
    with pytest.raises(ValueError):
        do.data_organize(input_df, 'wrong_type_value')
        
# test whethere the function works properly when all inputs are valid 
def test_data_organize(): 
    _, _, actual_output_osc, _, _, _ = do.data_organize(input_df, 'train')
    expected_osc_data = input_df.iloc[:, 2:43] 
    assert actual_output_osc.equals(expected_osc_data), "Function data_organize() didn't work properly."