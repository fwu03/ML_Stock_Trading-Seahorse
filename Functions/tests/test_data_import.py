# coding: utf-8
# This script is to test whether the data_import() function works properly. 
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# Date: Jun 19, 2019

import pandas as pd
import numpy as np
import pytest
import os
import sys

sys.path.append('../')
di = __import__('01_data_import')

# test whether the function handles properly when an invalid folder path is given as an input
def test_folder_path_nonexistent():
    with pytest.raises(NotADirectoryError):
        di.data_import('nonexistent_folder_path', ['GOOG', 'AMZN', 'MSFT', 'FB', 'AAPL'], 'train')
        
# test whether the function handles properly when the stock list parameter is in wrong data type
def test_stock_list_datatype_wrong():
    with pytest.raises(TypeError): 
        di.data_import('unit_test_data', {'stock': 'GOOG'}, 'train')
        
# test whether the function handles properly when an invalid dataset type value is passed in 
def test_dataset_type_wrong(): 
    with pytest.raises(ValueError):
        di.data_import('unit_test_data', ['GOOG', 'AMZN', 'MSFT', 'FB', 'AAPL'], 'whatever')

# test whether the function works properly when all inputs are valid
def test_data_import(): 
    actual_output_df, _ = di.data_import('unit_test_data', [], 'train') 
    # load in expected outcome data, which was manually prepared beforehand 
    expected_output_df = pd.read_csv('unit_test_data/di_expected_output.csv', header=0, index_col=0)
    assert actual_output_df.equals(expected_output_df), "Function data_import() didn't work properly."