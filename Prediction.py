#!/usr/bin/env python
# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: Jun 26, 2019
#
# PURPOSE: This script performs prediction on the returns of the signals in the Prediction
#          folder and provides a score for each signal to indicate if it is likely to be a gain
#          or a loss. It takes the choice of a saved model generated by the Train function as an argument.
#
# SCRIPT OUTPUT:
#   - A prediction result in the Prediction folder: The last column of the file, Score_rd,
#    is the predicted resulted for the test data (Prediction_yyyy_mm_dd_hhmm.csv).

# PACKAGES
# primary EDA code
import pandas as pd
import numpy as np
import os
import sys
# packages for plots
import matplotlib.pyplot as plt
import seaborn as sns
# warning ignore
import warnings
warnings.filterwarnings("ignore")
# packages for ML
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.interpolate import interp1d
import pickle
# others
import datetime
import time
# =====================================================================================

# VARIABLES
# define the path to the specific folders
path_model = 'Models/'
path_train_data = 'Train_data/'
path_test_data = 'Test_data/'
path_function = 'Functions/'
path_rubric = 'Training_report/rubric/'
path_pred = 'Prediction/'
# define names for all files
date = input("Please enter the model date you selected (yyyy_mm_dd_hhmm): ") #let the user select the model/rubric
rubric_name = path_rubric + 'Train_Rubric_' + date + '.csv'
model_name = path_model + 'Model_' + date + '.sav'
result_name = path_pred + 'Prediction_' + date + '.csv'

# FUNCTIONS & MODEL
# The following code will load all functions in the Function folder, these functions
# will be used in the data importing, data spliting, and feature engneering processes
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Loading functions')
sys.path.append(path_function)
data_import_module = __import__('01_data_import')
data_org_module = __import__('02_data_organize')
smooth_diff_module = __import__('03-1_smooth_generator')
smooth_fit_module = __import__('03-2_smooth_fit_based_generator')
deriv_module = __import__('04_derivative')
vol_module = __import__('05_volatility')
ratio_module = __import__('06_ratio')
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Successfully loaded all functions')

# The following code will load rubric and model from the training results based on usered selected date
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Loading train rubric and model')
model = pickle.load(open(model_name, 'rb'))
train_results = pd.read_csv(rubric_name)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Successfully loaded train rubric and model')

# LOAD DATA - TRAIN & TEST
# The following code will lode all train and test data, and create the special list of stock name index (stock_list)
# based on the train data's stock name, so that the test data will have the same stock name index as the train data.
# Also, we check the duplications between train and test data, and the program will remove the duplications and
# generate warning message if it finds any duplications.
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Loading train and test data')
print()
stock_list = []
df_train, stock_list = data_import_module.data_import(path_train_data, stock_list)
df_test, stock_list = data_import_module.data_import(path_test_data, stock_list, type='test')

# remove duplications
df_train = pd.concat([pd.Series([1] * len(df_train), name = 'type'), df_train], axis=1)
df_train = df_train.iloc[:,:-1]
df_test = pd.concat([pd.Series([0] * len(df_test), name = 'type'), df_test], axis=1)
df_comb = pd.concat([df_train, df_test])
n1 = len(df_comb)
df_comb.drop_duplicates(subset = df_comb.columns[2:], keep=False, inplace=True)
n2 = len(df_comb)
if n1-n2 != 0:
    print('Warning! Duplications between train and test data found')
    print('Automately removed ', n1-n2, 'duplicated data')

# split the updated test dataset after checking the duplications with train data
df = df_comb[df_comb.type == 0]
df = df.iloc[:,1:]

# split the test data into multiple groups
stock_gp, signal_gp, osc_gp, stk_gp, macd_gp, rtn_gp = data_org_module.data_organize(df, type='test')
print()
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Successfully loaded all data')

# FEATURES
# The following code is for feature engneering process, we calculate the:
#   - Derivatives: first and second derivatives for osc, stk, macd
#   - Smoothness: the smoothness level using fit method for osc
#   - Ratio: the ratio between osc&stk, osc&macd
#   - Volatility: the volatility of the stk

# Derivatives
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Calculating features')
print()
print('Derivatives calculation ---- 1/4')
first_deriv_macd = deriv_module.derivative(macd_gp, name="d_macd", relative = True, method='mean')
second_deriv_macd = deriv_module.derivative(first_deriv_macd, name="dd_macd")
first_deriv_osc = deriv_module.derivative(osc_gp, name="d_osc", relative = True, method='mean')
second_deriv_osc = deriv_module.derivative(first_deriv_osc, name="dd_osc")
first_deriv_stk = deriv_module.derivative(stk_gp, name="d_stk", relative = True, method='mean')
second_deriv_stk = deriv_module.derivative(first_deriv_stk, name="dd_stk")
third_deriv_stk = deriv_module.derivative(second_deriv_stk, name="ddd_stk")
# Smoothness
print('Smoothness calculation ---- 2/4')
smooth_osc = smooth_fit_module.smooth_fit_based_generator(osc_gp, "osc")
# Ratio
print('Ratio calculation ---- 3/4')
ratio_os = ratio_module.int_ratio(osc_gp, stk_gp, name = 'osc/stk')
ratio_om = ratio_module.int_ratio(osc_gp, macd_gp, name = 'osc/macd')
# Volatility
print('Volatility calculation ---- 4/4')
vol_stk = vol_module.volatility(stk_gp)

# Combine Features; Used all calculated features above for modeling
Feature_matrix_w_rtn = pd.concat([rtn_gp,
                                  signal_gp,
                                  stock_gp,
                                  smooth_osc,
                                  first_deriv_stk,
                                  second_deriv_stk,
                                  third_deriv_stk,
                                  first_deriv_osc,
                                  second_deriv_osc,
                                  first_deriv_macd,
                                  second_deriv_macd,
                                  vol_stk,
                                  ratio_os,
                                  ratio_om
                                  ], axis=1)
print()
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Successfully calcuated required features')

# PREDICTION
# The following code is used to predict the test data based on the training model and rubric user selected.
# Our methodology is to use the use the training result(score) as a rubric and then interpolate the score for the test
# data based on its predicted return, we will use the interpolcation package in Python with the 'nearest' method.

# Separate returns from Feature matrix
X_test = Feature_matrix_w_rtn.iloc[:, 1:]
rtn_test = Feature_matrix_w_rtn.iloc[:, 0]
signal_test = Feature_matrix_w_rtn.iloc[:, 1]

# predict the test data and create a result table
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Predicting on test data')
test_results = pd.DataFrame({'pred_rtn': model.predict(X_test),'signal': signal_test})
test_results = pd.concat([stock_gp, signal_gp, osc_gp, stk_gp, macd_gp, test_results], axis=1)
test_results = test_results.sort_values(by='pred_rtn')

# interpolate from train rubric
x1 = train_results['pred_rtn'].values
y1 = train_results['score'].values
x2, ind = np.unique(x1, return_index = True) #ensure all x&y pairs are unique
y2 = y1[ind]
f2 = interp1d(x2, y2, kind='nearest') #use nearest method - find closest y based on x

test_results['score'] = f2(test_results['pred_rtn'].values)
test_results['score_rd'] = np.round(test_results['score'],-1) #round the score into group
test_results = test_results.sort_index()
test_results = test_results.drop(['score', 'signal', 'signal_type', 'pred_rtn'], axis=1) #drop the meaningless columns

# update the stock index back to the stock names by using the stock_list
dictOfstock = {i : stock_list[i] for i in range(0, len(stock_list))}
test_results['stock_name'] = [dictOfstock[i] for i in test_results['stock_name'].values]

# output results
test_results.to_csv(result_name, index = False)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Successfully predicted on test data, results are saved under:', result_name)
