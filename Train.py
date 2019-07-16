#!/usr/bin/env python
# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: Jun 26, 2019
#
# PURPOSE: This script performs model training that applies the LightGBM Regressor
#          to the data in the Train_data folder. It then saves the results for prediction later.
#
# SCRIPT OUTPUT:
#   - A training report in the Training_report folder
#     (Train_Report_yyyy_mm_dd_hhmm.pdf)
#   - A trained model in the Model folder
#     (Model_yyyy_mm_dd_hhmm.sav)
#   - A rubric for prediction purpose in the Training_report/rubric folder
#     (Train_Rubric_yyyy_mm_dd_hhmm.csv)
#   - A screenshot for the training report in the Training_report/img folder
#     (Cross_Validation_Result_Boxplot_yyyy_mm_dd_hhmm.png)

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
# packages for generating report
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.units import inch
# others
import datetime
import time
import pickle
# =====================================================================================

# PATH
# define the path to the specific folders
path_model = 'Models/'
path_data = 'Train_data/'
path_function = 'Functions/'
path_report = 'Training_report/'


# FUNCTIONS
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

# LOAD TRAIN DATA
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Loading train data')
stock_list = []
df, stock_list = data_import_module.data_import(path_data, stock_list)

# drop small OSC values
len_before = df.shape[0]
df = df[(df.osc0 < -0.5) | (df.osc0 > 0.5)] #exclude the osc values between -0.5 and 0.5
df = df.reset_index(drop=True)
print("Removed", len_before - df.shape[0], 'records with small OSC')
print()
# data organization - split data into multiple groups for modeling purpose
stock_gp, signal_gp, osc_gp, stk_gp, macd_gp, rtn_gp = data_org_module.data_organize(df)
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
first_deriv_macd = deriv_module.derivative(macd_gp, name="d_macd", relative = True, method = 'mean')
second_deriv_macd = deriv_module.derivative(first_deriv_macd, name="dd_macd")
first_deriv_osc = deriv_module.derivative(osc_gp, name="d_osc", relative = True, method = 'mean')
second_deriv_osc = deriv_module.derivative(first_deriv_osc, name="dd_osc")
first_deriv_stk = deriv_module.derivative(stk_gp, name="d_stk", relative = True, method = 'mean')
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
print()
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Successfully calcuated required features')

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
                                  ratio_om],
                                  axis=1)


# MODEL - 10-FOLD CROSS VALIDATION
# The following code will perform the LGBM Modeling with 10-fold cross validation, in each fold, we will
# split the data into train and validation groups. The train data is used to train the model while the validation
# data is used to test the performance of the model. To capture the performance of the model, we store the validation
# results in all folds, and plot a corresponding boxplot for it.

valid_results_collection = pd.DataFrame()
label_gp = np.round(rtn_gp, 0)

# define 10-fold Cross Validation
n = 10
p = 0
kf = StratifiedKFold(n_splits = n, shuffle = True)
kf.get_n_splits(Feature_matrix_w_rtn, label_gp)

time.sleep(60)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Generating model with ' + str(n) + '-fold cross validation')
print()
# Within each fold
for train_index, test_index in kf.split(Feature_matrix_w_rtn, label_gp):
    # Define train/ validation set (convert X_w_rtn and y to np.array for indexing on the next line)
    X_train_w_rtn, X_valid_w_rtn = Feature_matrix_w_rtn.values[train_index], Feature_matrix_w_rtn.values[test_index]
    y_train, y_valid = rtn_gp.values[train_index], rtn_gp.values[test_index]

    # Separate returns and signal types from Feature matrix (convert them back to pandas)
    X_train = pd.DataFrame(X_train_w_rtn).iloc[:, 1:]
    X_valid = pd.DataFrame(X_valid_w_rtn).iloc[:, 1:]

    rtn_train = pd.DataFrame(X_train_w_rtn).iloc[:, 0]
    rtn_valid = pd.DataFrame(X_valid_w_rtn).iloc[:, 0]

    sig_train = pd.DataFrame(X_train_w_rtn).iloc[:, 1]
    sig_valid = pd.DataFrame(X_valid_w_rtn).iloc[:, 1]

    y_train = pd.Series(y_train)
    y_valid = pd.Series(y_valid)

    # Reset indices on all
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    rtn_train = rtn_train.reset_index(drop=True)

    X_valid = X_valid.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    rtn_valid = rtn_valid.reset_index(drop=True)

    # Train Model
    lg = lgb.LGBMRegressor(silent=True) #create model
    data_train = lgb.Dataset(X_train, label=y_train, categorical_feature=[0,1])
    #hypterparameters setting
    params = {
        'application': 'regression_l2',
        'num_leaves': 300,
        'learning_rate': 0.005,
        'max_depth': 100,
        'n_estimators': 100}
    CVmodel = lgb.train(params, data_train) #train model

    # Create train model table for rubric purpose, as the score for the validation set for refer to the score from train set
    train_pred_rtn = CVmodel.predict(X_train)
    train_scores = pd.DataFrame(train_pred_rtn).rank(pct = True) #use percentile ranking
    train_results = pd.concat([pd.DataFrame(train_pred_rtn), pd.DataFrame(train_scores), y_train, rtn_train, sig_train], axis =1)
    train_results.columns = ['pred_rtn', 'score', 'label', 'act_rtn', 'sig_ty'] #assign new column names
    train_results['label'] = train_results['label'] > 0 #label will be a boolean column; True/False
    train_results['score'] = train_results['score']*100 #assign score to be 0-100
    train_results = train_results.sort_values(by='pred_rtn') #sort the table by predicted return
    train_results['Score_rd'] = np.round(train_results['score'],-1) #round the score into groups

    # Extracts the probabilities of being a good trade
    valid_pred_rtn = CVmodel.predict(X_valid)
    valid_results = pd.concat([pd.DataFrame(valid_pred_rtn), y_valid, rtn_valid, sig_valid], axis =1)
    valid_results.columns = ['pred_rtn', 'label', 'act_rtn', 'sig_ty']

    # Interpolation for validation scores
    # as our methodology is to use the training result(score) as a rubric and then interpolate the score for the validation
    # from the training set based on the predicted return, we decide to use the interpolcation package in Python with the
    # 'nearest' method.

    # Setup the training result - x as predict return, y as score
    x1 = train_results['pred_rtn'].values
    y1 = train_results['score'].values
    x2, ind = np.unique(x1, return_index = True) #ensure all x&y pairs are unique
    y2 = y1[ind]
    f2 = interp1d(x2, y2, kind='nearest') #use nearest method - find closest y based on x

    valid_results['score'] = f2(valid_results['pred_rtn'].values)
    valid_results['label'] = valid_results['act_rtn'] > 0 #create label for validation - a boolean column
    valid_results['Score_rd'] = np.round(valid_results['score'],-1) #round the score into groups

    # Make Results Summary
    # grouping the validation results into five main types: StkUpRate, Count, Stk Mvt%, BuySig%, % of All Trades
    valid_results_summary = pd.concat([valid_results.groupby(['Score_rd']).mean()['label'],
                                       valid_results.groupby(['Score_rd']).count()['label'],
                                       valid_results.groupby(['Score_rd']).mean()['act_rtn'],
                                       valid_results.groupby(['Score_rd']).mean()['sig_ty']],axis=1)
    valid_results_summary.columns = ['StkUpRate', 'Count', 'Stk Mvt%', 'BuySig%']
    valid_results_summary['% of All Trades'] = np.round(valid_results_summary['Count']/np.sum(valid_results_summary['Count']),4)*100
    valid_results_summary = valid_results_summary.drop(['Count'], axis=1)

    # Store it in the Results Collection
    valid_results_collection = pd.concat([valid_results_collection, valid_results_summary])

    # Print Progress
    p = p + 1
    print('CV Modeling Progress: ', round(p*100/n, 0), '%')

# Validation Result - mean and standard deviation
valid_mean_summary = valid_results_collection.groupby(['Score_rd']).mean()
valid_std_summary = valid_results_collection.groupby(['Score_rd']).std()
valid_std_summary.columns = ['StkUpRate(std)','Stk Mvt%(std)','BuySig%(std)','% of All Trades(std)'] #rename columns for std table

# Running Time
currentDT = datetime.datetime.now()
strD = currentDT.strftime("%Y_%m_%d_%H%M")
strDT = currentDT.strftime("%Y-%m-%d %H:%M:%S")

# Names for all outputs
boxplot_name = path_report + "img/Cross_Validation_Result_Boxplot_" + strD + ".png"
report_name = path_report + 'Train_Report_' + strD + '.pdf'
rubric_name = path_report + 'rubric/Train_Rubric_' + strD + ".csv"
model_name = path_model + 'Model_' + strD + '.sav'

# Validation Box-Plot
# The following code is used to generate a boxplot for validation returns results, this plot shows
# the average returns of each score group. Each point represent the average return from one cross-validation run.
plt.figure(figsize=(15,8))
df_plot = valid_results_collection.reset_index()
ax = sns.stripplot(y=df_plot["Stk Mvt%"], x=df_plot["Score_rd"])
ax = sns.boxplot(y=df_plot["Stk Mvt%"], x=df_plot["Score_rd"], boxprops=dict(alpha=.1))
ax.axhline(np.mean(df.rtn), ls='--')
ax.axhline(0, ls='-')
ax.set(xlabel='Score')
ax.set_title('Validation Return Boxplot')
ax.figure.savefig(boxplot_name)
print('Cross validation return boxplot created:', boxplot_name)

# TRAINING REPORT
# The following code is used to generate the training report for the cross validation results in pdf format
# Styles setting
styles = getSampleStyleSheet()
styleN = styles['Normal']
styleH = styles['Heading4']
# Template for the document
doc = SimpleDocTemplate(report_name,
                        pagesize=landscape(letter),
                        rightMargin=72,leftMargin=72,
                        topMargin=72,bottomMargin=18)
# Table formatting
ts = [('ALIGN', (1,1), (-1,-1), 'CENTER'),
      ('GRID',(0,0),(-1,-1), 0.5, colors.grey),
      ('FONT', (0,0), (-1,0), 'Times-Bold'),
      ('FONT', (0,1), (0,-1), 'Times-Bold'),
      ('TEXTCOLOR',(0,0),(1,-1),colors.black),
      ('BACKGROUND', (0,0), (-1,0), colors.lavender)]

# Tables
df1 = valid_mean_summary.round(4) #valiation mean summary table
df2 = valid_std_summary.round(4) # validation standard deviation summary table
# explanation table for the column names
df3 = pd.DataFrame({'Column Name':['Score_rd','StkUpRate','Stk Mvt%','BuySig%','% of All Trades'],
                    'Meaning': ['Signals grouped by the score assigned by the model',
                                'Ratio of signals with positive returns in the group (0:0% - 1:100%)',
                                'Average return of the signals in the group',
                                'Ratio of signals that is a buy signal',
                                'Ratio of total signal in the group (0:none - 1:all signals in this group)']})
# reset index for the tables
df1 = df1.reset_index(drop=False)
df2 = df2.reset_index(drop=False)
df3 = df3.reset_index(drop=False)
# capture their values
list1 = [df1.columns[:,].values.astype(str).tolist()] + df1.values.tolist()
list2 = [df2.columns[:,].values.astype(str).tolist()] + df2.values.tolist()
list3 = [df3.columns[:,].values.astype(str).tolist()] + df3.values.tolist()
# create tables under designed format
table1 = Table(list1, style=ts)
table2 = Table(list2, style=ts)
table3 = Table(list3, style=ts)

# Images - get the boxplot of the cv result
img1 = Image(boxplot_name, 9*inch, 5*inch)

# Explanation for two tables and the boxplot
exp1 = 'The two tables above contains parameters (mean and standard deviation) to describe the signals with scores in a certain group, averaging over all the cross validation run.'
exp2 = 'The boxplot shows the average returns of each score group, each point represent the average return from one cross-validation run. A wider box means the signals in that group is relatively risker, and vice versa.'

# Generate Report
report = []
report.append(Paragraph("Training Model - Cross Validation Result", styles['Title']))
report.append(Spacer(1,0.5*inch))
report.append(Paragraph("Cross Validation Result - Mean", styleH))
report.append(table1)
report.append(Spacer(1,0.3*inch))
report.append(Paragraph("Cross Validation Result - Standard Deviation", styleH))
report.append(table2)
report.append(Spacer(1,0.2*inch))
report.append(Paragraph(exp1, styleN))
report.append(Spacer(1,0.3*inch))
report.append(table3)
report.append(Spacer(1,0.3*inch))
report.append(Paragraph("Cross Validation Result - Boxplot", styleH))
report.append(img1)
report.append(Spacer(1,0.2*inch))
report.append(Paragraph(exp2, styleN))
report.append(Spacer(1,0.3*inch))
report.append(Paragraph("Report Generated Time: " + strDT, styleN))

doc.build(report)
print('Training model report created:', report_name)
print()
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Successfully generated ' + str(n) + '-fold cross validation model')

time.sleep(60)
# MODEL - TRAINING SET
# The following code is used to train the model on the whole dataset we imported at begining;
# instead of spliting them into train and validation. The code will generate a rubric and model that can
# be directly used in the prediction process (Prediction.py)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Generating train model')
print()
label_gp = np.round(rtn_gp, 0)

# Separate returns and signal type from Feature matrix
X_train = pd.DataFrame(Feature_matrix_w_rtn).iloc[:, 1:]
rtn_train = pd.DataFrame(Feature_matrix_w_rtn).iloc[:, 0]
sig_train = pd.DataFrame(Feature_matrix_w_rtn).iloc[:, 1]
y_train = label_gp

# Training
lg = lgb.LGBMRegressor(silent=True) #create model
data_train = lgb.Dataset(X_train, label=y_train, categorical_feature=[0,1])
# hypterparameters setting
params = {
        'application': 'regression_l2',
        'num_leaves': 300,
        'learning_rate': 0.005,
        'max_depth': 100,
        'n_estimators': 100}
model = lgb.train(params, data_train) #train model

# Create training rubric that can be used for the prediction
train_pred_rtn = model.predict(X_train)
train_scores = pd.DataFrame(train_pred_rtn).rank(pct = True) #ranking the predicted returns under percentile
train_results = pd.concat([pd.DataFrame(train_pred_rtn), pd.DataFrame(train_scores), y_train, rtn_train, sig_train], axis =1)
train_results.columns = ['pred_rtn', 'score', 'label', 'act_rtn', 'sig_ty'] #rename columns
train_results['label'] = train_results['label'] > 0 #create label columns with boolean values
train_results['score'] = train_results['score']*100 #rearrange the score to be 0-100
train_results = train_results.sort_values(by='pred_rtn') #sorted the table based on the predicted return
train_results['Score_rd'] = np.round(train_results['score'],-1) #round the score into groups

# Generate model and rubric
pickle.dump(model, open(model_name, 'wb')) #save the model
train_results.to_csv(rubric_name, index = False) #save the rubric
print('Training model created:', model_name)
print('Training rubric created:', rubric_name)
print()
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '- Successfully completed the training process')
