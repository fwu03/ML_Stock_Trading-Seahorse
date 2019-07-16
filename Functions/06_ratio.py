# coding: utf-8
# TEAM: Simon Chiu, Gilbert Lei, Fan Wu, Linyang Yu
# DATE: Jun 14, 2019
#
# PURPOSE: The script takes into a two dataframes; data1 and data2, and calculates
#          and the ratio between them based on the number of columns selected
#
# INPUT:
#     - data1 (DataFrame): a dataframe where each rows indicates a signal time series;
#                          data1 should have the same shape as data2
#     - data2 (DataFrame): a dataframe where each rows indicates a signal time series;
#                          data2 should have the same shape as data1
#     - name (String): the assigned column name for the output dataframe
#     - num_col (Int): the number of columns from data1 and data2 will be considered
#                      for the ratio calculation
#
# OUTPUT:
#     - ratio_df (DataFrame): a dataframe contains the ratio between (data1/data2);
#                             For any zero values in data2, the function below considers
#                             their ratio as their data1 values.

# Packages Required
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def int_ratio(data1, data2, name = 'osc/stk', num_col = 5):

  # Input Check
  # check if the input is a dataframe
  if isinstance(data1, pd.DataFrame) == 0:
      raise Exception('time series should be in dataframe')

  if isinstance(data2, pd.DataFrame) == 0:
      raise Exception('time series should be in dataframe')

  # Check if there are null values
  if data1.isnull().values.any() or data2.isnull().values.any():
      raise Exception('data1/data2 contains NaN values')

  # Check if the shape of data frame match
  if data1.shape != data2.shape:
      raise Exception('data1 and data2 should have the same shape')

  # check the input of col number
  if num_col > data1.shape[1]:
      raise Exception('num_col should be less or equal to the number of columns in both data1 and data2')

  # Calculation
  temp_ratio = []
  # Replace all zeros with 0.00001 to avoid Inf when doing division
  data1 = data1.replace(0, 0.00001)
  data2 = data2.replace(0, 0.00001)

  for i in range(data1.shape[0]):
      y1 = pd.Series(data1.iloc[i,:num_col])
      y2 = pd.Series(data2.iloc[i,:num_col])
      # Ratio = y1/y2 for each point
      r = list(y1.values/y2.values)
      temp_ratio.append(r)

  # Assign Column Names
  col_name = []
  for i in range(num_col):
      col_name.append(name+str(i))

  ratio_df = pd.DataFrame(temp_ratio, columns=col_name)
  return ratio_df
