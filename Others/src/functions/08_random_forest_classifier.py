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
#     - Fitted random forest model
#     - A summary dataframe in trade class format

# Packages Required
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")


def random_forest_classifier(X_w_rtn, y, test_size=0.2, cv=5):

    results_summaries_collection = pd.DataFrame()
    results_prob_summaries_collection = pd.DataFrame()

    # n-fold Cross Validation
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X_w_rtn)

    # Within each fold
    for train_index, valid_index in kf.split(X_w_rtn):
        # Define train/ validation set (convert X_w_rtn and y to np.array for indexing on the next line)
        X_train_w_rtn, X_valid_w_rtn = X_w_rtn.values[train_index], X_w_rtn.values[valid_index]
        y_train, y_valid = y.values[train_index], y.values[valid_index]

        # Separate returns from Feature matrix (convert X_w_rtn and y back to pandas)
        X_train = pd.DataFrame(X_train_w_rtn).iloc[:, 1:]
        X_valid = pd.DataFrame(X_valid_w_rtn).iloc[:, 1:]

        rtn_train = pd.DataFrame(X_train_w_rtn).iloc[:, 0]
        rtn_valid = pd.DataFrame(X_valid_w_rtn).iloc[:, 0]

        y_train = pd.Series(y_train)
        y_valid = pd.Series(y_valid)

        # Reset indices on all
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        rtn_train = rtn_train.reset_index(drop=True)

        X_valid = X_valid.reset_index(drop=True)
        y_valid = y_valid.reset_index(drop=True)
        rtn_valid = rtn_valid.reset_index(drop=True)

        # Fit Random Forest
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        # Extracts the probabilities of being a good trade
        log_proba_set_valid = rf.predict_log_proba(X_valid)
        proba_valid = []
        for prob in range(len(log_proba_set_valid)):
            proba_valid.append(np.exp(log_proba_set_valid[prob][1]))

        # Get probabilities summary table
        results_prob = pd.concat([pd.DataFrame(proba_valid), y_valid, rtn_valid], axis =1)
        results_prob.columns = ['prob', 'label', 'return']
        results_prob['prob'] = np.round(results_prob['prob'],1)

        results_prob_summary = pd.concat([results_prob.groupby(['prob']).mean()['label'], results_prob.groupby(['prob']).count()['label'], results_prob.groupby(['prob']).mean()['return']],axis=1)
        results_prob_summary.columns = ['WinRate', 'Count', 'Avg. Return']
        results_prob_summary['% of All Trades'] = np.round(results_prob_summary['Count']/np.sum(results_prob_summary['Count']),4)*100

        # Classify the Classes of the Trades
        # Proba 1.0 => Excellent
        # Proba 0.8 - 1.0 => Great
        # Proba 0.6 - 0.8 => Good
        # Proba 0.0 - 0.6 => Average
        # Proba 0.0 => Sell

        trade_classes = ['Sell'] * len(proba_valid)
        for i in range(len(proba_valid)):
            if proba_valid[i] == 1.0:
                trade_classes[i] = 'Excellent'
            elif proba_valid[i] > 0.8:
                trade_classes[i] = 'Great'
            elif proba_valid[i] > 0.7:
                trade_classes[i] = 'Good'
            elif proba_valid[i] > 0.0:
                trade_classes[i] = 'Average'

        # Put Results Together
        results = pd.concat([pd.DataFrame(proba_valid), pd.DataFrame(trade_classes), y_valid, rtn_valid], axis =1)
        results.columns = ['prob', 'trade_class', 'label', 'return']

        # Make Results Summary
        results_summary = pd.concat([results.groupby(['trade_class']).mean()['label'], results.groupby(['trade_class']).count()['label'], results.groupby(['trade_class']).mean()['return']],axis=1)
        results_summary.columns = ['WinRate', 'Count', 'Avg. Return']
        results_summary['% of All Trades'] = np.round(results_summary['Count']/np.sum(results_summary['Count']),4)*100

        # Store it in the Results Collection
        results_prob_summaries_collection = pd.concat([results_prob_summaries_collection, results_prob_summary])
        results_summaries_collection = pd.concat([results_summaries_collection, results_summary])

    results_summaries_collection = results_summaries_collection.groupby(['trade_class']).mean().reindex(['Excellent', 'Great', 'Good', 'Average', 'Sell'])
    results_prob_summaries_collection = results_prob_summaries_collection.groupby(['prob']).mean()

    return rf, results_prob_summaries_collection, results_summaries_collection
