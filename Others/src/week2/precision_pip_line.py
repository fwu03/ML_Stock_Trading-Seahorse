# packages for basic python calculation
import pandas as pd
import numpy as np
import os

# warning ignore
import warnings
warnings.filterwarnings("ignore")

# packages for fft
import spectrum
from spectrum import Periodogram, data_cosine

# packages for ML
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


# defind function for data readin
def load_data(folder_path = "../data/buy/"):
    df_gp = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            stock = filename[0:4]
            # print("Loading stock data:", stock, ",")
            temp_df = pd.read_csv(os.path.join(folder_path, filename), delimiter= '\s+', header = None)
            temp_df.rename(columns={123:'rtn'}, inplace=True)
            temp_df = pd.concat([pd.Series([stock] * temp_df.shape[0], name = 'stock'), temp_df], axis=1)
            df_gp = pd.concat([df_gp, temp_df])
            continue
        else:
            continue

    return df_gp


# Extract feature for smoothness
# define function for psd calculation
def psd_generator(data, NFFT = 100, name = "osc"):
    freq = []
    for i in range(data.shape[0]):
        data_osc = data.iloc[i,:]
        p = Periodogram(data_osc, NFFT=NFFT)
        temp_list = list(p.psd)
        freq.append(temp_list)
    col_name = []
    for i in range(int(NFFT/2)+1):
        col_name.append("freq"+str(i))

    psd_df = pd.DataFrame(freq, columns=col_name)
    return psd_df

# define function to calculate smoothness more directly
def smooth_generator(data):
    smooth_list = []
    for i in range(data.shape[0]):
        smooth_list.append(np.var(abs(np.diff(data.iloc[i,:]))))
    smooth = pd.DataFrame(smooth_list, columns=["smooth"])

    return smooth

# define function for numerical differentiation
def derivative(data, space = 1, name = "macd"):
    dy = []
    for i in range(data.shape[0]):
        y = pd.Series(data.iloc[i,:])
        temp_dy = list(np.gradient(y, space))
        dy.append(temp_dy)

    col_name = []
    for i in range(data.shape[1]):
        col_name.append(name + "deriv"+ str(i))

    deriv_df = pd.DataFrame(dy, columns=col_name)

    return deriv_df



def precision_pipline(df_gp, model):

    # seperate indicators
    name_gp = df_gp.iloc[21:100, 0].reset_index()
    osc_gp = df_gp.iloc[21:100, 1:42].reset_index()
    stk_gp = df_gp.iloc[21:100, 42:83].reset_index()
    macd_gp = df_gp.iloc[21:100, 83:124].reset_index()
    rtn_gp = df_gp.iloc[21:100, 124]
    label_gp = np.sign(rtn_gp)
    label_gp = label_gp.map({1: 1, -1: 0, 0:0})
    results_gp = label_gp.map({1: 'EARN', 0: 'LOSS'})

    # Factor for smoothness
    # calculate the fft frequencies distribution for the oscillator
    psd_osc = psd_generator(osc_gp, NFFT = 100)
    smooth_osc = smooth_generator(osc_gp)

    # calculate the dy for macd
    first_deriv_macd = derivative(macd_gp)
    # calculate the ddy for macd
    second_deriv_macd = derivative(first_deriv_macd)


    # calculate the dy for osc (way to study curvature)
    first_deriv_osc = derivative(osc_gp)
    # calculate the ddy for osc
    second_deriv_osc = derivative(first_deriv_osc)

    Feature_matrix = pd.concat([osc_gp, stk_gp, macd_gp, psd_osc, smooth_osc, first_deriv_macd, second_deriv_macd, first_deriv_osc, second_deriv_osc], axis=1)
    # Feature_matrix.head()

    #Split test set
    X, X_test, y, y_test = train_test_split(Feature_matrix, label_gp, test_size=0.2)

    #Split train/validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    print("Training Accuracy:", model.score(X_train, y_train))
    print("Test Accuracy:", model.score(X_test, y_test))

    pre = model.predict(X_test)
    true = y_test
    df_ana = pd.DataFrame({"pre":pre,"true":true})



    percision = sum(df_ana["true"] & df_ana["pre"])/sum(df_ana["pre"])
    print("Test precision:", percision)
    return pre
