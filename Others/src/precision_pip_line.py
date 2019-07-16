# packages for basic python calculation
import pandas as pd
import numpy as np

# warning ignore
import warnings
warnings.filterwarnings("ignore")

# packages for fft
import spectrum
from spectrum import Periodogram, data_cosine

# packages for ML
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


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


def precision_pipline(file_path, model):

    # read in the file and seperate indicators
    df = pd.read_csv(file_path, delimiter="\s+", header = None)
    osc = df.iloc[21:,0:41].reset_index()
    stk = df.iloc[21:, 41:82].reset_index()
    macd = df.iloc[21:, 82:123].reset_index()
    rtn = df.iloc[21:,123]
    label = np.sign(rtn)
    label = label.map({1:1, -1:0, 0:0})

    results = label.map({1:"EARN", 0:"LOSS"})

    # calculate the fft frequencies for the oscillator
    psd_osc = psd_generator(osc, NFFT = 100)

    # calculate the dy for macd
    deriv_macd = derivative(macd)

    Feature_matrix = pd.concat([osc, stk, macd, psd_osc, deriv_macd], axis=1)
    # Feature_matrix.head()

    #Split test set
    X, X_test, y, y_test = train_test_split(Feature_matrix, label, test_size=0.2)

    #Split train/validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)

    print("Training Accuracy:", model.score(X_train, y_train))
    print("Test Accuracy:", model.score(X_test, y_test))

    pre = model.predict(X_test)
    true = y_test
    df_ana = pd.DataFrame({"pre":pre,
                          "true":true})



    percision = sum(df_ana["true"] & df_ana["pre"])/sum(df_ana["pre"])
    print("Test precision:", percision)
