from google.colab import drive
drive.mount('/content/gdrive')

# package loading
import pandas as pd
import numpy as np
import os

# warning ignore
import warnings
warnings.filterwarnings("ignore")

# packages for fft
#import spectrum
#from spectrum import Periodogram, data_cosine

# packages for ML
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

# package for ploting
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')

from random import sample

# loading the packages for CNN
from keras.datasets import mnist

from keras.layers import Dense, Dropout, Flatten, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling2D,BatchNormalization, Conv2D,MaxPooling2D

from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD

from keras.preprocessing.image import img_to_array, load_img

from keras.utils import np_utils
from keras.applications.inception_v3 import InceptionV3




# function for data loading
def load_data(folder_path = "../data/buy/"):
    df_gp = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            stock = filename[0:4]
            print("Loading stock data:", stock, ",")
            temp_df = pd.read_csv(os.path.join(folder_path, filename), delimiter= '\s+', header = None)
            temp_df.rename(columns={123:'rtn'}, inplace=True)
            temp_df = pd.concat([pd.Series([stock] * temp_df.shape[0], name = 'stock'), temp_df], axis=1)
            df_gp = pd.concat([df_gp, temp_df], ignore_index=True)
            df_gp.reset_index()
            continue
        else:
            continue

    # drop rows with NA
    rows_to_drop = []
    for i in range(df_gp.shape[0]):
        if sum(df_gp.iloc[i,:].isnull()):
            rows_to_drop.append(i)

    df_gp = df_gp.drop(rows_to_drop, axis=0)
    #df_gp = df_gp.iloc[:,1:]

    # remove duplicates
    df_gp = df_gp.drop_duplicates(subset=df_gp.columns.difference(['stock']))

    # change data type
    df_gp.iloc[:, 1:] = df_gp.iloc[:, 1:].astype(float)

    df_gp = df_gp.reset_index(drop = True)
    return df_gp

df_gp = load_data("/content/gdrive/My Drive/buy/")
# seperate indicators and returns
# name_gp = df_gp.iloc[:, 0]
osc_gp = df_gp.iloc[:, 1:42]
stk_gp = df_gp.iloc[:, 42:83]
macd_gp = df_gp.iloc[:, 83:124]
rtn_gp = df_gp.iloc[:, 124]
label_gp = np.sign(rtn_gp)
label_gp = label_gp.map({1: 1, -1: 0, 0:0})
results_gp = label_gp.map({1: 'EARN', 0: 'LOSS'})
label_gp = pd.DataFrame({"label": label_gp})


class OSC_analyzing_pipeline():

    """
    This class is designed and modified for the Seahorse program. It is applicable for Random Forest
    Classifier originally.

    Attributes:
    ----------------------------------------------------------------------------
    self.data: the input dataset that we will be using for training and testing
    self.model: the classifier we want to explore
    self.osc_gp: subset for the oscillator data
    * new addition: self.stk: subset for original stock data
    self.macd_gp: subset for the macd data
    self.rtn_gp: subset for the return values
    self.label_gp: the output for training and testing (the reponse variable)

    self.psd
    self.smooth
    self.first_derivitive_macd
    self.second_derivitive_macd
    self.first_derivitive_osc
    self.second_derivitive_osc
    self.partial_smooth

    self.feature_result_df: the feature matrix and label generated from the feature_generator function, it can be used to visual which feature is participated in model training
    self.X_train: to visual and further call the training dataset
    self.X_test: to visual and further call the test dataset
    self.y_train: to visual and further call the training dataset
    self.y_test: to visual and further call the test dataset

    self.report: generate the report to compare the winning rate with the original test
    The class object also contains all the attributes belongs to the model/classifier originally.

    Functions:
    ----------------------------------------------------------------------------
    * new addition: EDA_visualization(): function for visualization the record
    psd_calculator(): function calculate the fft values, callable with return value to be visualized
    smooth_calculator(): function for calculating the smoothness value, already modified and can handle the amplitude changes
    derivative_calculator(): function for calculating the derivitive for macd and osc
    features_generator(): function to decide which feature can be used in the final model training, the input is True/False for certain feature to be involved. In this function, all three function above will be called
    model_train(): training the model based on the feature matrix

    """

    # define the init
    def __init__(self, data):
        """
        Keyword argument:
        data -- (df) the input dataset for processing
        model -- (ML model) the classifier we are interested of, currently it is just the random forest
        """
        self.data = data

        # seperate indicators and returns
        # name_gp = df_gp.iloc[:, 0]
        self.osc_gp = self.data.iloc[:, 1:42]
        self.stk_gp = self.data.iloc[:, 42:83]
        self.macd_gp = self.data.iloc[:, 83:124]
        self.rtn_gp = self.data.iloc[:, 124]
        self.label_gp = np.sign(self.rtn_gp)
        self.label_gp = self.label_gp.map({1: 1, -1: 0, 0:0})
        self.results_gp = self.label_gp.map({1: 'EARN', 0: 'LOSS'})
        self.label_gp = pd.DataFrame({"label": self.label_gp})


    def EDA_visualization(self, index, osc_ind = True, stk_ind = False, macd_ind = False):
        if osc_ind:
            # plot Oscillator
            plt.figure(figsize=(12,4))
            plt.plot(range(41),  self.osc_gp.iloc[index,:])
            plt.gca().invert_xaxis()
            plt.axhline(y=0.0, color = "black",linestyle='--')
            plt.ylabel("Oscillator")
            plt.xlabel("Time")
            plt.title("Oscillator 3-Day Time Series")
            plt.show()


        if stk_ind:
            # plot stock price
            plt.figure(figsize=(12,4))
            plt.plot(range(41), self.stk_gp.iloc[index,:])
            plt.legend()
            plt.gca().invert_xaxis()
            plt.ylabel("Stock Price")
            plt.xlabel("Time")
            plt.title("Stock Price 3-Day Time Series")
            plt.show()

        if macd_ind:
        # plot MACD
            plt.figure(figsize=(12,4))
            plt.plot(range(41),  self.macd_gp.iloc[index,:])
            plt.legend()
            plt.gca().invert_xaxis()
            plt.axhline(y=0.0, color = "black",linestyle='--')
            plt.ylabel("MACD")
            plt.xlabel("Time")
            plt.title("MACD 3-Day Time Series")
            plt.show()

    # define function for psd calculation
    # define function to calculate smoothness more directly
    def smooth_calculator(self):
        smooth_list = []
        for i in range(self.osc_gp.shape[0]):
            amp = np.mean(abs(self.osc_gp.iloc[i,:]))
            if amp == 0:
                smooth_list.append(0)
            else:
                smooth_list.append(np.var(np.diff(self.osc_gp.iloc[i,:]))/amp)

        smooth = pd.DataFrame(smooth_list, columns=["smooth"])
        smooth.index = self.osc_gp.index

        return smooth

    # define the function to explore the influence of partial smoothness
    def smooth_partial(self):
        # get the nearest 10 points
        osc_10 = self.osc_gp.iloc[:,-10:]
        # calculate the smoothness of 10 points
        smooth_10 = []
        for i in range(osc_10.shape[0]):
            amp = np.mean(abs(osc_10.iloc[i,:]))
            if amp == 0:
                smooth_10.append(0)
            else:
                smooth_10.append(np.var(np.diff(osc_10.iloc[i,:]))/amp)


        # get the nearest 20 points
        osc_20 = self.osc_gp.iloc[:,-20:]
        smooth_20 = []
        for i in range(osc_20.shape[0]):
            amp = np.mean(abs(osc_20.iloc[i,:]))
            if amp == 0:
                smooth_20.append(0)
            else:
                smooth_20.append(np.var(np.diff(osc_20.iloc[i,:]))/(np.mean(abs(osc_20.iloc[i,:]))))

        # get the nearest 30 points
        osc_30 = self.osc_gp.iloc[:,-30:]
        smooth_30 = []
        for i in range(osc_30.shape[0]):
            amp = np.mean(abs(osc_30.iloc[i,:]))
            if amp == 0:
                smooth_30.append(0)
            else:
                smooth_30.append(np.var(np.diff(osc_30.iloc[i,:]))/(np.mean(abs(osc_30.iloc[i,:]))))

        partial_smooth = pd.DataFrame({"smooth_10": smooth_10,
                                      "smooth_20": smooth_20,
                                      "smooth_30": smooth_30,})

        partial_smooth.index = self.osc_gp.index
        return partial_smooth

    def amp_standardize(self, osc_std = True, stk_std = True, macd_std = True):

        # standardize osc
        if osc_std == True:
            norm_osc = []
            mean_amp = []

            for i in range(self.osc_gp.shape[0]):
                mean_amp_temp = np.mean(abs(self.osc_gp.iloc[i,:]))
                mean_amp.append(mean_amp_temp)
                if mean_amp == 0:
                    std = lambda x: x*mean_amp_temp
                    norm_osc_temp_list = list(map(std, self.osc_gp.iloc[i,:].tolist()))
                    norm_osc.append(norm_osc_temp_list)
                else:
                    std = lambda x: x/mean_amp_temp
                    norm_osc_temp_list = list(map(std, self.osc_gp.iloc[i,:].tolist()))
                    norm_osc.append(norm_osc_temp_list)


            col_name = []
            for i in range(len(norm_osc[0])):
                col_name.append("std_osc"+str(i))

            amp_osc = pd.DataFrame(norm_osc, columns=col_name)
            amp_osc["mean_amp_osc"] = mean_amp
            amp_osc.index = self.osc_gp.index

        if stk_std == True:
            norm_stk = []
            mean_amp_stk = []

            for i in range(self.stk_gp.shape[0]):
                mean_amp_temp = np.mean(abs(self.stk_gp.iloc[i,:]))
                mean_amp_stk.append(mean_amp_temp)
                if mean_amp_temp == 0:
                    std = lambda x: x*mean_amp_temp
                    norm_stk_temp_list = list(map(std, self.stk_gp.iloc[i,:].tolist()))
                    norm_stk.append(norm_stk_temp_list)
                else:
                    std = lambda x: x/mean_amp_temp
                    norm_stk_temp_list = list(map(std, self.stk_gp.iloc[i,:].tolist()))
                    norm_stk.append(norm_stk_temp_list)

            col_name = []
            for i in range(len(norm_stk[0])):
                col_name.append("std_stk"+str(i))

            amp_stk = pd.DataFrame(norm_stk, columns=col_name)
            amp_stk["mean_amp_stk"] = mean_amp_stk
            amp_stk.index = self.stk_gp.index

        if macd_std == True:
            norm_macd = []
            mean_amp_macd = []

            for i in range(self.macd_gp.shape[0]):
                mean_amp_temp = np.mean(abs(self.macd_gp.iloc[i,:]))
                mean_amp_macd.append(mean_amp_temp)
                if mean_amp_temp == 0:
                    std = lambda x: x*mean_amp_temp
                    norm_macd_temp_list = list(map(std, self.macd_gp.iloc[i,:].tolist()))
                    norm_macd.append(norm_macd_temp_list)
                else:
                    std = lambda x: x/mean_amp_temp
                    norm_macd_temp_list = list(map(std, self.macd_gp.iloc[i,:].tolist()))
                    norm_macd.append(norm_macd_temp_list)

            col_name = []
            for i in range(len(norm_stk[0])):
                col_name.append("std_macd"+str(i))

            amp_macd = pd.DataFrame(norm_macd, columns=col_name)
            amp_macd["mean_amp_macd"] = mean_amp_macd
            amp_macd.index = self.macd_gp.index

        return amp_osc, amp_stk, amp_macd

    def first_derivative_calculator(self, space = 1, name = "macd"):


        if name == "macd":
            dy = []
            for i in range(self.macd_gp.shape[0]):
                y = pd.Series(self.macd_gp.iloc[i,:])
                temp_dy = list(np.gradient(y, space))
                dy.append(temp_dy)

            col_name = []
            for i in range(self.macd_gp.shape[1]):
                col_name.append(name + "deriv"+ str(i))

            deriv_df = pd.DataFrame(dy, columns=col_name)
            deriv_df.index = self.macd_gp.index

        if name == "osc":
            dy = []
            for i in range(self.osc_gp.shape[0]):
                y = pd.Series(self.osc_gp.iloc[i,:])
                temp_dy = list(np.gradient(y, space))
                dy.append(temp_dy)

            col_name = []
            for i in range(self.osc_gp.shape[1]):
                col_name.append(name + "deriv"+ str(i))

            deriv_df = pd.DataFrame(dy, columns=col_name)
            deriv_df.index = self.osc_gp.index

        if name == "stk":
            dy = []
            for i in range(self.stk_gp.shape[0]):
                y = pd.Series(self.stk_gp.iloc[i,:])
                temp_dy = list(np.gradient(y, space))
                dy.append(temp_dy)

            col_name = []
            for i in range(self.stk_gp.shape[1]):
                col_name.append(name + "deriv"+ str(i))

            deriv_df = pd.DataFrame(dy, columns=col_name)
            deriv_df.index = self.stk_gp.index
        return deriv_df

    def second_derivative_calculator(self, space = 1, name = "macd"):

        if name == "macd":

            ddy = []
            for i in range(self.first_deriv_macd.shape[0]):
                y = pd.Series(self.first_deriv_macd.iloc[i,:])
                temp_ddy = list(np.gradient(y, space))
                ddy.append(temp_ddy)

            col_name = []
            for i in range(self.first_deriv_macd.shape[1]):
                col_name.append(name + "sec_deriv"+ str(i))

            sec_deriv_df = pd.DataFrame(ddy, columns=col_name)
            sec_deriv_df.index = self.first_deriv_macd.index

        if name == "osc":

            ddy = []
            for i in range(self.first_deriv_osc.shape[0]):
                y = pd.Series(self.first_deriv_osc.iloc[i,:])
                temp_ddy = list(np.gradient(y, space))
                ddy.append(temp_ddy)

            col_name = []
            for i in range(self.first_deriv_osc.shape[1]):
                col_name.append(name + "sec_deriv"+ str(i))

            sec_deriv_df = pd.DataFrame(ddy, columns=col_name)
            sec_deriv_df.index = self.first_deriv_osc.index


        return sec_deriv_df

    def var_stock_price(self):
        variance_stk = []
        for i in range(self.stk_gp.shape[0]):
            variance_stk.append(np.var(self.stk_gp.iloc[i,:]))

        variance = pd.DataFrame(variance_stk, columns=["stk_variance"])
        variance.index = self.stk_gp.index

        return variance

    def features_generator(self,
                           base_features = True,
                           smooth=True,
                           standardize = True,
                           curvature=True,
                           derivative = True,
                           partial_smooth = True,
                           var_stk = True):
    # Feature order:
    # 0-40, osc;
    # 41-81, stk;
    # 82-122: macd;
    # 123-173: freq
    # 174: smooth
    # 175-215: first derivitive of osc
    # 216-256: second derivitive of osc
    # 257-297: first derivitive of macd
    # 297-338: second derivitive of macd


        self.stock = pd.DataFrame(self.data.stock)
        self.stock = pd.get_dummies(self.stock)
        self.stock.index = self.data.stock.index

        Feature_matrix = self.data.iloc[:,1:-1]
        Feature_matrix = pd.merge(Feature_matrix, self.stock, left_index=True, right_index=True)



        # print(Feature_matrix)
        # Feature_matrix = pd.concat([Feature_matrix, psd_osc])

    # Factor for smoothness
        if smooth==True:
            print("now generating smoothness")
            self.smooth_osc = self.smooth_calculator()
            Feature_matrix = pd.merge(Feature_matrix, self.smooth_osc, left_index=True, right_index=True)

        if standardize == True:
            print("Normalizing the original osc, stk, macd")
            self.std_osc, self.std_stk, self.std_macd = self.amp_standardize()
            Feature_matrix = pd.merge(Feature_matrix, self.std_osc, left_index=True, right_index=True)
            Feature_matrix = pd.merge(Feature_matrix, self.std_stk, left_index=True, right_index=True)
            Feature_matrix = pd.merge(Feature_matrix, self.std_macd, left_index=True, right_index=True)



    # Factor for curvature
        if curvature == True:
            print("now generating osc derivitives")
        # calculate the dy for osc (way to study curvature)
            self.first_deriv_osc = self.first_derivative_calculator(name = "osc")
            Feature_matrix = pd.merge(Feature_matrix, self.first_deriv_osc, left_index=True, right_index=True)
        # calculate the ddy for osc
            self.second_deriv_osc = self.second_derivative_calculator(name = "osc")
            Feature_matrix = pd.merge(Feature_matrix, self.second_deriv_osc, left_index=True, right_index=True)

    # MACD dirivative
        if derivative == True:
            print("now generating first derivitives")
            # calculate the dy for macd
            self.first_deriv_macd = self.first_derivative_calculator(name = "macd")
            Feature_matrix = pd.merge(Feature_matrix, self.first_deriv_macd, left_index=True, right_index=True)
        # calculate the ddy for macd
            self.second_deriv_macd = self.second_derivative_calculator(name = "macd")
            Feature_matrix = pd.merge(Feature_matrix, self.second_deriv_macd, left_index=True, right_index=True)
        # Feature_matrix = pd.concat([Feature_matrix, first_deriv_macd,second_deriv_macd])
            self.first_deriv_stk = self.first_derivative_calculator(name = "stk")



    # partial smoothness
        if partial_smooth == True:
            print("now generating partial smoothness")
            self.partial_smooth = self.smooth_partial()
            Feature_matrix = pd.merge(Feature_matrix, self.partial_smooth, left_index=True, right_index=True)

         # variance of stk
        if var_stk == True:
            print("now generating stk volatility")
            self.variance_stk = self.var_stock_price()
            Feature_matrix = pd.merge(Feature_matrix, self.variance_stk, left_index=True, right_index=True)

        if base_features == False:
            Feature_matrix = Feature_matrix.iloc[:,:123]

        self.feature_result_df = pd.merge(Feature_matrix, self.label_gp, left_index=True, right_index=True)

# convert data to array
samples_full = []
for i in range(osc_gp.shape[0]):
  osc_list = osc_gp.iloc[i,:].tolist()
  stk_list = stk_gp.iloc[i,:].tolist()
  macd_list = macd_gp.iloc[i,:].tolist()
  temp_array = np.array((osc_list, stk_list, macd_list), dtype=float)
  samples_full.append(temp_array)

sample_y_full = label_gp["label"].tolist()
# complete dataset
X = np.array(samples_full)
y = np.array(sample_y_full)

# Train and test split
# get the index for validation set
index_val = sample(list(range(X.shape[0])), int(X.shape[0]*0.2))
# get the index for train set
index_train = list(set(list(range(X.shape[0]))) - set(index_val))

# the training dataset
sample_X_train = list(samples_full[i] for i in index_train)
sample_y_train = list(sample_y_full[i] for i in index_train)
sample_X_train = np.transpose(sample_X_train, (0,2,1))

# the test dataset
sample_X_val = list(samples_full[i] for i in index_val)
sample_y_val = list(sample_y_full[i] for i in index_val)
sample_X_val = np.transpose(sample_X_val, (0,2,1))

X_train = np.array(sample_X_train)
y_train = np.array(sample_y_train)
X_val = np.array(sample_X_val)
y_val = np.array(sample_y_val)

# construct the basic CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape = (41,3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])

# compile and fit the model
model.fit(X_train,y_train,
      validation_data=(X_val,y_val),
      epochs=5,
      verbose=True)


# constructed the concat CNN models
data_buy_norm = OSC_analyzing_pipeline(data_buy)
data_buy_norm.features_generator()

data_length = data_buy_norm.data.shape[0]
# smoothness
smoothness = data_buy_norm.smooth_osc

print("start processing the basic")
# sample for original data
samples_full_basic = []
for i in range(data_buy_norm.osc_gp.shape[0]):
    osc_list = data_buy_norm.osc_gp.iloc[i,:].tolist()
    stk_list = data_buy_norm.stk_gp.iloc[i,:].tolist()
    macd_list = data_buy_norm.macd_gp.iloc[i,:].tolist()
    temp_array = np.array((osc_list, stk_list, macd_list), dtype=float)
    samples_full_basic.append(temp_array)


print("start processing the deriv")

# sample for first derivitives for those three time series
samples_full_deriv = []
for i in range(data_buy_norm.first_deriv_macd.shape[0]):
    osc_list = data_buy_norm.first_deriv_osc.iloc[i,:].tolist()
    stk_list = data_buy_norm.first_deriv_stk.iloc[i,:].tolist()
    macd_list = data_buy_norm.first_deriv_macd.iloc[i,:].tolist()
    temp_array = np.array((osc_list, stk_list, macd_list), dtype=float)
    samples_full_deriv.append(temp_array)

print("start processing the norm")

# sample for normalizing data
samples_full_norm = []
for i in range(data_buy_norm.std_osc.shape[0]):
    osc_list = data_buy_norm.std_osc.iloc[i,:].tolist()
    stk_list = data_buy_norm.std_stk.iloc[i,:].tolist()
    macd_list = data_buy_norm.std_macd.iloc[i,:].tolist()
    temp_array = np.array((osc_list, stk_list, macd_list), dtype=float)
    samples_full_norm.append(temp_array)


print("creating index")
# get the index for validation set
index_val = sample(list(range(data_length)), int(data_length*0.2))
# get the index for train set
index_train = list(set(list(range(data_length))) - set(index_val))


print("train/test split for y")
# get the train and validation for y
sample_y_full = data_buy_norm.label_gp["label"].tolist()
sample_y_train = list(sample_y_full[i] for i in index_train)
sample_y_val = list(sample_y_full[i] for i in index_val)
y_train = np.array(sample_y_train)
y_val = np.array(sample_y_val)


print("train/test split for sample_basic")
# the training dataset for original dataset_ basic three factor
sample_X_train_basic = list(samples_full_basic[i] for i in index_train)
sample_X_train_basic = np.transpose(sample_X_train_basic, (0,2,1))
sample_X_val_basic = list(samples_full_basic[i] for i in index_val)
# what if we do not use the transpose
sample_X_val_basic = np.transpose(sample_X_val_basic, (0,2,1))
X_train_basic = np.array(sample_X_train_basic)
X_val_basic = np.array(sample_X_val_basic)

print("train/test spilt for sample_norm")
# the training dataset for normalized dataset_ norm three factor
sample_X_train_norm = list(samples_full_norm[i] for i in index_train)
sample_X_train_norm = np.transpose(sample_X_train_norm, (0,2,1))
sample_X_val_norm = list(samples_full_norm[i] for i in index_val)
# what if we do not use the transpose
sample_X_val_norm = np.transpose(sample_X_val_norm, (0,2,1))
X_train_norm = np.array(sample_X_train_norm)
X_val_norm = np.array(sample_X_val_norm)

print("train/test spilt for sample_deriv")

# # the training dataset for derivative dataset_ deriv three factor
sample_X_train_deriv = list(samples_full_deriv[i] for i in index_train)
sample_X_train_deriv = np.transpose(sample_X_train_deriv, (0,2,1))
sample_X_val_deriv = list(samples_full_deriv[i] for i in index_val)
# what if we do not use the transpose
sample_X_val_deriv = np.transpose(sample_X_val_deriv, (0,2,1))
X_train_deriv = np.array(sample_X_train_deriv)
X_val_deriv = np.array(sample_X_val_deriv)


print("train/test spilt for sample_smooth")

# the train/test for smoothness data
sample_X_train_smooth = smoothness.ix[index_train]

sample_X_val_smooth = smoothness.ix[index_val]
# what if we do not use the transpose
X_train_smooth = np.array(sample_X_train_smooth)
X_val_smooth = np.array(sample_X_val_smooth)


from keras.layers.merge import concatenate

# Since the upper code works fine
# test if the following code will work

convnet_basic_ts_in = Input(shape=(41,3))
convnet_basic_ts_dense_1 = Conv1D(filters=164, kernel_size=8, activation='relu', input_shape = (41,3))(convnet_basic_ts_in)
convnet_basic_ts_dense_2 = MaxPooling1D(pool_size=2)(convnet_basic_ts_dense_1)
convnet_basic_ts_dense_3 = Flatten()(convnet_basic_ts_dense_2)
convnet_basic_ts_dense_out = Dense(50, activation='relu')(convnet_basic_ts_dense_3)
convnet_basic_ts_dense_model = Model(convnet_basic_ts_in, convnet_basic_ts_dense_out)

convnet_norm_ts_in = Input(shape=(42,3))
convnet_norm_ts_dense_1 = Conv1D(filters=164, kernel_size=8, activation='relu', input_shape = (41,3))(convnet_norm_ts_in)
convnet_norm_ts_dense_2 = MaxPooling1D(pool_size=2)(convnet_norm_ts_dense_1)
convnet_norm_ts_dense_3 = Flatten()(convnet_norm_ts_dense_2)
convnet_norm_ts_dense_out = Dense(50, activation='relu')(convnet_norm_ts_dense_3)
convnet_norm_ts_dense_model = Model(convnet_norm_ts_in, convnet_norm_ts_dense_out)

convnet_deriv_ts_in = Input(shape=(41,3))
convnet_deriv_ts_dense_1 = Conv1D(filters=164, kernel_size=8, activation='relu', input_shape = (41,3))(convnet_deriv_ts_in)
convnet_deriv_ts_dense_2 = MaxPooling1D(pool_size=2)(convnet_deriv_ts_dense_1)
convnet_deriv_ts_dense_3 = Flatten()(convnet_deriv_ts_dense_2)
convnet_deriv_ts_dense_out = Dense(50, activation='relu')(convnet_deriv_ts_dense_3)
convnet_deriv_ts_dense_model = Model(convnet_deriv_ts_in, convnet_deriv_ts_dense_out)


concatenated = concatenate([convnet_basic_ts_dense_out, convnet_norm_ts_dense_out, convnet_deriv_ts_dense_out])
out = Dense(1, activation='sigmoid', name='output_layer')(concatenated)

merged_model = Model([convnet_basic_ts_in, convnet_norm_ts_in, convnet_deriv_ts_in], out)
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

merged_model.fit([X_train_basic, X_train_norm, X_train_deriv], y=y_train, epochs=10,
             verbose=1, validation_split=0.1, shuffle=True)


             
