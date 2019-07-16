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
    def psd_calculator(self, NFFT = 100, name = "osc"):
        """
        calculate the psd seq as new feature
        """
        freq = []
        for i in range(self.osc_gp.shape[0]):
            data_osc = self.osc_gp.iloc[i,:]
            p = Periodogram(data_osc, NFFT=NFFT)
            temp_list = list(p.psd)
            freq.append(temp_list)
        col_name = []
        for i in range(int(NFFT/2)+1):
            col_name.append("freq"+str(i))

        psd_df = pd.DataFrame(freq, columns=col_name)
        psd_df.index = self.osc_gp.index

        return psd_df

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

    def features_generator(self, psd=True, smooth=True, curvature=True, MACD_derivative = True, partial_smooth = True):

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


        Feature_matrix = self.data.iloc[:,1:-1]
        # calculate the fft frequencies distribution for the oscillator
        if psd==True:
            self.psd_osc = self.psd_calculator(NFFT = 100)
            Feature_matrix = pd.merge(Feature_matrix, self.psd_osc, left_index=True, right_index=True)

        # print(Feature_matrix)
        # Feature_matrix = pd.concat([Feature_matrix, psd_osc])

    # Factor for smoothness
        if smooth==True:
            self.smooth_osc = self.smooth_calculator()
            Feature_matrix = pd.merge(Feature_matrix, self.smooth_osc, left_index=True, right_index=True)

    # Factor for curvature
        if curvature == True:
        # calculate the dy for osc (way to study curvature)
            self.first_deriv_osc = self.first_derivative_calculator(name = "osc")
            Feature_matrix = pd.merge(Feature_matrix, self.first_deriv_osc, left_index=True, right_index=True)
        # calculate the ddy for osc
            self.second_deriv_osc = self.second_derivative_calculator(name = "osc")
            Feature_matrix = pd.merge(Feature_matrix, self.second_deriv_osc, left_index=True, right_index=True)

    # MACD dirivative
        if MACD_derivative == True:
            # calculate the dy for macd
            self.first_deriv_macd = self.first_derivative_calculator(name = "macd")
            Feature_matrix = pd.merge(Feature_matrix, self.first_deriv_macd, left_index=True, right_index=True)
        # calculate the ddy for macd
            self.second_deriv_macd = self.second_derivative_calculator(name = "macd")
            Feature_matrix = pd.merge(Feature_matrix, self.second_deriv_macd, left_index=True, right_index=True)
        # Feature_matrix = pd.concat([Feature_matrix, first_deriv_macd,second_deriv_macd])

    # partial smoothness
        if partial_smooth == True:
            self.partial_smooth = self.smooth_partial()
            Feature_matrix = pd.merge(Feature_matrix, self.partial_smooth, left_index=True, right_index=True)


        self.feature_result_df = pd.merge(Feature_matrix, self.label_gp, left_index=True, right_index=True)

    def model_train(self, model_type = "RandomForest", n_estimators = 20, max_depth = 80):

        if model_type == "RandomForest":
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        Feature_matrix = self.feature_result_df.iloc[:,:-1]
        label_gp = self.feature_result_df.iloc[:,-1]

        #Split test set
        self.X, self.X_test, self.y, self.y_test = train_test_split(Feature_matrix, label_gp, test_size=0.2)

        #Split train/validation set
        # self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=0.2)

        self.model.fit(self.X, self.y)

        scores = cross_val_score(self.model, self.X, self.y, cv=5)

        print("Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("Test Accuracy:", self.model.score(self.X_test, self.y_test))


        prob = self.model.predict_proba(self.X_test)
        # not quite familar and confortable with the log probability, just going to the normal probability

        # get the probbability of being 1
        proba_1 = []
        for probs in prob:
            proba_1.append(probs[1])

        df_result = pd.DataFrame({"predicted_prob":proba_1,
                             "true_label": self.y_test})

        df_count = df_result.groupby("predicted_prob").count()
        df_win = df_result.groupby("predicted_prob").sum()

        self.report = pd.DataFrame(df_win/df_count[["true_label"]])
        self.report["count"] = df_count

        self.report.columns = ["true_win_rate", "count"]

    #def feature_analyzing(self):
        
