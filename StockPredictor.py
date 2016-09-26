from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm, metrics, preprocessing
import datetime as dt
import pandas as pd
import numpy as np
import quandl
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import renders as rs
# Show matplotlib plots inline (nicely formatted in the notebook)
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key = '9gcG2sy8nDdewoUHUVrq'

class StockPredictor:

    def __init__(self, name):
        self.tickerSymbol = ''

    def loadData(self, tickerSymbol, startDate, endDate, reloadData=False, fileName='qandlData.csv'):

        self.tickerSymbol = tickerSymbol
        self.startDate = startDate
        self.endDate = endDate

        if reloadData:
            data = quandl.Dataset(tickerSymbol).data(
                params={'start_date': startDate, 'end_date': endDate}).to_pandas()

            # save as CSV to stop blowing up their API
            data.to_csv(fileName)

            # save then reload as the qandl date doesn't load right in Pandas
            data = pd.read_csv(fileName)
        else:
            data = pd.read_csv(fileName)

        # Due to differing markets and timezones, public holidays etc (e.g. BHP being an Australian stock,
        # ASX doesn't open on 26th Jan due to National Holiday) there are some gaps in the data.
        # from manual inspection and knowledge of the dataset, its safe to take the previous days' value
        data.fillna(method='ffill', inplace=True)
        self.data = data

    def prepareData(self, predictDate, metric = 'Adjusted Close', sequenceLength=5):


        # number day to predict ahead
        predictDate = dt.datetime.strptime(predictDate, "%Y-%m-%d")
        endDate = dt.datetime.strptime(self.endDate, "%Y-%m-%d")
        #predictAhead = abs((predictDate - endDate).days)

        self.numBdaysAhead = abs(np.busday_count(predictDate, endDate))
        print "bdays ahread", self.numBdaysAhead

        self.sequenceLength = sequenceLength
        self.predictAhead = self.numBdaysAhead
        self.metric = metric

        data = self.data
        # Calculate date delta
        data['Date'] = pd.to_datetime(data['Date'])
        data['date_delta'] = (data['Date'] - data['Date'].min()) / np.timedelta64(1, 'D')

        tslag = pd.DataFrame(index=data.index)

        for i in xrange(0, sequenceLength + self.numBdaysAhead):
            tslag["Lag%s" % str(i + 1)] = data[metric].shift(1 - i)

        tslag.shift(-2)
        tslag['date_delta'] = data['date_delta']

        # Specify which columns are of interest
        trainCols = ['date_delta']
        for i in xrange(0, sequenceLength):
            trainCols.append("Lag%s" % str(i + 1))
        labelCol = 'Lag' + str(sequenceLength + self.numBdaysAhead)

        # get the final row for predictions
        rowcalcs = tslag[trainCols]
        rowcalcs = rowcalcs.dropna()
        self.final_row_unscaled = rowcalcs.tail(1)


        # print tslag.tail(10)
        tslag.dropna(inplace=True)

        label = tslag[labelCol]
        new_data = tslag[trainCols]

        # print "NEW DATA", new_data.tail(1)
        self.scaler = preprocessing.StandardScaler().fit(new_data)
        scaled_data = pd.DataFrame(self.scaler.transform(new_data))

        # print "SCALED DATA", scaled_data.tail(1)
        self.scaled_data = scaled_data
        self.label = label


    def trainSVR(self):
        clf = svm.SVR()
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data, self.label, test_size=0.25, random_state=42)

        parameters = {'C': [1, 10], 'epsilon': [0.1, 1e-2, 1e-3]}
        r2_scorer = metrics.make_scorer(metrics.r2_score)

        cvsets = ShuffleSplit(X_train.shape[0], n_iter=10, random_state=42)
        grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=r2_scorer, cv=cvsets)
        grid_obj.fit(X_train, y_train)
        print "best svr params", grid_obj.best_params_

        predicttrain = grid_obj.predict(X_train)
        predicttest = grid_obj.predict(X_test)

        print "R2 score for training set (SVR): {:.4f}.".format(r2_score(predicttrain, y_train))
        print "R2 score for test set (SVR): {:.4f}.".format(r2_score(predicttest, y_test))
        self.model = grid_obj

    def predictSVR(self):

        inputSeq = self.scaler.transform(self.final_row_unscaled)
        print "inputseq", inputSeq

        inputSeq = pd.DataFrame(inputSeq)

        predicted = self.model.predict(inputSeq)[0]
        print "Predicted", predicted
        return predicted


    def trainNN(self):

        #data = self.scaler.inverse_transform(self.scaled_data.as_matrix())

        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data.as_matrix(), self.label.as_matrix(), test_size=0.25, random_state=42)

        # create model
        model = Sequential()
        model.add(Dense(180, input_dim=X_train.shape[1], init='normal', activation='relu'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.25))
        model.add(Dense(100, init='normal', activation='relu'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.05))
        model.add(Dense(1, init='normal', activation='linear'))

        print model.summary()
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='rmsprop')

        model.fit(X_train, y_train, nb_epoch=150, batch_size=150, verbose=2)

        predicttest = model.predict(X_test)
        predicttrain = model.predict(X_train)


        print "R2 score for training set (NN): {:.4f}.".format(r2_score(predicttrain, y_train))
        print "R2 score for test set (NN): {:.4f}.".format(r2_score(predicttest, y_test))
        self.model = model


    def predictNN(self):
        inputSeq = self.scaler.transform(self.final_row_unscaled)
        #inputSeq = self.final_row_unscaled.as_matrix()
        print "inputseq", inputSeq
        predicted = self.model.predict(inputSeq)[0][0]
        print "Predicted", predicted
        return predicted

    def trainRNN(self):

        colmn = self.data[self.metric]
        colmn = colmn.values
        print "last 3 cols", colmn[-1]
        self.maxlen = 5

        #self.step = 1
        self.step = self.numBdaysAhead

        self.batch_size = 25
        X = []
        y = []
        for i in range(0, len(colmn) - self.step-self.maxlen):
            X.append(colmn[i: i + self.maxlen])
            y.append(colmn[i + self.step+self.maxlen])
        print('nb sequences:', len(X))

        X = np.array(X)
        y = np.array(y)
        print "X and y", X, y

        X = np.reshape(X, X.shape + (1,))
        y = np.reshape(y, y.shape + (1,))

        print('X_train shape:', X.shape)
        print('X_test shape:', y.shape)

        print "X and y", X[-1], y[-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25, random_state=42)

        model = Sequential()
        model.add(LSTM(50,
                       batch_input_shape=(self.batch_size, self.maxlen, 1),
                       return_sequences=True))
        model.add(LSTM(50,
                       batch_input_shape=(self.batch_size, self.maxlen, 1),
                       return_sequences=False))
        model.add(Dense(1))

        print model.summary()
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='rmsprop')

        model.fit(X_train, y_train, nb_epoch=50, batch_size=self.batch_size, verbose=2)

        predicttest = model.predict(X_test)
        predicttrain = model.predict(X_train)

        print "R2 score for training set (NN): {:.4f}.".format(r2_score(predicttrain, y_train))
        print "R2 score for test set (NN): {:.4f}.".format(r2_score(predicttest, y_test))
        self.model = model

    def predictRNN(self):
        cols = self.data[self.metric].tail(self.batch_size)
        cols = cols.values

        '''
        predicted = 0
        for i in xrange(0, self.numBdaysAhead):
            X = []
            for i in range(0, len(cols), self.step):
                X.append(cols[i: i + self.maxlen])

            inputSeq = np.array(X)

            inputSeq = np.reshape(inputSeq, inputSeq.shape + (1,))
            #print "inputseq", inputSeq
            predicted = self.model.predict(inputSeq)[0][0]
            print "Predicted", predicted
            cols = cols[1:]
            cols = np.append(cols, predicted)
            #print "COLS", cols
        print "COLS", cols
        '''
        X = []
        for i in range(0, len(cols)-self.maxlen, self.step):
            X.append(cols[i: i + self.maxlen])

        inputSeq = np.array(X)
        print "test", inputSeq
        inputSeq = np.reshape(inputSeq, inputSeq.shape + (1,))
        # print "inputseq", inputSeq
        predicted = self.model.predict(inputSeq)[0][0]
        print "Predicted", predicted
        return predicted


