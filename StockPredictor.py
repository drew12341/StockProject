from sklearn import svm, metrics
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
        self.tickerSymbols = []

    def loadData(self, tickerSymbols, startDate, endDate, reloadData=False, metric=' - Adjusted Close', fileName='qandlData.csv'):
        self.tickerSymbols = tickerSymbols
        self.startDate = startDate
        self.endDate = endDate
        if reload:
            # merged_dataset = quandl.MergedDataset([('YAHOO/AX_BHP', {'column_index': [4]})])
            merged_dataset = quandl.MergedDataset(tickerSymbols)

            data_all = merged_dataset.data(
                params={'start_date': startDate, 'end_date': endDate}).to_pandas()

            data = pd.DataFrame(index=data_all.index)
            for symbol in tickerSymbols:
                data[symbol] = data_all[symbol + metric];

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

    def trainData(self,querySymbol, normalise=False, pca=False):
        data = self.data
        # take out the value we are trying to predict
        label = self.data[querySymbol]
        # don't need 'date' as its the order (index) we are mostly concerned with
        new_data = data.drop([querySymbol, 'Date'], axis=1)

        # split the data into train and test sets.  1/5th of the data is reserved for test
        X_train, X_test, y_train, y_test = train_test_split(new_data, label, test_size=0.2, random_state=42)

