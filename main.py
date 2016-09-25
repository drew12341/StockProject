from sklearn import svm, metrics, preprocessing

import datetime as dt
import pandas as pd
import numpy as np
import quandl
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.decomposition import PCA, FastICA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import renders as rs
# Show matplotlib plots inline (nicely formatted in the notebook)
import matplotlib.pyplot as plt

startDate = '2005-01-04'
endDate = dt.datetime.today().strftime("%Y-%m-%d")
#endDate = ''
metric = ' - Adjusted Close'

queryDate = '2016-09-24'

tickerSymbols = ['YAHOO/AX_BHP','YAHOO/AAPL','YAHOO/ASX_WBC_AX','YAHOO/INDEX_AORD','YAHOO/DRAM','YAHOO/INDEX_GOX']
query = 'YAHOO/AX_BHP'

def days_between(d1, d2):
    d1 = dt.datetime.strptime(d1, "%Y-%m-%d")
    d2 = dt.datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


sequenceLength = 5
predictAhead = days_between(queryDate, endDate)

quandl.ApiConfig.api_key = '9gcG2sy8nDdewoUHUVrq'

fileName = 'qandlData.csv'



reload = False
if reload:
    #merged_dataset = quandl.MergedDataset([('YAHOO/AX_BHP', {'column_index': [4]})])
    merged_dataset = quandl.MergedDataset(tickerSymbols)

    data_all = merged_dataset.data(
        params={'start_date': startDate, 'end_date': dt.datetime.today().strftime("%Y-%m-%d")}).to_pandas()

    data = pd.DataFrame(index=data_all.index)
    for symbol in tickerSymbols:
        data[symbol] = data_all[symbol + metric];

    # save as CSV to stop blowing up their API
    data.to_csv(fileName)
    # save then reload as the qandl date doesn't load right in Pandas
    data = pd.read_csv(fileName)
else:
    data = pd.read_csv('qandlData.csv')

# have the data - now lets look for any correlation within the data using PCA
# Due to differing markets and timezones, public holidays etc (e.g. BHP being an Australian stock,
# ASX doesn't open on 26th Jan due to National Holiday) there are some gaps in the data.
# from manual inspection and knowledge of the dataset, its safe to take the previous days' value

data.fillna(method='ffill', inplace=True)
#catch where there is no previous value - just replace with 0
data.fillna(0.000001, inplace=True)

#take out the value we are trying to predict
label = data[query]
#don't need 'date' as its the order (index) we are mostly concerned with
new_data = data.drop([query, 'Date'], axis = 1)


#scaled_data = new_data.apply(np.log)
#scaled_data = new_data
scaled_data = preprocessing.scale(new_data)
scaled_data = pd.DataFrame(scaled_data)



parameters = {'C': [1,10], 'epsilon':[0.1, 1e-2, 1e-3]}
r2_scorer = metrics.make_scorer(metrics.r2_score)

clf = svm.SVR(C=10, epsilon=0.1)

#split the data into train and test sets.  1/5th of the data is reserved for test
X_train, X_test, y_train, y_test = train_test_split(scaled_data, label, test_size=0.25, random_state=42)


cvsets = ShuffleSplit(X_train.shape[0], n_iter=10, random_state=42)
#grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=r2_scorer, cv=cvsets)
grid_obj = svm.SVR(C=10, epsilon=0.1)
grid_obj.fit(X_train, y_train)
#print "best params", grid_obj.best_params_

predicttrain = grid_obj.predict(X_train)
predicttest = grid_obj.predict(X_test)

print "R2 score for training set: {:.4f}.".format(r2_score(predicttrain, y_train))
print "R2 score for test set: {:.4f}.".format(r2_score(predicttest, y_test))
