import datetime as dt
import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing, svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score

startDate = '2005-01-04'
endDate = dt.datetime.today().strftime("%Y-%m-%d")
#endDate = ''
metric = ' - Adjusted Close'

quandl.ApiConfig.api_key = '9gcG2sy8nDdewoUHUVrq'

fileName = 'qandlTestData.csv'

query = 'YAHOO/AX_BHP'

reload = False
if reload:
    #merged_dataset = quandl.MergedDataset([('YAHOO/AX_BHP', {'column_index': [4]})])
    merged_dataset = quandl.MergedDataset([query])

    data_all = merged_dataset.data(
        params={'start_date': startDate, 'end_date': dt.datetime.today().strftime("%Y-%m-%d")}).to_pandas()

    data = pd.DataFrame(index=data_all.index)
    data[query] = data_all[query + metric];

    # save as CSV to stop blowing up their API
    data.to_csv(fileName)
    # save then reload as the qandl date doesn't load right in Pandas
    data = pd.read_csv(fileName)
else:
    data = pd.read_csv(fileName)

#simple regression
data['Date'] = pd.to_datetime(data['Date'])
data['date_delta'] = (data['Date'] - data['Date'].min())  / np.timedelta64(1,'D')

print data.tail(10)

label = data[query]
new_data = data['date_delta']

scaled_data = preprocessing.scale(new_data)
scaled_data = pd.DataFrame(scaled_data)

clf = svm.SVR(C=10, epsilon=0.1)
X_train, X_test, y_train, y_test = train_test_split(scaled_data, label, test_size=0.25, random_state=42)
clf.fit(X_train, y_train)

predicttrain = clf.predict(X_train)
predicttest = clf.predict(X_test)

print "R2 score for training set (SIMPL): {:.4f}.".format(r2_score(predicttrain, y_train))
print "R2 score for test set (SIMPL): {:.4f}.".format(r2_score(predicttest, y_test))

## alternative approach: sliding scale

#number of days empirically determined as having logical sequence
sequenceLength = 5
#number day to predict ahead
predictAhead = 2

tslag = pd.DataFrame(index=data.index)

for i in xrange(0,sequenceLength+predictAhead):
    tslag["Lag%s" % str(i + 1)] = data[query].shift(1 - i)



#tslag = tslag.shift(sequenceLength+predictAhead-1)
tslag.shift(-2)
tslag['date_delta'] = data['date_delta']

tslag.dropna(inplace=True)

print "tslag size", len(tslag)

print tslag.tail()

label = tslag['Lag'+str(sequenceLength+predictAhead)]
new_data = tslag.ix[:, 0:sequenceLength]

scaled_data = preprocessing.scale(new_data)
scaled_data = pd.DataFrame(scaled_data)


clf = svm.SVR(C=10, epsilon=0.1)
X_train, X_test, y_train, y_test = train_test_split(scaled_data, label, test_size=0.25, random_state=42)

clf.fit(X_train, y_train)

predicttrain = clf.predict(X_train)
predicttest = clf.predict(X_test)

print "R2 score for training set (SLID): {:.4f}.".format(r2_score(predicttrain, y_train))
print "R2 score for test set (SLID): {:.4f}.".format(r2_score(predicttest, y_test))


