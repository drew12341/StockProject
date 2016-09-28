import datetime as dt
import pandas as pd
import quandl
import numpy as np
import StockPredictor

startDate = '2005-01-04'
#endDate = dt.datetime.today().strftime("%Y-%m-%d")
endDate = '2016-07-14'
queryDate = '2016-07-28'
# options of: Open   High    Low  Close      Volume  Adjusted Close
metric = 'Adjusted Close'
screeningMetric = ' - Adjusted Close'

quandl.ApiConfig.api_key = '9gcG2sy8nDdewoUHUVrq'


fileName = 'qandlScreenData.csv'

stocks_query = ['YAHOO/AX_BHP','YAHOO/AAPL','YAHOO/ASX_WBC_AX','YAHOO/ASX_TLS_AX','YAHOO/ASX_RIO_AX']

reloadData = False

if reloadData:
    #merged_dataset = quandl.MergedDataset([('YAHOO/AX_BHP', {'column_index': [4]})])
    merged_dataset = quandl.MergedDataset(stocks_query)

    data_all = merged_dataset.data(
        params={'start_date': startDate, 'end_date': dt.datetime.today().strftime("%Y-%m-%d")}).to_pandas()

    data = pd.DataFrame(index=data_all.index)
    for q in stocks_query:
        data[q] = data_all[q + screeningMetric]

    # save as CSV to stop blowing up their API
    data.to_csv(fileName)
    # save then reload as the qandl date doesn't load right in Pandas
    data = pd.read_csv(fileName)
else:
    data = pd.read_csv(fileName)

#calculate the difference in daily values
#for q in query:
#    data[q] = data[q] - data[q].shift(-1)

data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

data = data.drop('Date', 1)

data = data.pct_change()

outliers = pd.DataFrame(index=data.index)

outliers = pd.DataFrame( np.where( (data > 1.5*data.quantile(0.75)) | (data < 1.5*data.quantile(0.25)), 1, 0 ), columns=data.columns )



res = data.describe().transpose()

res['variance'] = data.var()
res['outliers'] = outliers.sum()

print res

print "SELECTED STOCK", res.sort_values(by=['variance','outliers'], ascending=[False, True]).transpose().keys()[0]

#now we have the selected stock, play ball.

tickerSymbol = res.sort_values(by=['variance','outliers'], ascending=[False, True]).transpose().keys()[0]


#backtest - query the data, and then query the API to see how close it was to the correct value


fileName = "backTest.csv"
if reloadData:
    data = quandl.Dataset(tickerSymbol).data(
        params={'start_date': '2016-01-01', 'end_date': queryDate}).to_pandas()
    # save as CSV to stop blowing up their API
    data.to_csv(fileName)
    # save then reload as the qandl date doesn't load right in Pandas
    data = pd.read_csv(fileName)
else:
    data = pd.read_csv(fileName)

#print data[metric]

print "Actual",data[metric][data['Date'] == queryDate].values[0]
actual = data[metric][data['Date'] == queryDate].values[0]
endDatePrice = data[metric][data['Date'] == endDate].values[0]

def varianceOfReturn(endPrice, actualPrice, predictedPrice):
    t1 = abs(actualPrice- endPrice)
    p1 = abs(predictedPrice-actualPrice)
    return (p1/t1)*100.0

sp = StockPredictor.StockPredictor('abc')

sp.loadData(tickerSymbol, startDate, endDate, reloadData=reloadData, fileName='qandlData.csv')
sp.prepareData(queryDate, metric=metric, sequenceLength=5)
'''
sp.trainLinearRegression()
predicted = sp.predictLinearRegression()
print "Actual:", actual, "Predicted by Linear Regression", predicted
print "Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0)

sp.trainSVR()
predicted = sp.predictSVR()
print "Actual:", actual, "Predicted by SVR", predicted
print "Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0)
'''

sp.trainNN()
predicted = sp.predictNN()
print "Actual:", actual, "Predicted by NNet", predicted
print "Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0)


sp.trainRNN()
predicted = sp.predictRNN()
print "end date price", endDatePrice
print "Actual:", actual, "Predicted by RNN", predicted
print "Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0)
print "Variance of return:{:.4f} %".format(varianceOfReturn(endDatePrice,actual,predicted))