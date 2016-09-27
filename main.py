import datetime as dt
import pandas as pd
import quandl


import StockPredictor

startDate = '2005-01-04'
#endDate = dt.datetime.today().strftime("%Y-%m-%d")
endDate = '2016-07-14'
queryDate = '2016-07-28'
# options of: Open   High    Low  Close      Volume  Adjusted Close
metric = 'Adjusted Close'



tickerSymbol = 'YAHOO/AX_BHP'

quandl.ApiConfig.api_key = '9gcG2sy8nDdewoUHUVrq'

#backtest - query the data, and then query the API to see how close it was to the correct value
reloadData = True

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

sp = StockPredictor.StockPredictor('abc')

sp.loadData(tickerSymbol, startDate, endDate, reloadData=reloadData, fileName='qandlData.csv')
sp.prepareData(queryDate, metric=metric, sequenceLength=5)

sp.trainLinearRegression()
predicted = sp.predictLinearRegression()
print "Actual:", actual, "Predicted by SVR", predicted
print "Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0)

sp.trainSVR()
predicted = sp.predictSVR()
print "Actual:", actual, "Predicted by SVR", predicted
print "Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0)

sp.trainNN()
predicted = sp.predictNN()
print "Actual:", actual, "Predicted by SVR", predicted
print "Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0)


sp.trainRNN()
predicted = sp.predictRNN()
print "Actual:", actual, "Predicted by RNN", predicted
print "Percent Difference:{:.4f} %".format(abs((actual-predicted)/actual)*100.0)