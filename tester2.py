import pandas as pd
import quandl
import datetime as dt
import numpy as np


quandl.ApiConfig.api_key = '9gcG2sy8nDdewoUHUVrq'


startDate = '2005-01-04'
endDate = dt.datetime.today().strftime("%Y-%m-%d")
#endDate = ''
metric = ' - Adjusted Close'

quandl.ApiConfig.api_key = '9gcG2sy8nDdewoUHUVrq'

fileName = 'qandlTestData.csv'

query = ['YAHOO/AX_BHP','YAHOO/AAPL','YAHOO/ASX_WBC_AX','YAHOO/ASX_TLS_AX','YAHOO/ASX_RIO_AX']

reload = False
if reload:
    #merged_dataset = quandl.MergedDataset([('YAHOO/AX_BHP', {'column_index': [4]})])
    merged_dataset = quandl.MergedDataset(query)

    data_all = merged_dataset.data(
        params={'start_date': startDate, 'end_date': dt.datetime.today().strftime("%Y-%m-%d")}).to_pandas()

    data = pd.DataFrame(index=data_all.index)
    for q in query:
        data[q] = data_all[q + metric];

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

'''
telstra = pd.DataFrame(data['YAHOO/ASX_TLS_AX'])
print "TELSTRA" , pd.DataFrame(np.where( (telstra > 1.5*telstra.quantile(0.75)) | (telstra < 1.5*telstra.quantile(0.25)), 1, 0 ), columns=telstra.columns ).sum()


aapl = pd.DataFrame(data['YAHOO/AAPL'])
print "AAPL" , pd.DataFrame(np.where( (aapl > 1.5*aapl.quantile(0.75)) | (aapl < 1.5*aapl.quantile(0.25)), 1, 0 ), columns=aapl.columns ).sum()
'''

#print outliers.sum().sort_values()
#print "SELECTED STOCK", outliers.sum().sort_values().keys()[0]

res = data.describe().transpose()

res['variance'] = data.var()
res['outliers'] = outliers.sum()


print res.sort_values(by=['variance','outliers'], ascending=[False, True])
print "SELECTED STOCK", res.sort_values(by=['variance','outliers'], ascending=[False, True]).transpose().keys()[0]
