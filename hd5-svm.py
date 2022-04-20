import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression
from decimal import Decimal

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas_datareader.data as web
import datetime as dt

dataset=pd.read_hdf('./ICE-3001 Individual Project/g-research-crypto-forecasting/train.h5', low_memory=False, 
                      dtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32', 'Open': 'float64', 'High': 'float64', 
                              'Low': 'float64', 'Close': 'float64', 'Volume': 'float64', 'VWAP': 'float64'})
asset_details = pd.read_csv('./ICE-3001 Individual Project/g-research-crypto-forecasting/asset_details.csv')

dataset=dataset.drop(columns=['index'])

npdata = np.array(dataset)
data_len = len(npdata)




btc = dataset[dataset["Asset_ID"]==1].set_index("timestamp") # Asset_ID = 1 for Bitcoin
btc_len = len(btc)


totimestamp = lambda s: np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))

btc_timestamps = btc.index.to_numpy()
btc_np = np.array(btc)

testnp = np.insert(btc_np[0], 0, btc_timestamps[0], axis=0)

btc_close = btc_np[:,[0,6]]

btc_close = np.empty([btc_len, 3])

btc_fin = []

for i in range(btc_len-360):
    try:
        btc_pred = btc.at[btc_timestamps[i] + 360, "Close"]
        btc_fin.append([btc_timestamps[i], btc_np[i][5], btc_pred])
    except:
        continue;

btc_close = np.asarray(btc_fin)

print(btc_close[0][0].astype('datetime64[s]'))
print(btc_close[0][1].astype('float64'))
print(btc_close[0][2].astype('float64'))



#for i in range(20):
#   print(f"Date and time: {btc_close[i][0].astype('datetime64[s]')}, Close: {btc_close[i][1].astype('float64')}, Prediction: {btc_close[i][2].astype('float64')} ")
    
print("\n\n\n\n\n")    

reversed_btc_close = reversed(btc_close)
    
for i in range(20):
    print(f"Date and time: {reversed_btc_close[-i][0].astype('datetime64[s]')}, Close: {reversed_btc_close[-i][1].astype('float64')}, Prediction: {reversed_btc_close[-i][2].astype('float64')} ")


