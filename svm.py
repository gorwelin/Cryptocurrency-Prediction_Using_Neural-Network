import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression
from decimal import Decimal

dataset = pd.read_csv('ICE-3001 Individual Project/g-research-crypto-forecasting/train2.csv', low_memory=False, 
                      dtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32', 'Open': 'float64', 'High': 'float64', 
                              'Low': 'float64', 'Close': 'float64', 'Volume': 'float64', 'VWAP': 'float64'})
asset_details = pd.read_csv('ICE-3001 Individual Project/g-research-crypto-forecasting/asset_details.csv')

data = dataset.head()
npdata = np.array(dataset)
data_len = len(npdata)

btc = dataset[dataset["Asset_ID"]==1].set_index("timestamp") # Asset_ID = 1 for Bitcoin
btc_len = len(btc)

totimestamp = lambda s: np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))

btc_timestamps = btc.index.to_numpy()
btc_np = np.array(btc)

testnp = np.insert(btc_np[0], 0, btc_timestamps[0], axis=0)


for i in range(100):
      print(int(btc_timestamps[i]))


btc_close = btc_np[:,[0,6]]

btc_close = np.empty([btc_len, 3])

print(btc.tail)

for i in range(btc_len-361):
      btc_pred = btc.at[btc_timestamps[i] + 360, "Close"]
      btc_temp = np.array([btc_timestamps[i], btc_np[i][6], btc_pred])
      #btc_temp = btc_np[i][:,[0,6]]
      #btc_close[i] = np.insert(btc_temp, 0, btc_timestamps[0], axis=0)
      if(i < 10):
            print(btc_temp)

#btc_pred_index = btc_np[0][0] + 360
#print(btc_np[0][0])
#print(btc_pred_index)

btc_pred_arr = np.zeros(len(btc_close))

#for i in range(len(btc_close - 900)):
#      btc_pred = btc_np[i + 360][6]
#      btc_pred_arr[i] = btc_pred

#print(btc_pred_arr)

#for i in range(20):
#      btc_pred_index = btc[i][0] + 360
#      btc_pref = btc[i][0 == btc_pred_index]
#      
#      print(btc_pref)
      #btc_close = np.append(btc_close[i], ["test"])


#for i in range(240):
      #print(btc_close[i])
#  print("Timestamp: " + str(btc_close[i][0]))
#  print("Timestamp: " + str(btc_close[i][0].astype('datetime64[s]')) + ", close: " + str(btc_close[i][1]))
  
print(btc)