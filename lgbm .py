import pandas as pd
import numpy as np
from datetime import datetime
from lightgbm import LGBMRegressor
import gresearch_crypto
import traceback
import time
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.model_selection import train_test_split

path = "/kaggle/input/g-research-crypto-forecasting/"
df_train = pd.read_csv(path + "train.csv")
df_test = pd.read_csv(path + "example_test.csv")
df_asset_details = pd.read_csv(path + "asset_details.csv")
df_supp_train = pd.read_csv(path + "supplemental_train.csv")

df_train.head()

totimestamp = lambda s: np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))


btc = df_train[df_train["Asset_ID"]==1].set_index("timestamp") # Asset_ID = 1 for Bitcoin
eth = df_train[df_train["Asset_ID"]==6].set_index("timestamp") # Asset_ID = 6 for Ethereum
bnb = df_train[df_train["Asset_ID"]==0].set_index("timestamp") # Asset_ID = 0 for Binance Coin
ada = df_train[df_train["Asset_ID"]==3].set_index("timestamp") # Asset_ID = 3 for Cardano

beg_btc = datetime.fromtimestamp(btc.index[0]).strftime("%A, %B %d, %Y %I:%M:%S") 
end_btc = datetime.fromtimestamp(btc.index[-1]).strftime("%A, %B %d, %Y %I:%M:%S") 
beg_eth = datetime.fromtimestamp(eth.index[0]).strftime("%A, %B %d, %Y %I:%M:%S") 
end_eth = datetime.fromtimestamp(eth.index[-1]).strftime("%A, %B %d, %Y %I:%M:%S")
beg_bnb = datetime.fromtimestamp(eth.index[0]).strftime("%A, %B %d, %Y %I:%M:%S") 
end_bnb = datetime.fromtimestamp(eth.index[-1]).strftime("%A, %B %d, %Y %I:%M:%S")
beg_ada = datetime.fromtimestamp(eth.index[0]).strftime("%A, %B %d, %Y %I:%M:%S") 
end_ada = datetime.fromtimestamp(eth.index[-1]).strftime("%A, %B %d, %Y %I:%M:%S")

print('Bitcoin data goes from ', beg_btc, ' to ', end_btc) 
print('Ethereum data goes from ', beg_eth, ' to ', end_eth)
print('Binance coin data goes from ', beg_bnb, ' to ', end_bnb) 
print('Cardano data goes from ', beg_ada, ' to ', end_ada)


plt.figure(figsize=(8,6))
sns.heatmap(btc[['Count','Open','High','Low','Close','Volume','VWAP','Target']].corr(), 
            vmin=-1.0, vmax=1.0, annot=True, cmap='coolwarm', linewidths=0.1)
plt.show()


btc_mini = btc.iloc[-200:] # Select recent data rows
eth_mini = eth.iloc[-200:]

fig = go.Figure(data=[go.Candlestick(x=btc_mini.index, open=btc_mini['Open'], high=btc_mini['High'], low=btc_mini['Low'], close=btc_mini['Close'])])
fig.update_xaxes(title_text="$")
fig.update_yaxes(title_text="Index")
fig.update_layout(title="Bitcoin Price, 200 Last Minutes")
fig.show()

fig = go.Figure(data=[go.Candlestick(x=eth_mini.index, open=eth_mini['Open'], high=eth_mini['High'], low=eth_mini['Low'], close=eth_mini['Close'])])
fig.update_xaxes(title_text="$")
fig.update_yaxes(title_text="Index")
fig.update_layout(title="Ethereum Price, 200 Last Minutes")
fig.show()






f = plt.figure(figsize=(15,4))

# fill NAs for BTC and ETH
btc = btc.reindex(range(btc.index[0],btc.index[-1]+60,60),method='pad')
eth = eth.reindex(range(eth.index[0],eth.index[-1]+60,60),method='pad')

ax = f.add_subplot(121)
plt.plot(btc['Close'], color='yellow', label='BTC')
plt.legend()
plt.xlabel('Time (timestamp)')
plt.ylabel('Bitcoin')

ax2 = f.add_subplot(122)
ax2.plot(eth['Close'], color='purple', label='ETH')
plt.legend()
plt.xlabel('Time (timestamp)')
plt.ylabel('Ethereum')

plt.tight_layout()
plt.show()


data =df_train[-10000:]
check = pd.DataFrame()
for i in data.Asset_ID.unique():
    check[i] = data[data.Asset_ID==i]['Target'].reset_index(drop=True) 
    
plt.figure(figsize=(10,8))
sns.heatmap(check.dropna().corr(), vmin=-1.0, vmax=1.0, annot=True, cmap='coolwarm', linewidths=0.1)
plt.show()






def hlco_ratio(df): 
    return (df['High'] - df['Low'])/(df['Close']-df['Open'])
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['hlco_ratio'] = hlco_ratio(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat

# train test split df_train into 80% train rows and 20% valid rows
train_data = df_train
# train_data = df_train.sample(frac = 0.8)
# valid_data = df_train.drop(train_data.index)

def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    df = df.sample(frac=0.2)
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_proc = df_proc.dropna(how="any")
    
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]   
    model = LGBMRegressor()
    model.fit(X, y)
    return X, y, model

Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    X, y, model = get_Xy_and_model_for_asset(train_data, asset_id)       
    try:
        Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model
    except: 
        Xs[asset_id], ys[asset_id], models[asset_id] = None, None, None 
        
        
        
parameters = {
    # 'max_depth': range (2, 10, 1),
    'num_leaves': range(21, 161, 10),
    'learning_rate': [0.1, 0.01, 0.05]
}

new_models = {}
for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print("GridSearchCV for: " + asset_name)
    grid_search = GridSearchCV(
        estimator=get_Xy_and_model_for_asset(df_train, asset_id)[2], # bitcoin
        param_grid=parameters,
        n_jobs = -1,
        cv = 5,
        verbose=True
    )
    grid_search.fit(Xs[asset_id], ys[asset_id])
    new_models[asset_id] = grid_search.best_estimator_
    grid_search.best_estimator_
    
    
for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Tuned model for {asset_name:<1} (ID={asset_id:})")
    print(new_models[asset_id])
    
    
    
    
    
    
    
    
# df_pred = []

#for j , row in valid_data.iterrows():        
#    if new_models[row['Asset_ID']] is not None:
#        model = new_models[row['Asset_ID']]
#        x_test = get_features(row)
#        y_pred = model.predict(pd.DataFrame([x_test]))[0]
#        df_pred.append(y_pred)
#    else: 
#        df_pred.append(0)

#print(df_pred)



# # We will simplify things and use correlation (without weights) for evaluation, and consider only BTC.
#print('Test score for BTC: ', f"{np.corrcoef(df_pred[df_pred["Asset_ID"]==1].set_index("timestamp")["Target"], valid_data[valid_data["Asset_ID"]==1].set_index("timestamp")["Target"]):.2f}")


env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():        
        if new_models[row['Asset_ID']] is not None:
            try:
                model = new_models[row['Asset_ID']]
                x_test = get_features(row)
                y_pred = model.predict(pd.DataFrame([x_test]))[0]
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
            except:
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
                traceback.print_exc()
        else: 
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0  
    
    env.predict(df_pred)