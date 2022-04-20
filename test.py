import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('./ICE-3001 Individual Project/g-research-crypto-forecasting/train.csv', low_memory=False, 
                      dtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32', 'Open': 'float64', 'High': 'float64', 
                              'Low': 'float64', 'Close': 'float64', 'Volume': 'float64', 'VWAP': 'float64'})


details=pd.read_csv('./ICE-3001 Individual Project/g-research-crypto-forecasting/asset_details.csv')




from tqdm import tqdm
def compute(df):
    
    R=list()
    c=list(df['Close'])
    for i in range(df.shape[0]):
        future=c[min([i+16,df.shape[0]-1])]
        past=c[min([i+1,df.shape[0]-1])]
        R.append(future/past)
    R=np.array(R)

    df['pred']=R-1
    return df
crops=list()
for a in tqdm(list(range(14))):
    
    crops.append(compute(df[df['Asset_ID']==a]))
    
conc=pd.concat(crops)
conc=conc.reset_index()

j=np.array(list(conc['Target'].isnull()))
new_targets=np.where(j,conc['pred'],conc['Target'])

conc['Target']=new_targets
conc=conc.drop(columns=['pred'],axis=1)
conc.head()


for i in range(14):
    dfcrop=conc[conc['Asset_ID']==i]
    print('Percentage of values not nan',(1-(np.sum((dfcrop['Target'].isnull()).astype(int))/dfcrop.shape[0]))*100)

conc.to_csv('train.csv')
del conc,crops

df=pd.read_csv('train.csv')
#df=df.drop(columns=['index'])
print(df.head())