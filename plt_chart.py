import pandas as pd
import numpy as np
import os
import glob
import random
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('./data/train/train.csv')
submission = pd.read_csv('./data/sample_submission.csv')
print(np.tan(0.5))
def preprocess_data(data, is_train=True):
    temp = data.copy()
    temp = temp[['Hour','Minute', 'TARGET', 'DHI','DNI','WS', 'RH', 'T']]
    temp = temp.assign(GHI=lambda x: x['DHI'] + x['DNI'] * np.cos(((180 * (x['Hour']+1) / 24) - 90)/180*np.pi))
    temp = temp.assign(TIME=lambda x: (x['Hour']+1+x['Minute']/60))
    temp = temp[['TIME','TARGET','GHI','DHI','DNI','RH','T','WS']]
   
    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        return temp.iloc[:-96]

    elif is_train==False:  
        return temp.iloc[-48:, :]

df_train = preprocess_data(train)

X_test = df_train
plt.figure(1)
plt.scatter(X_test[['TIME']], X_test[['TARGET']],alpha=0.05)
plt.figure(2)
plt.scatter(X_test[['TARGET']], X_test[['GHI']],alpha=0.05)
plt.show()