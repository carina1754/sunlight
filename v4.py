import pandas as pd
import numpy as np
import os
import glob
import random
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('/Users/jungsuuann/Downloads/data/train/train.csv')
submission = pd.read_csv('/Users/jungsuuann/Downloads/data/sample_submission.csv')
print(np.tan(0.5))
def preprocess_data(data, is_train=True):
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI','DNI','WS', 'RH', 'T']]
    temp = temp.assign(GHI=lambda x: x['DHI'] + x['DNI'] * np.cos(((180 * (x['Hour']+1) / 24) - 90)/180*np.pi))
    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        return temp.iloc[:-96]

    elif is_train==False:  
        return temp.iloc[-48:, :]

df_train = preprocess_data(train)
print(df_train[:48])
test = []

for i in range(81):
    file_path = '/Users/jungsuuann/Downloads/data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False).iloc[-48:]
    test.append(temp)

X_test = df_train
print(X_test[:48])

plt.scatter(X_test[['TARGET']], X_test[['GHI']], alpha=0.1)
plt.show()