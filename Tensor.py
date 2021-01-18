import pandas as pd
import numpy as np
import os
import glob
import random
import math
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('./data/train/train.csv')
submission = pd.read_csv('./data/sample_submission.csv')

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'Minute','TARGET', 'DHI','DNI','WS', 'RH', 'T']]
    temp = temp.assign(GHI=lambda x: x['DHI'] + x['DNI'] * np.cos(((180 * (x['Hour']+1+x['Minute']/60) / 24) - 90)/180*np.pi))
    temp = temp[['Hour','TARGET','GHI','DHI','DNI','RH','T','WS']]
    
    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        return temp.iloc[:-96] 

    elif is_train==False:  
        return temp.iloc[-48:, :]

df_train = preprocess_data(train)
test = []

for i in range(81):
    file_path = './data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False).iloc[-48:]
    test.append(temp)

X_test = pd.concat(test)

from sklearn.model_selection import train_test_split
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
x_test_1,x_valid_1,y_test_1,y_valid_1 = train_test_split(x_test_1, y_test_1, test_size=0.5, random_state=0)

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)
x_test_2,x_valid_2,y_test_2,y_valid_2 = train_test_split(x_test_2, y_test_2, test_size=0.5, random_state=0)

from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

def training(q, X_train, Y_train,X_valid,Y_valid, X_test):
    model = Sequential([
    Dense(512, activation='relu'), 
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='softmax')
    ])
    
    # 3. 훈련
    model.compile(loss=lambda y,pred: quantile_loss(q,y,pred),optimizer='adam',metrics=[lambda y, pred: quantile_loss(q, y, pred)])
    model.fit(X_train,Y_train, epochs=10,validation_data=(X_valid, Y_valid))
    pred = pd.Series(np.ravel(model.predict(X_test), order='C'))
    return pred, model

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def train_data(X_train, Y_train, X_valid, Y_valid, X_test):
    
    train_models=[]
    train_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = training(q, X_train, Y_train, X_valid, Y_valid, X_test)
        train_models.append(model)
        train_actual_pred = pd.concat([train_actual_pred,pred],axis=1)

    train_actual_pred.columns=quantiles
    
    return train_models, train_actual_pred

models_1, results_1 = train_data(x_train_1, y_train_1, x_valid_1, y_valid_1, X_test)
models_2, results_2 = train_data(x_train_2, y_train_2, x_valid_2, y_valid_2, X_test)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values

submission.to_csv('./data/submission.csv', index=False)