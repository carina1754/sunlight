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
    temp = temp[['Hour', 'TARGET','GHI','DHI','DNI','RH','T','WS']]
    
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
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
from sklearn.model_selection import GridSearchCV, train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

param_grid = {
    'max_depth': [4],
    'max_features': [3],
    'n_estimators': [10000],
    'learning_rate': [0.1,0.05],
    'loss':['quantile']
}

from sklearn.ensemble import GradientBoostingRegressor

# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # (a) Modeling  
    gd_model = GradientBoostingRegressor(random_state=1,alpha=q)
    model = GridSearchCV(estimator = gd_model, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)             
    
    model.fit(X_train, Y_train)
    X_test = scaler.transform(X_test)
    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))
    print(pred)
    return pred, model

def train_data(X_train, Y_train, X_valid, Y_valid, X_test):
    
    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles
    
    return LGBM_models, LGBM_actual_pred

models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values

submission.to_csv('./data/submission.csv', index=False)