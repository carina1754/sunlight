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

from sklearn.model_selection import RandomizedSearchCV, train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.2, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.2, random_state=0)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

from lightgbm import LGBMRegressor

gridParams = {
    'learning_rate': [0.11,0.1,0.07],
    'num_leaves': [1000],
    'boosting_type' : ['gbdt'],
    'objective' : ['quantile'],
    'max_depth' : [-1], 
    'subsample' : [0.8],
    'min_split_gain' : [0.01,0.05,0.027],
    'metric':['quantile'],
    'scale_pos_weight' : [1],
    'n_estimators': [10000],
    'bagging_fraction':[0.8]
    }
# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    # (a) Modeling  
    model = LGBMRegressor(alpha=q)
    grid = RandomizedSearchCV(model,gridParams,verbose=1,cv=10,n_jobs = -1,n_iter=10)                   
                         
    grid.fit(X_train, Y_train, eval_metric = ['quantile'],eval_set=[(X_valid, Y_valid)],verbose=500)

    # (b) Predictions
    print(grid.best_params_)
    pred = pd.Series(grid.predict(X_test).round(2))
    return pred, grid

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