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
    temp = temp[['Hour', 'TARGET', 'DHI','DNI','WS', 'RH', 'T']]
    temp = temp.assign(GHI=lambda x: x['DHI'] + x['DNI'] * np.cos(((180 * (x['Hour']+1) / 24) - 90)/180*np.pi))
    #temp = temp.assign(WP=lambda y: 742.9 + 176.5*y['T'] + 3.562*y['WS']- 13.14*y['T']*y['T'] - 0.7466*y['T']*y['WS']-0.151*y['WS']*y['WS'])
    temp = temp[['Hour', 'TARGET','GHI','WS', 'RH', 'T']]
    
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
    file_path = './data/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False).iloc[-48:]
    test.append(temp)

X_test = pd.concat(test)
print(X_test[:48])

from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

from lightgbm import LGBMRegressor

params = {
    'objective': 'quantile',
    'metric': 'quantile',
    #'max_depth': 4,
    #'num_leaves': 15,
    'learning_rate': 0.027,
    'n_estimators': 10000,
    'boosting_type': 'gbdt'
}

# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    # (a) Modeling  
    model = LGBMRegressor(alpha=q, bagging_fraction=0.7, subsample=0.7,**params)                   
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300,verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))
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