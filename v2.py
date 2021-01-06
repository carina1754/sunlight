
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.backend import mean, maximum
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)

x = pd.read_csv('/Users/jungsuuann/Downloads/data/train/train.csv')
y = pd.read_csv('/Users/jungsuuann/Downloads/data/submission.csv')

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for q in q_lst:
  model = Sequential()
  model.add(Dense(10))
  model.add(Dense(1))   
  model.compile(loss=lambda y,pred: quantile_loss(q,y,pred), optimizer='adam')
  model.fit(x,y, epoch=300)
