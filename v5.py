import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# 데이터
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
    temp = temp[['Hour','Minute', 'TARGET', 'DHI','DNI','WS', 'RH', 'T']]
    temp = temp.assign(GHI=lambda x: x['DHI'] + x['DNI'] * np.cos(((180 * (x['Hour']+1) / 24) - 90)/180*np.pi))
    temp = temp[['Hour','Minute', 'TARGET', 'DHI','DNI','WS', 'RH', 'T','GHI']]
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

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [72., 88., 92., 100., 71.]
 
# 변수
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설
hypothesis = x1*w1 + x2*w2 + x3*w3 + b
print (w1) 
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))


# cost function 최소화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)


train = optimizer.minimize(cost)
 
# 세션 생성
with tf.Session() as sess:
    # 사용할 변수 선언
    sess.run(tf.global_variables_initializer())
 
    for step in range(100001):
        cost_val, hy_val, _,w1_val,w2_val,w3_val,b_val, = sess.run([cost, hypothesis, train,w1,w2,w3,b],
                                       feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
print(w1_val)
print(w2_val)
print(w3_val)
print(b_val)