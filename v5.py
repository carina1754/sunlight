import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import MinMaxScaler
tf.disable_v2_behavior()

x1_data = []#time
x2_data = []
x3_data = []
x4_data = []
x5_data = []
x6_data = []#tempertaure
y_data = []#target


input_file = './data/train/train.csv'

with open(input_file) as file:
  file.readline()
  for line in file:
    temp1 = 0
    temp1_1 = float(line.split(',')[1])#hour
    temp1_2 = float(line.split(',')[2])#minute
    temp2 = float(line.split(',')[3])#
    temp3 = float(line.split(',')[4])#
    temp4 = float(line.split(',')[5])#
    temp5 = float(line.split(',')[6])#
    temp6 = float(line.split(',')[7])#temperature
    tempY = float(line.split(',')[8])#target

    if tempY == 0: continue

    if temp1_2 == 0: 
      temp1 = temp1_1
    else : temp1 = temp1_1 + 0.5

    x1_data.append(temp1)
    x2_data.append(temp2)
    x3_data.append(temp3)
    x4_data.append(temp4)
    x5_data.append(temp5)
    x6_data.append(temp6)
    y_data.append(tempY)


scaler = MinMaxScaler()
scaler.fit(x1_data)
scaler.fit(x2_data)

print(x1_data)
print(x2_data)


# 변수
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
x4 = tf.placeholder(tf.float32)
x5 = tf.placeholder(tf.float32)
x6 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
w4 = tf.Variable(tf.random_normal([1]), name='weight4')
w5 = tf.Variable(tf.random_normal([1]), name='weight5')
w6 = tf.Variable(tf.random_normal([1]), name='weight6')
b = tf.Variable(tf.random_normal([1]), name='bias')
 
# 가설
hypothesis = x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 + b
 
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# cost function 최소화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(1001):
  cost_val, hy_val,_  = sess.run([cost, hypothesis, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, x4: x4_data, x5: x5_data, x6: x6_data, Y: y_data})
  if step % 50 == 0:
    print(step, "\nCost: ", cost_val, "\nPrediction:\n", hy_val)
  #if cost_val < 7 : break


#print(w1_value)
#print(w2_value)
#print(w3_value)
#print(b_value)
print(sess.run(w1),sess.run(w2),sess.run(w3),sess.run(w4),sess.run(w5),sess.run(w6),sess.run(b))