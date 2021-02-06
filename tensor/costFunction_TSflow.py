"""Tensorflow로 cost Function 구현"""
# tf.square
#tf.reduce_mean
#np.linspace

import tensorflow as tf
import numpy as np

X = np.array([1,2,3])
Y = np.array([1,2,3])

def cost(W,X,Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))

W_values = np.linspace(-3, 5, num=15)

for feed_W in W_values:
    curr_cost = cost(feed_W, X, Y)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

