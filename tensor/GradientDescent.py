import tensorflow as tf
import numpy as np

# 랜덤시드 초기
tf.random.set_seed(0)

X = np.array([1.,2., 3., 4.])
Y = np.array([1., 3., 5., 7.])
# 정규분포 이용 W에 할
W = tf.Variable(tf.random.normal([1], -100., 100.))


for step in range(300000):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    alpha = 0.00001
    # cost함수, 즉 ( Wx -y)^2을 W에 대하여 미분하면 2(Wx-y)x이므로 gradient에 해당 형태를 만들어주는
    #print(tf.multiply(tf.multiply(W, X) - Y, X))
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))

    # descent가 새롭게 W에 할당될 값. 원래의 W값에서 alpha와 미분값을 곱한 값을 빼준다.
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)


    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(
            step, cost.numpy(), W.numpy()[0]))

