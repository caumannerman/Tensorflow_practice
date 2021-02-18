import numpy as np
import tensorflow as tf

# 학습데이터 준비
x_data = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(10,1)
t_data = np.array([0,0,0,0,0,0,1,1,1,1]).reshape(10,1)

# 임의의 직선 z = Wx + b 정의 (임의의 값으로 가중치 W, 바이어스 b 초기화)
W = np.random.rand(1,1)
b = np.random.rand(1)
print("W = ", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss_func(x, t):
    delta = 1e-7

    z = np.dot(x,W) + b
    y = sigmoid(z)

    return -np.sum(t * np.log(y+delta) + (1-t) * np.log((1-y) + delta))

def numerical_derivative(f,x):
    delta_x = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
# f(x+delta_x)
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
# f(x + delta_x)
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val
        it.iternext()
    return grad


def error_val(x, t):
    delta = 1e-7

    z = np.dot(x,W) + b
    y = sigmoid(z)

    return -np.sum(t*np.log(y + delta) + (1-t) * np.log((1-y) + delta) )

def predict(x):

    z = np.dot(x,W) + b
    y = sigmoid(z)

    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result

# learning rate초기화 및 손실함수가 최소 될 때까지 W, b 업데이트
learning_rate = 1e-2
f = lambda x : loss_func(x_data, t_data)
print("Initial error value = ", error_val(x_data, t_data), "Initial W = ", W, "\n", ", b = ", b)

for step in range(10001):
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)

    if ( step % 400 == 0):
        print( " step = ", step, "error value = ", error_val(x_data, t_data), "W = ", W, ", b = ", b)

r, logi = predict(3)
print(r, logi)

rr, loggi = predict(17)
print(rr, loggi)