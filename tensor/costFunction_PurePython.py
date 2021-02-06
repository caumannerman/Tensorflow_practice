import numpy as np
#cost function implementation by pure python
X = np.array([1,2,3])
Y = np.array([1,2,3])

def cost(W,X,Y):
    c = 0
    for i in range(len(X)):
        c += (W * X[i] - Y[i]) ** 2
    return c/len(X)


for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = cost(feed_W, X, Y)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
