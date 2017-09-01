# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 00:51:05 2017

@author: panhaiqi
"""

import numpy as np
import matplotlib.pyplot as plt



def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost_multi(X, y, theta):
    m = y.shape[0]  
    return 0.5 / m * np.sum((np.dot(X, theta) - y) ** 2)


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = y.shape[0]  
    J_history = np.zeros((num_iters, 1))

    for iter in range(0, num_iters):
        theta = theta - alpha / m * np.dot(X.T, (np.dot(X, theta) - y))
        J_history[iter] = compute_cost_multi(X, y, theta)
    return theta, J_history

def normaleqn(X, y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return theta


## ================ Part 1: Feature Normalization ================

data = np.loadtxt('E:\ml_coursera\machine-learning-ex1\ex1\ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2:3]
m = y.shape[0]

print('First 10 examples from the dataset: \n')
for tp in np.concatenate((X[0:10, :], y[0:10, :]), axis=1):
    print(' x = [{0:.0f} {1:.0f}], y = {2:.0f}'.format(tp[0], tp[1], tp[2]))


print('Normalizing Features ...\n')

X, mu, sigma = feature_normalize(X)


X = np.concatenate((np.ones((m, 1)), X), axis=1)

## ================ Part 2: Gradient Descent ================
print('Running gradient descent ...\n')


alpha = 0.1
num_iters = 100


theta = np.zeros((X.shape[1], 1))
theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)


plt.close()
plt.figure()
plt.plot(list(range(1, J_history.size + 1)), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()


print('Theta computed from gradient descent: ')
print(np.array_str(theta).replace('[', ' ').replace(']', ' '))
print('\n')


price = np.dot(np.concatenate(([1], (np.array([1650, 3]) - mu) / sigma)), theta)[0]

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${0:f}\n'.format(price))


## ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')


data = np.loadtxt('E:\ml_coursera\machine-learning-ex1\ex1\ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2:3]
m = y.shape[0]


X = np.concatenate((np.ones((m, 1)), X), axis=1)


theta = normaleqn(X, y)
print('Theta computed from the normal equations: ')
print(np.array_str(theta).replace('[', ' ').replace(']', ' '))
print('\n')


price = np.dot(np.array([1,1650,3]), theta)[0]

# ============================================================
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n ${0:f}\n'.format(price))
