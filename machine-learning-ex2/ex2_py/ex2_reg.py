# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:39:11 2017

@author: panhaiqi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy


def cost_function_reg(theta, X, y, ld):
    m = y.shape[0]
    d1 = X.shape[1]
    d2 = y.shape[1]
    theta = theta.reshape((d1, d2))
    h = sigmoid(X.dot(theta))
    if theta.size > 0:
        theta2 = theta
        theta2[0, :] = 0
    J = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / m + ld/m/2 * np.sum(theta2 ** 2)
    grad = (h - y).T.dot(X).T / m + ld / m * theta2
    return J, grad.flatten()



    
data = np.loadtxt('E:\ml_coursera\machine-learning-ex2\ex2\ex2data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2:3]

plotdata(X, y)



plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')



plt.legend(['y = 1', 'y = 0'])
plt.show()

## =========== Part 1: Regularized Logistic Regression ============


X = map_feature(X[:,:1], X[:,1:2])


initial_theta = np.zeros((X.shape[1], 1))


ld = 1


[cost, grad] = cost_function_reg(initial_theta, X, y, ld)

print('Cost at initial theta (zeros): {0:f}\n'.format(cost))



## ============= Part 2: Regularization and Accuracies =============

initial_theta = np.zeros((X.shape[1], 1)) + 0.01


ld = 1


res = scipy.optimize.minimize(cost_function_reg, initial_theta, args=(X, y, ld), method='BFGS', jac=True, options={'maxiter': 400})
theta = res['x']


plot_decision_boundary(theta, X, y)
plt.title(print('lambda = {0:g}'.format(ld)))


plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
plt.show()


p = predict(theta, X)

print('Train Accuracy: {0:f}\n'.format(np.mean(p == y.flatten()) * 100))