# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:18:50 2017

@author: panhaiqi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize



def plotdata(X, y):
    plt.figure()
    pos = y == 1
    neg = y == 0
    plt.plot(X[:, :1][pos], X[:, 1:2][pos], 'k+', linewidth=2, markersize=7)
    plt.plot(X[:, :1][neg], X[:, 1:2][neg], 'ko', markerfacecolor='y', markersize=7)



def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g



def cost_function(theta, X, y):
    m = y.shape[0]
    d1 = X.shape[1]
    d2 = y.shape[1]
    theta = theta.reshape((d1, d2))
    h = sigmoid(X.dot(theta))
    J = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / m
    grad = (h - y).T.dot(X) / m
    return J, grad.flatten()



def map_feature(X1, X2):
    degree = 6
    out = np.ones(X1[:, :1].shape)
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out = np.hstack((out, (X1 ** (i - j)) * (X2 ** j)))
    return out



def plot_decision_boundary(theta, X, y):
    
    plotdata(X[:, 1:3], y)
    if theta.ndim == 2 and (theta.shape[0] == 1 or theta.shape[1] == 1):
        theta = theta.flatten()

    if X.shape[1] <= 3:
        
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        
        plt.plot(plot_x, plot_y)

        
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    else:
        
        u = np.linspace(-1, 1.5, 50).reshape(-1, 1)
        v = np.linspace(-1, 1.5, 50).reshape(-1, 1)

        z = np.zeros((u.size, v.size))
        
        for i in range(0, u.size):
            for j in range(0, v.size):
                z[i, j] = map_feature(u[i:i + 1, :1], v[j:j + 1, :1]).dot(theta)

        z = z.T  

        
        plt.contour(u.flatten(), v.flatten(), z, [0], linewidth=2)



def predict(theta, X):
    m = X.shape[0]
    p = sigmoid(X.dot(theta)) >= 0.5
    return p



if __name__ == "__main__":

    data = np.loadtxt('E:\ml_coursera\machine-learning-ex2\ex2\ex2data1.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2:3]

    ## ==================== Part 1: Plotting ====================
    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

    plotdata(X, y)

   
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()



    ## ============ Part 2: Compute Cost and Gradient ============
    
    m, n = X.shape

   
    X = np.hstack((np.ones((m, 1)), X))

    
    initial_theta = np.zeros((n + 1, 1))

    
    cost, grad = cost_function(initial_theta, X, y)

    print('Cost at initial theta (zeros): {0:f}'.format(cost))
    print('Gradient at initial theta (zeros): ')
    print(np.array_str(grad.reshape(-1, 1)).replace('[', ' ').replace(']', ' '))



    ## ============= Part 3: Optimizing using fmin  =============

    
    initial_theta += 0.01
    res = scipy.optimize.minimize(cost_function, initial_theta, args=(X, y), method='BFGS', jac=True,
                                  options={'maxiter': 400})
    theta = res['x']

    
    print('Cost at theta found by optimization function: %f\n', res['fun'])
    print('theta: \n')
    print(np.array_str(theta).replace('[', ' ').replace(']', ' '))

    
    plot_decision_boundary(theta, X, y)

    
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()



    ## ============== Part 4: Predict and Accuracies ==============

    
    prob = sigmoid(np.array([1, 45, 85]).dot(theta))
    print('For a student with scores 45 and 85, we predict an admission probability of {0:f}\n'.format(prob))

   
    p = predict(theta, X)

    print('Train Accuracy: {0:f}\n'.format(np.mean((p == y.flatten())) * 100))

