# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt



def plotdata(x, y):
    plt.plot(x, y, 'rx', markersize=10)
    plt.pause(0.0001)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')



def computeCost(X, y, theta):
    m = y.shape[0]
    return 0.5 / m * np.sum((np.dot(X, theta) - y) ** 2)



def gradientDescent(X, y, theta, alpha, num_iters):
  
    m = y.shape[0] 
    J_history = np.zeros((num_iters, 1))

    for iter in range(0, num_iters):
        theta = theta - alpha / m * (np.dot(X.T, (np.dot(X, theta) - y)))
        
        J_history[iter, 0] = compute_cost(X, y, theta)

    return theta, J_history


## ======================= Part 2: Plotting =======================

data = np.loadtxt('E:\ml_coursera\machine-learning-ex1\ex1\ex1data1.txt', delimiter=',')
X = data[:, :1]
y = data[:, 1:2]
m = y.shape[0]  

plt.close()
plotdata(X, y)


print('Running Gradient Descent ...\n')

X = np.concatenate((np.ones((m, 1)), X), axis=1)  
theta = np.zeros((2, 1))  


iterations = 1500
alpha = 0.01


print('cost:')
print(computeCost(X, y, theta))


theta, J_history = gradientDescent(X, y, theta, alpha, iterations)


print('Theta found by gradient descent: ')
print('{0:f} {1:f} \n'.format(theta[0, 0], theta[1, 0]))


plt.plot(X[:, 1], np.dot(X, theta), '-')
plt.legend(['Training data', 'Linear regression'])

plt.pause(0.0001)
plt.show()


predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {0:f}\n'.format(predict1[0] * 10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {0:f}\n'.format(predict2[0] * 10000))


## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')


theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)


J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))


for i in range(0, theta0_vals.shape[0]):
    for j in range(0, theta1_vals.shape[0]):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = computeCost(X, y, t)



J_vals = J_vals.T


plt.close()

fig = plt.figure()
ax = fig.gca(projection='3d')
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(theta0_vals, theta1_vals, J_vals)
plt.xlabel('theta_0')
plt.ylabel('theta_1')


plt.figure()

plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
plt.show()
