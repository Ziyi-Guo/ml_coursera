import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.io, scipy.optimize




def linear_reg_cost_function(X, y, theta, ld):

    m = y.shape[0]
    d1 = X.shape[1]
    d2 = y.shape[1]

    theta = theta.reshape((d1, d2))

    J = 0.5 / m * (np.sum((X.dot(theta) - y) ** 2) + ld * np.sum(theta[1:, :] ** 2))
    grad = 1 / m * X.T.dot(X.dot(theta) - y) + ld / m * np.vstack((np.zeros((1, theta.shape[1])), theta[1:, :]))
    return J, grad.flatten()



def train_linear_reg(X, y, ld):

    initial_theta = np.zeros((X.shape[1], 1))

    cost_function = lambda t: linear_reg_cost_function(X, y, t, ld)

    options = {'maxiter': 200, 'disp': True, 'gtol': 0.05}


    ret = scipy.optimize.minimize(cost_function, initial_theta, options=options, jac=True, method='BFGS')
    return ret['x']


def learning_curve(X, y, Xval, yval, ld):
    m = X.shape[0]

    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for i in range(1, m + 1):
        print(i)
        theta = train_linear_reg(X[:i, :], y[:i, :], ld).reshape(-1, 1)
        error_train[i - 1, 0] = 0.5 / i * np.sum((X[:i, :].dot(theta) - y[:i, :]) ** 2)
        error_val[i - 1, 0] = 0.5 / yval.shape[0] * np.sum((Xval.dot(theta) - yval) ** 2)
    return error_train, error_val

def poly_features(X, p):
    X_poly = X ** np.array(range(1, p + 1))
    return X_poly

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

def plotfit(min_x, max_x, mu, sigma, theta, p):
    nnum = np.floor((max_x - min_x + 40) / 0.05)
    x = np.linspace(min_x - 15, nnum * 0.05 + min_x - 15, nnum).reshape(-1, 1)


    X_poly = poly_features(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma


    X_poly = np.hstack((np.ones((x.shape[0], 1)), X_poly))

    plt.plot(x, X_poly.dot(theta), '--', linewidth=2)

    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).T


    error_train = np.zeros((lambda_vec.size, 1))
    error_val = np.zeros((lambda_vec.size, 1))

    for i in range(0, lambda_vec.size):
        ld = lambda_vec[i]
        theta = train_linear_reg(X, y, ld).reshape(X.shape[1], y.shape[1])
        error_train[i] = 0.5 / X.shape[0] * np.sum((X.dot(theta) - y) ** 2)
        error_val[i] = 0.5 / yval.shape[0] * np.sum((Xval.dot(theta) - yval) ** 2)
    return lambda_vec, error_train, error_val

def validation_curve(X, y, Xval, yval):
    # Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).T

    # Initialize return values
    error_train = np.zeros((lambda_vec.size, 1))
    error_val = np.zeros((lambda_vec.size, 1))

    for i in range(0, lambda_vec.size):
        ld = lambda_vec[i]
        theta = train_linear_reg(X, y, ld).reshape(X.shape[1], y.shape[1])
        error_train[i] = 0.5 / X.shape[0] * np.sum((X.dot(theta) - y) ** 2)
        error_val[i] = 0.5 / yval.shape[0] * np.sum((Xval.dot(theta) - yval) ** 2)
    return lambda_vec, error_train, error_val

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format}, edgeitems=50, linewidth=150)
    ## =========== Part 1: Loading and Visualizing Data =============
    # Load Training Data
    print('Loading and Visualizing Data ...\n')

    # Load from ex5data1:
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = scipy.io.loadmat('E:\ml_coursera\machine-learning-ex5\ex5\ex5data1', matlab_compatible=True)
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']

    # m = Number of examples
    m = X.shape[0]

    # Plot training data
    plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()

    print('Program paused. Press enter to continue.\n')
    input()

    ## =========== Part 2: Regularized Linear Regression Cost =============
    theta = np.array([[1], [1]])
    J = linear_reg_cost_function(np.hstack((np.ones((m, 1)), X)), y, theta, 1)[0]

    print('Cost at theta = [1 ; 1]: {0:f} \n(this value should be about 303.993192)\n'.format(J))

    print('Program paused. Press enter to continue.\n')

    ## =========== Part 3: Regularized Linear Regression Gradient =============
    J, grad = linear_reg_cost_function(np.hstack((np.ones((m, 1)), X)), y, theta, 1)

    print('Gradient at theta = [1 ; 1]:  [{0:f}; {1:f}] \n(this value should be about [-15.303016; 598.250744])\n'.format(
        grad[0], grad[1]))

    print('Program paused. Press enter to continue.\n')

    ## =========== Part 4: Train Linear Regression =============
    # Train linear regression with ld = 0
    ld = 0

    theta = train_linear_reg(np.hstack((np.ones((m, 1)), X)), y, ld)

    # Plot fit over the data
    plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(X, np.hstack((np.ones((m, 1)), X)).dot(theta), '--', linewidth=2)
    plt.show()

    print('Program paused. Press enter to continue.\n')
    input()

    ## =========== Part 5: Learning Curve for Linear Regression =============
    ld = 0

    error_train, error_val = learning_curve(np.hstack((np.ones((m, 1)), X)), y,
                                            np.hstack((np.ones((Xval.shape[0], 1)), Xval)), yval, ld)

    plt.plot(np.array(range(1, m + 1)), error_train, np.array(range(1, m + 1)), error_val)
    plt.title('Learning curve for linear regression')
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 150])
    plt.show()

    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(0, m):
        print('  \t{0:d}\t\t{1:f}\t{2:f}'.format(i, error_train[i, 0], error_val[i, 0]))

    print('Program paused. Press enter to continue.\n')
    input()

    ## =========== Part 6: Feature Mapping for Polynomial Regression =============
    p = 8

    # Map X onto Polynomial Features and Normalize
    X_poly = poly_features(X, p)
    X_poly, mu, sigma = feature_normalize(X_poly)  # Normalize
    X_poly = np.hstack((np.ones((m, 1)), X_poly))  # Add Ones

    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = poly_features(Xtest, p)
    X_poly_test = X_poly_test - mu
    X_poly_test = X_poly_test / sigma
    X_poly_test = np.hstack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))  # Add Ones

    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = poly_features(Xval, p)
    X_poly_val = X_poly_val - mu
    X_poly_val = X_poly_val / sigma
    X_poly_val = np.hstack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))  # Add Ones

    print('Normalized Training Example 1:\n')
    print(X_poly[0, :])

    print('\nProgram paused. Press enter to continue.\n')
    input()

    ## =========== Part 7: Learning Curve for Polynomial Regression =============
    ld = 0
    theta = train_linear_reg(X_poly, y, ld)

    # Plot training data and fit
    plt.figure()
    plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
    plotfit(np.min(X), np.max(X), mu, sigma, theta, p)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = {0:f})'.format(ld))

    plt.figure()
    error_train, error_val = learning_curve(X_poly, y, X_poly_val, yval, ld)
    plt.plot(np.array(range(1, m + 1)), error_train, np.array(range(1, m + 1)), error_val)

    plt.title('Polynomial Regression Learning Curve (lambda = {0:f})'.format(ld))
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 100])
    plt.legend(['Train', 'Cross Validation'])
    plt.show()

    print('Polynomial Regression (lambda = {0:f})\n\n'.format(ld))
    print('# Training Examples\tTrain Error\tCross Validation Error\n')
    for i in range(0, m):
        print('  \t{0:d}\t\t{1:f}\t{2:f}\n'.format(i + 1, error_train[i, 0], error_val[i, 0]))

    print('Program paused. Press enter to continue.\n')
    input()

    ## =========== Part 8: Validation for Selecting Lambda =============
    lambda_vec, error_train, error_val = validation_curve(X_poly, y, X_poly_val, yval)

    plt.plot(lambda_vec, error_train, lambda_vec, error_val)
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.show()

    print('lambda\t\tTrain Error\tValidation Error\n')
    for i in range(0, lambda_vec.size):
        print(' {0:f}\t{1:f}\t{2:f}\n'.format(lambda_vec[i], error_train[i, 0], error_val[i, 0]))

    print('Program paused. Press enter to continue.\n')