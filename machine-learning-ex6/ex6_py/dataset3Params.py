import numpy as np
from sklearn.svm import SVC


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.
#    C = 1
#    sigma = 0.3

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#
    C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    minError = sys.maxsize
    finalC = 0
    finalSigma = 0

    clf = SVC(kernel='rbf')

    for i in C:
        for j in sigma:
            clf = clf.set_params(C=i, gamma=1 / (2 * j * j))
            clf.fit(X, y.ravel())
            predictions = clf.predict(Xval)
            error = np.mean(predictions.reshape(-1, 1) != yval)
            if error <= minError:
                minError = error
                finalC = i
                finalSigma = j
    return finalC, finalSigma
# =========================================================================
 #   return C, sigma
