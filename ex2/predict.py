from numpy import round
import numpy as np
from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    theta = np.transpose(theta)
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters.
#               You should set p to a vector of 0's and 1's
#   
    p = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if sigmoid(np.dot(X[i],theta)) >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
      

# =========================================================================

    return p