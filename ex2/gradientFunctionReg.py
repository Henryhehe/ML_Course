from numpy import asfortranarray, squeeze, asarray
import numpy as np

from gradientFunction import gradientFunction
from sigmoid import sigmoid

def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples
    grad = np.zeros(theta.shape[0])
    theta = np.transpose(theta)
    sum_1 = 0
    X = X.values
    y = y.values
    #calcuate the theta_0 
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
    for i in range(theta.shape[0]):
        if i == 0:
            for j in range(m):
                sum_1 += (sigmoid(np.dot(X[j],theta)) - y[j]) * X[j,i]
        else:
            for j in range(m):
                sum_1 += (sigmoid(np.dot(X[j],theta)) - y[j]) * X[j,i] + Lambda*theta[i]
        grad[i] = sum_1/m
        sum_1 = 0

# =============================================================

    return grad