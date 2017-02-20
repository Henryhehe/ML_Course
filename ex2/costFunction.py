from numpy import log
from sigmoid import sigmoid
import numpy as np

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and theku
    gradient of the cost w.r.t. to the parameters."""

# Initialize some useful values
    m = y.size # number of training examples
    sum_1 = 0
    J = 0
    theta = np.transpose(theta)
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#
# Note: grad should have the same dimensions as theta
    for i in range(m):
        sum_1 += -y[i] * log(sigmoid(np.dot(X[i],theta))) - (1-y[i])* log(1- sigmoid(np.dot(X[i],theta)))
        
        
        
    J = sum_1/m
    return J
