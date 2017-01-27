import numpy as np



def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = y.size
    J = 0
    theta_0 = theta[0]
    theta_1 = theta[1]
    h = lambda x: theta_0 + theta_1 * x
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
    #there are theta0 and theta1 as two variables 
    #h_theate = theta0 + theta * x 
    #
    for i in np.arange(m):
        J += (h(X[i,1]) - y[i])**2
    J = J / (2*m)
    
# =========================================================================
    return J
