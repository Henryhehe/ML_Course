import numpy as np
from computeCost import computeCost



def computeTheta0(theta0,theta1,X,y,alpha,size):
    h = lambda x: theta0 + theta1 * x
    total = 0
    for i in np.arange(size):
        total += (h(X[i,1]) - y[i])
    return theta0 - alpha * (total) / size

def computeTheta1(theta0,theta1,X,y,alpha,size):
    h = lambda x: theta0 + theta1 * x
    total = 0
    for i in np.arange(size):
        total += (h(X[i,1]) - y[i])* X[i,1]
    return theta1 - alpha * (total) / size

    
def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples
    temp0 = 0
    temp1 = 0

    for i in np.arange(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        temp0 = computeTheta0(theta[0],theta[1],X,y,alpha,m)
        temp1 = computeTheta1(theta[0],theta[1],X,y,alpha,m)
        #updata both theta values
        theta[0] = temp0
        theta[1] = temp1

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    print J_history
    return theta, J_history
