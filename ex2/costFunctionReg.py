from costFunction import costFunction
import numpy as np

#Since the Reg version is using Panda(why?!)
#It's probably makes more sense to make a speical function just for that....
def costFunctionPanda(theta, X,y):
    return 1
    

def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples
    #the sum
    sum_1 = 0
    sum_2 = 0
    #the cost
    J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
    sum_1 = costFunction(theta,X.values,y.values)
    for i in range(theta.shape[0]):
        sum_2 += theta[i]**2
                 
    J = sum_1 + (sum_2 * Lambda)/(2*m)

# =============================================================
    return J
