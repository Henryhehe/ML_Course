import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).
    g = 1 / (1 + np.exp(-z))
# =============================================================
    return g

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
#print sigmoid(0)