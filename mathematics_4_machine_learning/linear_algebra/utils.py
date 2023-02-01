import numpy as np

def matrix_is_singular(matrix):
    """Return True if the matrix is singular, False otherwise.
    """

    return np.linalg.det(matrix) == 0
