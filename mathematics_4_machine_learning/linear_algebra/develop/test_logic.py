import unittest
import numpy as np

from mathematics_4_machine_learning.linear_algebra.utils import augmented_to_ref

# create a test for augmented_to_ref function
class TestClass(unittest.TestCase):
    def test_augmented_to_ref(self):
        A = np.array([
            [2, -1, 1, 1],
            [1, 2, -1, -1],
            [-1, 2, 2, 2],
            [1, -1, 2, 1]
        ], dtype=np.dtype(float))
        b = np.array([6, 3, 14, 8], dtype=np.dtype(float))
        A_ref = np.array([
            [1, 2, -1, -1, 3],
            [0, 1, 4, 3, 22],
            [0, 0, 1, 3, 7],
            [0, 0, 0, 1, 1]
        ], dtype=np.dtype(float))
        A_out = augmented_to_ref(A, b)
        print(A_out)
        print(A_ref)
        print(np.linalg.solve(A_ref[:, :4], np.array([3, 22, 7, 1], dtype=np.dtype(float))))

        self.assertTrue(np.allclose(A_out, A_ref))