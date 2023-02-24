import unittest
import numpy as np
import numpy.linalg as la
import sympy as sp

sp.init_printing(use_unicode=True)

from mathematics_4_machine_learning.linear_algebra.utils import augmented_to_ref


class TestLinearAlgebraQuiz(unittest.TestCase):
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
        print(la.solve(A_ref[:, :4], np.array([3, 22, 7, 1], dtype=np.dtype(float))))

        self.assertTrue(np.allclose(A_out, A_ref))

    def test_distance(self):
        A = np.array([1, 0, 7])
        B = np.array([0, -1, 2])

        print(la.norm(A))
        print(la.norm(B))
        print(la.norm(A - B))

    def test_maximum_norm(self):
        A = np.array([0, 0, 0, 0])
        B = np.array([1, 0, -2, 0, 1])
        C = np.array([1, 2, 3])
        D = np.array([2, 2, 2, 2])
        E = np.array([2, 5])

        print(la.norm(A, ord=np.inf))
        print(la.norm(B, ord=np.inf))
        print(la.norm(C, ord=np.inf))
        print(la.norm(D, ord=np.inf))
        print(la.norm(E, ord=np.inf))

    def test_dot_product_on_vectors(self):
        def dot_product(A, B):
            dotp = 0
            for x, y in zip(A, B):
                print(x * y)
                dotp += x * y

            return dotp

        A = np.array([-1, 5, 2])
        B = np.array([-3, 6, -4])
        res = dot_product(A, B)
        print(res)

    def test_dot_product_on_matrices(self):
        A = np.array([
            [2, -1],
            [3, -3]
        ])
        B = np.array([
            [5, -2],
            [0, 1]
        ])
        print(A)
        print(B)

        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                print(A[i, :] * B[:, j])
                print(np.sum(A[i, :] * B[:, j]))

    def test_single_dot_product(self):
        A = np.array([-9, -1])
        B = np.array([-3, -5])
        print(A)
        print(B)
        print(A * B)
        print(np.sum(A * B))

    def test_determinant(self):
        A = np.array([
            [1, 2, -1],
            [1, 0, 1],
            [0, 1, 0]
        ])
        print(la.det(A))

    def test_inverse(self):
        A = np.array([
            [1, 2, -1],
            [1, 0, 1],
            [0, 1, 0]
        ])
        print(la.inv(A))

    def test_inverse_multiply_by_identity_matrix(self):
        A = np.array([
            [1, 2, -1],
            [1, 0, 1],
            [0, 1, 0]
        ])
        A_inv = la.inv(A)
        # identity matrix
        I = np.eye(3)
        print(A_inv @ I)

    def test_identity_matrix_is_singular(self):
        I = np.eye(3)
        print(la.det(I))

    def test_linear_transformation(self):
        W = np.array([
            [1, 2, -1],
            [1, 0, 1],
            [0, 1, 0]
        ])
        v = np.array([5, -2, 0])
        print(W @ v)
        print(la.det(W))

    def test_column_as_vectors_multiplication(self):
        Z = np.array([
            [3, 5, 2],
            [1, 2, 2],
            [-7, 1, 0]
        ])
        # extract the first column and the third column as separate vectors
        v0 = Z[:, 0]
        v2 = Z[:, 2]
        print(v0 @ v2)

    def test_3d_matrix_multilplication(self):
        A = np.array([
            [5, 2, 3],
            [-1, -3, 2],
            [0, 1, -1]
        ])
        B = np.array([
            [1, 0, -4],
            [2, 1, 0],
            [8, -1, 0]
        ])
        AB = A @ B
        print(AB)
        print(round(la.det(la.inv(AB)), 2))

    def test_extract_eigenvalues_and_eigenvectors_from_matrix(self):
        A = np.array([
            [9, 4],
            [4, 3]
        ])
        w, v = la.eig(A)
        print('eigenvalues')
        print(w)
        print('eigenvectors')
        print(v)

    def test_extract_eigenvalues_and_eigenvectors_from_matrix2(self):
        A = np.array([
            [2, 1],
            [-3, 6]
        ])
        w, v = la.eig(A)
        print('eigenvalues')
        print(w)
        print('eigenvectors')
        print(v)

    def test_extract_eigenvalues_and_eigenvectors_from_matrix3(self):
        A = sp.Matrix([
            [1, 2],
            [0, 4]
        ])
        print('eigenvalues')
        print(A.eigenvects()[0][0], A.eigenvects()[1][0])
        print('eigenvectors')
        print(A.eigenvects()[0][2], A.eigenvects()[1][2])

    def test_matrix_transformation(self):
        A = np.array([
            [1, 2],
            [3, 4]
        ])
        # define a matrix to reflect through the y-axis
        R = np.array([
            [-1, 0],
            [0, 1]
        ])
        # define a matrix to shear along the x-axis by 0.5
        S = np.array([
            [1, 0.5],
            [0, 1]
        ])
        # define a matrix where all elements on the main diagonal should be equal to 0 and the entries in each column must add to one with n=5
        P = np.array([
            [0, 0.2, 0.2, 0.3, 0.5],
            [0.3, 0, 0.2, 0.5, 0.25],
            [0.3, 0.25, 0, 0.15, 0.1],
            [0.35, 0.5, 0.2, 0, 0.15],
            [0.05, 0.05, 0.4, 0.05, 0]
        ])

        # check that all columns sum to 1
        print(np.sum(P, axis=0))
