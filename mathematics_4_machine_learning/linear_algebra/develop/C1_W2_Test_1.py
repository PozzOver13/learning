
import numpy as np

from mathematics_4_machine_learning.linear_algebra.utils import matrix_is_singular

# 1. Solve the system of equations using the method of elimination and select the correct answer.
A = np.array([[1, 1], [-6, 2]], dtype=int)

b = np.array([[4], [16]], dtype=int)

system_1 = np.hstack((A, b))

system_1[1] = 6 * system_1[0] + system_1[1]

system_1[1] = 1/8 * system_1[1]

# 2. For the questions 2-3, calculate the determinant of the matrices and determine if the matrices are singular or non-singular:

A = np.array([4, -3, 7, -8], dtype=int).reshape(2, 2)
print(A)

np.linalg.det(A)

# 3. For the questions 2-3, calculate the determinant of the matrices and determine if the matrices are singular or non-singular:

A = np.array([-3, 8, 1, 2, 2, -1, -5, 6, 2], dtype=int).reshape(3, 3)
print(A)

round(np.linalg.det(A))

matrix_is_singular(A)

# 6. In the following matrix:
a = 2
b = 1
c = 2

A = np.array([a, a, b, c]).reshape(2, 2)
print(A)

np.linalg.det(A)

# Luis went yesterday to the bank to find out the interest rate of three different financial instruments.
# I. s + c + z = 10000 (s, c, and z are amounts in savings, cds, and bonds respectively)
# II. 0.02s + 0.03c + 0.04z = 260, or 2s + 3c + 4z =26000
# III. s-2c=0 (because s=2c, “he put twice as much money in the savings account as in the CDs”)

# 3c + z = 10000
# 7c + 4z =26000
# s=2c

A = np.array([3, 1, 7, 4], dtype=int).reshape(2, 2)

b = np.array([10000, 26000], dtype=int)

np.linalg.solve(A, b)



# 7. The following system of equations is solved using the method of elimination:

A = np.array([7, 5, 3, 3, 2, 5, 1, 2, 1]).reshape(3, 3)

b = np.array([120, 70, 20], dtype=int)

matrix_is_singular(A)

np.linalg.det(A)

# solve the linear system and round the result to 2 decimal places
res = np.linalg.solve(A, b)

# compute rank of A
np.linalg.matrix_rank(A)

B = np.array([0, 1, 1, 2, 4, 2, 1, 2, 1], dtype=int).reshape(3, 3)
C = np.array([7.5, 5, 12.5, 3, 2, 5, 0, 0, 0], dtype=int).reshape(3, 3)

np.linalg.matrix_rank(C)

A_test = A.copy()

A_test[0] = A_test[0] * 3

A_test[0, 1]


A = np.array([
        [2, -1, 1, 1],
        [1, 2, -1, -1],
        [-1, 2, 2, 2],
        [1, -1, 2, 1]
    ], dtype=np.dtype(float))
b = np.array([6, 3, 14, 8], dtype=np.dtype(float))


