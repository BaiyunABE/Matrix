import numpy as np
from sympy import Matrix, exp, cos


def matrix_function(f, A):
    A = Matrix(A)
    P, J = A.jordan_form()
    for i in range(A.shape[0]):
        J[i, i] = f(J[i, i])
    return np.dot(np.dot(P, J), P.inv())


A = np.array([
    [4., 6., 0.],
    [-3., -5., 0.],
    [-3., -6., 1.]
])
print(f'expA={matrix_function(exp, A)}')
print(f'cosA={matrix_function(cos, A)}')
