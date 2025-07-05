import numpy as np
import sympy as sp


def matrix_function(f, A):
    A = sp.Matrix(A)
    P, J = A.jordan_form()
    idx = 0
    x = sp.symbols('x')
    while idx < J.shape[0]:
        m = 1
        while idx + m < J.shape[0] and J[idx + m, idx + m] == J[idx, idx] and J[idx + m - 1, idx + m] == 1.:
            m += 1
        C = 1
        for i in range(m):
            for j in range(i, m):
                J[j - i + idx, j + idx] = C * sp.diff(f(x), x, i).subs(x, J[j - i + idx, j + idx])
            C /= (i + 1)
        idx += m
    return np.dot(np.dot(P, J), P.inv())


A = np.array([
    [2., 1., 0., 0.],
    [0., 2., 0., 0.],
    [0., 0., 1., 1.],
    [0., 0., 0., 1.]
])
fA = matrix_function(sp.log, A)
print(f'lnA={fA}')
