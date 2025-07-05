import numpy as np


def lu_decomposition(A):
    L = np.eye(A.shape[0])
    U = A.copy()
    for i in range(A.shape[0] - 1):
        for j in range(i + 1, A.shape[0]):
            L[j, i] = U[j, i] / U[i, i]
            U[j] -= L[j, i] * U[i]
    return L, U


def ldu_decomposition(A):
    L, U = lu_decomposition(A)
    D = np.eye(A.shape[0])
    for i in range(A.shape[0]):
        D[i, i] = U[i, i]
        U[i] = U[i] / D[i, i] if D[i, i] != 0 else np.eye(A.shape[0])[i]
    return L, D, U


def doolittle_decomposition(A):
    L, D, U = ldu_decomposition(A)
    for i in range(A.shape[0]):
        U[i] *= D[i, i]
    return L, U


def crout_decomposition(A):
    L, D, U = ldu_decomposition(A)
    for i in range(A.shape[0]):
        L[i] *= D[i, i]
    return L, U


def cholesky_decomposition(A):
    G, D, GT = ldu_decomposition(A)
    for i in range(A.shape[0]):
        G[i] *= D[i, i] ** 0.5
    return G


def forward_substitution(L, b):
    y = np.zeros(L.shape[0])
    y[0] = b[0] / L[0, 0]
    for i in range(1, L.shape[0]):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def back_substitution(U, y):
    x = np.zeros(U.shape[0])
    x[-1] = y[-1] / U[-1, -1]
    for i in range(1, U.shape[0]):
        x[-1 - i] = (y[-1 - i] - np.dot(U[-1 - i, -i:U.shape[0]], x[-i:U.shape[0]])) / U[-1 - i, -1 - i]
    return x


A = np.array([
    [1., -1., -1.],
    [2., -1., -3.],
    [3., 2., -5.]
])
b = np.array([2., 1., 0.])
L, U = lu_decomposition(A)
y = forward_substitution(L, b)
x = back_substitution(U, y)
print(f'x={x}')
