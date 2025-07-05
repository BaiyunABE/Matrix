import numpy as np


def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for i in range(n):
        Q[:, i] = A[:, i]
        R[i, i] = 1
        for j in range(i):
            R[j, i] = np.dot(A[:, i], Q[:, j]) / np.dot(Q[:, j], Q[:, j])
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]
    for i in range(n):
        R[i] = R[i] * np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]
    return Q, R


def back_substitution(R, y):
    x = np.zeros(R.shape[0])
    x[-1] = y[-1] / R[-1, -1]
    for i in range(1, R.shape[0]):
        x[-1 - i] = (y[-1 - i] - np.dot(R[-1 - i, -i:R.shape[0]], x[-i:R.shape[0]])) / R[-1 - i, -1 - i]
    return x


A = np.array([
    [-1, 2, 1],
    [1, 0, 1],
    [0, -2, -2],
    [3, 2, 5]
])
b = np.array([1, 0, -1, 1])
Q, R = qr_decomposition(A)
QH = Q.transpose()
print(np.dot(QH, Q))
x = back_substitution(R, np.dot(QH, b))
print(f'x={x}')
print(Q)
print(np.dot(A, x))
