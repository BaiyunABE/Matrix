import numpy as np


def full_rank_decomposition(A):
    m, n = A.shape
    A = np.hstack((A, np.eye(m)))
    i, j = 0, 0
    for i in range(m):
        while A[i, j] == 0 and j < n:
            for k in range(i + 1, m):
                if A[k, j] != 0:
                    A[[i, k]] = A[[k, i]]
                    break
            j = j + 1 if A[i, j] == 0 else j
        if j == n:
            break
        for k in range(i + 1, m):
            A[k] -= A[i] * A[k, j] / A[i, j]
    return np.linalg.inv(A[:, n:n + m])[:, :i], A[:i, :n]


def pseudo_inv_full_rank(A):
    F, G = full_rank_decomposition(A)
    FH = F.transpose()
    GH = G.transpose()
    F_pseudo_inv = np.dot(np.linalg.inv(np.dot(FH, F)), FH)
    G_pseudo_inv = np.dot(GH, np.linalg.inv(np.dot(G, GH)))
    return G_pseudo_inv @ F_pseudo_inv


def pseudo_inv_svd(A):
    U, S, Vt = np.linalg.svd(A)
    m, n = A.shape
    Sp = np.zeros((n, m))
    for i in range(len(S)):
        if S[i] > 1e-10:
            Sp[i, i] = 1 / S[i]
    return Vt.T @ Sp @ U.T


A = np.array([
    [-1, 2, 1],
    [1, 0, 1],
    [0, -2, -2],
    [3, 2, 5]
])
b = np.array([[1], [0], [-1], [1]])
A_pseudo_inv = pseudo_inv_full_rank(A)
print(f'x={A_pseudo_inv @ b}')
A_pseudo_inv = pseudo_inv_svd(A)
print(f'x={A_pseudo_inv @ b}')
