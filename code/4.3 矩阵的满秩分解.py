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


A = np.array([
    [-1., 0., 1., 2.],
    [1., 2., -1., 1.],
    [2., 2., -2., -1.]
])
F, G = full_rank_decomposition(A)
print(f'F={F}')
print(f'G={G}')
