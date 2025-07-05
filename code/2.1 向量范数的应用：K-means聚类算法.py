import numpy as np


def kmeans(vectors, centroids, K, max_iter=100):
    global labels
    for _ in range(max_iter):
        distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([vectors[labels == i].mean(axis=0) for i in range(K)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids


vectors = np.array([
    [1, 2, 3],
    [1.5, 2.5, 3.5],
    [10, 11, 12],
    [10.5, 11.5, 12.5],
    [100, 101, 102],
    [100.5, 101.5, 102.5]
])
# 测试分成 2 类
K = 2
initial_centroids_idx = np.random.choice(vectors.shape[0], K, replace=False)
centroids = vectors[initial_centroids_idx]
labels, centroids = kmeans(vectors, centroids, K)
print("分类标签：", labels)
print("类别中心点：\n", centroids)
# 测试分成 3 类
K = 3
initial_centroids_idx = np.random.choice(vectors.shape[0], K, replace=False)
centroids = vectors[initial_centroids_idx]
labels, centroids = kmeans(vectors, centroids, K)
print("分类标签：", labels)
print("类别中心点：\n", centroids)
