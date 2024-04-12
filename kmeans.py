import numpy as np

def kmeans(X, k, max_iters=100):
    cluster_ids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        clusters = np.array([np.argmin(np.linalg.norm(x-cluster_ids, axis=1)) for x in X])
        new_cluster_ids = np.array([X[clusters==i].mean(axis=0) for i in range(k)])
        if np.all(new_cluster_ids == cluster_ids):
            break
        cluster_ids = new_cluster_ids
    return cluster_ids, clusters

if __name__ == "__main__":
    X = np.array([[1,1],[1,2], [9,9],[10,10], [2,3]])
    cluster_ids, clusters = kmeans(X, k=2, max_iters=100)
    print(cluster_ids, clusters)
