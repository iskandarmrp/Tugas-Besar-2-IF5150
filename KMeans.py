import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def calculate_wcss(self):
        wcss = 0
        for cluster_idx, cluster in enumerate(self.clusters):
            centroid = self.centroids[cluster_idx]
            for sample_idx in cluster:
                sample = self.X[sample_idx]
                wcss += euclidean_distance(sample, centroid)**2
        return wcss

def elbow_method(X, max_clusters=10):
    wcss_values = []
    
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(K=k)
        kmeans.predict(X)
        wcss = kmeans.calculate_wcss()
        wcss_values.append(wcss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss_values, marker='o')
    plt.title('Metode Elbow')
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    
    diffs = np.diff(wcss_values)
    optimal_k = np.argmin(diffs) + 2
    
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Jumlah Cluster Optimal = {optimal_k}')
    plt.legend()
    plt.show()
    
    return wcss_values, optimal_k