import numpy as np
import pandas as pd

class DBScan:

    def __init__(self, alpha, beta, n, d):
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.d = d

    def numerical_distance(self, X1, X2):
        column_list = X1.columns
        sum = 0
        for col in column_list:
            sum+=((X1[col]-X2[col])**2) 
        return sum**0.5
    
    def categorical_distance(self, X1, X2):
        column_list = X1.columns
        sum = 0
        for col in column_list:
            if (X1[col]!=X2[col]):
                sum+=1
        return sum
    
    def split_numerical_categorical(self, X):
        numerical_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        numerical_data = X[numerical_cols]
        categorical_data = X[categorical_cols]
        return numerical_data, categorical_data
    
    def build_adjacency_list(self, X, d):
        X_num, X_cat = self.split_numerical_categorical(X)
        n = X.shape[0]
        adjacency_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n): 
                num_distance = np.linalg.norm(X_num.iloc[i] - X_num.iloc[j])
                cat_distance = (X_cat.iloc[i] != X_cat.iloc[j]).sum()
                distance = self.alpha * num_distance + self.beta * cat_distance
                if distance <= self.d:  
                    adjacency_list[i].append(j)  
                    adjacency_list[j].append(i)  
        return adjacency_list
    
    def clustering(self, X, n, d):
        adjacency_list = self.build_adjacency_list(X, d)
        clusters = {}
        index_visiting = set(range(len(adjacency_list)))
        while index_visiting:
            current_node = index_visiting.pop()
            temp_visiting = {current_node}
            cluster = set()
            while temp_visiting:
                node = temp_visiting.pop()
                cluster.add(node)

                neighbors = [neighbor for neighbor in adjacency_list[node] if neighbor in index_visiting]
                temp_visiting.update(neighbors)
                index_visiting.difference_update(neighbors)
            if len(cluster) >= n:
                clusters[len(clusters) + 1] = cluster
        return clusters
    
    def fit(self, X, n, d):
        self.n = n
        self.d = d
        clusters = self.clustering(X, n, d)
        X['cluster'] = 'outlier'
        for cluster_id, cluster_members in clusters.items():
            for member in cluster_members:
                X.at[member, 'cluster'] = cluster_id
        return X