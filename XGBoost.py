import numpy as np
import pandas as pd

class XGBoost:
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, reg_lambda=1.0):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda
        self.trees = []

    class DecisionTree:
        def __init__(self, max_depth):
            self.max_depth = max_depth
            self.tree = None

        def fit(self, X, y, gradients):
            # Build the decision tree (simplified for regression)
            self.tree = self._build_tree(X, y, gradients, depth=0)

        def predict(self, X):
            # Predict using the built tree
            return np.array([self._predict_single(x, self.tree) for x in X], dtype='int64')

        def _build_tree(self, X, y, gradients, depth):
            if depth >= self.max_depth or len(X) <= 1:
                return {'value': -np.mean(gradients)}

            # Find the best split
            best_split = None
            best_loss = float('inf')
            n_samples, n_features = X.shape

            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left_idx = X[:, feature] <= threshold
                    right_idx = X[:, feature] > threshold

                    if len(X[left_idx]) == 0 or len(X[right_idx]) == 0:
                        continue

                    loss = self._split_loss(y, gradients, left_idx, right_idx)
                    if loss < best_loss:
                        best_loss = loss
                        best_split = (feature, threshold, left_idx, right_idx)

            # If no valid split is found, return a leaf node
            if best_split is None:
                return {'value': -np.mean(gradients)}

            feature, threshold, left_idx, right_idx = best_split
            left_tree = self._build_tree(X[left_idx], y[left_idx], gradients[left_idx], depth + 1)
            right_tree = self._build_tree(X[right_idx], y[right_idx], gradients[right_idx], depth + 1)

            return {
                'feature': feature,
                'threshold': threshold,
                'left': left_tree,
                'right': right_tree
            }

        def _split_loss(self, y, gradients, left_idx, right_idx):
            # Use a simple loss: variance reduction weighted by gradients
            grad_left = gradients[left_idx]
            grad_right = gradients[right_idx]

            loss_left = np.sum(grad_left**2)
            loss_right = np.sum(grad_right**2)

            return loss_left + loss_right

        def _predict_single(self, x, tree):
            if 'value' in tree:
                return tree['value']

            if x[tree['feature']] <= tree['threshold']:
                return self._predict_single(x, tree['left'])
            else:
                return self._predict_single(x, tree['right'])

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        predictions = np.zeros_like(y)

        for _ in range(self.n_estimators):
            gradients = self._compute_gradients(y, predictions)
            tree = self.DecisionTree(self.max_depth)
            tree.fit(X, y, gradients)
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X)

    def predict(self, X):
        X = np.array(X)
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

    def _compute_gradients(self, y, predictions):
        # For regression: gradients = -2 * (y - predictions)
        return -2 * (y - predictions)

