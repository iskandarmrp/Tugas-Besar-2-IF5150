import numpy as np
import math

class NaiveBayes:
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.unique_classes = np.unique(y)
        num_classes = len(self.unique_classes)
        self.class_means = np.zeros((num_classes, num_features), dtype=np.float64)
        self.class_vars = np.zeros((num_classes, num_features), dtype=np.float64)
        self.class_priors = np.zeros(num_classes, dtype=np.float64)

        for idx, class_label in enumerate(self.unique_classes):
            data_class = X[y == class_label]
            self.class_means[idx, :] = data_class.mean(axis=0)
            self.class_vars[idx, :] = data_class.var(axis=0)
            self.class_priors[idx] = data_class.shape[0] / float(num_samples)
            

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        class_posteriors = []

        for idx, class_label in enumerate(self.unique_classes):
            log_prior = np.log(self.class_priors[idx])
            log_likelihood = np.sum(np.log(self._calculate_pdf(idx, x)))
            log_posterior = log_likelihood + log_prior
            class_posteriors.append(log_posterior)

        return self.unique_classes[np.argmax(class_posteriors)]

    def _calculate_pdf(self, class_idx, x):
        x = np.asarray(x)
        mean = self.class_means[class_idx]
        var = self.class_vars[class_idx]
        var = np.maximum(var, 1e-9)
        val = -1*((x - mean) ** 2) / (2 * var)
        numerator = np.array([math.exp(i) for i in val])
        denominator = np.sqrt(2 * np.pi * var)
        return np.maximum(numerator / denominator, 1e-9)