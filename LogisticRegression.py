import pandas as pd
import numpy as np
import math
from RandomForest import RandomForest
from NaiveBayes import NaiveBayes
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate=0.01, ephocs=1000):
        self.learning_rate = learning_rate
        self.ephocs = ephocs
        

        self.randomForest = RandomForest(3)
        self.naiveBayes = NaiveBayes()

    def sigmoid(self, z):
        return 1 / (1 + np.array([math.exp(x) for x in z]))

    def initialize_weights(self, n_features):
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = np.array(0, dtype=np.float64)

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.initialize_weights(2)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.randomForest.fit(X_train, y_train)

        self.naiveBayes.fit(X_train, y_train)

        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))

        df = pd.DataFrame({'Feature_1': [None] * n_samples})

        df['Feature_1'] = self.randomForest.predict(X)
        df['Feature_2'] = self.naiveBayes.predict(X)

        df = np.array(df, dtype=np.float64)
        for i in range(self.ephocs):
            linear_model = np.dot(df, self.weights) + self.bias

            p = self.sigmoid(linear_model)

            y_predicted = [1 if prob >= 0.5 else 0 for prob in np.log10(p / (1 - p))]

            y_predicted = np.array(y_predicted, dtype=np.float64)
            y = np.array(y, dtype=np.float64)

            dw = (1 / n_samples) * np.dot(df.T, (y_predicted-y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _predict(self, X):
        X_1 = pd.DataFrame([X])
        x_1 = self.randomForest._predict(X_1)      
        x_2 = self.naiveBayes._predict(X)
        
        x = np.array([x_1,x_2])

        linear_model = np.dot(x, self.weights) + self.bias            
        p = 1 / (1 + math.exp(linear_model))

        return 1 if np.log10(p / (1 - p)) >= 0.5 else 0 
    
    def predict(self, X):
        return [self._predict(x) for x in X]