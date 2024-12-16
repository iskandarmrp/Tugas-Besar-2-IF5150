from DTL import DTL
from collections import Counter
import numpy as np
import pandas as pd

class RandomForest():

    def __init__(self, width):
        self.width = width
        self.foots = {}

    def buildRandomForest(self, S, Y):
        width = self.width
        rows = len(S)
        
        for i in range(width):
            sample = S.sample(n=rows, replace=True, random_state=42)
            tree = DTL()
            self.foots[str(i)] = tree.buildTree(sample, Y)
        
        return self
    
    def _predict(self, X):
        predictions = []
        if isinstance(X, pd.DataFrame):
            pass
        else:
            X = pd.DataFrame(X)

        for i in self.foots.keys():
            root = self.foots[i]
            pred = root.inferenceTree(root, X)
            predictions.append(pred.foots)
        
        count = Counter(predictions)
        mode_prediction = count.most_common(1)[0][0]
        return mode_prediction
    
    def predict(self, X):
        y_pred = []
        for row in X:
            row_df = pd.DataFrame([row])
            prediction = self._predict(row_df) 
            y_pred.append(prediction)

        return y_pred
    
    def fit(self, X, Y):
        Y = pd.Series(Y, name='label')
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = pd.concat([X, Y], axis=1)   
        return self.buildRandomForest(X, 'label')