import numpy as np
import pandas as pd
import math
from collections import Counter



class DTL:
        def __init__(self, name=''):
            self.name = name
            self.foots = {}

        def addFoots(self, S, Y):
            newFoots = DTL()
            newFoots = newFoots.buildTree(S, Y)
            return newFoots

        def addResult(self, name, value):
            newFoots = DTL(name)
            newFoots.foots = value
            return newFoots

        def buildTree(self, S, Y):
            if entropy(S[Y]) == 0:
                return self.addResult(Y, S[Y].mode()[0])
            elif len(S.columns) == 1:
                return self.addResult(Y, S[Y].mode()[0]) 
            else:
                feature = max_gain(S, Y) 
                self.name = feature
                for value in S[feature].unique():
                    newS = S.loc[S[feature] == value].drop(columns=feature)
                    self.foots[value] = self.addFoots(newS, Y) 
            return self           

        def inferenceTree(self, root, X):
            while isinstance(root.foots, dict):
                if root.name not in X.columns: 
                    break
                if X[root.name].iloc[0] not in root.foots.keys():
                    key, root = next(iter(root.foots.items()))
                else:
                    root = root.foots[X[root.name].iloc[0]] 
            return root
        
        def fit(self, X, Y):
            Y = pd.Series(Y, name='label')
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            X = pd.concat([X, Y], axis=1)   
            return self.buildTree(X, 'label')

        def predict(self, X):
            X = pd.DataFrame(X)
            result = []
            for _, row in X.iterrows():
                y = self.inferenceTree(self, row)
                result.append(y.foots)
            return result
        

def entropy(S):
    class_list = S.unique()
    temp_count = 0
    for i in range(len(class_list)):
        p = S.loc[S == class_list[i]].shape[0] / S.shape[0]
        temp_count+= -1*(p*math.log2(p))
    return float(temp_count)

def gain(S, target:str, feature: str):
    value_list = S[feature].unique()
    temp_count = 0
    for i in range(len(value_list)):
        temp_S = S.loc[S[feature]==value_list[i]]
        p = temp_S.shape[0] / S.shape[0]
        temp_val = p*entropy(temp_S[target])
        temp_count+=temp_val
    return entropy(S[target])-temp_count

def max_gain(S, target):
    S_feature = S.drop(columns=target)
    column_list = S_feature.columns
    max_column = (column_list[0],gain(S, target, column_list[0]))
    for i in range(1,len(column_list)):
        gain_value = gain(S, target, column_list[i])
        if max_column[1] < gain_value:
            max_column = (column_list[i], gain_value)

    return max_column[0]


