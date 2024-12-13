import numpy as np
import pandas as pd
import math



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
                return self.addResult(Y, S[Y].iloc[0])
            elif len(S.columns) == 1:
                return self.addResult(Y, S[Y].mode()[0]) 
            else:
                feature = max_gain(S, Y) 
                self.name = feature
                values_list = S[feature].value_counts()
                filtered_values = [value for value in values_list.index if value in S[feature].values]
                sorted_values = filtered_values
                for value in sorted_values:
                    newS = S.loc[S[feature] == value].drop(columns=feature)
                    self.foots[value] = self.addFoots(newS, Y) 
            return self

        def inferenceTree(self, root, X):
            while isinstance(root.foots, dict):
                if root.name not in X: 
                    raise ValueError(f"Feature '{root.name}' not found in input data.")
                if X[root.name] not in root.foots:
                    key, root = next(iter(root.foots.items()))
                else:
                    root = root.foots[X[root.name]] 
            return root

        def predict(self, X):
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


