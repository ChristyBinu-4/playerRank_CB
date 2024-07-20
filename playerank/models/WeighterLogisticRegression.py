from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Weighter_LR(BaseEstimator):
    def __init__(self, label_type='w-dl', random_state=42):
        self.label_type_ = label_type
        self.random_state_ = random_state

    def fit(self, dataframe, target, scaled=False, var_threshold=0.001, filename='weights.json'):
        feature_names = list(dataframe.columns)
        sel = VarianceThreshold(var_threshold)
        X = sel.fit_transform(dataframe)

        selected_feature_names = [feature_names[i] for i, var in enumerate(list(sel.variances_)) if var > var_threshold]
        print("[Weighter] filtered features:", [(feature_names[i], var) for i, var in enumerate(list(sel.variances_)) if var <= var_threshold])
        dataframe = pd.DataFrame(X, columns=selected_feature_names)
        if self.label_type_ == 'w-dl':
            y = dataframe[target].apply(lambda x: 1 if x > 0 else -1)
        elif self.label_type_ == 'wd-l':
            y = dataframe[target].apply(lambda x: 1 if x >= 0 else -1)
        else:
            y = dataframe[target].apply(lambda x: 1 if x > 0 else 0 if x == 0 else 2)

        X = dataframe.loc[:, dataframe.columns != target].values
        y = y.values

        self.X = X
        self.y = y

        if scaled:
            X = StandardScaler().fit_transform(X)

        self.feature_names_ = dataframe.loc[:, dataframe.columns != target].columns
        self.clf_ = LogisticRegression(max_iter=10000, random_state=self.random_state_, class_weight='balanced')

        f1_score = np.mean(cross_val_score(self.clf_, X, y, cv=2, scoring='f1_weighted'))
        self.f1_score = f1_score

        self.clf_.fit(X, y)

        outcome = 0
        if self.label_type_ == 'w-d-l':
            outcome = 1

        importances = self.clf_.coef_[outcome]

        sum_importances = sum(np.abs(importances))
        self.weights_ = importances / sum_importances

        features_and_weights = {}
        for feature, weight in sorted(zip(self.feature_names_, self.weights_), key=lambda x: x[1]):
            features_and_weights[feature] = weight
        json.dump(features_and_weights, open('%s' % filename, 'w'))

    def get_weights(self):
        return self.weights_

    def get_feature_names(self):
        return self.feature_names_
