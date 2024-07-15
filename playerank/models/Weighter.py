# /usr/local/bin/python
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Weighter(BaseEstimator):
    """Automatic weighting of performance features

    Parameters
    ----------
    label_type: str
        the label type associated to the game outcome.
        options: w-dl (victory vs draw or defeat), wd-l (victory or draw vs defeat),
                 w-d-l (victory, draw, defeat)
    random_state : int
        RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    feature_names_ : array, [n_features]
        names of the features
    label_type_: str
        the label type associated to the game outcome.
        options: w-dl (victory vs draw or defeat), wd-l (victory or draw vs defeat),
                 w-d-l (victory, draw, defeat)
    clf_: LinearSVC object
        the object of the trained classifier
    weights_ : array, [n_features]
        weights of the features computed by the classifier
    random_state_: int
        RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by 'np.random'.
    """
    def __init__(self, label_type='w-dl', random_state=42):
        self.label_type_ = label_type
        self.random_state_ = random_state

    def fit(self, dataframe, target, scaled=False, var_threshold = 0.001 , filename='weights.json'):
        """
        Compute weights of features.

        Parameters
        ----------
            dataframe : pandas DataFrame
                a dataframe containing the feature values and the target values

            target: str
               a string indicating the name of the target variable in the dataframe

            scaled: boolean
                True if X must be normalized, False otherwise
                (optional)

            filename: str
                the name of the files to be saved (the json file containing the feature weights,
                )
                default: "weights"
        """
        ##feature selection by variance, to delete outlier features
        feature_names = list(dataframe.columns)
        # normalize the data and then eliminate the variables with zero variance
        sel = VarianceThreshold(var_threshold)
        X = sel.fit_transform(dataframe)

        selected_feature_names = [feature_names[i] for i, var in enumerate(list(sel.variances_)) if var > var_threshold]
        print ("[Weighter] filtered features:", [(feature_names[i],var) for i, var in enumerate(list(sel.variances_)) if var <= var_threshold])
        dataframe = pd.DataFrame(X, columns=selected_feature_names)
        if self.label_type_ == 'w-dl':
            y = dataframe[target].apply(lambda x: 1 if x > 0 else -1)
        elif self.label_type_ == 'wd-l':
            y = dataframe[target].apply(lambda x: 1 if x >= 0 else -1 )
        else:
            y = dataframe[target].apply(lambda x: 1 if x > 0 else 0 if x==0 else 2)

        X = dataframe.loc[:, dataframe.columns != target].values
        y = y.values

        self.X = X
        self.y = y

        if scaled:
            X = StandardScaler().fit_transform(X)

        self.feature_names_ = dataframe.loc[:, dataframe.columns != target].columns
        self.clf_ = LinearSVC(fit_intercept=True, dual = False,  max_iter = 100000,random_state=self.random_state_, class_weight='balanced')

        f1_score = np.mean(cross_val_score(self.clf_, X, y, cv=2, scoring='f1_weighted'))
        self.f1_score_ = f1_score

        self.clf_.fit(X, y)

        outcome = 0
        if self.label_type_ == 'w-d-l':
            outcome = 1

        importances = self.clf_.coef_[outcome]  

        sum_importances = sum(np.abs(importances))
        self.weights_ = importances / sum_importances

        ## Save the computed weights into a json file
        features_and_weights = {}
        for feature, weight in sorted(zip(self.feature_names_, self.weights_),key = lambda x: x[1]):
            features_and_weights[feature]=  weight
        json.dump(features_and_weights, open('%s' %filename, 'w'))
        ## Save the object
        #pkl.dump(self, open('%s.pkl' %filename, 'wb'))

    def get_weights(self):
        return self.weights_

    def get_feature_names(self):
        return self.feature_names_
    
    def plot_graph(self):
        X = self.X
        y = self.y
        # Using PCA to reduce the data to 2D
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        
        # Fitting SVM on reduced data
        clf_reduced = LinearSVC(fit_intercept=True, dual=False, max_iter=100000, random_state=self.random_state_)
        clf_reduced.fit(X_reduced, y)

        # Plotting the data points
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='winter', marker='o')

        # Plotting the decision boundary
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30))
        Z = clf_reduced.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

        # Plotting the weight vector
        coef = clf_reduced.coef_[0]
        intercept = clf_reduced.intercept_[0]
        def decision_boundary(x):
            return -(coef[0] * x + intercept) / coef[1]
        
        x_vals = np.array(ax.get_xlim())
        y_vals = decision_boundary(x_vals)
        plt.plot(x_vals, y_vals, 'k-')

        # Plotting the weight vector as an arrow
        origin = np.array([0, 0])
        plt.quiver(*origin, coef[0], coef[1], scale=5, color='r', width=0.005)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Linear SVC Decision Boundary and Weight Vector in Reduced 2D Space')
        plt.show()
