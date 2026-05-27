"""Hyperparameter search grids and base model instances for all classifiers.

Each model gets two things defined here: a parameter grid (used by GridSearch
and RandomSearch) and a base model instance with sensible defaults. Keeping
these together makes it straightforward to add or adjust search spaces without
touching the tuning logic itself.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


# Support Vector Classifier - RBF kernel, sweep C and gamma
SVCParams = {
    'kernel': ['rbf'],
    'gamma': [0.1, 0.01, 0.001],
    'C': [0.1, 1.0, 10.0],
}
SVCModel = SVC(probability=True)


# XGBoost - tree depth, regularisation, and sampling rates
XGBParams = {
    'min_child_weight': [3, 5, 8, 10],
    'gamma': [0.5, 0.8, 1.0, 1.2, 2.0, 3.0],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
    'max_depth': [1, 2, 4, 5, 7, 8],
}
XGBModel = XGBClassifier(eval_metric='logloss')


# Random Forest - 'auto' was removed in sklearn 1.3; use 'sqrt' or 'log2' instead
RFParams = {
    'max_features': ['sqrt', 'log2'],
    'n_estimators': [50, 100, 200, 400, 600],
    'max_samples': [0.6, 0.8],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=6)],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 3, 5, 8, 10],
}
RFModel = RandomForestClassifier(n_jobs=-1)


# Logistic Regression - regularisation strength and penalty type
LRParams = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l2'],          # liblinear supports l1 but newton-cg/lbfgs need l2
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}
LRModel = LogisticRegression(max_iter=1000)


# Naive Bayes - only the variance smoothing floor is tunable
NBParams = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
NBModel = GaussianNB()


# K-Nearest Neighbours - neighbour count, weighting, and distance metric
KNNParams = {
    'n_neighbors': range(1, 16, 2),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}
KNNModel = KNeighborsClassifier()
