"""Logistic Regression wrapper with automated hyperparameter tuning."""

import warnings
warnings.filterwarnings('ignore')

from src.models.ModelTuning import Tuner
from models.ModelParams import LRParams, LRModel


class LogRegression:
    """Tune and fit a Logistic Regression classifier.

    Runs both GridSearch and RandomSearch then keeps the estimator that
    achieved the higher cross-validated accuracy.
    """

    def __init__(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def TunedModel(self):
        tuner = Tuner(LRParams, LRModel, self.X_train, self.y_train)
        grid_score, grid_model = tuner.GridSearch()
        rand_score, rand_model = tuner.RandomSearch()

        best = grid_model if grid_score >= rand_score else rand_model
        return best.fit(self.X_train, self.y_train)
