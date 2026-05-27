"""Hyperparameter search wrappers: GridSearch and RandomSearch.

The Tuner class applies both strategies to a given model and parameter grid,
printing the best result from each.  The caller picks whichever strategy
produced the higher cross-validated accuracy.
"""

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class Tuner:
    """Run grid search and random search on a classifier.

    Parameters
    ----------
    params:   hyperparameter grid or distribution dict.
    model:    an unfitted scikit-learn estimator.
    X_train:  training features.
    y_train:  training labels (1-D array).
    """

    # 3-fold CV is a pragmatic choice - enough folds to reduce variance
    # without making the search prohibitively slow on larger datasets.
    CV_FOLDS = 3
    QUICK_MODE = False

    def __init__(self, params, model, X_train, y_train) -> None:
        self.params = params
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def GridSearch(self):
        """Exhaustive search over every combination in *params*.

        Returns
        -------
        (best_score, best_estimator) tuple.
        """
        if self.QUICK_MODE:
            # In quick mode we rely on RandomSearch only to save time.
            return 0.0, self.model

        search = GridSearchCV(
            estimator=self.model,
            param_grid=self.params,
            scoring='accuracy',
            cv=self.CV_FOLDS,
            n_jobs=-1,
            error_score=0,
            verbose=0,
        )
        result = search.fit(self.X_train, self.y_train)
        print(f"  Grid search   -> best accuracy: {result.best_score_:.4f}  params: {result.best_params_}")
        return search.best_score_, search.best_estimator_

    def RandomSearch(self):
        """Randomised search over a sample of *params* combinations.

        Returns
        -------
        (best_score, best_estimator) tuple.
        """
        n_iter = 8 if self.QUICK_MODE else 20
        search = RandomizedSearchCV(
            self.model,
            param_distributions=self.params,
            scoring='accuracy',
            cv=self.CV_FOLDS,
            n_jobs=-1,
            n_iter=n_iter,
            random_state=42,
            verbose=0,
        )
        result = search.fit(self.X_train, self.y_train)
        print(f"  Random search -> best accuracy: {result.best_score_:.4f}  params: {result.best_params_}")
        return search.best_score_, search.best_estimator_
