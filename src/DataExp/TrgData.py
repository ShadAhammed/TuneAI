"""Data preparation: statistics, scaling, and train/test splitting.

The DataPreparation class takes a raw DataFrame (loaded from any Excel file)
and prepares it for machine learning.  It assumes the last column is the
target variable - no column name is hardcoded, so the same class works for
any dataset regardless of how the columns are named.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin


class DataPreparation:
    """Wraps a raw DataFrame and exposes data-readiness helpers.

    Parameters
    ----------
    data: a pandas DataFrame where the last column is the binary target.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("The supplied DataFrame is empty.")
        self.data = data
        self._target_column = data.columns[-1]

    def DataStat(self) -> None:
        """Print a concise summary of the loaded dataset."""
        n_rows, n_cols = self.data.shape
        n_features = n_cols - 1
        n_positive = int(self.data[self._target_column].sum())
        n_negative = n_rows - n_positive

        print(f"Dataset shape   : {n_rows} samples, {n_features} features")
        print(f"Target column   : '{self._target_column}'")
        print(f"Positive cases  : {n_positive}")
        print(f"Negative cases  : {n_negative}")
        print(f"Class balance   : {n_positive / n_rows:.1%} positive\n")

    def split_data(
        self,
        test_size: float,
        scaler: TransformerMixin,
        random_state: int = 42,
    ):
        """Scale features and split into train and test sets.

        Parameters
        ----------
        test_size:
            Fraction of the data reserved for testing (e.g. 0.3 = 30 %).
        scaler:
            Any fitted scikit-learn scaler (e.g. MinMaxScaler()).
        random_state:
            Seed for reproducible splits.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

        X = self.data.iloc[:, :-1]
        y = self.data[[self._target_column]]   # Keep as DataFrame so column names travel with it

        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=self.data.index, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # y_train as a flat array is what scikit-learn classifiers expect
        y_train_arr = y_train.values.ravel()

        print(
            f"Train split: {len(X_train)} samples  |  "
            f"Test split: {len(X_test)} samples\n"
        )
        return X_train, X_test, y_train_arr, y_test
