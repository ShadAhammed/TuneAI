"""Legacy batch runner — kept for reference.

The preferred way to use TuneAI is to run ``python run.py`` from the
project root, which opens a file-picker and handles a single dataset
interactively.

This file shows how to programmatically run multiple datasets in sequence
without the GUI dialog.  Replace the file paths below with paths that
exist on your local machine.  The data folder is intentionally excluded
from the repository (see .gitignore) to protect participant privacy.
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.DataExp.TrgData import DataPreparation
from models.MLModels import ModelRunner


def run_dataset(filepath: str, label: str) -> None:
    """Load, prepare, and evaluate a single dataset.

    Parameters
    ----------
    filepath:  absolute path to the Excel file.
    label:     short name used in console output and chart titles.
    """
    if not os.path.exists(filepath):
        print(f'[SKIP] File not found: {filepath}')
        return

    print(f'\n{"=" * 60}')
    print(f'  Dataset: {label}')
    print(f'{"=" * 60}')

    data = pd.read_excel(filepath, index_col=0)
    prep = DataPreparation(data)
    prep.DataStat()

    X_train, X_test, y_train, y_test = prep.split_data(
        test_size=0.3,
        scaler=MinMaxScaler(),
    )

    runner = ModelRunner(X_train, y_train)
    runner.RunModel(X_test, y_test, dataset_label=label)


# ---------------------------------------------------------------------------
# Add your dataset paths here.  Keep real data out of the repository.
# ---------------------------------------------------------------------------
DATASETS = [
    # (r'C:\path\to\your\data\dataset1.xlsx', 'My Dataset'),
]


if __name__ == '__main__':
    if not DATASETS:
        print('No datasets configured.  Add entries to the DATASETS list in this file,')
        print('or use "python run.py" for an interactive file-picker instead.')
        sys.exit(0)

    for path, label in DATASETS:
        run_dataset(path, label)

    print('\nBatch run complete.')
