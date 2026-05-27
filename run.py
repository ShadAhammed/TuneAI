"""TuneAI - entry point.

Run without arguments to pick a file via dialog, or pass a path directly:

    python run.py
    python run.py --file data/FilteredPsyData.xlsx
    python run.py --file data/FilteredPsyData.xlsx --quick

The --quick flag subsamples large datasets and uses lighter hyperparameter
searches so you can verify the pipeline end-to-end in reasonable time.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.DataExp.TrgData import DataPreparation
from models.MLModels import ModelRunner


_BANNER = """
============================================================
          TuneAI - ML Model Comparison Suite
  ANN | SVM | XGBoost | Random Forest | LR | NB | KNN
============================================================
"""


def _pick_file() -> str:
    """Open a native file-picker and return the chosen path."""
    import tkinter as tk
    from tkinter.filedialog import askopenfilename

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    path = askopenfilename(
        title='Select your training data (Excel file)',
        filetypes=[
            ('Excel files', '*.xlsx *.xls'),
            ('All files', '*.*'),
        ],
    )
    root.destroy()

    if not path:
        print('No file selected. Exiting.')
        sys.exit(0)

    return path


def _apply_quick_mode() -> None:
    """Tell tuning modules to use faster (lighter) search settings."""
    from src.models import ModelTuning
    from src.models import ANN as ann_module

    ModelTuning.Tuner.QUICK_MODE = True
    ModelTuning.Tuner.CV_FOLDS = 2
    ann_module.ANN.QUICK_MODE = True


def _maybe_subsample(data: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Return a stratified subset when the dataset exceeds *max_rows*."""
    if len(data) <= max_rows:
        return data

    target = data.iloc[:, -1]
    subset, _ = train_test_split(
        data,
        train_size=max_rows,
        stratify=target,
        random_state=42,
    )
    print(f'Quick mode: using {len(subset)} of {len(data)} samples.\n')
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description='TuneAI model comparison suite')
    parser.add_argument(
        '--file', '-f',
        help='Path to Excel data file (skips file-picker dialog)',
    )
    parser.add_argument(
        '--label', '-l',
        help='Dataset label for charts (defaults to file name)',
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Faster run: subsample to 5000 rows and lighter tuning',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='results',
        help='Folder for saved dashboard image (default: results)',
    )
    args = parser.parse_args()

    print(_BANNER)

    if args.quick:
        _apply_quick_mode()

    if args.file:
        filepath = os.path.abspath(args.file)
        if not os.path.isfile(filepath):
            print(f'File not found: {filepath}')
            sys.exit(1)
    else:
        print('Please select your Excel data file ...\n')
        filepath = _pick_file()

    dataset_label = args.label or os.path.splitext(os.path.basename(filepath))[0]

    print(f'Dataset : {dataset_label}')
    print(f'File    : {filepath}')
    if args.quick:
        print('Mode    : quick (lighter hyperparameter search)\n')
    else:
        print()

    try:
        data = pd.read_excel(filepath, index_col=0)
    except Exception as exc:
        print(f'Error reading file: {exc}')
        sys.exit(1)

    if args.quick:
        data = _maybe_subsample(data, max_rows=5000)

    prep = DataPreparation(data)
    prep.DataStat()
    X_train, X_test, y_train, y_test = prep.split_data(
        test_size=0.3,
        scaler=MinMaxScaler(),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    safe_label = dataset_label.replace(' ', '_')
    dashboard_path = os.path.join(args.output_dir, f'dashboard_{safe_label}.png')

    dataset_info = {
        'n_train':       len(X_train),
        'n_test':        len(X_test),
        'n_features':    X_train.shape[1],
        'class_balance': round(float(data.iloc[:, -1].mean()), 4),
        'target_column': data.columns[-1],
    }

    runner = ModelRunner(X_train, y_train)
    runner.RunModel(
        X_test, y_test,
        dataset_label=dataset_label,
        dashboard_path=dashboard_path,
        results_dir=args.output_dir,
        dataset_info=dataset_info,
    )

    print('All done. Open the dashboard image or the plot window above.\n')


if __name__ == '__main__':
    main()
