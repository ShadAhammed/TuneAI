"""Persistence layer for TuneAI run results.

Every time ModelRunner finishes a dataset, it calls save_results() to write
a JSON file under the results/ directory.  The Streamlit dashboard reads
those files with load_results() so it can display any past run interactively
without re-training the models.

JSON schema summary
-------------------
{
  "label":        str,
  "timestamp":    ISO-8601 string,
  "dataset_info": { "n_train", "n_test", "n_features", "class_balance", "target_column" },
  "models": [
    {
      "name":              str,
      "accuracy":          float,
      "precision":         float,
      "recall":            float,
      "f1":                float,
      "auc":               float | null,
      "avg_precision":     float | null,
      "fpr":               [float] | null,   -- ROC false-positive rates
      "tpr":               [float] | null,   -- ROC true-positive rates
      "precision_curve":   [float] | null,   -- PR precision values
      "recall_curve":      [float] | null,   -- PR recall values
      "confusion_matrix":  [[int]] | null    -- 2x2 matrix
    },
    ...
  ]
}
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def save_results(
    results_dir: str,
    dataset_label: str,
    dataset_info: dict,
    report_frames: list,
    roc_data: list | None = None,
    confusion_matrices: list | None = None,
) -> str:
    """Serialise a complete run to a JSON file and return its path.

    Parameters
    ----------
    results_dir:
        Folder where the file will be written.
    dataset_label:
        Human-readable name for the run (used as part of the file name).
    dataset_info:
        Dict with keys n_train, n_test, n_features, class_balance,
        target_column.
    report_frames:
        List of single-row DataFrames from ModelPerformance.clf_report().
    roc_data:
        Optional list of curve dicts from ModelRunner._collect_curve_data().
    confusion_matrices:
        Optional list of (model_name, np.ndarray) pairs.
    """
    safe_label = dataset_label.replace(' ', '_').replace('/', '_')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f'{safe_label}_results.json')

    summary = pd.concat(report_frames, ignore_index=True)

    roc_map = {rd['name']: rd for rd in roc_data} if roc_data else {}
    cm_map  = {name: cm for name, cm in confusion_matrices} if confusion_matrices else {}

    models_out = []
    for _, row in summary.iterrows():
        name = row['Model']
        rd   = roc_map.get(name)
        cm   = cm_map.get(name)

        models_out.append({
            'name':            name,
            'accuracy':        round(float(row['Accuracy']), 4),
            'precision':       round(float(row['Precision']), 4),
            'recall':          round(float(row['Recall']), 4),
            'f1':              round(float(row['F1-Score']), 4),
            'auc':             round(float(rd['auc']), 4)          if rd else None,
            'avg_precision':   round(float(rd['avg_precision']), 4) if rd else None,
            'fpr':             rd['fpr'].tolist()               if rd else None,
            'tpr':             rd['tpr'].tolist()               if rd else None,
            'precision_curve': rd['precision_curve'].tolist()   if rd else None,
            'recall_curve':    rd['recall_curve'].tolist()      if rd else None,
            'confusion_matrix': cm.tolist()                     if cm is not None else None,
        })

    payload = {
        'label':        dataset_label,
        'timestamp':    datetime.now().isoformat(timespec='seconds'),
        'dataset_info': dataset_info,
        'models':       models_out,
    }

    with open(filepath, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)

    return filepath


def load_results(filepath: str) -> dict:
    """Load a JSON results file and convert curve lists back to numpy arrays."""
    with open(filepath, encoding='utf-8') as fh:
        data = json.load(fh)

    array_keys = ('fpr', 'tpr', 'precision_curve', 'recall_curve')
    for model in data.get('models', []):
        for key in array_keys:
            if model.get(key) is not None:
                model[key] = np.array(model[key])
        if model.get('confusion_matrix') is not None:
            model['confusion_matrix'] = np.array(model['confusion_matrix'])

    return data


def list_results(results_dir: str) -> list[str]:
    """Return sorted list of JSON result file paths in results_dir."""
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith('_results.json')
    )
