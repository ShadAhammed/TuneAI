"""ModelRunner — orchestrates all seven classifiers end-to-end.

Instantiate with training data, call RunModel() with test data and a
dataset label.  After all models finish, the runner:
  1. Generates a static PNG dashboard.
  2. Saves a full JSON results file (metrics + ROC/PR curves + confusion
     matrices) so the interactive Streamlit dashboard can load it later.
  3. Prints a formatted summary table.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from src.models.ANN import ANN
from src.models.SVM import SupportVector
from src.models.XGB import XtremeGrad
from src.models.RF import RandomForest
from src.models.LR import LogRegression
from src.models.NB import NaiveBayes
from src.models.KNN import KNearestNeighbour
from src.visualization.Performance import ModelPerformance
from src.results_io import save_results


class ModelRunner:
    """Runs every available classifier against the supplied dataset."""

    def __init__(self, X_train: pd.DataFrame, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self._report_frames: list[pd.DataFrame] = []
        self._roc_data: list[dict] = []
        self._confusion_matrices: list[tuple[str, np.ndarray]] = []

    @staticmethod
    def _get_positive_class_scores(model, X_test) -> np.ndarray | None:
        """Return positive-class probability scores for ROC/PR computation."""
        if hasattr(model, 'predict_proba'):
            try:
                return model.predict_proba(X_test)[:, 1]
            except Exception:
                pass
        try:
            raw = model.predict(X_test, verbose=0)
            if hasattr(raw, 'ndim') and raw.ndim > 1:
                return raw.ravel()
        except TypeError:
            try:
                raw = model.predict(X_test)
                if hasattr(raw, 'ndim') and raw.ndim > 1:
                    return raw.ravel()
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _collect_curve_data(self, model_name: str, model, X_test, y_true: np.ndarray) -> None:
        """Compute ROC and Precision-Recall curves and store for the dashboard."""
        y_score = self._get_positive_class_scores(model, X_test)
        if y_score is None:
            return
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
        self._roc_data.append({
            'name':            model_name,
            'fpr':             fpr,
            'tpr':             tpr,
            'auc':             auc(fpr, tpr),
            'precision_curve': precision_vals,
            'recall_curve':    recall_vals,
            'avg_precision':   average_precision_score(y_true, y_score),
        })

    def _evaluate(self, model_name: str, trained_model, X_test, y_test) -> None:
        """Predict, draw confusion matrix, collect curve data, store report."""
        print(f'\n{"=" * 55}')
        print(f'  {model_name}')
        print(f'{"=" * 55}')

        prediction = trained_model.predict(X_test)
        if prediction.ndim > 1:
            prediction = np.round(prediction).astype(int).ravel()

        y_true = np.asarray(y_test).ravel()

        conf_matrix = metrics.confusion_matrix(y_true, prediction)
        self._confusion_matrices.append((model_name, conf_matrix))

        perf = ModelPerformance(X_test, y_test)
        perf.draw_confusion_matrix(conf_matrix, model_name)

        report = ModelPerformance.clf_report(y_true, prediction, model_name)
        self._report_frames.append(report)

        self._collect_curve_data(model_name, trained_model, X_test, y_true)

    def RunModel(
        self,
        X_test: pd.DataFrame,
        y_test,
        dataset_label: str = 'Dataset',
        dashboard_path: str | None = None,
        results_dir: str = 'results',
        dataset_info: dict | None = None,
    ) -> None:
        """Train, tune, and evaluate all seven classifiers in sequence."""
        print(f'\nRunning TuneAI on: {dataset_label}\n')

        ann = ANN(self.X_train, self.y_train)
        self._evaluate('ANN', ann.ANN_model(tuner_id=2), X_test, y_test)

        svm_clf = SupportVector(self.X_train, self.y_train)
        self._evaluate('SVM', svm_clf.TunedModel(), X_test, y_test)

        xgb_clf = XtremeGrad(self.X_train, self.y_train)
        self._evaluate('XGBoost', xgb_clf.TunedModel(), X_test, y_test)

        rf_clf = RandomForest(self.X_train, self.y_train)
        self._evaluate('Random Forest', rf_clf.TunedModel(), X_test, y_test)

        lr_clf = LogRegression(self.X_train, self.y_train)
        self._evaluate('Logistic Regression', lr_clf.TunedModel(), X_test, y_test)

        nb_clf = NaiveBayes(self.X_train, self.y_train)
        self._evaluate('Naive Bayes', nb_clf.TunedModel(), X_test, y_test)

        knn_clf = KNearestNeighbour(self.X_train, self.y_train)
        self._evaluate('KNN', knn_clf.TunedModel(), X_test, y_test)

        # Static PNG dashboard
        ModelPerformance.generate_dashboard(
            self._report_frames,
            dataset_label,
            save_path=dashboard_path,
            roc_data=self._roc_data if self._roc_data else None,
        )

        # JSON results file — loaded by the interactive Streamlit dashboard
        json_path = save_results(
            results_dir=results_dir,
            dataset_label=dataset_label,
            dataset_info=dataset_info or {},
            report_frames=self._report_frames,
            roc_data=self._roc_data if self._roc_data else None,
            confusion_matrices=self._confusion_matrices if self._confusion_matrices else None,
        )
        print(f'Results saved to: {json_path}')

        self._print_summary(dataset_label)

    def _print_summary(self, dataset_label: str) -> None:
        if not self._report_frames:
            return
        summary = pd.concat(self._report_frames, ignore_index=True)
        if self._roc_data:
            auc_map = {rd['name']: round(rd['auc'], 4) for rd in self._roc_data}
            summary['ROC-AUC'] = summary['Model'].map(auc_map)
        print(f'\n\n{"=" * 70}')
        print(f'  Performance Summary - {dataset_label}')
        print(f'{"=" * 70}')
        print(tabulate(summary, headers='keys', tablefmt='simple', floatfmt='.4f'))
        print()
