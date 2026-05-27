"""Model performance utilities: confusion matrices and a professional dashboard.

ModelPerformance handles three responsibilities:
  1. Drawing a per-model confusion matrix during the training loop.
  2. Building a one-row performance report (precision, recall, F1, accuracy)
     that ModelRunner accumulates across all classifiers.
  3. Rendering a professional multi-panel dashboard at the end of a run,
     including ROC-AUC curves, Precision-Recall curves, a bar chart
     comparison, and a color-coded metrics summary table.
"""

import itertools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report


# Seven visually distinct colors — one per classifier in the same order
# that ModelRunner trains them.  Used consistently across ROC, PR, and
# bar chart panels so a reader can cross-reference panels easily.
_PALETTE = [
    '#2196F3',  # ANN            — blue
    '#4CAF50',  # SVM            — green
    '#FF5722',  # XGBoost        — deep orange
    '#9C27B0',  # Random Forest  — purple
    '#FF9800',  # Logistic Reg.  — amber
    '#00BCD4',  # Naive Bayes    — cyan
    '#F44336',  # KNN            — red
]


class ModelPerformance:
    """Per-model evaluation helper, attached to a specific test split.

    Parameters
    ----------
    X_test: held-out feature matrix.
    y_test: held-out labels — DataFrame, Series, or 1-D array.
    """

    def __init__(self, X_test, y_test) -> None:
        self.X_test = X_test
        self.y_test = y_test

    def draw_confusion_matrix(self, matrix: np.ndarray, model_name: str = '') -> None:
        """Render a single confusion matrix figure, show briefly, then close."""
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

        # Derive class label strings; fall back to integers when ambiguous
        try:
            if hasattr(self.y_test, 'columns'):
                class_names = list(self.y_test.columns)
            elif hasattr(self.y_test, 'name') and self.y_test.name:
                class_names = [str(self.y_test.name)]
            else:
                class_names = [str(i) for i in range(matrix.shape[0])]
        except Exception:
            class_names = [str(i) for i in range(matrix.shape[0])]

        if len(class_names) != matrix.shape[0]:
            class_names = [str(i) for i in range(matrix.shape[0])]

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='left')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        thresh = matrix.max() / 2.0
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            ax.text(
                j, i, str(matrix[i, j]),
                ha='center', va='center',
                color='white' if matrix[i, j] > thresh else 'black',
            )

        ax.set_ylabel('True label', size=12)
        ax.set_xlabel('Predicted label', size=12)
        title = f'Confusion Matrix - {model_name}' if model_name else 'Confusion Matrix'
        ax.set_title(title, pad=14)
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)

    @staticmethod
    def clf_report(y_test, prediction, model_name: str) -> pd.DataFrame:
        """Return a one-row DataFrame with weighted-average performance metrics.

        Columns: Model, Precision, Recall, F1-Score, Accuracy.
        """
        report = classification_report(
            y_test, prediction, output_dict=True, zero_division=0
        )
        df = pd.DataFrame(report).T

        accuracy = df.loc['accuracy', 'precision'] if 'accuracy' in df.index else np.nan
        weighted = df.loc['weighted avg'] if 'weighted avg' in df.index else df.iloc[-1]

        return pd.DataFrame([{
            'Model':     model_name,
            'Precision': round(float(weighted['precision']), 4),
            'Recall':    round(float(weighted['recall']), 4),
            'F1-Score':  round(float(weighted['f1-score']), 4),
            'Accuracy':  round(float(accuracy), 4),
        }])

    # ------------------------------------------------------------------
    # Professional dashboard
    # ------------------------------------------------------------------

    @staticmethod
    def generate_dashboard(
        report_frames: list,
        dataset_label: str = 'Dataset',
        save_path: str | None = None,
        roc_data: list | None = None,
    ) -> None:
        """Render a professional multi-panel performance dashboard.

        Parameters
        ----------
        report_frames:
            List of single-row DataFrames from clf_report(), one per model.
        dataset_label:
            Title shown across the top of the figure.
        save_path:
            Optional path to save the figure (PNG, PDF, etc.).
        roc_data:
            Optional list of dicts produced by ModelRunner._collect_curve_data().
            Each dict must have: name, fpr, tpr, auc,
            precision_curve, recall_curve, avg_precision.
            When present, the dashboard gains ROC and PR curve panels.
        """
        if not report_frames:
            return

        summary = pd.concat(report_frames, ignore_index=True)
        has_roc = bool(roc_data)

        # Build the figure layout
        if has_roc:
            fig = plt.figure(figsize=(17, 15), facecolor='white')
            gs = gridspec.GridSpec(
                3, 2,
                height_ratios=[1.05, 0.95, 0.60],
                hspace=0.50,
                wspace=0.32,
                left=0.06, right=0.97,
                top=0.92, bottom=0.04,
            )
            ax_roc = fig.add_subplot(gs[0, 0])
            ax_pr  = fig.add_subplot(gs[0, 1])
            ax_bar = fig.add_subplot(gs[1, :])
            ax_tbl = fig.add_subplot(gs[2, :])
        else:
            fig, ax_bar = plt.subplots(figsize=(14, 6), facecolor='white')

        # --- Panel 1: ROC-AUC curves ------------------------------------
        if has_roc:
            ModelPerformance._draw_roc_panel(ax_roc, roc_data)

        # --- Panel 2: Precision-Recall curves ---------------------------
        if has_roc:
            ModelPerformance._draw_pr_panel(ax_pr, roc_data)

        # --- Panel 3: Bar chart comparison ------------------------------
        ModelPerformance._draw_bar_panel(ax_bar, summary, roc_data)

        # --- Panel 4: Metrics summary table -----------------------------
        if has_roc:
            ModelPerformance._draw_table_panel(ax_tbl, summary, roc_data)

        # Figure title
        fig.suptitle(
            f'TuneAI  |  Professional Performance Dashboard  |  {dataset_label}',
            fontsize=13,
            fontweight='bold',
            y=0.975 if has_roc else 1.02,
        )

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f'\nDashboard saved to: {save_path}\n')

        plt.show(block=False)
        plt.pause(3)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Private panel helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_roc_panel(ax: plt.Axes, roc_data: list) -> None:
        """Draw ROC curves for every model on a single axes."""
        # Diagonal reference line for a random classifier
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random (AUC = 0.500)')

        for i, rd in enumerate(roc_data):
            color = _PALETTE[i % len(_PALETTE)]
            ax.plot(
                rd['fpr'], rd['tpr'],
                lw=2.0, color=color,
                label=f"{rd['name']} (AUC = {rd['auc']:.3f})",
            )

        ax.fill_between([0, 1], [0, 1], alpha=0.04, color='grey')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.03])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title('ROC-AUC Curves', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=7.5, framealpha=0.92)
        ax.grid(True, linestyle='--', alpha=0.35)

    @staticmethod
    def _draw_pr_panel(ax: plt.Axes, roc_data: list) -> None:
        """Draw Precision-Recall curves for every model on a single axes."""
        for i, rd in enumerate(roc_data):
            color = _PALETTE[i % len(_PALETTE)]
            ax.plot(
                rd['recall_curve'], rd['precision_curve'],
                lw=2.0, color=color,
                label=f"{rd['name']} (AP = {rd['avg_precision']:.3f})",
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.03])
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title('Precision-Recall Curves', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', fontsize=7.5, framealpha=0.92)
        ax.grid(True, linestyle='--', alpha=0.35)

    @staticmethod
    def _draw_bar_panel(ax: plt.Axes, summary: pd.DataFrame, roc_data: list | None) -> None:
        """Draw a grouped bar chart with all metrics per model."""
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        available    = [c for c in metrics_cols if c in summary.columns]
        models       = summary['Model'].tolist()

        x         = np.arange(len(models))
        bar_width = 0.17
        n_metrics = len(available)
        offsets   = np.linspace(
            -(n_metrics - 1) * bar_width / 2,
             (n_metrics - 1) * bar_width / 2,
            n_metrics,
        )
        bar_colors = ['#3F51B5', '#43A047', '#E53935', '#8E24AA']

        for metric, offset, color in zip(available, offsets, bar_colors):
            values = summary[metric].tolist()
            bars   = ax.bar(
                x + offset, values, bar_width,
                label=metric, color=color, alpha=0.82,
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=6.5,
                )

        # Highlight the best-accuracy model with a subtle shaded column
        if 'Accuracy' in summary.columns:
            best_idx = int(summary['Accuracy'].idxmax())
            ax.axvspan(best_idx - 0.44, best_idx + 0.44, alpha=0.09, color='gold', zorder=0)
            ax.text(
                best_idx, 1.07, 'Best',
                ha='center', fontsize=8,
                color='#B8860B', fontweight='bold',
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18, ha='right', fontsize=9)
        ax.set_ylim(0.0, 1.14)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Model Performance Comparison', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.grid(axis='y', linestyle='--', alpha=0.35)

    @staticmethod
    def _draw_table_panel(ax: plt.Axes, summary: pd.DataFrame, roc_data: list) -> None:
        """Render a color-coded metrics summary table."""
        ax.axis('off')

        auc_map = {rd['name']: rd['auc'] for rd in roc_data}

        col_labels = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        cell_text  = []
        for _, row in summary.iterrows():
            name    = row['Model']
            auc_val = auc_map.get(name, float('nan'))
            cell_text.append([
                name,
                f"{row['Accuracy']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1-Score']:.4f}",
                f"{auc_val:.4f}" if not np.isnan(auc_val) else 'N/A',
            ])

        def _score_color(val_str: str) -> str:
            """Map a numeric score to a background color."""
            try:
                v = float(val_str)
                if v >= 0.87:
                    return '#C8E6C9'   # light green — excellent
                elif v >= 0.75:
                    return '#FFF9C4'   # light yellow — good
                else:
                    return '#FFCDD2'   # light red — needs attention
            except ValueError:
                return '#FFFFFF'

        # First column (model name) gets a neutral background
        cell_colors = [
            ['#E8EAF6'] + [_score_color(v) for v in row[1:]]
            for row in cell_text
        ]
        header_color = ['#3F51B5'] * len(col_labels)

        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            colColours=header_color,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        tbl.auto_set_column_width(list(range(len(col_labels))))

        # White bold text for header row
        for j in range(len(col_labels)):
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')

        ax.set_title(
            'Performance Metrics Summary   (green >= 0.87  |  yellow >= 0.75  |  red < 0.75)',
            fontsize=9,
            pad=8,
            color='#555555',
        )
