"""TuneAI Interactive Dashboard.

Run with:
    streamlit run app.py

The dashboard has two modes:
  1. Browse — pick any previously completed run from the sidebar and
     explore all its metrics interactively (ROC curves, PR curves,
     confusion matrices, and the metrics table).
  2. Run new data — upload an Excel file, configure options, and let
     TuneAI train, tune, and evaluate all seven classifiers.  Results
     are saved automatically so you can come back to them later.

Excel file requirements
-----------------------
  - First column: row index.
  - Middle columns: numeric features.
  - Last column: binary target (0 / 1).
"""

import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sure imports resolve from the project root regardless of how
# Streamlit is invoked.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.results_io import list_results, load_results

# -------------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------------
st.set_page_config(
    page_title='TuneAI Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Global style overrides ────────────────────────────────────────────────
# Times New Roman 15 px throughout; dark navy chrome.
st.markdown("""
<style>
    /* Base font: Times New Roman 15 px across all text */
    html, body, [class*="css"], .stMarkdown, .stText,
    .stDataFrame, .stTable, label, p, span, div {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 15px !important;
    }

    /* Headings */
    h1 { font-size: 1.7rem !important;
         font-family: "Times New Roman", Times, serif !important; }
    h2, h3 { font-size: 1.25rem !important;
              font-family: "Times New Roman", Times, serif !important; }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-family: "Times New Roman", Times, serif !important;
        color: #4fc3f7 !important;
        white-space: nowrap;
        overflow: visible;
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-family: "Times New Roman", Times, serif !important;
        color: #a0b8cc !important;
        white-space: nowrap;
        overflow: visible;
    }
    [data-testid="stMetricDelta"] {
        font-size: 13px !important;
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Page padding */
    .block-container { padding-top: 1.2rem; padding-bottom: 0.8rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        height: 36px; padding: 0 20px;
        font-size: 14px !important;
        font-family: "Times New Roman", Times, serif !important;
        font-weight: 600;
        color: #a0b8cc !important;
    }
    .stTabs [aria-selected="true"] { color: #4fc3f7 !important; }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #081527 !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 15px !important;
    }

    /* Widget labels */
    .stSelectbox > label,
    .stSlider > label,
    .stCheckbox > label,
    .stTextInput > label {
        font-size: 14px !important;
        color: #8bb5cc !important;
    }

    /* File uploader — keep only the widget's own label, no extras */
    [data-testid="stFileUploaderDropzoneInput"] + div span {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 11px !important;
    }

    /* Divider line */
    hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# Consistent color per model — same palette as the static PNG dashboard
_PALETTE = [
    '#2196F3', '#4CAF50', '#FF5722', '#9C27B0',
    '#FF9800', '#00BCD4', '#F44336',
]

RESULTS_DIR = os.path.join(_ROOT, 'results')


# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
with st.sidebar:
    st.title('TuneAI')
    st.caption('ML Model Comparison Suite')
    st.divider()

    # -- Browse existing results -----------------------------------------
    st.subheader('Existing Results')
    result_files = list_results(RESULTS_DIR)
    result_labels = {
        os.path.basename(f).replace('_results.json', '').replace('_', ' '): f
        for f in result_files
    }

    selected_label = st.selectbox(
        'Choose a dataset',
        options=list(result_labels.keys()),
        index=0 if result_labels else None,
    )

    st.divider()

    # -- Run new data ----------------------------------------------------
    st.subheader('Run New Dataset')
    uploaded_file = st.file_uploader(
        'Excel file (.xlsx) — last column = target (0/1)',
        type=['xlsx', 'xls'],
    )

    if uploaded_file:
        custom_label = st.text_input(
            'Dataset label',
            value=os.path.splitext(uploaded_file.name)[0],
        )
        test_size = st.slider(
            'Test split (%)', min_value=10, max_value=40, value=30, step=5
        ) / 100
        quick_mode = st.checkbox(
            'Quick mode (subsample to 5 000 rows, lighter tuning)',
            value=True,
        )
        run_button = st.button('Run Analysis', type='primary', use_container_width=True)
    else:
        run_button = False
        custom_label = ''
        test_size = 0.3
        quick_mode = True

    st.divider()
    st.caption('Results are saved automatically after each run.')


# -------------------------------------------------------------------------
# Handle "Run Analysis" click
# -------------------------------------------------------------------------
if run_button and uploaded_file is not None:
    # Lazy import — only needed when the user actually runs a dataset
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from src.DataExp.TrgData import DataPreparation
    from models.MLModels import ModelRunner
    from src.models.ModelTuning import Tuner
    from src.models.ANN import ANN as ANNClass

    st.info('Analysis started — this takes several minutes. Do not close this tab.')
    progress_bar = st.progress(0, text='Loading data ...')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        data = pd.read_excel(tmp_path, index_col=0)
    except Exception as exc:
        st.error(f'Could not read the file: {exc}')
        st.stop()
    finally:
        os.unlink(tmp_path)

    if quick_mode:
        if len(data) > 5000:
            target = data.iloc[:, -1]
            data, _ = train_test_split(
                data, train_size=5000, stratify=target, random_state=42
            )
        Tuner.QUICK_MODE = True
        Tuner.CV_FOLDS = 2
        ANNClass.QUICK_MODE = True

    progress_bar.progress(10, text='Preparing features ...')

    prep = DataPreparation(data)
    X_train, X_test, y_train, y_test = prep.split_data(
        test_size=test_size, scaler=MinMaxScaler()
    )

    dataset_info = {
        'n_train':       len(X_train),
        'n_test':        len(X_test),
        'n_features':    X_train.shape[1],
        'class_balance': round(float(data.iloc[:, -1].mean()), 4),
        'target_column': data.columns[-1],
    }

    progress_bar.progress(15, text='Training models (ANN first — this takes the longest) ...')

    safe_label = custom_label.replace(' ', '_')
    dashboard_path = os.path.join(RESULTS_DIR, f'dashboard_{safe_label}.png')

    runner = ModelRunner(X_train, y_train)

    # Streamlit can't intercept print() calls, so we run silently and
    # update progress at key milestones.
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runner.RunModel(
            X_test, y_test,
            dataset_label=custom_label,
            dashboard_path=dashboard_path,
            results_dir=RESULTS_DIR,
            dataset_info=dataset_info,
        )

    progress_bar.progress(100, text='Done!')
    time.sleep(0.5)
    progress_bar.empty()

    st.success(f'Analysis complete!  Results saved to results/{safe_label}_results.json')
    st.rerun()


# -------------------------------------------------------------------------
# Main display area
# -------------------------------------------------------------------------
st.title('TuneAI — Interactive Performance Dashboard')

if not result_labels:
    st.info(
        'No results yet. Upload an Excel file in the sidebar and click '
        '"Run Analysis" to get started.'
    )
    st.stop()

# Load the chosen result
results = load_results(result_labels[selected_label])
models  = results['models']
info    = results.get('dataset_info', {})
label   = results['label']
ts      = results.get('timestamp', '')

# Header row: dataset metadata
st.subheader(label)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Train samples', f"{info.get('n_train', '—'):,}")
col2.metric('Test samples',  f"{info.get('n_test', '—'):,}")
col3.metric('Features',       info.get('n_features', '—'))
col4.metric('Class balance',  f"{info.get('class_balance', 0):.1%}")
col5.metric('Run date', ts[:10] if ts else '—')

st.divider()

# Build a tidy summary DataFrame for reuse across tabs
summary = pd.DataFrame([{
    'Model':     m['name'],
    'Accuracy':  m['accuracy'],
    'Precision': m['precision'],
    'Recall':    m['recall'],
    'F1-Score':  m['f1'],
    'ROC-AUC':   m['auc'] if m['auc'] is not None else float('nan'),
    'Avg Prec.': m['avg_precision'] if m['avg_precision'] is not None else float('nan'),
} for m in models])

has_curves = any(m['fpr'] is not None for m in models)
has_cm     = any(m['confusion_matrix'] is not None for m in models)

model_color = {m['name']: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}


# -------------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------------
tab_overview, tab_roc, tab_pr, tab_cm, tab_raw = st.tabs([
    'Overview',
    'ROC-AUC Curves',
    'Precision-Recall Curves',
    'Confusion Matrices',
    'Raw Metrics',
])


# === Tab 1: Overview =====================================================
with tab_overview:
    # Key metric cards: best model for each criterion
    best_acc = summary.loc[summary['Accuracy'].idxmax()]
    best_auc = summary.loc[summary['ROC-AUC'].idxmax()]
    best_f1  = summary.loc[summary['F1-Score'].idxmax()]

    c1, c2, c3 = st.columns(3)
    c1.metric('Best Accuracy',  f"{best_acc['Accuracy']:.4f}",  best_acc['Model'])
    c2.metric('Best ROC-AUC',   f"{best_auc['ROC-AUC']:.4f}",  best_auc['Model'])
    c3.metric('Best F1-Score',  f"{best_f1['F1-Score']:.4f}",  best_f1['Model'])

    st.markdown('####  Model Comparison')

    metric_choice = st.multiselect(
        'Metrics to display',
        options=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        default=['Accuracy', 'F1-Score', 'ROC-AUC'],
    )

    fig_bar = go.Figure()
    for metric in metric_choice:
        if metric not in summary.columns:
            continue
        vals = summary[metric].tolist()
        fig_bar.add_trace(go.Bar(
            name=metric,
            x=summary['Model'].tolist(),
            y=vals,
            text=[f'{v:.4f}' for v in vals],
            textposition='outside',
            textfont=dict(size=13),
        ))

    fig_bar.update_layout(
        barmode='group',
        title=dict(text=f'Performance Comparison - {label}',
                   font=dict(size=15, color='#cdd9e5', family='Times New Roman')),
        xaxis_title='Classifier',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.15], color='#a0b8cc', tickfont=dict(family='Times New Roman', size=13)),
        xaxis=dict(color='#a0b8cc', tickfont=dict(family='Times New Roman', size=13)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(family='Times New Roman', size=13), bgcolor='rgba(0,0,0,0)'),
        font=dict(family='Times New Roman', color='#cdd9e5'),
        height=420,
        plot_bgcolor='#0e2040',
        paper_bgcolor='#0a1628',
        margin=dict(t=60, b=40),
    )
    fig_bar.update_xaxes(showgrid=False, linecolor='#1e3a5f')
    fig_bar.update_yaxes(showgrid=True, gridcolor='#1e3a5f')
    st.plotly_chart(fig_bar, use_container_width=True)


# === Tab 2: ROC-AUC Curves ===============================================
with tab_roc:
    if not has_curves:
        st.info(
            'ROC curves are not available for this pre-loaded result. '
            'They are generated automatically when you run a new dataset '
            'using the sidebar upload.'
        )
    else:
        fig_roc = go.Figure()

        # Random baseline
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='grey', width=1.5),
            name='Random classifier (AUC = 0.500)',
            hoverinfo='skip',
        ))

        for m in models:
            if m['fpr'] is None:
                continue
            fig_roc.add_trace(go.Scatter(
                x=m['fpr'].tolist(),
                y=m['tpr'].tolist(),
                mode='lines',
                name=f"{m['name']} (AUC = {m['auc']:.3f})",
                line=dict(color=model_color[m['name']], width=2.2),
                hovertemplate=(
                    f"<b>{m['name']}</b><br>"
                    'FPR: %{x:.3f}<br>'
                    'TPR: %{y:.3f}<extra></extra>'
                ),
            ))

        fig_roc.update_layout(
            title=dict(text=f'ROC-AUC Curves - {label}',
                       font=dict(size=15, color='#cdd9e5', family='Times New Roman')),
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1], showgrid=True, gridcolor='#1e3a5f',
                       color='#a0b8cc', tickfont=dict(family='Times New Roman', size=13)),
            yaxis=dict(range=[0, 1.02], showgrid=True, gridcolor='#1e3a5f',
                       color='#a0b8cc', tickfont=dict(family='Times New Roman', size=13)),
            legend=dict(x=0.62, y=0.06, bgcolor='rgba(10,22,40,0.85)',
                        bordercolor='#1e3a5f', borderwidth=1,
                        font=dict(family='Times New Roman', size=11, color='#cdd9e5')),
            font=dict(family='Times New Roman', color='#cdd9e5'),
            height=500,
            plot_bgcolor='#0e2040',
            paper_bgcolor='#0a1628',
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption('Hover over any curve to read exact FPR / TPR values. '
                   'Click legend items to show/hide individual models.')


# === Tab 3: Precision-Recall Curves ======================================
with tab_pr:
    if not has_curves:
        st.info(
            'Precision-Recall curves are not available for this pre-loaded result. '
            'They are generated automatically when you run a new dataset '
            'using the sidebar upload.'
        )
    else:
        fig_pr = go.Figure()

        for m in models:
            if m['precision_curve'] is None:
                continue
            fig_pr.add_trace(go.Scatter(
                x=m['recall_curve'].tolist(),
                y=m['precision_curve'].tolist(),
                mode='lines',
                name=f"{m['name']} (AP = {m['avg_precision']:.3f})",
                line=dict(color=model_color[m['name']], width=2.2),
                hovertemplate=(
                    f"<b>{m['name']}</b><br>"
                    'Recall: %{x:.3f}<br>'
                    'Precision: %{y:.3f}<extra></extra>'
                ),
            ))

        fig_pr.update_layout(
            title=dict(text=f'Precision-Recall Curves - {label}',
                       font=dict(size=15, color='#cdd9e5', family='Times New Roman')),
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1], showgrid=True, gridcolor='#1e3a5f',
                       color='#a0b8cc', tickfont=dict(family='Times New Roman', size=13)),
            yaxis=dict(range=[0, 1.02], showgrid=True, gridcolor='#1e3a5f',
                       color='#a0b8cc', tickfont=dict(family='Times New Roman', size=13)),
            legend=dict(x=0.01, y=0.15, bgcolor='rgba(10,22,40,0.85)',
                        bordercolor='#1e3a5f', borderwidth=1,
                        font=dict(family='Times New Roman', size=11, color='#cdd9e5')),
            font=dict(family='Times New Roman', color='#cdd9e5'),
            height=500,
            plot_bgcolor='#0e2040',
            paper_bgcolor='#0a1628',
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        st.caption('Average Precision (AP) is the area under the PR curve — '
                   'higher is better, especially on imbalanced datasets.')


# === Tab 4: Confusion Matrices ==========================================
with tab_cm:
    if not has_cm:
        st.info(
            'Confusion matrices are not available for this pre-loaded result. '
            'They are saved automatically when you run a new dataset '
            'using the sidebar upload.'
        )
    else:
        model_names_with_cm = [m['name'] for m in models if m['confusion_matrix'] is not None]
        selected_model = st.selectbox('Select model', options=model_names_with_cm)

        m_data = next(m for m in models if m['name'] == selected_model)
        cm     = m_data['confusion_matrix']

        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])
        acc    = (tp + tn) / (tp + tn + fp + fn)
        sens   = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec   = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0

        col_cm, col_stats = st.columns([1, 1])

        with col_cm:
            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                colorscale='Blues',
                showscale=True,
                text=[[str(tn), str(fp)], [str(fn), str(tp)]],
                texttemplate='%{text}',
                textfont=dict(size=20, color='black'),
                hoverinfo='skip',
            ))
            fig_cm.update_layout(
                title=dict(text=f'Confusion Matrix - {selected_model}',
                           font=dict(size=15, color='#cdd9e5', family='Times New Roman')),
                xaxis=dict(side='top', color='#a0b8cc',
                           tickfont=dict(family='Times New Roman', size=13)),
                yaxis=dict(color='#a0b8cc',
                           tickfont=dict(family='Times New Roman', size=13)),
                font=dict(family='Times New Roman', color='#cdd9e5'),
                height=380,
                margin=dict(t=80, b=30, l=30, r=30),
                plot_bgcolor='#0e2040',
                paper_bgcolor='#0a1628',
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_stats:
            st.markdown(f'### {selected_model} — Derived Metrics')
            st.metric('Accuracy',    f'{acc:.4f}')
            st.metric('Sensitivity (Recall)', f'{sens:.4f}')
            st.metric('Specificity', f'{spec:.4f}')
            st.metric('Precision',   f'{prec:.4f}')
            st.markdown('---')
            st.markdown(f'**True Positives:** {tp}')
            st.markdown(f'**True Negatives:** {tn}')
            st.markdown(f'**False Positives:** {fp}')
            st.markdown(f'**False Negatives:** {fn}')


# === Tab 5: Raw Metrics =================================================
with tab_raw:
    st.markdown('#### All Models — Full Metrics Table')

    def _color_cell(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return ''
        if val >= 0.87:
            return 'background-color: #C8E6C9; color: #1b5e20'
        elif val >= 0.75:
            return 'background-color: #FFF9C4; color: #795548'
        else:
            return 'background-color: #FFCDD2; color: #b71c1c'

    numeric_cols = [c for c in summary.columns if c != 'Model']
    styled = (
        summary.style
        .map(_color_cell, subset=numeric_cols)
        .format({c: '{:.4f}' for c in numeric_cols})
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([{
            'selector': 'th',
            'props': [
                ('background-color', '#3F51B5'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
            ]
        }])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.caption(
        'Color key: green >= 0.87  |  yellow >= 0.75  |  red < 0.75. '
        'ROC-AUC and Avg Prec. may be NaN for pre-loaded results without '
        'saved curve data.'
    )

    st.download_button(
        label='Download as CSV',
        data=summary.to_csv(index=False).encode(),
        file_name=f'{label.replace(" ", "_")}_metrics.csv',
        mime='text/csv',
    )
