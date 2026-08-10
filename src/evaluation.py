"""
evaluation.py
=============
Output Layer — Evaluation, Visualisation & Results Persistence

Provides:
  - plot_confusion_matrix()     binary and multi-class CM figures
  - plot_roc_curve()            multi-class ROC curves
  - plot_feature_importance()   XGBoost feature-importance bar chart
  - save_results()              write metrics.csv, model_comparison.csv, per-class reports
  - save_preprocessing_artifacts()  save MI selector, Scaler, PCA, LabelEncoder
  - print_final_summary()       formatted console table
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')           # headless / non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, classification_report,
    roc_curve, auc, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    normal_class_idx: int = 0,
    save_dir: str = "assets",
    prefix: str = "",
) -> None:
    """
    Save both a binary and a multi-class confusion matrix to *save_dir*.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Binary CM ---
    y_true_bin = np.where(y_true == normal_class_idx, 0, 1)
    y_pred_bin = np.where(y_pred == normal_class_idx, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
    ConfusionMatrixDisplay(cm_bin, display_labels=['Normal', 'Attack']).plot(
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title(f"{prefix}Binary Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}binary_cm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Multi-class CM ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_multi = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm_multi, display_labels=class_names).plot(
        ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45
    )
    ax.set_title(f"{prefix}Multi-Class Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}multiclass_cm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# ROC Curve (multi-class)
# ---------------------------------------------------------------------------

def plot_roc_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    scaler=None,
    pca=None,
    title: str = "ROC Curve",
    save_dir: str = "assets",
    save_path: str = None,
    prefix: str = "",
) -> None:
    """
    Plot multi-class ROC curves (one-vs-rest).

    If *scaler* and *pca* are provided, X_test is transformed through
    them before calling predict_proba (needed when the model was trained
    on scaled/PCA'd data but X_test is still in MI-selected space).
    """
    os.makedirs(save_dir, exist_ok=True)

    X = X_test
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)

    classes = np.unique(y_test)
    n_classes = len(classes)
    y_bin = label_binarize(y_test, classes=classes)

    try:
        y_score = model.predict_proba(X)
    except Exception:
        print("  [plot_roc_curve] Model has no predict_proba - skipping.")
        return

    # Compute per-class ROC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Weighted average AUC
    weighted_auc = np.mean([roc_auc[i] for i in range(n_classes)])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        ax.plot(fpr[i], tpr[i], color=color, lw=1.5,
                label=f'{cls_name} (AUC={roc_auc[i]:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"{title}  [Weighted AUC={weighted_auc:.3f}]")
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        path = os.path.join(save_dir, f"{prefix}roc_curve.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature Importance (XGBoost)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model,
    n_components: int = 10,
    save_dir: str = "assets",
) -> None:
    """Plot XGBoost feature importance scores for the PCA components."""
    os.makedirs(save_dir, exist_ok=True)

    try:
        importances = model.feature_importances_
    except AttributeError:
        print("  [feature_importance] Model has no feature_importances_ - skipping.")
        return

    labels = [f"PC{i+1}" for i in range(len(importances))]
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(importances)),
           importances[indices],
           color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([labels[i] for i in indices], rotation=45, ha='right')
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Feature Importance Score")
    ax.set_title("XGBoost Feature Importance (PCA Components)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    plt.tight_layout()

    path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Preprocessing Artifact Persistence
# ---------------------------------------------------------------------------

def save_preprocessing_artifacts(
    selector=None,
    scaler=None,
    pca=None,
    le=None,
    categorical_encoder=None,
    save_dir: str = "artifacts",
) -> None:
    """
    Save fitted preprocessing artifacts for inference reproducibility.

    Saves: MI selector, StandardScaler, PCA, LabelEncoder.
    """
    os.makedirs(save_dir, exist_ok=True)

    if selector is not None:
        path = os.path.join(save_dir, "mi_selector.joblib")
        joblib.dump(selector, path)

    if scaler is not None:
        path = os.path.join(save_dir, "scaler.joblib")
        joblib.dump(scaler, path)

    if pca is not None:
        path = os.path.join(save_dir, "pca.joblib")
        joblib.dump(pca, path)

    if le is not None:
        path = os.path.join(save_dir, "label_encoder.joblib")
        joblib.dump(le, path)

    if categorical_encoder is not None:
        path = os.path.join(save_dir, "categorical_encoder.joblib")
        joblib.dump(categorical_encoder, path)

    print(f"  Preprocessing artifacts saved -> {save_dir}/")


# ---------------------------------------------------------------------------
# Results Persistence
# ---------------------------------------------------------------------------

def save_results(
    all_test_results: list,
    cv_results: dict,
    y_true: np.ndarray = None,
    y_pred_dict: dict = None,
    class_names: list = None,
    results_dir: str = "results",
) -> None:
    """
    Write model_comparison.csv (blind-test) and metrics.csv (CV).

    Parameters
    ----------
    all_test_results : list of single-row DataFrames from each trainer
    cv_results       : dict  {'HGB': df, 'XGBoost': df, 'LogReg': df}
    y_true           : ground truth labels for per-class reports (optional)
    y_pred_dict      : dict  {'HGB': preds, 'XGBoost': preds, 'LogReg': preds}
    class_names      : list of class names for per-class reports
    results_dir      : output directory
    """
    os.makedirs(results_dir, exist_ok=True)

    # model_comparison.csv — one row per model, blind test metrics
    comparison_df = pd.concat(all_test_results, ignore_index=True)
    path = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(path, index=False, float_format='%.4f')
    print(f"  Saved: {path}")

    # metrics.csv — all CV fold metrics concatenated
    rows = []
    for model_name, df in cv_results.items():
        df_copy = df.copy()
        df_copy.insert(0, 'Model', model_name)
        rows.append(df_copy)

    if rows:
        metrics_df = pd.concat(rows, ignore_index=True)
        path = os.path.join(results_dir, "metrics.csv")
        metrics_df.to_csv(path, index=False, float_format='%.4f')
        print(f"  Saved: {path}")

    # Per-class classification reports
    if y_true is not None and y_pred_dict and class_names:
        for model_name, y_pred in y_pred_dict.items():
            report = classification_report(
                y_true, y_pred, target_names=class_names,
                output_dict=True, zero_division=0,
            )
            report_df = pd.DataFrame(report).transpose()
            report_path = os.path.join(
                results_dir, f"{model_name.lower()}_per_class_report.csv"
            )
            report_df.to_csv(report_path, float_format='%.4f')
            print(f"  Saved: {report_path}")


def save_ablation_tables(
    summary_rows: list,
    cv_rows: list,
    results_dir: str,
    modes: list,
) -> None:
    """
    Persist ablation comparison tables for paper-ready export.

    Writes into *results_dir*:
      ablation_test_metrics.csv   — long form: one row per (Model, Preprocessing)
      ablation_cv_metrics.csv     — long form: per-experiment mean CV metrics
      ablation_<metric>.csv       — pivot: Model rows × experiment columns

    Parameters
    ----------
    summary_rows : list of dicts with keys Model, Preprocessing + test metrics
    cv_rows      : list of dicts with keys Model, Preprocessing + cv_* means
    results_dir  : output directory
    modes        : ordered experiment labels (columns of the pivots)
    """
    os.makedirs(results_dir, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    cv_df = pd.DataFrame(cv_rows)

    test_summary_path = os.path.join(results_dir, 'ablation_test_metrics.csv')
    summary_df.to_csv(test_summary_path, index=False, float_format='%.4f')
    print(f"  Saved: {test_summary_path}")

    cv_summary_path = os.path.join(results_dir, 'ablation_cv_metrics.csv')
    cv_df.to_csv(cv_summary_path, index=False, float_format='%.4f')
    print(f"  Saved: {cv_summary_path}")

    metadata_cols = {'Model', 'Experiment', 'Preprocessing', 'MI', 'PCA', 'KMeansSMOTE'}
    metric_cols = [c for c in summary_df.columns if c not in metadata_cols]
    for metric in metric_cols:
        try:
            pivot = summary_df.pivot(index='Model', columns='Preprocessing', values=metric)
        except Exception:
            continue
        pivot = pivot.reindex(columns=modes)
        metric_path = os.path.join(results_dir, f'ablation_{metric}.csv')
        pivot.to_csv(metric_path, float_format='%.4f')
        print(f"  Saved: {metric_path}")

    cv_metric_cols = [c for c in cv_df.columns if c.startswith('cv_')]
    for metric in cv_metric_cols:
        try:
            pivot = cv_df.pivot(index='Model', columns='Preprocessing', values=metric)
        except Exception:
            continue
        pivot = pivot.reindex(columns=modes)
        metric_path = os.path.join(results_dir, f'ablation_{metric}.csv')
        pivot.to_csv(metric_path, float_format='%.4f')
        print(f"  Saved: {metric_path}")


def build_model_ablation_rows(
    model_name: str,
    experiments_root: str = "results",
) -> tuple:
    """
    Aggregate per-experiment results for one model into table rows.

    Reads  results/<model>/<experiment>/test_metrics.json  and
           results/<model>/<experiment>/cv_metrics.csv  for each of the
    seven ablation presets (in canonical order) and returns
    (summary_rows, cv_rows) ready for save_ablation_tables.
    """
    from src.experiment_config import ABLATION_ORDER, ABLATION_DISPLAY_NAMES

    from src.experiment_config import ABLATION_PRESETS

    records = []
    for exp in ABLATION_ORDER:
        exp_dir = os.path.join(experiments_root, model_name, exp)
        config_path = os.path.join(exp_dir, "experiment_config.json")
        tm_path = os.path.join(exp_dir, "test_metrics.json")
        cv_path = os.path.join(exp_dir, "cv_metrics.csv")
        model_path = os.path.join(exp_dir, f"{model_name.lower()}_model.joblib")
        missing = [
            name for name, path in (
                ('experiment_config.json', config_path),
                ('test_metrics.json', tm_path),
                ('cv_metrics.csv', cv_path),
                (os.path.basename(model_path), model_path),
            ) if not os.path.isfile(path)
        ]
        if missing:
            raise ValueError(
                f"Cannot aggregate {model_name}: experiment '{exp}' is missing "
                f"{', '.join(missing)} under {exp_dir}."
            )

        with open(config_path, 'r') as f:
            config = json.load(f)
        expected = ABLATION_PRESETS[exp]
        if config.get('experiment') != exp or config.get('experiment_name') != exp:
            raise ValueError(
                f"Cannot aggregate {model_name}: configuration in '{exp}' does not "
                "identify the matching experiment."
            )
        for key, value in expected.items():
            if config.get(key) is not value:
                raise ValueError(
                    f"Cannot aggregate {model_name}: configuration mismatch for '{exp}': "
                    f"{key} must be {value}."
                )

        required_metadata = ('seed', 'feature_selection_k', 'pca_variance',
                             'balancer', 'cv_folds')
        absent = [key for key in required_metadata if key not in config]
        if absent:
            raise ValueError(
                f"Cannot aggregate {model_name}: configuration for '{exp}' is missing "
                f"reproducibility metadata: {', '.join(absent)}."
            )
        if config['balancer'] != 'kmeans':
            raise ValueError(
                f"Cannot aggregate {model_name}: configuration for '{exp}' must use KMeansSMOTE."
            )

        records.append((exp, config, tm_path, cv_path))

    reference = records[0][1]
    reproducibility_keys = ('seed', 'feature_selection_k', 'pca_variance',
                            'balancer', 'cv_folds', 'balancer_k_neighbors',
                            'balancer_rus_cap')
    for exp, config, _, _ in records[1:]:
        inconsistent = [
            key for key in reproducibility_keys
            if config.get(key) != reference.get(key)
        ]
        if inconsistent:
            raise ValueError(
                f"Cannot aggregate {model_name}: reproducibility metadata differs for "
                f"'{exp}': {', '.join(inconsistent)}."
            )

    summary_rows, cv_rows = [], []
    for exp, _, tm_path, cv_path in records:
        with open(tm_path, 'r') as f:
            tm = json.load(f)

        cv_df = pd.read_csv(cv_path)
        cv_means = {
            f"cv_{c}": float(cv_df[c].mean())
            for c in ('accuracy', 'precision', 'recall', 'f1', 'auc')
            if c in cv_df.columns
        }

        summary_rows.append({
            "Model": model_name,
            "Experiment": ABLATION_DISPLAY_NAMES[exp],
            "Preprocessing": ABLATION_DISPLAY_NAMES[exp],
            "MI": "Yes" if ABLATION_PRESETS[exp]['use_mi'] else "No",
            "PCA": "Yes" if ABLATION_PRESETS[exp]['use_pca'] else "No",
            "KMeansSMOTE": "Yes" if ABLATION_PRESETS[exp]['use_balancing'] else "No",
            **tm,
        })
        cv_rows.append({
            "Model": model_name,
            "Experiment": ABLATION_DISPLAY_NAMES[exp],
            "Preprocessing": ABLATION_DISPLAY_NAMES[exp],
            "MI": "Yes" if ABLATION_PRESETS[exp]['use_mi'] else "No",
            "PCA": "Yes" if ABLATION_PRESETS[exp]['use_pca'] else "No",
            "KMeansSMOTE": "Yes" if ABLATION_PRESETS[exp]['use_balancing'] else "No",
            **cv_means,
        })

    return summary_rows, cv_rows


def save_model_ablation_tables(
    model_name: str,
    results_root: str = "results",
) -> None:
    """
    Build + save the ablation comparison tables for a single model.

    The ablation_test_metrics.csv / ablation_cv_metrics.csv contain exactly
    seven rows (Raw, MI, MI+KMeansSMOTE, PCA, PCA+KMeansSMOTE, MI+PCA,
    MI+PCA+KMeansSMOTE) once all seven experiments have completed.
    """
    from src.experiment_config import ABLATION_ORDER, ABLATION_DISPLAY_NAMES

    summary_rows, cv_rows = build_model_ablation_rows(model_name, results_root)
    modes = [ABLATION_DISPLAY_NAMES[e] for e in ABLATION_ORDER]
    save_ablation_tables(
        summary_rows, cv_rows,
        results_dir=os.path.join(results_root, model_name),
        modes=modes,
    )


def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    normal_class_idx: int = 0,
) -> dict:
    """
    Compute binary + multiclass metrics for the final test evaluation.

    Returns dict with:
        accuracy, precision, recall, f1, auc           (existing keys)
        multi_acc, macro_f1, weighted_f1              (multiclass)
        binary_acc, binary_f1, binary_auc             (Normal vs Attack)
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'multi_acc': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    y_true_bin = np.where(y_true == normal_class_idx, 0, 1)
    y_pred_bin = np.where(y_pred == normal_class_idx, 0, 1)
    metrics['binary_acc'] = accuracy_score(y_true_bin, y_pred_bin)
    metrics['binary_f1'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='weighted'
            )
        except Exception:
            metrics['auc'] = 0.0
        try:
            p_attack = 1.0 - np.asarray(y_proba)[:, normal_class_idx]
            metrics['binary_auc'] = roc_auc_score(y_true_bin, p_attack)
        except Exception:
            metrics['binary_auc'] = 0.0
    else:
        metrics['auc'] = 0.0
        metrics['binary_auc'] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Console Summary
# ---------------------------------------------------------------------------

def print_final_summary(all_test_results: list) -> None:
    """Print a formatted comparison table to stdout."""
    df = pd.concat(all_test_results, ignore_index=True)
    print("\n" + "=" * 72)
    print("  FINAL MODEL COMPARISON - Blind 20 % Test Set")
    print("=" * 72)
    float_cols = [c for c in df.columns if c != 'Model']
    fmt = {c: '{:.4f}'.format for c in float_cols}
    print(df.to_string(index=False, formatters=fmt))
    print("=" * 72)
