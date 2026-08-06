"""
train_logistic.py
=================
Phase 10 -- Logistic Regression (multinomial / saga)

Pipeline (leakage-free):
  1. Per-fold CV: MI → Scaler → PCA → K-means SMOTE → train → eval
  2. After CV:  Full train → MI fit → Scaler fit → PCA fit → K-means SMOTE → retrain
  3. Single evaluation on locked test set
"""

import numpy as np
import gc
import os
import json
import warnings
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score)

from src.cross_validation import run_cv
from src.balancing import balance_full_train
from src.feature_pipeline import PreprocessingConfig, fit_transform_full
from src.evaluation import plot_confusion_matrix, plot_roc_curve


MODEL_NAME = "LogReg"


def _train_and_evaluate(X_tr, y_tr, X_val, y_val, random_state=42):
    """Train Logistic Regression on balanced data, return metrics dict."""
    model = LogisticRegression(
        solver='saga',
        max_iter=50, random_state=random_state, n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred,
                                     average='weighted', zero_division=0),
        'recall': recall_score(y_val, y_pred,
                               average='weighted', zero_division=0),
        'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0),
        'macro_f1': f1_score(y_val, y_pred, average='macro', zero_division=0),
    }
    try:
        metrics['auc'] = roc_auc_score(y_val, model.predict_proba(X_val),
                                       multi_class='ovr', average='weighted')
    except Exception:
        metrics['auc'] = 0.0
    return model, metrics, y_pred


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    n_splits: int = 5,
    mi_k: int = 15,
    pca_variance: float = 0.95,
    k_neighbors: int = 3,
    random_state: int = 42,
    balancer: str = "kmeans",
    make_plots: bool = True,
    normal_class_idx: int = 0,
    use_mi: bool = True,
    use_pca: bool = True,
    use_balancing: bool = True,
    experiment_name: str = None,
    preprocessing_mode: str = None,
    output_dir: str = None,
) -> dict:
    """
    Full LogReg pipeline: CV → final retrain → test eval.

    All transformers (MI, Scaler, PCA) are fit on training data only.

    Returns
    -------
    dict with keys: cv_metrics, test_metrics, y_test_pred, model
    """
    print(f"\n{'='*60}")
    print(f"  {MODEL_NAME}  Logistic Regression (saga / multinomial)")
    print(f"{'='*60}")

    model_params = dict(
        solver='saga',
        max_iter=50, random_state=random_state, n_jobs=-1,
    )
    preprocessing_config = PreprocessingConfig(
        use_mi=use_mi,
        use_pca=use_pca,
        use_balancing=use_balancing,
        mi_k=mi_k,
        pca_n_components=pca_variance,
        random_state=random_state,
    )
    run_name = experiment_name or preprocessing_mode or preprocessing_config.experiment_name
    save_dir = output_dir or os.path.join("results", MODEL_NAME, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- Step 1: Per-fold CV (MI + Scaler + PCA + K-means SMOTE inside) ---
    print(f"\n  === Cross-Validation ({n_splits} folds) ===")
    cv_metrics, _, _, _ = run_cv(
        X_train, y_train,
        model_class=LogisticRegression,
        model_params=model_params,
        n_splits=n_splits,
        mi_k=mi_k,
        pca_variance=pca_variance,
        use_mi=use_mi,
        use_pca=use_pca,
        use_balancing=use_balancing,
        k_neighbors=k_neighbors,
        random_state=random_state,
        strategy=balancer,
        normal_class_idx=normal_class_idx,
    )

    print(f"\n  CV Results ({MODEL_NAME}):")
    for k, v in cv_metrics.items():
        print(f"    {k:>10s}: {np.mean(v):.4f} (+/- {np.std(v):.4f})")

    cv_metrics_path = os.path.join(save_dir, "cv_metrics.csv")
    pd.DataFrame(cv_metrics).to_csv(cv_metrics_path, index=False, float_format='%.4f')
    print(f"  CV metrics saved -> {cv_metrics_path}")

    # --- Step 2: Full train → MI → Scaler → PCA → K-means SMOTE → retrain ---
    print(f"\n  === Final Retrain on Full Training Set ===")

    final_data = fit_transform_full(
        X_train, y_train, X_test, y_test,
        config=preprocessing_config,
        balance_fn=balance_full_train if use_balancing else None,
        balance_kwargs=dict(
            strategy=balancer,
            k_neighbors=k_neighbors,
            random_state=random_state,
        ) if use_balancing else None,
    )

    X_train_b = final_data['X_train']
    y_train_b = final_data['y_train']
    X_test_p = final_data['X_test']
    selector = final_data['selector']
    scaler = final_data['scaler']
    pca = final_data['pca']

    if selector is not None:
        print(f"    MI selected: {selector.k} features")
    if pca is not None:
        print(f"    PCA components: {X_train_b.shape[1]}")
    if selector is None and pca is None:
        print(f"    Raw features: {X_train_b.shape[1]}")

    model, test_metrics, y_test_pred = _train_and_evaluate(
        X_train_b, y_train_b, X_test_p, y_test, random_state=random_state,
    )
    y_test_binary = (y_test != normal_class_idx).astype(int)
    y_pred_binary = (y_test_pred != normal_class_idx).astype(int)
    test_metrics['binary_accuracy'] = accuracy_score(y_test_binary, y_pred_binary)
    test_metrics['binary_f1'] = f1_score(y_test_binary, y_pred_binary, zero_division=0)

    print(f"\n  {MODEL_NAME} Test Metrics:")
    for k, v in test_metrics.items():
        print(f"    {k:>10s}: {v:.4f}")

    # --- Step 3: Save artifacts ---
    model_path = os.path.join(save_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"\n  Model saved -> {model_path}")

    test_metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  Test metrics saved -> {test_metrics_path}")

    if make_plots:
        plot_confusion_matrix(
            y_test, y_test_pred, class_names,
            normal_class_idx=normal_class_idx, save_dir=save_dir,
            prefix="",
        )
        plot_roc_curve(
            model, X_test_p, y_test, class_names,
            scaler=None, pca=None,
            title=f"{MODEL_NAME} ROC Curve (Test Set)",
            save_dir=save_dir, prefix="",
        )

    results = {
        'model': model,
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics,
        'y_test_pred': y_test_pred,
        'selector': selector,
        'scaler': scaler,
        'pca': pca,
        'preprocessing_mode': preprocessing_mode or preprocessing_config.mode_name,
        'experiment_name': run_name,
        'save_dir': save_dir,
    }

    del X_train_b, y_train_b; gc.collect()
    return results
