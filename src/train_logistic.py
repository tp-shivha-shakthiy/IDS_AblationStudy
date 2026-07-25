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
import warnings
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score)

from src.cross_validation import run_cv
from src.balancing import balance_full_train
from src.evaluation import plot_confusion_matrix, plot_roc_curve


MODEL_NAME = "LogReg"


def _train_and_evaluate(X_tr, y_tr, X_val, y_val):
    """Train Logistic Regression on balanced data, return metrics dict."""
    model = LogisticRegression(
        multi_class='multinomial', solver='saga',
        max_iter=50, random_state=42, n_jobs=-1,
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
        multi_class='multinomial', solver='saga',
        max_iter=50, random_state=42, n_jobs=-1,
    )

    # --- Step 1: Per-fold CV (MI + Scaler + PCA + K-means SMOTE inside) ---
    print(f"\n  === Cross-Validation ({n_splits} folds) ===")
    cv_metrics, _, _, _ = run_cv(
        X_train, y_train,
        model_class=LogisticRegression,
        model_params=model_params,
        n_splits=n_splits,
        mi_k=mi_k,
        pca_variance=pca_variance,
        k_neighbors=k_neighbors,
        random_state=random_state,
        strategy="kmeans",
    )

    print(f"\n  CV Results ({MODEL_NAME}):")
    for k, v in cv_metrics.items():
        print(f"    {k:>10s}: {np.mean(v):.4f} (+/- {np.std(v):.4f})")

    # --- Step 2: Full train → MI → Scaler → PCA → K-means SMOTE → retrain ---
    print(f"\n  === Final Retrain on Full Training Set ===")

    selector = SelectKBest(score_func=mutual_info_classif, k=mi_k)
    selector.fit(X_train, y_train)
    X_train_mi = selector.transform(X_train)
    X_test_mi = selector.transform(X_test)
    print(f"    MI selected: {X_train_mi.shape[1]} features")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_mi)

    pca = PCA(n_components=pca_variance, random_state=random_state)
    X_train_p = pca.fit_transform(X_train_s)
    print(f"    PCA components: {X_train_p.shape[1]}")

    X_train_b, y_train_b = balance_full_train(
        X_train_p, y_train, strategy="kmeans",
        k_neighbors=k_neighbors, random_state=random_state,
    )

    X_test_p = pca.transform(scaler.transform(X_test_mi))

    model, test_metrics, y_test_pred = _train_and_evaluate(
        X_train_b, y_train_b, X_test_p, y_test,
    )

    print(f"\n  {MODEL_NAME} Test Metrics:")
    for k, v in test_metrics.items():
        print(f"    {k:>10s}: {v:.4f}")

    # --- Step 3: Save artifacts ---
    save_dir = os.path.join("models", "artifacts", MODEL_NAME)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{MODEL_NAME.lower()}_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n  Model saved → {model_path}")

    plot_confusion_matrix(
        y_test, y_test_pred, class_names,
        normal_class_idx=0, save_dir=save_dir,
        prefix=f"{MODEL_NAME.lower()}_",
    )
    plot_roc_curve(
        model, X_test_mi, y_test, class_names,
        scaler=scaler, pca=pca,
        title=f"{MODEL_NAME} ROC Curve (Test Set)",
        save_dir=save_dir, prefix=f"{MODEL_NAME.lower()}_",
    )

    results = {
        'model': model,
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics,
        'y_test_pred': y_test_pred,
        'selector': selector,
        'scaler': scaler,
        'pca': pca,
    }

    del X_train_mi, X_test_mi, X_train_s, X_train_p
    del X_train_b, y_train_b; gc.collect()
    return results
