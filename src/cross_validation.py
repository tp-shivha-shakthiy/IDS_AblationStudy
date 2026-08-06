"""
cross_validation.py
===================
Phase 6 -- Cross-Validation Training Helpers

Two levels of helpers:

  train_and_score_fold(model, X_tr, y_tr, X_val, y_val)
      Train once on already-balanced fold data and return metrics.
      Low-level building block used by train_*.py.

  run_cv(X_train, y_train, model_class, model_params, ...)
      Full per-fold pipeline:
        MI fit on fold train → transform fold train + val
        StandardScaler fit on fold train → transform fold train + val
        PCA fit on fold train → transform fold train + val
        K-means SMOTE on fold train only
        train → eval

      Returns mean metrics plus fitted selector/scaler/pca for retraining.
"""

import numpy as np
import gc
import time
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score)

from src.balancing import balance_training_fold
from src.feature_selection import fit_mi_selector


# ===================================================================
# Low-level: train a single model on an already-balanced fold
# ===================================================================

def train_and_score_fold(model, X_tr, y_tr, X_val, y_val,
                         use_sample_weight: bool = False):
    """
    Fit *model* on (X_tr, y_tr) and score on (X_val, y_val).

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc (if available)
    """
    sample_weight = None
    if use_sample_weight:
        classes, counts = np.unique(y_tr, return_counts=True)
        n_samples = len(y_tr)
        n_classes = len(classes)
        sample_weight = n_samples / (n_classes * counts)
        sw_map = dict(zip(classes, sample_weight))
        sample_weight = np.array([sw_map[c] for c in y_tr])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sample_weight is not None:
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
        else:
            model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

    auc = 0.0
    try:
        auc = roc_auc_score(y_val, model.predict_proba(X_val),
                            multi_class='ovr', average='weighted')
    except Exception:
        pass

    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc}


# ===================================================================
# Full per-fold pipeline: MI → Scaler → PCA → K-means SMOTE → train
# ===================================================================

def run_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_class,
    model_params: dict,
    n_splits: int = 5,
    mi_k: int = 15,
    pca_variance: float = 0.95,
    k_neighbors: int = 3,
    random_state: int = 42,
    use_sample_weight: bool = False,
    strategy: str = "kmeans",
    n_clusters: int = 20,
):
    """
    Run Stratified K-Fold CV with per-fold leakage-free preprocessing.

    For each fold:
      1. Fit MI selector on fold train → transform fold train + val
      2. Fit StandardScaler on fold train → transform fold train + val
      3. Fit PCA on scaled fold train → transform fold train + val
      4. K-means SMOTE on fold train only
      5. Train model → evaluate on val

    Parameters
    ----------
    X_train       : array (N, F)  raw preprocessed features (pre-MI)
    y_train       : array (N,)    labels
    model_class   : class         sklearn-compatible estimator class
    model_params  : dict          kwargs for model_class(...)
    n_splits      : int
    mi_k          : int           top-k MI features to select per fold
    pca_variance  : float         cumulative variance to retain
    k_neighbors   : int           SMOTE neighbour count
    random_state  : int
    use_sample_weight : bool
    strategy      : 'kmeans' | 'smote'
    n_clusters    : int           K-means clusters for K-means SMOTE

    Returns
    -------
    metrics       : dict  {metric_name: [fold_values]}
    selector      : fitted SelectKBest  (last fold, for retrain reference)
    scaler        : fitted StandardScaler (last fold)
    pca           : fitted PCA            (last fold)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    metrics = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'auc': [],
    }

    selector, scaler, pca = None, None, None

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n    Fold {fold+1}/{n_splits}")
        t0 = time.time()

        X_tr, X_val = X_train[trn_idx], X_train[val_idx]
        y_tr, y_val = y_train[trn_idx], y_train[val_idx]

        # --- 1. MI Feature Selection fitted on fold train only ---
        fold_selector = fit_mi_selector(X_tr, y_tr, k=mi_k, random_state=random_state)
        X_tr_mi = fold_selector.transform(X_tr)
        X_val_mi = fold_selector.transform(X_val)

        # --- 2. StandardScaler fitted on fold train only ---
        fold_scaler = StandardScaler()
        X_tr_s = fold_scaler.fit_transform(X_tr_mi)
        X_val_s = fold_scaler.transform(X_val_mi)

        # --- 3. PCA fitted on scaled fold train only ---
        fold_pca = PCA(n_components=pca_variance, random_state=random_state)
        X_tr_p = fold_pca.fit_transform(X_tr_s)
        X_val_p = fold_pca.transform(X_val_s)

        print(f"      MI k={mi_k} | PCA components: {X_tr_p.shape[1]}")

        # --- 4. Balancing on fold train only ---
        X_tr_b, y_tr_b = balance_training_fold(
            X_tr_p, y_tr,
            strategy=strategy,
            k_neighbors=k_neighbors,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        print(f"      Balanced: {X_tr_b.shape[0]:,} samples")

        # --- 5. Train + eval ---
        model = model_class(**model_params)
        fold_metrics = train_and_score_fold(
            model, X_tr_b, y_tr_b, X_val_p, y_val,
            use_sample_weight=use_sample_weight,
        )

        for k in metrics:
            metrics[k].append(fold_metrics[k])

        elapsed = time.time() - t0
        print(f"      Acc={fold_metrics['accuracy']:.4f}  "
              f"F1={fold_metrics['f1']:.4f}  "
              f"AUC={fold_metrics['auc']:.4f}  ({elapsed:.1f}s)")

        # keep last fold's transformers for final retraining
        selector, scaler, pca = fold_selector, fold_scaler, fold_pca

        del X_tr, X_val, X_tr_mi, X_val_mi, X_tr_s, X_val_s
        del X_tr_p, X_val_p, X_tr_b, y_tr_b, model; gc.collect()

    return metrics, selector, scaler, pca
