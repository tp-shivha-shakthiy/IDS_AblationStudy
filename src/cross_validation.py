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
from src.preprocessing import fit_categorical_encoder, transform_features


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
    rus_cap: int = 0,
    fold_cache: dict = None,
    use_mi: bool = True,
    use_pca: bool = True,
    use_balancing: bool = True,
):
    """
    Run Stratified K-Fold CV with per-fold leakage-free preprocessing.

    For each fold:
      1. Fit MI selector on fold train → transform fold train + val (if use_mi)
      2. Fit StandardScaler on fold train → transform fold train + val (always)
      3. Fit PCA on scaled fold train → transform fold train + val (if use_pca)
      4. K-means SMOTE on fold train only (if use_balancing)
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
    rus_cap       : int           if >0, cap each class to this many samples
                                  before oversampling (speed/RAM saving)
    fold_cache    : dict          optional cache of preprocessed folds shared
                                  across models with identical preprocessing
    use_mi        : bool          apply per-fold MI feature selection
    use_pca       : bool          apply per-fold PCA
    use_balancing : bool          apply per-fold K-means SMOTE / SMOTE

    Returns
    -------
    metrics       : dict  {metric_name: [fold_values]}
    selector      : fitted SelectKBest  (last fold, None when use_mi=False)
    scaler        : fitted StandardScaler (last fold)
    pca           : fitted PCA            (last fold, None when use_pca=False)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    metrics = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'auc': [],
    }

    selector, scaler, pca = None, None, None

    cache_key = (n_splits, mi_k, pca_variance, k_neighbors, n_clusters,
                 random_state, strategy, rus_cap, use_mi, use_pca,
                 use_balancing)

    if fold_cache is not None and cache_key in fold_cache:
        folds = fold_cache[cache_key]
        print(f"    [run_cv] Reusing {len(folds)} preprocessed folds from cache")
    else:
        # --- Preprocessing: MI -> Scaler -> PCA -> balance (per fold, once) ---
        folds = []
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n    Fold {fold+1}/{n_splits} (preprocessing)")
            t0 = time.time()

            if hasattr(X_train, 'iloc'):
                X_tr_raw, X_val_raw = X_train.iloc[trn_idx], X_train.iloc[val_idx]
                categorical_encoder = fit_categorical_encoder(X_tr_raw)
                X_tr = transform_features(X_tr_raw, categorical_encoder)
                X_val = transform_features(X_val_raw, categorical_encoder)
            else:
                X_tr, X_val = X_train[trn_idx], X_train[val_idx]
            y_tr, y_val = y_train[trn_idx], y_train[val_idx]

            # --- 1. MI Feature Selection fitted on fold train only ---
            if use_mi:
                fold_selector = fit_mi_selector(X_tr, y_tr, k=mi_k,
                                                random_state=random_state)
                X_tr_mi = fold_selector.transform(X_tr)
                X_val_mi = fold_selector.transform(X_val)
            else:
                fold_selector = None
                X_tr_mi, X_val_mi = X_tr, X_val

            # --- 2. StandardScaler fitted on fold train only ---
            fold_scaler = StandardScaler()
            X_tr_s = fold_scaler.fit_transform(X_tr_mi)
            X_val_s = fold_scaler.transform(X_val_mi)

            # --- 3. PCA fitted on scaled fold train only ---
            if use_pca:
                fold_pca = PCA(n_components=pca_variance, random_state=random_state)
                X_tr_p = fold_pca.fit_transform(X_tr_s)
                X_val_p = fold_pca.transform(X_val_s)
            else:
                fold_pca = None
                X_tr_p, X_val_p = X_tr_s, X_val_s

            transform_desc = " | ".join(
                part for part in (
                    f"MI k={mi_k}" if use_mi else None,
                    f"PCA components: {X_tr_p.shape[1]}" if use_pca else None,
                ) if part
            )
            print(f"      {transform_desc or 'Raw features: ' + str(X_tr_p.shape[1])}")

            # --- 4. Balancing on fold train only ---
            if use_balancing:
                X_tr_b, y_tr_b = balance_training_fold(
                    X_tr_p, y_tr,
                    strategy=strategy,
                    k_neighbors=k_neighbors,
                    n_clusters=n_clusters,
                    random_state=random_state,
                    rus_cap=rus_cap,
                )
                elapsed = time.time() - t0
                print(f"      Balanced: {X_tr_b.shape[0]:,} samples  ({elapsed:.1f}s)")
            else:
                X_tr_b, y_tr_b = X_tr_p, y_tr

            folds.append(dict(
                X_tr_b=X_tr_b, y_tr_b=y_tr_b,
                X_val_p=X_val_p, y_val=y_val,
                selector=fold_selector, scaler=fold_scaler, pca=fold_pca,
            ))

            del X_tr, X_val, X_tr_mi, X_val_mi, X_tr_s, X_val_s, X_tr_p
            gc.collect()

        if fold_cache is not None:
            fold_cache[cache_key] = folds

    # --- Training: fit model + evaluate on each preprocessed fold ---
    for fold, fold_data in enumerate(folds):
        print(f"\n    Fold {fold+1}/{n_splits} (training)")
        t0 = time.time()

        model = model_class(**model_params)
        fold_metrics = train_and_score_fold(
            model, fold_data['X_tr_b'], fold_data['y_tr_b'],
            fold_data['X_val_p'], fold_data['y_val'],
            use_sample_weight=use_sample_weight,
        )

        for k in metrics:
            metrics[k].append(fold_metrics[k])

        elapsed = time.time() - t0
        print(f"      Acc={fold_metrics['accuracy']:.4f}  "
              f"F1={fold_metrics['f1']:.4f}  "
              f"AUC={fold_metrics['auc']:.4f}  ({elapsed:.1f}s)")

        # keep last fold's transformers for final retraining reference
        selector = fold_data['selector']
        scaler = fold_data['scaler']
        pca = fold_data['pca']

        del model; gc.collect()

    return metrics, selector, scaler, pca
