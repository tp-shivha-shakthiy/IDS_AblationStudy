"""
balancing.py
============
Class Balancing — Single Source of Truth

Provides two functions:

  balance_training_fold(X_train, y_train, ...)
      Per-fold balancing for cross-validation.
      Receives ONLY the fold's training data.

  balance_full_train(X_train, y_train, ...)
      Final retrain balancing on the full training set.
      Must NEVER receive validation or test data.

Both support:
  strategy="kmeans"  (default) — KMeansSMOTE (imblearn) cluster-aware oversampling
  strategy="smote"              — regular SMOTE
"""

import time
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import MiniBatchKMeans


def _kms_fit_resample(X, y, adj_k, random_state, stage="balancing"):
    """
    Run KMeansSMOTE with visible progress logging.

    Prints the input class distribution before fitting (flushed immediately so
    background runs show it is alive), then the elapsed time and resulting class
    distribution after fitting.
    """
    counts_before = dict(sorted(Counter(y).items()))
    total_before = len(y)
    label = f"KMeansSMOTE ({stage})"
    print(
        f"    {label}: {total_before:,} samples | before "
        f"-> {counts_before}", flush=True,
    )
    t0 = time.perf_counter()
    kms = KMeansSMOTE(
        cluster_balance_threshold=0.0,
        k_neighbors=adj_k,
        kmeans_estimator=MiniBatchKMeans(n_init='auto', random_state=random_state),
        random_state=random_state,
        n_jobs=1,
    )
    X_res, y_res = kms.fit_resample(X, y)
    dt = time.perf_counter() - t0
    counts_after = dict(sorted(Counter(y_res).items()))
    print(
        f"    {label}: done in {dt:.1f}s | after -> {counts_after}",
        flush=True,
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Per-fold balancing (used inside cross_validation.run_cv)
# ---------------------------------------------------------------------------

def balance_training_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "kmeans",
    k_neighbors: int = 3,
    n_clusters: int = 20,
    random_state: int = 42,
    rus_cap: int = 0,
    stage: str = "fold",
) -> tuple:
    """
    Balance a single training fold.  Must receive ONLY training data.

    Parameters
    ----------
    X_train      : array (n_fold, F)  fold training features
    y_train      : array (n_fold,)    fold training labels
    strategy     : 'kmeans' | 'smote'
    k_neighbors  : int   SMOTE neighbour count
    n_clusters   : int   K-means cluster count (ignored when strategy='smote')
    random_state : int
    rus_cap      : int   if >0, cap each class to this many samples before
                         oversampling (matches Tier 2 DL pipeline behaviour)

    Returns
    -------
    X_balanced, y_balanced
    """
    X_use, y_use = X_train, y_train

    if rus_cap > 0:
        class_counts = Counter(y_use)
        under_strategy = {c: min(cnt, rus_cap) for c, cnt in class_counts.items()}
        rus = RandomUnderSampler(
            sampling_strategy=under_strategy, random_state=random_state,
        )
        X_use, y_use = rus.fit_resample(X_use, y_use)

    if strategy == "kmeans":
        minority_count = min(Counter(y_use).values())
        adj_k = min(k_neighbors, minority_count - 1)
        adj_k = max(adj_k, 1)
        X_res, y_res = _kms_fit_resample(
            X_use, y_use, adj_k, random_state, stage=stage,
        )
    else:
        minority_count = min(Counter(y_use).values())
        adj_k = min(k_neighbors, minority_count - 1)
        adj_k = max(adj_k, 1)
        sm = SMOTE(random_state=random_state, k_neighbors=adj_k)
        X_res, y_res = sm.fit_resample(X_use, y_use)

    return X_res, y_res


# ---------------------------------------------------------------------------
# Full-training-set balancing (used for final model retrain)
# ---------------------------------------------------------------------------

def balance_full_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "kmeans",
    k_neighbors: int = 3,
    n_clusters: int = 20,
    random_state: int = 42,
    rus_cap: int = 0,
) -> tuple:
    """
    Balance the full training set for final model retraining.

    Must NEVER be called with validation or test data.

    Parameters
    ----------
    X_train      : array (N, F)  full training features (already MI/scaled/PCA'd)
    y_train      : array (N,)    full training labels
    strategy     : 'kmeans' | 'smote'
    k_neighbors  : int   SMOTE neighbour count
    n_clusters   : int   K-means cluster count (ignored when strategy='smote')
    random_state : int
    rus_cap      : int   if >0, cap each class before oversampling

    Returns
    -------
    X_balanced, y_balanced
    """
    X_balanced, y_balanced = balance_training_fold(
        X_train, y_train,
        strategy=strategy,
        k_neighbors=k_neighbors,
        n_clusters=n_clusters,
        random_state=random_state,
        rus_cap=rus_cap,
        stage="final retrain",
    )
    print(f"    Balanced training: {X_train.shape[0]:,} -> "
          f"{X_balanced.shape[0]:,} samples")
    return X_balanced, y_balanced
