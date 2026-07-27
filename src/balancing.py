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
  strategy="kmeans"  (default) — MiniBatchKMeans cluster pre-processing + SMOTE
  strategy="smote"              — regular SMOTE
"""

import numpy as np
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from sklearn.cluster import MiniBatchKMeans


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

    Returns
    -------
    X_balanced, y_balanced
    """
    from collections import Counter
    minority_count = min(Counter(y_train).values())
    if minority_count < 2:
        raise ValueError(
            "Balancing requires at least two samples in every class; "
            "reduce n_splits or remove singleton classes first."
        )
    actual_k = min(k_neighbors, minority_count - 1)
    if strategy == "kmeans":
        # KMeansSMOTE uses cluster membership to decide where to synthesize
        # samples.  The previous implementation fitted KMeans and discarded
        # its labels, making this option indistinguishable from plain SMOTE.
        sm = KMeansSMOTE(
            k_neighbors=max(actual_k, 1),
            kmeans_estimator=MiniBatchKMeans(
                n_clusters=min(n_clusters, len(X_train)), batch_size=2048,
                random_state=random_state, n_init='auto',
            ),
            cluster_balance_threshold=0.0,
            random_state=random_state,
            n_jobs=1,
        )
    elif strategy == "smote":
        sm = SMOTE(random_state=random_state, k_neighbors=max(actual_k, 1))
    else:
        raise ValueError("strategy must be 'kmeans' or 'smote'")
    X_res, y_res = sm.fit_resample(X_train, y_train)
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
    )
    print(f"    Balanced training: {X_train.shape[0]:,} -> "
          f"{X_balanced.shape[0]:,} samples")
    return X_balanced, y_balanced
