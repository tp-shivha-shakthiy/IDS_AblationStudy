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


def _kmeans_smote_sparsity_failure_reason(error: Exception):
    """Classify only KMeansSMOTE's known cluster-sparsity failure modes.

    In imbalanced-learn 0.14.x, KMeansSMOTE computes per-cluster sampling
    weights as ``cluster_sparsities / cluster_sparsities.sum()``.  A valid
    cluster made entirely of duplicate minority samples has sparsity zero.  If
    every valid cluster is like that, the sum is zero, yielding a NaN weight
    and this specific conversion error when the sampler allocates samples.
    """
    if isinstance(error, ValueError) and "cannot convert float NaN to integer" in str(error):
        return "kmeans_zero_sparsity"

    # ``_ArrayMemoryError`` is a MemoryError subclass.  Restrict this fallback
    # to a traceback through imbalanced-learn's sparsity routine so memory
    # failures in KMeans fitting, SMOTE itself, or unrelated caller code still
    # fail visibly instead of being treated as a balancing degeneracy.
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if (
            isinstance(error, MemoryError)
            and frame.f_code.co_name == "_find_cluster_sparsity"
            and frame.f_code.co_filename.replace("\\", "/").endswith("_smote/cluster.py")
        ):
            return "kmeans_sparsity_memory_error"
        traceback = traceback.tb_next
    return None


def _balancing_audit(stage, counts_before, adj_k, n_clusters, random_state,
                     cluster_balance_threshold, n_jobs):
    """Build JSON-serializable metadata for one common balancing invocation."""
    return {
        "stage": stage,
        "class_counts_before": {str(label): int(count) for label, count in counts_before.items()},
        "primary_method_attempted": "KMeansSMOTE",
        "primary_parameters": {
            "k_neighbors": int(adj_k),
            "n_clusters": int(n_clusters),
            "cluster_balance_threshold": float(cluster_balance_threshold),
            "random_state": int(random_state),
            "n_jobs": int(n_jobs),
        },
        "effective_minibatch_kmeans_nonempty_clusters": None,
        "fallback_used": False,
        "failure_reason": None,
        "fallback_method": None,
        "fallback_parameters": None,
    }


def _record_effective_cluster_count(audit, sampler):
    """Record nonempty fitted MiniBatchKMeans clusters when fit has completed."""
    labels = getattr(getattr(sampler, "kmeans_estimator_", None), "labels_", None)
    if labels is not None:
        audit["effective_minibatch_kmeans_nonempty_clusters"] = int(np.unique(labels).size)


def _kms_fit_resample(X, y, adj_k, n_clusters, random_state, cluster_balance_threshold=0.0,
                      n_jobs=1, stage="balancing"):
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
    audit = _balancing_audit(
        stage, counts_before, adj_k, n_clusters, random_state,
        cluster_balance_threshold, n_jobs,
    )
    t0 = time.perf_counter()
    kms = KMeansSMOTE(
        cluster_balance_threshold=cluster_balance_threshold,
        k_neighbors=adj_k,
        kmeans_estimator=MiniBatchKMeans(
            n_clusters=n_clusters, n_init='auto', random_state=random_state,
        ),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    try:
        X_res, y_res = kms.fit_resample(X, y)
    except (ValueError, MemoryError) as error:
        _record_effective_cluster_count(audit, kms)
        fallback_reason = _kmeans_smote_sparsity_failure_reason(error)
        if fallback_reason is None:
            raise

        # This fallback is intentionally limited to KMeansSMOTE's documented
        # zero/non-finite cluster-weight allocation failure.  Standard SMOTE
        # retains the intended class-oversampling semantics while avoiding
        # cluster-density weighting, and is deterministic with these same
        # effective neighbour and seed settings.
        audit.update({
            "fallback_used": True,
            "failure_reason": fallback_reason,
            "failure_detail": str(error),
            "fallback_method": "SMOTE",
            "fallback_parameters": {
                "k_neighbors": int(adj_k),
                "random_state": int(random_state),
            },
        })
        print(
            f"    {label}: {fallback_reason}; "
            f"falling back deterministically to SMOTE (k_neighbors={adj_k}, "
            f"random_state={random_state})",
            flush=True,
        )
        X_res, y_res = SMOTE(
            random_state=random_state, k_neighbors=adj_k,
        ).fit_resample(X, y)
    else:
        _record_effective_cluster_count(audit, kms)
    dt = time.perf_counter() - t0
    counts_after = dict(sorted(Counter(y_res).items()))
    print(
        f"    {label}: done in {dt:.1f}s | after -> {counts_after}",
        flush=True,
    )
    audit["class_counts_after"] = {
        str(label): int(count) for label, count in sorted(Counter(y_res).items())
    }
    return X_res, y_res, audit


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
    cluster_balance_threshold: float = 0.0,
    n_jobs: int = 1,
    stage: str = "fold",
    return_audit: bool = False,
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
    X_balanced, y_balanced, or (when ``return_audit=True``) a third,
    JSON-serializable balancing audit dictionary.
    """
    X_use, y_use = X_train, y_train
    if not np.isfinite(np.asarray(X_use)).all():
        raise ValueError("Balancing input contains non-finite feature values.")

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
        X_res, y_res, audit = _kms_fit_resample(
            X_use, y_use, adj_k, n_clusters, random_state,
            cluster_balance_threshold=cluster_balance_threshold,
            n_jobs=n_jobs, stage=stage,
        )
    else:
        minority_count = min(Counter(y_use).values())
        adj_k = min(k_neighbors, minority_count - 1)
        adj_k = max(adj_k, 1)
        sm = SMOTE(random_state=random_state, k_neighbors=adj_k)
        X_res, y_res = sm.fit_resample(X_use, y_use)
        audit = {
            "stage": stage,
            "class_counts_before": {
                str(label): int(count) for label, count in sorted(Counter(y_use).items())
            },
            "primary_method_attempted": "SMOTE",
            "primary_parameters": {
                "k_neighbors": int(adj_k), "random_state": int(random_state),
            },
            "fallback_used": False,
            "failure_reason": None,
            "fallback_method": None,
            "fallback_parameters": None,
            "class_counts_after": {
                str(label): int(count) for label, count in sorted(Counter(y_res).items())
            },
        }

    if return_audit:
        return X_res, y_res, audit
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
    cluster_balance_threshold: float = 0.0,
    n_jobs: int = 1,
    return_audit: bool = False,
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
    X_balanced, y_balanced, or (when ``return_audit=True``) a third,
    JSON-serializable balancing audit dictionary.
    """
    balanced = balance_training_fold(
        X_train, y_train,
        strategy=strategy,
        k_neighbors=k_neighbors,
        n_clusters=n_clusters,
        random_state=random_state,
        rus_cap=rus_cap,
        cluster_balance_threshold=cluster_balance_threshold,
        n_jobs=n_jobs,
        stage="final retrain",
        return_audit=return_audit,
    )
    if return_audit:
        X_balanced, y_balanced, audit = balanced
    else:
        X_balanced, y_balanced = balanced
    print(f"    Balanced training: {X_train.shape[0]:,} -> "
          f"{X_balanced.shape[0]:,} samples")
    if return_audit:
        return X_balanced, y_balanced, audit
    return X_balanced, y_balanced
