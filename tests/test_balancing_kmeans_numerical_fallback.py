"""Regression coverage for KMeansSMOTE's zero-sparsity allocation failure."""

from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest

from src.balancing import balance_training_fold


def test_kmeans_smote_zero_sparsity_uses_auditable_deterministic_fallback():
    """Duplicate 4-D minority samples yield zero KMeansSMOTE cluster density.

    imbalanced-learn divides the all-zero sparsity vector by zero and raises
    ``ValueError: cannot convert float NaN to integer``.  The common balancing
    layer must retain deterministic oversampling via its narrowly scoped SMOTE
    fallback and expose the event for result auditing.
    """
    X = np.vstack([np.zeros((12, 4)), np.ones((48, 4))]).astype(np.float32)
    y = np.array([0] * 12 + [1] * 48)

    with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
        X_resampled, y_resampled, audit = balance_training_fold(
            X, y, strategy="kmeans", k_neighbors=3, n_clusters=20,
            random_state=42, return_audit=True,
        )

    assert X_resampled.shape == (96, 4)
    assert Counter(y_resampled) == Counter({0: 48, 1: 48})
    assert audit["primary_method_attempted"] == "KMeansSMOTE"
    assert audit["effective_minibatch_kmeans_nonempty_clusters"] == 2
    assert audit["fallback_used"] is True
    assert audit["failure_reason"] == "kmeans_zero_sparsity"
    assert audit["fallback_method"] == "SMOTE"
    assert audit["fallback_parameters"] == {"k_neighbors": 3, "random_state": 42}


def test_kmeans_smote_sparsity_memory_error_uses_same_deterministic_fallback():
    """Only a MemoryError raised by imblearn's sparsity routine may fall back."""
    from imblearn.over_sampling._smote import cluster as cluster_module

    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(-3, 0.2, size=(12, 4)),
        rng.normal(3, 0.2, size=(48, 4)),
    ]).astype(np.float32)
    y = np.array([0] * 12 + [1] * 48)

    def memory_exhausted(*args, **kwargs):
        raise np._core._exceptions._ArrayMemoryError((21284, 21284), np.dtype("float64"))

    with patch.object(cluster_module, "pairwise_distances", memory_exhausted):
        _, y_resampled, audit = balance_training_fold(
            X, y, strategy="kmeans", k_neighbors=3, n_clusters=2,
            random_state=42, return_audit=True,
        )

    assert Counter(y_resampled) == Counter({0: 48, 1: 48})
    assert audit["failure_reason"] == "kmeans_sparsity_memory_error"
    assert audit["fallback_method"] == "SMOTE"


def test_unrelated_kmeans_smote_exception_is_not_silently_fallbacked(monkeypatch):
    """The narrow fallback must not mask an unrelated sampler failure."""
    class FailingSampler:
        def __init__(self, **kwargs):
            pass

        def fit_resample(self, X, y):
            raise ValueError("unrelated KMeansSMOTE failure")

    import src.balancing as balancing

    monkeypatch.setattr(balancing, "KMeansSMOTE", FailingSampler)
    X = np.arange(120, dtype=np.float32).reshape(60, 2)
    y = np.array([0] * 12 + [1] * 48)
    with pytest.raises(ValueError, match="unrelated KMeansSMOTE failure"):
        balance_training_fold(X, y, n_clusters=2, random_state=42)
