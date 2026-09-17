"""Regression coverage for KMeansSMOTE's zero-sparsity allocation failure."""

from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest

from src.balancing import balance_training_fold
from src.presplit_cv_checkpoints import (
    build_cv_config_fingerprint,
    checkpoint_path,
    load_compatible_fold_checkpoint,
    save_fold_checkpoint,
)


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


def test_checkpoint_is_reused_only_with_an_exact_fingerprint(tmp_path):
    """A completed fold is reusable only for the exact presplit configuration."""
    config = {
        "experiment": "E3_spearman", "model": "HGB", "random_state": 42,
        "k_neighbors": 3, "rus_cap": 15000,
        "spearman_selection_rule": "top_k", "spearman_selection_value": 4,
    }
    dimensions = {"prepared_dimension": 43, "selected_dimension": 4, "final_dimension": 4}
    selected_features = {"selected_features": ["ct_state_ttl", "sttl", "state", "dload"]}
    fingerprint = build_cv_config_fingerprint(config, dimensions, selected_features)
    saved = save_fold_checkpoint(
        str(tmp_path), 1, {"accuracy": 0.75}, {"failure_reason": None},
        1.25, 2.5, config, dimensions, fingerprint,
    )

    assert checkpoint_path(str(tmp_path), 1).endswith("cv_fold_1.json")
    assert saved["fold_number"] == 1
    reused = load_compatible_fold_checkpoint(
        str(tmp_path), 1, fingerprint, config, dimensions,
    )
    assert reused["validation_metrics"] == {"accuracy": 0.75}

    changed_config = {**config, "rus_cap": 0}
    changed_fingerprint = build_cv_config_fingerprint(
        changed_config, dimensions, selected_features,
    )
    with pytest.raises(ValueError, match="Refusing to reuse incompatible CV checkpoint"):
        load_compatible_fold_checkpoint(
            str(tmp_path), 1, changed_fingerprint, changed_config, dimensions,
        )
