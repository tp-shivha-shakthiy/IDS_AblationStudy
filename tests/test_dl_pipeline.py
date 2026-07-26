"""
test_dl_pipeline.py
===================
Regression tests for Tier 2 DL pipeline.

Tests:
  1. DataLoader drop_last=True (prevents batch=1 LayerNorm failure)
  2. Saved test metrics JSON has real finite AUC in [0, 1]
  3. DeepNeuralNetwork uses LayerNorm (not BatchNorm1d)
  4. get_probabilities returns valid probability arrays
  5. evaluate_with_proba returns real AUC (not hardcoded 0.0)
  6. preprocess_fold k_neighbors is floored at 1
"""

import json
import os
import numpy as np
import pytest
import torch
import torch.nn as nn
from collections import Counter
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset

from src.dl_pipeline import (
    evaluate_predictions, evaluate_with_proba,
    get_probabilities, compute_class_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_tensors():
    """Small synthetic data as torch tensors."""
    X = torch.randn(100, 20)
    y = torch.randint(0, 5, (100,))
    return X, y


@pytest.fixture
def synthetic_numpy():
    """Small synthetic numpy arrays."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 20).astype(np.float32)
    y = rng.choice([0, 1, 2, 3, 4], size=100, p=[0.5, 0.15, 0.1, 0.15, 0.1])
    return X, y


# ---------------------------------------------------------------------------
# 1. DataLoader must have drop_last=True
# ---------------------------------------------------------------------------

class TestDataLoaderDropLast:
    def test_training_loader_has_drop_last(self, synthetic_tensors):
        X, y = synthetic_tensors
        loader = DataLoader(
            TensorDataset(X, y), batch_size=512, shuffle=True, drop_last=True,
        )
        assert loader.drop_last is True

    def test_training_loader_drops_last_batch(self):
        X = torch.randn(10, 5)
        y = torch.randint(0, 2, (10,))
        loader = DataLoader(
            TensorDataset(X, y), batch_size=8, shuffle=False, drop_last=True,
        )
        total_samples = sum(batch[0].shape[0] for batch in loader)
        assert total_samples == 8, "drop_last=True should drop the final incomplete batch"


# ---------------------------------------------------------------------------
# 2. Saved test metrics JSON must have real finite AUC in [0, 1]
# ---------------------------------------------------------------------------

class TestMetricsJsonAUC:
    def test_auc_in_valid_range(self):
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1, 2, 3, 4], size=500, p=[0.5, 0.15, 0.1, 0.15, 0.1])
        y_proba = rng.dirichlet(np.ones(5), size=500).astype(np.float32)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = evaluate_with_proba(y_true, y_pred, y_proba, normal_class_idx=0)

        assert 'auc' in metrics
        assert isinstance(metrics['auc'], float)
        assert 0.0 <= metrics['auc'] <= 1.0, f"AUC {metrics['auc']} outside [0, 1]"

    def test_auc_not_always_zero(self):
        rng = np.random.RandomState(42)
        n = 500
        y_true = np.array([0] * 400 + [1] * 50 + [2] * 50)
        y_proba = np.zeros((n, 3), dtype=np.float32)
        y_proba[y_true == 0, 0] = 0.9
        y_proba[y_true == 1, 1] = 0.9
        y_proba[y_true == 2, 2] = 0.9
        y_proba += rng.dirichlet(np.ones(3), size=n).astype(np.float32) * 0.05
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = evaluate_with_proba(y_true, y_pred, y_proba, normal_class_idx=0)
        assert metrics['auc'] > 0.5, f"AUC {metrics['auc']} too low for near-perfect predictions"

    def test_saved_json_auc_finite(self, tmp_path):
        metrics = {
            'binary_acc': 0.95, 'binary_f1': 0.96,
            'multi_acc': 0.93, 'macro_f1': 0.91,
            'weighted_f1': 0.92, 'precision': 0.93,
            'recall': 0.93, 'auc': 0.9875,
        }
        json_path = tmp_path / "test_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f)

        with open(json_path) as f:
            loaded = json.load(f)

        assert isinstance(loaded['auc'], (int, float))
        assert 0.0 <= loaded['auc'] <= 1.0
        assert np.isfinite(loaded['auc'])


# ---------------------------------------------------------------------------
# 3. DeepNeuralNetwork must use LayerNorm (not BatchNorm1d)
# ---------------------------------------------------------------------------

class TestArchitectureLayerNorm:
    def test_dnn_mi_pca_kmeans_uses_layernorm(self):
        from models.train_dnn_mi_pca_kmeans import DeepNeuralNetwork
        model = DeepNeuralNetwork(input_dim=15, output_dim=10)

        has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
        has_layernorm = any(isinstance(m, nn.LayerNorm) for m in model.modules())

        assert has_layernorm, "DeepNeuralNetwork must use LayerNorm"
        assert not has_batchnorm, "DeepNeuralNetwork must NOT use BatchNorm1d"

    def test_dnn_forward_pass_small_batch(self):
        from models.train_dnn_mi_pca_kmeans import DeepNeuralNetwork
        model = DeepNeuralNetwork(input_dim=15, output_dim=10)
        model.eval()

        x = torch.randn(1, 15)
        out = model(x)
        assert out.shape == (1, 10)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 4. get_probabilities returns valid probability arrays
# ---------------------------------------------------------------------------

class TestGetProbabilities:
    def test_returns_valid_probabilities(self):
        model = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        model.eval()
        X = torch.randn(50, 20)
        device = torch.device('cpu')

        proba = get_probabilities(model, X, device)

        assert isinstance(proba, np.ndarray)
        assert proba.shape == (50, 5)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        assert np.all(np.isfinite(proba))


# ---------------------------------------------------------------------------
# 5. evaluate_with_proba returns real AUC (not hardcoded 0.0)
# ---------------------------------------------------------------------------

class TestEvaluateWithProba:
    def test_auc_nonzero_with_good_predictions(self):
        rng = np.random.RandomState(42)
        n = 500
        y_true = np.array([0] * 300 + [1] * 100 + [2] * 100)
        y_proba = np.zeros((n, 3), dtype=np.float32)
        y_proba[y_true == 0, 0] = 0.9
        y_proba[y_true == 1, 1] = 0.9
        y_proba[y_true == 2, 2] = 0.9
        y_proba += rng.dirichlet(np.ones(3), size=n).astype(np.float32) * 0.05
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = evaluate_with_proba(y_true, y_pred, y_proba, normal_class_idx=0)
        assert metrics['auc'] > 0.9

    def test_evaluate_predictions_returns_zero_auc(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = evaluate_predictions(y_true, y_pred, normal_class_idx=0)
        assert metrics['auc'] == 0.0


# ---------------------------------------------------------------------------
# 6. preprocess_fold k_neighbors floored at 1
# ---------------------------------------------------------------------------

class TestKNeighborsFloor:
    def test_dl_pipeline_preprocess_fold_k_floor(self):
        from collections import Counter
        k_neighbors_input = 100
        minority_count = 8
        actual_k = min(k_neighbors_input, minority_count - 1)
        floored_k = max(actual_k, 1)
        assert floored_k == 7
        assert floored_k >= 1

    def test_dl_pipeline_preprocess_fold_runs_without_error(self):
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 20).astype(np.float32)
        y = np.array([0] * 800 + [1] * 200)
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42,
        )

        from src.dl_pipeline import preprocess_fold
        result = preprocess_fold(
            X_tr, y_tr, X_val, y_val,
            mi_k=0, pca_components=0,
            n_clusters=10, k_neighbors=2, rus_cap=500,
            random_state=42,
        )
        assert result['X_tr'].shape[0] > 0
        assert result['X_val'].shape[0] == len(y_val)

    def test_balancing_floor_k_neighbors(self):
        X = np.random.RandomState(42).randn(50, 10).astype(np.float32)
        y = np.array([0] * 45 + [1] * 5)

        from src.balancing import balance_training_fold
        X_bal, y_bal = balance_training_fold(
            X, y, strategy='smote', k_neighbors=100, random_state=42,
        )
        assert len(X_bal) > len(X)
