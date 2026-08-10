"""
test_dl_pipeline.py
===================
Tests for DL model architectures, preprocessing, evaluation, and integration.

Covers:
  1. DL model forward pass — output shape and gradient flow
  2. DL model save/load roundtrip — state_dict persistence
  3. Feature selection — fit_mi_selector and apply_feature_selection
  4. Evaluation — evaluate_predictions, evaluate_with_proba, get_probabilities
  5. Experiment config — build_experiment_config schema
  6. Integration — synthetic data through full CV pipeline
"""

import os
import sys
import json
import importlib
import numpy as np
import pytest
import torch
import tempfile
import joblib
from unittest.mock import patch

from sklearn.model_selection import train_test_split

# Add models/ to path so hyphenated files can be loaded via importlib
_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)


def _import_hyphenated(filename, module_attr):
    """Import a class from a hyphenated filename using importlib."""
    spec = importlib.util.spec_from_file_location(
        filename.replace(".py", ""),
        os.path.join(_models_dir, filename),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, module_attr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset():
    """Small reproducible dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 15).astype(np.float32)
    y = rng.choice([0, 1, 2], size=500, p=[0.5, 0.3, 0.2])
    return X, y


@pytest.fixture
def split_data(synthetic_dataset):
    """Stratified 80/20 split."""
    X, y = synthetic_dataset
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# ---------------------------------------------------------------------------
# 1. DL model forward pass
# ---------------------------------------------------------------------------

class TestDLModelForwardPass:
    def test_dnn_output_shape(self):
        """DNN must output logits of shape (batch, num_classes)."""
        from models.train_dnn import DeepNeuralNetwork

        model = DeepNeuralNetwork(input_dim=15, output_dim=10)
        X = torch.randn(32, 15)
        out = model(X)
        assert out.shape == (32, 10)

    def test_dnn_mi_pca_kmeans_output_shape(self):
        """DNN_MI_PCA_KMeans must output correct shape."""
        from models.train_dnn_mi_pca_kmeans import DeepNeuralNetwork

        model = DeepNeuralNetwork(input_dim=15, output_dim=10)
        X = torch.randn(32, 15)
        out = model(X)
        assert out.shape == (32, 10)

    def test_lstm_output_shape(self):
        """BiLSTM must output correct shape."""
        from models.train_LSTM import BiLSTMNetwork

        model = BiLSTMNetwork(input_dim=15, output_dim=10)
        X = torch.randn(32, 15)
        out = model(X)
        assert out.shape == (32, 10)

    def test_bilstm_output_shape(self):
        """WeightedBiLSTM must output correct shape."""
        WeightedBiLSTM = _import_hyphenated("train_Bi-LSTM.py", "WeightedBiLSTM")

        model = WeightedBiLSTM(input_dim=15, output_dim=10)
        X = torch.randn(32, 15)
        out = model(X)
        assert out.shape == (32, 10)

    def test_multi_task_output_shapes(self):
        """MultiTaskHierarchicalDNN must return (binary, multi) outputs."""
        MultiTaskHierarchicalDNN = _import_hyphenated(
            "train_Bi-LSTM_shared-feature-extractor.py", "MultiTaskHierarchicalDNN"
        )

        model = MultiTaskHierarchicalDNN(input_dim=15, num_classes=10)
        X = torch.randn(32, 15)
        bin_out, multi_out = model(X)
        assert bin_out.shape == (32, 2)
        assert multi_out.shape == (32, 10)

    def test_dnn_gradients_flow(self):
        """DNN must produce gradients on backward pass."""
        from models.train_dnn import DeepNeuralNetwork

        model = DeepNeuralNetwork(input_dim=15, output_dim=5)
        X = torch.randn(16, 15)
        y = torch.randint(0, 5, (16,))
        out = model(X)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# 2. DL model save/load roundtrip
# ---------------------------------------------------------------------------

class TestDLModelSaveLoad:
    def test_dnn_save_load_roundtrip(self):
        """Saved DNN state_dict must load back identically."""
        from models.train_dnn import DeepNeuralNetwork

        model = DeepNeuralNetwork(input_dim=15, output_dim=5)
        model.eval()
        X = torch.randn(10, 15)
        with torch.no_grad():
            before = model(X).numpy()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            path = f.name

        model2 = DeepNeuralNetwork(input_dim=15, output_dim=5)
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            after = model2(X).numpy()

        np.testing.assert_array_equal(before, after)
        os.unlink(path)

    def test_lstm_save_load_roundtrip(self):
        """Saved LSTM state_dict must load back identically."""
        from models.train_LSTM import BiLSTMNetwork

        model = BiLSTMNetwork(input_dim=15, output_dim=5)
        model.eval()
        X = torch.randn(10, 15)
        with torch.no_grad():
            before = model(X).numpy()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            path = f.name

        model2 = BiLSTMNetwork(input_dim=15, output_dim=5)
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            after = model2(X).numpy()

        np.testing.assert_array_equal(before, after)
        os.unlink(path)


# ---------------------------------------------------------------------------
# 3. Feature selection
# ---------------------------------------------------------------------------

class TestFeatureSelection:
    def test_fit_mi_selector_returns_selectkbest(self):
        """fit_mi_selector must return a fitted SelectKBest."""
        from src.feature_selection import fit_mi_selector

        X = np.random.randn(200, 10).astype(np.float32)
        y = np.random.choice([0, 1, 2], size=200)
        selector = fit_mi_selector(X, y, k=5)

        assert hasattr(selector, 'scores_')
        assert selector.k == 5
        X_mi = selector.transform(X)
        assert X_mi.shape == (200, 5)

    def test_fit_mi_selector_with_sampling(self):
        """fit_mi_selector with sample_frac must still produce valid selector."""
        from src.feature_selection import fit_mi_selector

        X = np.random.RandomState(42).randn(500, 10).astype(np.float32)
        y = np.random.RandomState(42).choice([0, 1, 2], size=500)
        selector = fit_mi_selector(X, y, k=5, sample_frac=0.3)

        assert hasattr(selector, 'scores_')
        X_mi = selector.transform(X)
        assert X_mi.shape == (500, 5)

    def test_apply_feature_selection(self):
        """apply_feature_selection must transform with fitted selector."""
        from src.feature_selection import fit_mi_selector, apply_feature_selection

        X = np.random.RandomState(42).randn(200, 10).astype(np.float32)
        y = np.random.RandomState(42).choice([0, 1], size=200)
        selector = fit_mi_selector(X, y, k=5)

        X_mi = apply_feature_selection(X, selector)
        assert X_mi.shape == (200, 5)


# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------

class TestEvaluationHelpers:
    def test_evaluate_with_proba_auc_above_0_5(self):
        """evaluate_with_proba must return AUC > 0.5 for correlated predictions."""
        from src.dl_pipeline import evaluate_with_proba

        rng = np.random.RandomState(42)
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        y_proba = rng.dirichlet(np.ones(3), size=10)
        for i, c in enumerate(y_true):
            y_proba[i, c] += 3.0
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        y_pred = np.argmax(y_proba, axis=1)

        m = evaluate_with_proba(y_true, y_pred, y_proba)
        assert m['auc'] > 0.5
        assert m['multi_acc'] > 0.0

    def test_evaluate_predictions_binary(self):
        """evaluate_predictions must compute binary metrics correctly."""
        from src.dl_pipeline import evaluate_predictions

        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])

        m = evaluate_predictions(y_true, y_pred, normal_class_idx=0)
        assert m['binary_acc'] == pytest.approx(0.8)
        assert 'auc' in m


# ---------------------------------------------------------------------------
# 5. Experiment config schema
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_tier1_config_has_required_keys(self):
        """Tier 1 config must have all standard keys."""
        from src.experiment_config import build_experiment_config

        cfg = build_experiment_config(
            model_name="XGBoost",
            model_params={"n_estimators": 100},
            tier=1,
        )
        for key in ["model", "tier", "seed", "cv_folds", "balancer",
                     "feature_selection", "scaler", "pca_variance",
                     "model_hyperparameters", "timestamp", "git_commit"]:
            assert key in cfg, f"Missing key: {key}"
        assert cfg["tier"] == 1
        assert cfg["model"] == "XGBoost"

    def test_tier2_config_has_required_keys(self):
        """Tier 2 config must have all standard keys."""
        from src.experiment_config import build_experiment_config

        cfg = build_experiment_config(
            model_name="DNN",
            model_params={"layers": [64, 32]},
            tier=2,
            dl_extra={"epochs": 10},
        )
        for key in ["model", "tier", "seed", "cv_folds", "balancer",
                     "feature_selection", "scaler", "dl_extra",
                     "timestamp", "git_commit"]:
            assert key in cfg, f"Missing key: {key}"
        assert cfg["tier"] == 2
        assert cfg["ablation_scope"] == "excluded_tier2"
        assert cfg["dl_extra"]["epochs"] == 10

    def test_config_json_serializable(self):
        """Config must be JSON-serializable without errors."""
        from src.experiment_config import build_experiment_config

        cfg = build_experiment_config(model_name="TestModel", tier=1)
        json_str = json.dumps(cfg, indent=2)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# 6. Integration — full CV pipeline on synthetic data
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_cv_pipeline_with_hgb(self, split_data):
        """Full CV pipeline with HGB must produce per-fold metrics."""
        from src.model_training import MODEL_REGISTRY, _ensure_registry, _train_and_evaluate
        from src.cross_validation import run_cv
        from src.feature_selection import fit_mi_selector
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from src.balancing import balance_full_train

        X_train, X_test, y_train, y_test = split_data
        _ensure_registry()
        entry = MODEL_REGISTRY["HGB"]

        cv_metrics, selector, scaler, pca = run_cv(
            X_train, y_train,
            model_class=entry["model_class"],
            model_params=entry["params"],
            n_splits=3, mi_k=5,
            pca_variance=0.95, k_neighbors=2,
            random_state=42, strategy="kmeans",
        )

        for k, v in cv_metrics.items():
            assert len(v) == 3

        assert selector is not None
        assert scaler is not None
        assert pca is not None

        # Final retrain path
        selector2 = fit_mi_selector(X_train, y_train, k=5, random_state=42)
        X_tr_mi = selector2.transform(X_train)
        X_te_mi = selector2.transform(X_test)
        scaler2 = StandardScaler()
        X_tr_s = scaler2.fit_transform(X_tr_mi)
        pca2 = PCA(n_components=0.95, random_state=42)
        X_tr_p = pca2.fit_transform(X_tr_s)
        X_tr_b, y_tr_b = balance_full_train(
            X_tr_p, y_train, strategy="kmeans", k_neighbors=2, random_state=42,
        )
        X_te_p = pca2.transform(scaler2.transform(X_te_mi))

        model, test_m, y_pred = _train_and_evaluate(
            entry["model_class"], entry["params"],
            X_tr_b, y_tr_b, X_te_p, y_test,
        )
        assert 0.0 <= test_m['accuracy'] <= 1.0
        assert len(y_pred) == len(y_test)
