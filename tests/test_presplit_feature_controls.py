"""Regression coverage for the E4/E5 presplit feature-ablation controls."""

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.feature_selection import fit_mi_selector
from src.presplit_cv_checkpoints import build_cv_config_fingerprint
from src import presplit_feature_ablation_protocol as protocol


class _ZeroModel:
    """Fast, pickleable classifier used to observe runner protocol calls."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0], dtype=self.classes_.dtype)

    def predict_proba(self, X):
        probabilities = np.zeros((len(X), len(self.classes_)))
        probabilities[:, 0] = 1.0
        return probabilities


@pytest.fixture
def prepared_data():
    """A 43-feature, balanced stand-in for the prepared UNSW representation."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(150, 43)).astype(np.float32)
    y = np.repeat(np.arange(3), 50)
    return X, y


def test_e4_is_the_existing_mi_transform_without_pca(prepared_data):
    X, y = prepared_data
    with patch.object(
        protocol,
        "fit_mi_selector",
        wraps=fit_mi_selector,
    ) as mi_fit, patch.object(
        protocol,
        "PCA",
        side_effect=AssertionError("E4 must not instantiate PCA"),
    ):
        X_e4, dimensions, artifacts, _ = (
            protocol.apply_presplit_feature_processing(X, y, "E4_mi_only", mi_k=15)
        )

    assert mi_fit.call_count == 1
    assert mi_fit.call_args.kwargs["k"] == 15
    assert mi_fit.call_args.kwargs["random_state"] == 42
    assert dimensions == {
        "prepared_dimension": 43,
        "mi_dimension": 15,
        "final_dimension": 15,
    }
    assert X_e4.shape == (len(y), 15)
    assert artifacts["selector"] is not None
    assert "pca" not in artifacts
    assert len(artifacts["selected_features"]["selected_features"]) == 15


def test_e5_reuses_e2_independent_pca_branch_and_never_fits_mi(prepared_data):
    X, y = prepared_data
    with patch.object(
        protocol,
        "_fit_parallel_pca_branch",
        wraps=protocol._fit_parallel_pca_branch,
    ) as pca_branch:
        X_e2, e2_dimensions, _, _ = (
            protocol.apply_presplit_feature_processing(
                X, y, "E2_parallel_mi_pca", mi_k=15, pca_variance=0.95,
            )
        )
        with patch.object(
            protocol,
            "fit_mi_selector",
            side_effect=AssertionError("E5 must not fit MI"),
        ):
            X_e5, e5_dimensions, e5_artifacts, _ = (
                protocol.apply_presplit_feature_processing(
                    X, y, "E5_pca_only", pca_variance=0.95,
                )
            )

    assert pca_branch.call_count == 2
    assert e5_dimensions == {
        "prepared_dimension": 43,
        "pca_dimension": e2_dimensions["pca_branch_dimension"],
        "final_dimension": e2_dimensions["pca_branch_dimension"],
    }
    assert X_e5.shape[1] == e2_dimensions["pca_branch_dimension"]
    np.testing.assert_allclose(X_e5, X_e2[:, 15:])
    assert e5_artifacts["scaler"] is not None
    assert e5_artifacts["pca"] is not None
    assert "selector" not in e5_artifacts
    assert "selected_features" not in e5_artifacts


def test_existing_e1_e2_e3_transform_outputs_remain_unchanged(prepared_data):
    """Compare E1/E2 with their pre-control transformation definitions."""
    X, y = prepared_data
    selector = fit_mi_selector(X, y, k=15, random_state=42)
    X_mi = selector.transform(X)

    expected_e1 = PCA(n_components=0.95, random_state=42).fit_transform(
        StandardScaler().fit_transform(X_mi)
    ).astype(np.float32)
    expected_e2 = np.concatenate(
        [
            X_mi,
            PCA(n_components=0.95, random_state=42).fit_transform(
                StandardScaler().fit_transform(X)
            ),
        ],
        axis=1,
    ).astype(np.float32)

    X_e1, e1_dimensions, _, _ = protocol.apply_presplit_feature_processing(
        X, y, "E1_sequential_mi_pca", mi_k=15, pca_variance=0.95,
    )
    X_e2, e2_dimensions, _, _ = protocol.apply_presplit_feature_processing(
        X, y, "E2_parallel_mi_pca", mi_k=15, pca_variance=0.95,
    )
    X_e3, e3_dimensions, _, _ = protocol.apply_presplit_feature_processing(
        X,
        y,
        "E3_spearman",
        spearman_selection_rule="top_k",
        spearman_selection_value=4,
    )

    np.testing.assert_allclose(X_e1, expected_e1)
    np.testing.assert_allclose(X_e2, expected_e2)
    assert e1_dimensions["mi_dimension"] == 15
    assert e2_dimensions["mi_branch_dimension"] == 15
    assert e2_dimensions["final_dimension"] == expected_e2.shape[1]
    assert X_e3.shape == (len(y), 4)
    assert e3_dimensions["final_dimension"] == 4


def test_controls_are_transformed_before_the_outer_split(monkeypatch, prepared_data):
    X, y = prepared_data
    transformed = np.full((len(y), 15), 3.0, dtype=np.float32)
    observed = {"processed": False}

    def fake_processing(*args, **kwargs):
        observed["processed"] = True
        return transformed, {"final_dimension": 15}, {}, 0.0

    def stop_after_split(X_final, split_y, random_state):
        assert observed["processed"]
        assert X_final is transformed
        assert split_y is y
        assert random_state == 42
        raise RuntimeError("outer split observed")

    def fake_calibrated_processing(*args, **kwargs):
        X_final, dimensions, artifacts, elapsed = fake_processing(*args, **kwargs)
        return X_final, dimensions, artifacts, elapsed, {
            "manifest_path": "fixture.json", "manifest_sha256": "a" * 64,
            "protocol": "fixture", "dataset_fingerprint": "fixture",
            "prepared_dimension": 43, "mi_k": 10,
            "pca_parameterization": "component_count", "pca_components": 5,
            "spearman_selection_rule": "top_k",
            "spearman_calibration_method": "fixture", "spearman_k": 4,
        }

    monkeypatch.setattr(
        protocol, "apply_calibrated_presplit_feature_processing",
        fake_calibrated_processing,
    )
    monkeypatch.setattr("src.dimensionality_reduction.split_data", stop_after_split)
    with pytest.raises(RuntimeError, match="outer split observed"):
        protocol.run_presplit_ablation_experiment(
            X,
            y,
            ["Normal", "AttackA", "AttackB"],
            "E4_mi_only",
            calibration_manifest="fixture.json",
        )


def test_e1_to_e5_have_distinct_resume_fingerprints():
    fingerprints = {
        build_cv_config_fingerprint(
            {"experiment": experiment, "model": "HGB", "random_state": 42},
            {"prepared_dimension": 43, "final_dimension": 15},
            None,
        )
        for experiment in protocol.EXPERIMENTS
    }
    assert len(fingerprints) == 5


def test_all_existing_classifier_choices_remain_available_to_controls():
    # Classical models use MODEL_REGISTRY; DL adapters receive input_dim from
    # X.shape[1], so neither path pins an E4/E5-specific feature width.
    protocol._ensure_registry()
    assert {"HGB", "LogReg", "XGBoost"} <= set(protocol.MODEL_REGISTRY)
    assert set(protocol.DL_MODELS) == {
        "DNN", "LSTM", "BiLSTM", "BiLSTMSharedFeature",
    }

    torch = pytest.importorskip("torch")
    from src.presplit_ablation_models import _spec

    for model_name in protocol.DL_MODELS:
        model, *_ = _spec(model_name, input_dim=15, classes=3, device=torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros((2, 15), dtype=torch.float32))
        logits = output[1] if isinstance(output, tuple) else output
        assert logits.shape == (2, 3)


@pytest.mark.parametrize("experiment", ("E4_mi_only", "E5_pca_only"))
def test_controls_preserve_existing_balancing_configuration(
    monkeypatch,
    prepared_data,
    tmp_path,
    experiment,
):
    X, y = prepared_data
    calls = []

    def fake_balance(X_values, y_values, **kwargs):
        calls.append(kwargs)
        return X_values, y_values, {"method": "unchanged-test-double"}

    monkeypatch.setattr(protocol, "balance_training_fold", fake_balance)
    monkeypatch.setattr(protocol, "balance_full_train", fake_balance)
    monkeypatch.setattr(protocol, "_ensure_registry", lambda: None)
    monkeypatch.setitem(
        protocol.MODEL_REGISTRY,
        "HGB",
        {"model_class": _ZeroModel, "params": {}},
    )
    monkeypatch.setattr(protocol, "_latency", lambda *args: {})

    def fake_calibrated_processing(X_values, y_values, *args, **kwargs):
        return X_values, {"prepared_dimension": 43, "final_dimension": 43}, {}, 0.0, {
            "manifest_path": "fixture.json", "manifest_sha256": "a" * 64,
            "protocol": "fixture", "dataset_fingerprint": "fixture",
            "prepared_dimension": 43, "mi_k": 10,
            "pca_parameterization": "component_count", "pca_components": 5,
            "spearman_selection_rule": "top_k",
            "spearman_calibration_method": "fixture", "spearman_k": 4,
        }

    monkeypatch.setattr(
        protocol, "apply_calibrated_presplit_feature_processing",
        fake_calibrated_processing,
    )

    protocol.run_presplit_ablation_experiment(
        X,
        y,
        ["c0", "c1", "Normal"],
        experiment,
        model_name="HGB",
        n_splits=2,
        rus_cap=15000,
        results_root=str(tmp_path),
        calibration_manifest="fixture.json",
    )

    assert len(calls) == 3  # two CV folds plus final 80% retraining
    assert all(call["strategy"] == "kmeans" for call in calls)
    assert all(call["k_neighbors"] == 3 for call in calls)
    assert all(call["random_state"] == 42 for call in calls)
    assert all(call["rus_cap"] == 15000 for call in calls)
    assert all(call["return_audit"] is True for call in calls)
