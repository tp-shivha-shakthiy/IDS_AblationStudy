"""Focused tests for classifier-free presplit parameter calibration."""

import inspect
import json

import numpy as np
import pytest

from src import presplit_feature_calibration as calibration


def _curved_values(size=43):
    return np.concatenate(([10.0, 7.0, 1.0], np.linspace(0.99, 0.01, size - 3)))


def test_mi_scores_every_feature_before_calculating_k(monkeypatch):
    observed = {}

    class Selector:
        scores_ = _curved_values()
        mi_n_neighbors_ = 3

    def fake_selector(X, y, **kwargs):
        observed.update(kwargs)
        return Selector()

    monkeypatch.setattr(calibration, "fit_mi_selector", fake_selector)
    profile = calibration.calibrate_mutual_information(
        np.zeros((12, 43), dtype=np.float32),
        np.arange(12) % 3,
        [f"f{index}" for index in range(43)],
        np.zeros(43, dtype=bool),
    )

    assert observed["k"] == 43
    assert observed["random_state"] == 42
    assert len(profile["all_feature_scores"]) == 43
    assert profile["selected_k"] != 15


def test_knee_detection_is_deterministic_and_ties_choose_smaller_rank():
    selected_a, curve_a, _ = calibration._normalized_knee(
        [1.0, 0.5, 0.5, 0.0],
        method_name="test",
    )
    selected_b, curve_b, _ = calibration._normalized_knee(
        [1.0, 0.5, 0.5, 0.0],
        method_name="test",
    )
    assert selected_a == selected_b == 1  # index 1 is the smaller tied cutoff, k=2
    assert curve_a == curve_b


def test_flat_profiles_fail_without_a_historical_default():
    with pytest.raises(calibration.CalibrationError, match="non-flat knee"):
        calibration._normalized_knee([0.0, 0.0, 0.0, 0.0], method_name="test")


def test_pca_component_count_is_derived_from_complete_variance_spectrum():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(160, 43)).astype(np.float32)
    X *= np.geomspace(10.0, 0.01, 43)
    profile = calibration.calibrate_pca(X)

    assert len(profile["components"]) == 43
    assert 1 < profile["selected_component_count"] < 43
    selected = profile["components"][profile["selected_component_count"] - 1]
    assert profile["selected_cumulative_variance"] == selected[
        "cumulative_explained_variance_ratio"
    ]
    assert "n_components=" not in inspect.getsource(calibration.calibrate_pca)
    assert "0.95" not in inspect.getsource(calibration.calibrate_pca)


def test_spearman_k_is_recomputed_from_scores_not_hardcoded():
    y = np.arange(120, dtype=float)
    X = np.column_stack(
        [
            y,
            y * 0.8 + np.sin(y),
            y * 0.2 + np.cos(y),
            *[
                np.random.RandomState(index).normal(size=len(y))
                for index in range(40)
            ],
        ]
    ).astype(np.float32)
    names = [f"f{index}" for index in range(X.shape[1])]
    profile = calibration.calibrate_spearman(X, y, names)
    recomputed = calibration._rank_profile(
        np.abs(profile["coefficients"]),
        names,
        method_name="Spearman",
    )
    assert profile["selected_k"] == recomputed["selected_k"]
    assert "= 4" not in inspect.getsource(calibration.calibrate_spearman)


def test_calibration_module_has_no_classifier_or_metric_result_dependency():
    source = inspect.getsource(calibration)
    assert "MODEL_REGISTRY" not in source
    assert "test_metrics.json" not in source
    assert "cv_metrics" not in source


def test_manifest_records_provenance_and_selected_values(monkeypatch, tmp_path):
    prepared = np.zeros((9, 3), dtype=np.float32)
    names = ["a", "b", "c"]
    mask = np.array([False, True, False])
    profile = {
        "selected_k": 2,
        "selected_features_rank_order": ["a", "b"],
        "ranking": [],
        "normalized_curve": [],
        "maximum_distance": 0.2,
        "selection_method": calibration.KNEE_METHOD,
    }
    pca = {
        "selected_component_count": 2,
        "selected_cumulative_variance": 0.8,
        "components": [
            {"component": 1, "explained_variance_ratio": 0.5, "cumulative_explained_variance_ratio": 0.5},
            {"component": 2, "explained_variance_ratio": 0.3, "cumulative_explained_variance_ratio": 0.8},
        ],
        "normalized_curve": [],
        "maximum_distance": 0.2,
        "selection_method": calibration.KNEE_METHOD,
        "recommended_frozen_parameterization": "component_count",
    }
    spearman = {**profile, "coefficients": [0.4, 0.3, 0.2], "absolute_coefficients": [0.4, 0.3, 0.2]}
    monkeypatch.setattr(calibration, "prepare_presplit_calibration_features", lambda X: (prepared, names, mask, None))
    monkeypatch.setattr(calibration, "calibrate_mutual_information", lambda *args: dict(profile))
    monkeypatch.setattr(calibration, "calibrate_pca", lambda *args: dict(pca))
    monkeypatch.setattr(calibration, "calibrate_spearman", lambda *args: dict(spearman))

    manifest = calibration.calibrate_presplit_feature_parameters(
        object(),
        np.arange(9) % 3,
        ["Attack", "Normal", "Other"],
        dataset_files=[{"name": "fixture.csv", "bytes": 12}],
    )
    calibration.write_calibration_artifacts(
        manifest,
        tmp_path / "manifest.json",
        tmp_path / "report",
    )

    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["prepared_dimension"] == 3
    assert saved["mi"]["selected_k"] == 2
    assert saved["pca"]["selected_component_count"] == 2
    assert saved["spearman"]["selected_k"] == 2
    assert saved["dataset"]["prepared_feature_fingerprint"]
    assert (tmp_path / "report" / "calibration_report.md").is_file()


def test_existing_experiments_remain_registered_unchanged():
    from src.presplit_feature_ablation_protocol import EXPERIMENTS

    assert EXPERIMENTS[:3] == (
        "E1_sequential_mi_pca",
        "E2_parallel_mi_pca",
        "E3_spearman",
    )
