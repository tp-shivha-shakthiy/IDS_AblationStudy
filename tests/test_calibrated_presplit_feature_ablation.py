"""Regression coverage for the frozen calibration -> E1-E5 integration."""

import json

import numpy as np
import pytest
from scipy.stats import spearmanr
from unittest.mock import patch

from src import presplit_feature_ablation_protocol as protocol
from src import presplit_feature_calibration as calibration
from src.feature_selection import fit_mi_selector
from src.presplit_cv_checkpoints import build_cv_config_fingerprint


@pytest.fixture
def calibrated_fixture(tmp_path):
    rng = np.random.RandomState(7)
    X = rng.normal(size=(150, 43)).astype(np.float32)
    y = np.repeat(np.arange(3), 50)
    names = [f"feature_{index}" for index in range(43)]
    selector = fit_mi_selector(X, y, k=10, random_state=42)
    mi_names = [names[index] for index in selector.get_support(indices=True)]
    correlation = np.asarray(spearmanr(X, y, axis=0).statistic[:-1, -1], dtype=float)
    ranking = np.argsort(-np.abs(np.nan_to_num(correlation)), kind="stable")
    manifest = {
        "protocol": calibration.CALIBRATION_PROTOCOL,
        "prepared_dimension": 43,
        "feature_names": names,
        "feature_name_order_hash": calibration.hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest(),
        "class_names": ["Attack", "Normal", "Other"],
        "dataset": {
            "prepared_feature_fingerprint": calibration._fingerprint_prepared_data(
                X, y, names
            )
        },
        "mi": {"selected_k": 10, "selected_features_rank_order": mi_names},
        "pca": {
            "selected_component_count": 5,
            "recommended_frozen_parameterization": "component_count",
        },
        "spearman": {
            "selected_k": 4,
            "selection_method": calibration.KNEE_METHOD,
            "selected_features_rank_order": [names[index] for index in ranking[:4]],
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return X, y, manifest["class_names"], path


@pytest.mark.parametrize(
    ("experiment", "expected_dimension"),
    (
        ("E1_sequential_mi_pca", 5),
        ("E2_parallel_mi_pca", 15),
        ("E3_spearman", 4),
        ("E4_mi_only", 10),
        ("E5_pca_only", 5),
    ),
)
def test_final_transforms_take_all_feature_parameters_from_manifest(
    calibrated_fixture, experiment, expected_dimension
):
    X, y, class_names, manifest = calibrated_fixture
    with patch.object(
            protocol, "_fit_parallel_pca_branch",
            wraps=protocol._fit_parallel_pca_branch,
    ) as parallel_pca:
        X_final, dimensions, artifacts, _, frozen = (
            protocol.apply_calibrated_presplit_feature_processing(
                X, y, class_names, experiment, manifest
            )
        )
        if experiment in {"E2_parallel_mi_pca", "E5_pca_only"}:
            assert parallel_pca.call_args.kwargs["pca_components"] == frozen["pca_components"]

    assert X_final.shape == (len(y), expected_dimension)
    assert dimensions["final_dimension"] == expected_dimension
    assert frozen["mi_k"] == 10
    assert frozen["pca_parameterization"] == "component_count"
    assert frozen["pca_components"] == 5
    assert frozen["spearman_selection_rule"] == "top_k"
    assert frozen["spearman_k"] == 4
    assert artifacts["calibration_manifest"]["manifest_sha256"]


def test_manifest_validation_rejects_missing_or_incompatible_data(calibrated_fixture):
    X, y, class_names, manifest = calibrated_fixture
    invalid = manifest.with_name("invalid.json")
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(calibration.CalibrationError, match="protocol"):
        calibration.load_and_validate_frozen_calibration_manifest(
            invalid, X, y, class_names
        )

    changed = X.copy()
    changed[0, 0] += 1
    with pytest.raises(calibration.CalibrationError, match="fingerprint"):
        calibration.load_and_validate_frozen_calibration_manifest(
            manifest, changed, y, class_names
        )


def test_checkpoint_fingerprint_binds_manifest_and_component_parameterization():
    base = {
        "experiment": "E1_sequential_mi_pca",
        "model": "HGB",
        "random_state": 42,
        "calibration": {
            "manifest_sha256": "new-manifest",
            "dataset_fingerprint": "dataset",
            "mi_k": 10,
            "pca_parameterization": "component_count",
            "pca_components": 5,
            "spearman_selection_rule": "top_k",
            "spearman_k": 4,
        },
    }
    old = {**base, "calibration": {**base["calibration"], "mi_k": 15,
                                    "pca_parameterization": "variance_threshold",
                                    "pca_components": 0.95}}
    dimensions = {"prepared_dimension": 43, "final_dimension": 5}
    assert build_cv_config_fingerprint(base, dimensions, None) != (
        build_cv_config_fingerprint(old, dimensions, None)
    )


def test_final_runner_requires_a_manifest(calibrated_fixture):
    X, y, class_names, _ = calibrated_fixture
    with pytest.raises(calibration.CalibrationError, match="require a frozen"):
        protocol.run_presplit_ablation_experiment(
            X, y, class_names, "E4_mi_only", n_splits=2
        )


@pytest.mark.parametrize(
    ("experiment", "kwargs", "match"),
    (
        ("E2_parallel_mi_pca", {}, "MI k"),
        ("E4_mi_only", {}, "MI k"),
        ("E5_pca_only", {}, "PCA parameter"),
    ),
)
def test_presplit_transforms_reject_unfrozen_feature_parameters(
    calibrated_fixture, experiment, kwargs, match,
):
    X, y, _, _ = calibrated_fixture
    with pytest.raises(calibration.CalibrationError, match=match):
        protocol.apply_presplit_feature_processing(X, y, experiment, **kwargs)
