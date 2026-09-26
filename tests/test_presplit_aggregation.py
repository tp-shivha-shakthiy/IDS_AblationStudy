"""Validation for final calibrated E2/E4/E5 aggregation."""

import hashlib
import json

import pytest

from aggregate_presplit_feature_ablation import (
    FINAL_EXPERIMENTS,
    FINAL_MODELS,
    aggregate_completed_results,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(root, experiment, model, calibration):
    directory = root / experiment / model
    directory.mkdir(parents=True)
    _write_json(directory / "test_metrics.json", {"multiclass_accuracy": 0.9})
    _write_json(directory / "latency.json", {})
    _write_json(directory / "feature_dimensions.json", {"final_dimension": 5})
    _write_json(directory / "resolved_config.json", {"calibration": calibration})
    _write_json(directory / "environment.json", {})


def _write_manifest(path):
    _write_json(path, {
        "mi": {"selected_k": 10},
        "pca": {"selected_component_count": 5},
        "dataset": {"prepared_feature_fingerprint": "dataset"},
    })
    return {
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mi_k": 10,
        "pca_components": 5,
        "pca_parameterization": "component_count",
        "dataset_fingerprint": "dataset",
    }


def test_final_aggregation_rejects_missing_runs(tmp_path):
    manifest = tmp_path / "manifest.json"
    calibration = _write_manifest(manifest)
    _write_run(tmp_path / "results", "E2_parallel_mi_pca", "HGB", calibration)

    with pytest.raises(ValueError, match="incomplete"):
        aggregate_completed_results(
            tmp_path / "results", require_complete=True, calibration_manifest=manifest,
        )


def test_final_aggregation_requires_one_frozen_manifest_and_writes_table(tmp_path):
    manifest = tmp_path / "manifest.json"
    calibration = _write_manifest(manifest)
    root = tmp_path / "results"
    for experiment in FINAL_EXPERIMENTS:
        for model in FINAL_MODELS:
            _write_run(root, experiment, model, calibration)

    rows = aggregate_completed_results(
        root, require_complete=True, calibration_manifest=manifest,
    )

    assert len(rows) == 18
    assert (root / "comparison.csv").is_file()
