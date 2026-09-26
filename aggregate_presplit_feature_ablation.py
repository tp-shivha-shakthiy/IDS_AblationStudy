"""Read completed presplit-ablation artifacts and write one comparison CSV."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.presplit_feature_ablation_protocol import CALIBRATED_RESULTS_ROOT


COLUMNS = ("experiment", "model", "final_dimension", "multiclass_accuracy", "macro_f1", "weighted_f1",
           "macro_recall", "binary_accuracy", "binary_precision", "binary_recall", "binary_f1", "binary_auc",
           "feature_preprocessing_seconds", "balancing_seconds", "training_seconds", "model_load_seconds",
           "cold_start_prediction_ms", "single_sample_mean_ms", "single_sample_median_ms", "single_sample_p95_ms",
           "warm_inference_ms_per_sample", "throughput_samples_per_second", "model_size_mb")
REQUIRED = ("test_metrics.json", "latency.json", "feature_dimensions.json", "resolved_config.json", "environment.json")
FINAL_EXPERIMENTS = ("E2_parallel_mi_pca", "E4_mi_only", "E5_pca_only")
FINAL_MODELS = ("HGB", "LogReg", "XGBoost", "DNN", "LSTM", "BiLSTM")


def _read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_contract(path):
    manifest_path = Path(path)
    payload = _read(manifest_path)
    return {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "mi_k": int(payload["mi"]["selected_k"]),
        "pca_components": int(payload["pca"]["selected_component_count"]),
        "dataset_fingerprint": payload["dataset"]["prepared_feature_fingerprint"],
    }


def _validate_final_study(records, contract):
    expected = {
        (experiment, model)
        for experiment in FINAL_EXPERIMENTS
        for model in FINAL_MODELS
    }
    received = {(record["experiment"], record["model"]) for record in records}
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    duplicates = len(received) != len(records)
    if missing or unexpected or duplicates:
        raise ValueError(
            "Final E2/E4/E5 study is incomplete or contains unexpected runs: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}."
        )
    for record in records:
        calibration = record["config"].get("calibration", {})
        required = {
            "manifest_sha256": contract["manifest_sha256"],
            "dataset_fingerprint": contract["dataset_fingerprint"],
            "mi_k": contract["mi_k"],
            "pca_components": contract["pca_components"],
            "pca_parameterization": "component_count",
        }
        mismatches = {
            key: (calibration.get(key), expected_value)
            for key, expected_value in required.items()
            if calibration.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(
                f"{record['experiment']}/{record['model']} does not consume "
                f"the frozen calibration manifest: {mismatches}."
            )


def aggregate_completed_results(
    results_root=CALIBRATED_RESULTS_ROOT,
    *,
    require_complete=False,
    calibration_manifest="configs/presplit_feature_calibration.json",
):
    root = Path(results_root)
    records = []
    if root.exists():
        for directory in root.glob("*/*"):
            if not directory.is_dir() or not all((directory / name).is_file() for name in REQUIRED):
                continue
            metrics, latency, dimensions, config = (
                _read(directory / name) for name in REQUIRED[:4]
            )
            row = {"experiment": directory.parent.name, "model": directory.name,
                   "final_dimension": dimensions.get("final_dimension")}
            row.update({"multiclass_accuracy": metrics.get("multiclass_accuracy", metrics.get("multi_acc")),
                        "macro_f1": metrics.get("macro_f1"), "weighted_f1": metrics.get("weighted_f1"),
                        "macro_recall": metrics.get("macro_recall"),
                        "binary_accuracy": metrics.get("binary_accuracy", metrics.get("binary_acc")),
                        "binary_precision": metrics.get("binary_precision"), "binary_recall": metrics.get("binary_recall"),
                        "binary_f1": metrics.get("binary_f1"), "binary_auc": metrics.get("binary_auc")})
            row.update({name: latency.get(name) for name in COLUMNS if name in latency})
            records.append({**row, "config": config})
    experiment_order = {name: index for index, name in enumerate(FINAL_EXPERIMENTS)}
    model_order = {name: index for index, name in enumerate(FINAL_MODELS)}
    records.sort(
        key=lambda record: (
            experiment_order.get(record["experiment"], len(experiment_order)),
            model_order.get(record["model"], len(model_order)),
            record["experiment"],
            record["model"],
        )
    )
    if require_complete:
        _validate_final_study(records, _manifest_contract(calibration_manifest))
    rows = [{key: value for key, value in record.items() if key != "config"} for record in records]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=CALIBRATED_RESULTS_ROOT)
    parser.add_argument("--calibration-manifest", default="configs/presplit_feature_calibration.json")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require exactly E2/E4/E5 x HGB/LogReg/XGBoost/DNN/LSTM/BiLSTM with one frozen manifest.",
    )
    args = parser.parse_args()
    rows = aggregate_completed_results(
        args.results_root,
        require_complete=args.require_complete,
        calibration_manifest=args.calibration_manifest,
    )
    print(f"completed rows: {len(rows)}")
