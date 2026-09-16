"""Read completed presplit-ablation artifacts and write one comparison CSV."""

import argparse
import csv
import json
from pathlib import Path

from src.presplit_feature_ablation_protocol import RESULTS_ROOT


COLUMNS = ("experiment", "model", "final_dimension", "multiclass_accuracy", "macro_f1", "weighted_f1",
           "macro_recall", "binary_accuracy", "binary_precision", "binary_recall", "binary_f1", "binary_auc",
           "feature_preprocessing_seconds", "balancing_seconds", "training_seconds", "model_load_seconds",
           "cold_start_prediction_ms", "single_sample_mean_ms", "single_sample_median_ms", "single_sample_p95_ms",
           "warm_inference_ms_per_sample", "throughput_samples_per_second", "model_size_mb")
REQUIRED = ("test_metrics.json", "latency.json", "feature_dimensions.json", "resolved_config.json", "environment.json")


def _read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_completed_results(results_root=RESULTS_ROOT):
    root = Path(results_root)
    rows = []
    if root.exists():
        for directory in root.glob("*/*"):
            if not directory.is_dir() or not all((directory / name).is_file() for name in REQUIRED):
                continue
            metrics, latency, dimensions = (_read(directory / name) for name in REQUIRED[:3])
            row = {"experiment": directory.parent.name, "model": directory.name,
                   "final_dimension": dimensions.get("final_dimension")}
            row.update({"multiclass_accuracy": metrics.get("multiclass_accuracy", metrics.get("multi_acc")),
                        "macro_f1": metrics.get("macro_f1"), "weighted_f1": metrics.get("weighted_f1"),
                        "macro_recall": metrics.get("macro_recall"),
                        "binary_accuracy": metrics.get("binary_accuracy", metrics.get("binary_acc")),
                        "binary_precision": metrics.get("binary_precision"), "binary_recall": metrics.get("binary_recall"),
                        "binary_f1": metrics.get("binary_f1"), "binary_auc": metrics.get("binary_auc")})
            row.update({name: latency.get(name) for name in COLUMNS if name in latency})
            rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    args = parser.parse_args()
    print(f"completed rows: {len(aggregate_completed_results(args.results_root))}")
