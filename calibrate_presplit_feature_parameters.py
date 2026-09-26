"""Run the complete-dataset, classifier-free presplit calibration stage."""

import argparse
from pathlib import Path

from src.preprocessing import load_and_prepare
from src.presplit_feature_calibration import (
    calibrate_presplit_feature_parameters,
    write_calibration_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--manifest",
        default="configs/presplit_feature_calibration.json",
    )
    parser.add_argument(
        "--results-dir",
        default="results/feature_calibration",
    )
    parser.add_argument(
        "--historical-spearman",
        default="results/feature_calibration/spearman_calibration.json",
    )
    args = parser.parse_args()

    X, y, encoder = load_and_prepare(args.data_dir)
    dataset_files = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
        }
        for path in sorted(Path(args.data_dir).glob("UNSW-NB15_*.csv"))
    ]
    manifest = calibrate_presplit_feature_parameters(
        X,
        y,
        list(encoder.classes_),
        dataset_files=dataset_files,
        historical_spearman_path=args.historical_spearman,
    )
    plot_paths = write_calibration_artifacts(
        manifest,
        args.manifest,
        args.results_dir,
    )
    print(f"manifest: {args.manifest}")
    print(f"MI k: {manifest['mi']['selected_k']}")
    print(f"PCA components: {manifest['pca']['selected_component_count']}")
    print(f"Spearman k: {manifest['spearman']['selected_k']}")
    print(f"plots: {len(plot_paths)}")


if __name__ == "__main__":
    main()
