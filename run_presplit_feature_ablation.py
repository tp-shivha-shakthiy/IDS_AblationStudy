"""CLI for the explicitly non-leakage-free presplit ablation protocol."""
import argparse

from src.preprocessing import load_and_prepare
from src.presplit_feature_ablation_protocol import (
    CALIBRATED_RESULTS_ROOT, EXPERIMENTS,
    run_presplit_ablation_experiment,
)


DEFAULT_CALIBRATION_MANIFEST = "configs/presplit_feature_calibration.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--model", choices=("HGB", "LogReg", "XGBoost", "DNN", "LSTM", "BiLSTM", "BiLSTMSharedFeature"), default="HGB")
    parser.add_argument("--cap", type=int, default=0)
    parser.add_argument("--results-root", default=CALIBRATED_RESULTS_ROOT)
    parser.add_argument(
        "--calibration-manifest", default=DEFAULT_CALIBRATION_MANIFEST,
        help="Frozen full-data feature calibration required for final E1-E5 runs.",
    )
    parser.add_argument("--aggregate", action="store_true", help="Aggregate completed runs only; never trains.")
    parser.add_argument("--resume", action="store_true", help="Reuse only compatible completed CV fold checkpoints.")
    args = parser.parse_args()
    if args.aggregate:
        from aggregate_presplit_feature_ablation import aggregate_completed_results
        print(len(aggregate_completed_results(args.results_root)))
        return
    if args.experiment is None:
        parser.error("--experiment is required unless --aggregate is used.")
    X, y, encoder = load_and_prepare(args.data_dir)
    result = run_presplit_ablation_experiment(
        X, y, list(encoder.classes_), args.experiment,
        model_name=args.model, rus_cap=args.cap, results_root=args.results_root,
        calibration_manifest=args.calibration_manifest,
        resume=args.resume,
    )
    print(result["save_dir"])


if __name__ == "__main__":
    main()
