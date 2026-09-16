"""CLI for the explicitly non-leakage-free presplit ablation protocol."""
import argparse

from src.preprocessing import load_and_prepare
from src.presplit_feature_ablation_protocol import (
    EXPERIMENTS, SPEARMAN_SELECTION_RULES, RESULTS_ROOT,
    run_presplit_ablation_experiment, validate_spearman_selection,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--model", choices=("HGB", "LogReg", "XGBoost", "DNN", "LSTM", "BiLSTM", "BiLSTMSharedFeature"), default="HGB")
    parser.add_argument("--quick", type=int, default=0)
    parser.add_argument("--cap", type=int, default=0)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--spearman-selection-rule", choices=SPEARMAN_SELECTION_RULES)
    parser.add_argument("--spearman-selection-value", type=float)
    parser.add_argument("--aggregate", action="store_true", help="Aggregate completed runs only; never trains.")
    args = parser.parse_args()
    if args.aggregate:
        from aggregate_presplit_feature_ablation import aggregate_completed_results
        print(len(aggregate_completed_results(args.results_root)))
        return
    if args.experiment is None:
        parser.error("--experiment is required unless --aggregate is used.")
    if args.experiment == "E3_spearman":
        try:
            validate_spearman_selection(args.spearman_selection_rule, args.spearman_selection_value)
        except ValueError as exc:
            parser.error(str(exc))
    X, y, encoder = load_and_prepare(args.data_dir)
    if args.quick and args.quick < len(X):
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=args.quick, stratify=y, random_state=42)
    result = run_presplit_ablation_experiment(
        X, y, list(encoder.classes_), args.experiment,
        model_name=args.model, rus_cap=args.cap, results_root=args.results_root,
        spearman_selection_rule=args.spearman_selection_rule,
        spearman_selection_value=args.spearman_selection_value,
    )
    print(result["save_dir"])


if __name__ == "__main__":
    main()
