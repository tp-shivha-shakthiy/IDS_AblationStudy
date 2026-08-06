"""
main.py
=======
INTRUSION DETECTION SYSTEM — Tier 1 Pipeline Orchestrator
==========================================================

The preprocessing layer is configurable while remaining leakage-free:

  Phase 3  Preprocessing             (preprocessing.py)
  Phase 4  Stratified 80/20 Split    (dimensionality_reduction.py)
  Phase 5  CV: configurable MI / PCA / Scaler / KMeans-SMOTE
  Phase 6  HGB Training + Test Eval  (train_hgb.py)
  Phase 7  XGBoost Training + Eval   (train_xgboost.py)
  Phase 8  LogReg Training + Eval    (train_logistic.py)
  Output   Ablation Tables + Configs

Critical invariants (no data leakage):
  ✦ MI selection is fitted on fold-train only (inside CV loop)
  ✦ StandardScaler is fitted on fold-train / full-train only
  ✦ PCA is fitted on fold-train / full-train only
  ✦ K-means SMOTE is applied to fold-train / full-train only — never val or test
  ✦ Test set is locked and never touched until final evaluation

Usage
-----
  python main.py                             # default: MI + PCA + Balancing
  python main.py --experiment raw            # single named preset
  python main.py --experiment mi_pca_balancing
  python main.py --ablation preprocessing    # run all seven presets sequentially
  python main.py --preprocessing raw         # legacy alias (balancing stays ON)
  python main.py --preprocessing all         # legacy alias for every mode
  python main.py --balancer smote            # use regular SMOTE instead
  python main.py --skip-plots                # skip matplotlib output
"""

import argparse

import numpy as np
import pandas as pd

from src.preprocessing import load_and_preprocess
from src.dimensionality_reduction import split_data
from src.train_hgb import train_and_evaluate as train_hgb
from src.train_xgboost import train_and_evaluate as train_xgb
from src.train_logistic import train_and_evaluate as train_logreg
from src.evaluation import save_ablation_tables
from src.experiment_config import (
    build_model_config,
    build_experiment_run_dir,
    save_experiment_config,
)
from src.feature_pipeline import (
    experiment_preset_from_string,
    resolve_experiments,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="UNSW-NB15 Intrusion Detection System (Tier 1)"
    )
    parser.add_argument(
        '--data-dir', default='data/raw',
        help='Directory containing UNSW-NB15_1.csv ... UNSW-NB15_4.csv'
    )
    parser.add_argument(
        '--balancer', choices=['kmeans', 'smote'], default='kmeans',
        help='Class-balancing strategy (default: kmeans)'
    )
    parser.add_argument(
        '--experiment',
        choices=['raw', 'mi', 'mi_balancing', 'pca', 'pca_balancing', 'mi_pca', 'mi_pca_balancing'],
        default=None,
        help='Named preprocessing experiment preset (default: mi_pca_balancing)'
    )
    parser.add_argument(
        '--preprocessing',
        choices=['raw', 'mi', 'pca', 'mi+pca', 'all'],
        default=None,
        help=(
            'Legacy preprocessing alias for backward compatibility; '
            'balancing stays at its default (ON). Use --experiment / '
            '--ablation for the seven official presets.'
        )
    )
    parser.add_argument(
        '--ablation',
        choices=['preprocessing'],
        default=None,
        help='Run the full preprocessing ablation suite'
    )
    parser.add_argument(
        '--n-splits', type=int, default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--mi-k', type=int, default=15,
        help='Top-k MI features to retain per fold (default: 15)'
    )
    parser.add_argument(
        '--pca-variance', type=float, default=0.95,
        help='Cumulative PCA variance to retain (default: 0.95)'
    )
    parser.add_argument(
        '--skip-plots', action='store_true',
        help='Skip saving confusion matrices and feature importance plots'
    )
    parser.add_argument(
        '--results-dir', default='results',
        help='Directory for experiment outputs (default: results)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    X_processed, y_multi, le = load_and_preprocess(data_dir=args.data_dir)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = split_data(X_processed, y_multi)
    del X_processed

    smallest_class = int(np.bincount(y_train).min())
    if args.n_splits < 2 or args.n_splits > smallest_class:
        raise ValueError(
            f"--n-splits must be between 2 and {smallest_class}, the smallest training-class size."
        )
    normal_class_idx = int(np.where(le.classes_ == 'Normal')[0][0])

    experiment_names = resolve_experiments(
        experiment=args.experiment,
        preprocessing=args.preprocessing,
        ablation=args.ablation,
    )

    for experiment_name in experiment_names:
        cfg = experiment_preset_from_string(
            experiment_name,
            default_mi_k=args.mi_k,
            default_pca_n_components=args.pca_variance,
        )
        if cfg.use_mi and not 1 <= args.mi_k <= X_train.shape[1]:
            raise ValueError(
                f"--mi-k must be between 1 and {X_train.shape[1]} for this dataset."
            )
        if cfg.use_pca and not 0 < args.pca_variance <= 1:
            raise ValueError("--pca-variance must be in the interval (0, 1].")

    model_specs = [
        ('HGB', train_hgb),
        ('XGBoost', train_xgb),
        ('LogReg', train_logreg),
    ]

    summary_rows = []
    cv_rows = []

    for experiment_name in experiment_names:
        preprocessing_cfg = experiment_preset_from_string(
            experiment_name,
            default_mi_k=args.mi_k,
            default_pca_n_components=args.pca_variance,
        )
        print(f"\n=== Experiment: {preprocessing_cfg.experiment_name} ===")

        for model_name, trainer in model_specs:
            run_dir = build_experiment_run_dir(
                args.results_dir, model_name, preprocessing_cfg.experiment_name
            )
            results = trainer(
                X_train, y_train, X_test, y_test,
                class_names=class_names,
                n_splits=args.n_splits,
                mi_k=args.mi_k,
                pca_variance=args.pca_variance,
                balancer=args.balancer,
                make_plots=not args.skip_plots,
                normal_class_idx=normal_class_idx,
                use_mi=preprocessing_cfg.use_mi,
                use_pca=preprocessing_cfg.use_pca,
                use_balancing=preprocessing_cfg.use_balancing,
                experiment_name=preprocessing_cfg.experiment_name,
                preprocessing_mode=preprocessing_cfg.mode_name,
                output_dir=run_dir,
            )

            config = build_model_config(
                model_name=model_name,
                mi_k=args.mi_k,
                pca_variance=args.pca_variance,
                n_splits=args.n_splits,
                balancer=args.balancer,
                k_neighbors=3,
                use_mi=preprocessing_cfg.use_mi,
                use_pca=preprocessing_cfg.use_pca,
                use_balancing=preprocessing_cfg.use_balancing,
                experiment_name=preprocessing_cfg.experiment_name,
                preprocessing_mode=preprocessing_cfg.mode_name,
            )
            save_experiment_config(config, save_dir=run_dir)

            test_row = {'Model': model_name, 'Preprocessing': preprocessing_cfg.experiment_name}
            test_row.update(results['test_metrics'])
            summary_rows.append(test_row)

            cv_df = pd.DataFrame(results['cv_metrics'])
            cv_row = {'Model': model_name, 'Preprocessing': preprocessing_cfg.experiment_name}
            cv_row.update({f'cv_{col}': float(cv_df[col].mean()) for col in cv_df.columns})
            cv_rows.append(cv_row)

    save_ablation_tables(summary_rows, cv_rows, args.results_dir, experiment_names)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
