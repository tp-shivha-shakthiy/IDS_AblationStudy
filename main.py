"""
main.py
=======
INTRUSION DETECTION SYSTEM — Tier 1 Pipeline Orchestrator
==========================================================

Correct (leakage-free) pipeline:

  Phase 3  Preprocessing             (preprocessing.py)
  Phase 4  Stratified 80/20 Split    (dimensionality_reduction.py)
  Phase 5  CV: MI→Scaler→PCA→Kmeans-SMOTE  (cross_validation.py) — per fold
  Phase 6  HGB Training + Test Eval  (train_hgb.py)
  Phase 7  XGBoost Training + Eval   (train_xgboost.py)
  Phase 8  LogReg Training + Eval    (train_logistic.py)
  Output   Evaluation + Plots        (evaluation.py)

Critical invariants (no data leakage):
  ✦ MI selection is fitted on fold-train only (inside CV loop)
  ✦ StandardScaler is fitted on fold-train / full-train only
  ✦ PCA is fitted on fold-train / full-train only
  ✦ K-means SMOTE is applied to fold-train / full-train only — never val or test
  ✦ Test set is locked and never touched until final evaluation

Usage
-----
  python main.py                        # default: mi_pca_balancing experiment
  python main.py --data-dir /path/csv   # custom raw data path
  python main.py --experiment raw       # ablation preset (7 available)
   python main.py --aggregate-ablation    # validate and build final 7-row tables
  python main.py --cap 15000             # cap each class before SMOTE (speed/RAM)
  python main.py --quick 200000          # verify pipeline on a stratified sample
  python main.py --skip-plots            # skip matplotlib output
"""

import argparse
import sys
import time
import os
import numpy as np
import pandas as pd

# Windows consoles default to cp1252; reconfigure so Unicode output never crashes.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from src.preprocessing            import load_and_prepare
from src.dimensionality_reduction import split_data
from src.train_hgb                import train_and_evaluate as train_hgb
from src.train_xgboost            import train_and_evaluate as train_xgboost
from src.train_logistic           import train_and_evaluate as train_logistic
from src.evaluation               import (
    save_results,
    save_preprocessing_artifacts,
    print_final_summary,
    save_model_ablation_tables,
    build_model_ablation_rows,
)
from src.experiment_config        import (
    ABLATION_PRESETS,
    ABLATION_DISPLAY_NAMES,
    resolve_experiment,
    save_experiment_config,
    build_model_config,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="UNSW-NB15 Intrusion Detection System (Tier 1)"
    )
    parser.add_argument(
        '--data-dir', default='data/raw',
        help='Directory containing UNSW-NB15_1.csv … UNSW-NB15_4.csv'
    )
    parser.add_argument(
        '--experiment', choices=list(ABLATION_PRESETS.keys()), default=None,
        help='Ablation preset. Exactly one of: '
             + ', '.join(ABLATION_PRESETS.keys())
             + ' (default: mi_pca_balancing).'
    )
    parser.add_argument(
        '--aggregate-ablation', action='store_true',
        help='Validate all seven experiments and generate final ablation tables'
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
        '--cap', type=int, default=0,
        help='Cap each class to N samples before oversampling '
             '(0 = no cap; e.g. 15000 for fast runs within 20GB RAM)'
    )
    parser.add_argument(
        '--quick', type=int, default=0,
        help='Run on a stratified random sample of N rows '
             '(e.g. --quick 200000) to verify the pipeline end-to-end'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.aggregate_ablation:
        model_names = ('HGB', 'XGBoost', 'LogReg')
        # Validate every model before writing any table for this aggregation run.
        for name in model_names:
            build_model_ablation_rows(name, experiments_root="results")
        for name in model_names:
            save_model_ablation_tables(name, results_root="results")
        print("\nAblation comparison tables generated.")
        return

    # Resolve the ablation preset (default = current behaviour, backward compat)
    experiment = args.experiment or "mi_pca_balancing"
    flags = resolve_experiment(experiment)
    use_mi, use_pca, use_balancing = (
        flags["use_mi"], flags["use_pca"], flags["use_balancing"],
    )
    print(f"\n  Experiment: {experiment}  "
          f"(MI={'on' if use_mi else 'off'}, "
          f"PCA={'on' if use_pca else 'off'}, "
          f"KMeansSMOTE={'on' if use_balancing else 'off'})")

    # Presets are the single source of truth for preprocessing. Refuse
    # contradictory low-level knobs so an experiment can't be silently altered.
    if experiment != "mi_pca_balancing" and (
        args.mi_k != 15 or args.pca_variance != 0.95
    ):
        raise ValueError(
            f"--experiment {experiment} fixes the preprocessing preset. "
            "Remove --mi-k / --pca-variance overrides."
        )

    # ------------------------------------------------------------------
    # Phase 3 — Preprocessing  (deterministic, no fit on test)
    # ------------------------------------------------------------------
    X_raw, y_multi, le = load_and_prepare(data_dir=args.data_dir)
    class_names = list(le.classes_)

    if args.quick > 0 and args.quick < X_raw.shape[0]:
        from sklearn.model_selection import train_test_split
        X_raw, _, y_multi, _ = train_test_split(
            X_raw, y_multi,
            train_size=args.quick, stratify=y_multi, random_state=42,
        )
        print(f"  [quick] Stratified sample: {X_raw.shape[0]:,} rows")

    # ------------------------------------------------------------------
    # Phase 4 — Stratified 80/20 Holdout Split  (NO fitting here)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(X_raw, y_multi)
    del X_raw

    # ------------------------------------------------------------------
    # Phase 5 — Per-fold CV is handled inside each trainer:
    #   MI fit on fold train → transform fold train + val   (if use_mi)
    #   StandardScaler fit on fold train → transform both    (always)
    #   PCA fit on fold train → transform fold train + val  (if use_pca)
    #   K-means SMOTE on fold train only                     (if use_balancing)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 6 — HistGradientBoosting  (CV + retrain + test eval)
    # ------------------------------------------------------------------
    fold_cache = {}   # shared so HGB/XGBoost/LogReg reuse preprocessed folds
    t0 = time.time()
    hgb_results = train_hgb(
        X_train, y_train, X_test, y_test,
        class_names=class_names,
        n_splits=args.n_splits,
        mi_k=args.mi_k,
        pca_variance=args.pca_variance,
        rus_cap=args.cap,
        fold_cache=fold_cache,
        use_mi=use_mi,
        use_pca=use_pca,
        use_balancing=use_balancing,
        experiment=experiment,
    )
    print(f"  [main] HGB completed in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Phase 7 — XGBoost  (CV + retrain + test eval)
    # ------------------------------------------------------------------
    t0 = time.time()
    xgb_results = train_xgboost(
        X_train, y_train, X_test, y_test,
        class_names=class_names,
        n_splits=args.n_splits,
        mi_k=args.mi_k,
        pca_variance=args.pca_variance,
        rus_cap=args.cap,
        fold_cache=fold_cache,
        use_mi=use_mi,
        use_pca=use_pca,
        use_balancing=use_balancing,
        experiment=experiment,
    )
    print(f"  [main] XGBoost completed in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Phase 8 — Logistic Regression  (CV + retrain + test eval)
    # ------------------------------------------------------------------
    t0 = time.time()
    lr_results = train_logistic(
        X_train, y_train, X_test, y_test,
        class_names=class_names,
        n_splits=args.n_splits,
        mi_k=args.mi_k,
        pca_variance=args.pca_variance,
        rus_cap=args.cap,
        fold_cache=fold_cache,
        use_mi=use_mi,
        use_pca=use_pca,
        use_balancing=use_balancing,
        experiment=experiment,
    )
    print(f"  [main] LogReg completed in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Output Layer — Model Comparison (per experiment)
    # ------------------------------------------------------------------
    all_test_results = []
    cv_results = {}
    y_pred_dict = {}

    for name, res in [('HGB', hgb_results), ('XGBoost', xgb_results),
                      ('LogReg', lr_results)]:
        row = {'Model': name}
        row.update(res['test_metrics'])
        all_test_results.append(pd.DataFrame([row]))
        cv_results[name] = pd.DataFrame(res['cv_metrics'])
        y_pred_dict[name] = res['y_test_pred']

    print_final_summary(all_test_results)
    save_results(
        all_test_results, cv_results,
        y_true=y_test, y_pred_dict=y_pred_dict,
        class_names=class_names,
        results_dir=os.path.join("results", experiment),
    )

    # --- Save preprocessing artifacts (incl. label encoder) per experiment ---
    model_dirs = {
        'HGB': hgb_results,
        'XGBoost': xgb_results,
        'LogReg': lr_results,
    }
    for name, res in model_dirs.items():
        save_preprocessing_artifacts(
            selector=res['selector'],
            scaler=res['scaler'],
            pca=res['pca'],
            le=le,
            categorical_encoder=res['categorical_encoder'],
            save_dir=res['save_dir'],
        )

    # --- Save experiment config for the experiment (per model) ---
    for name, res in model_dirs.items():
        cfg = build_model_config(
            model_name=name,
            experiment_name=experiment,
            preprocessing_mode=experiment,
            use_mi=use_mi,
            use_pca=use_pca,
            use_balancing=use_balancing,
            mi_k=args.mi_k,
            pca_variance=args.pca_variance,
            n_splits=args.n_splits,
            balancer="kmeans",
            k_neighbors=3,
            rus_cap=args.cap,
        )
        save_experiment_config(cfg, save_dir=res['save_dir'])

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
