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
  python main.py                        # default data/raw/
  python main.py --data-dir /path/csv   # custom raw data path
  python main.py --balancer smote        # use regular SMOTE instead
  python main.py --cap 15000             # cap each class before SMOTE (speed/RAM)
  python main.py --quick 200000          # verify pipeline on a stratified sample
  python main.py --skip-plots            # skip matplotlib output
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd

# Windows consoles default to cp1252; reconfigure so Unicode output never crashes.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from src.preprocessing            import load_and_preprocess
from src.dimensionality_reduction import split_data
from src.train_hgb                import train_and_evaluate as train_hgb
from src.train_xgboost            import train_and_evaluate as train_xgboost
from src.train_logistic           import train_and_evaluate as train_logistic
from src.evaluation               import (
    save_results,
    save_preprocessing_artifacts,
    print_final_summary,
)
from src.experiment_config        import build_model_config, save_experiment_config


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
        '--balancer', choices=['kmeans', 'smote'], default='kmeans',
        help='Class-balancing strategy (default: kmeans)'
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

    # ------------------------------------------------------------------
    # Phase 3 — Preprocessing  (deterministic, no fit on test)
    # ------------------------------------------------------------------
    X_processed, y_multi, le = load_and_preprocess(data_dir=args.data_dir)
    class_names = list(le.classes_)

    if args.quick > 0 and args.quick < X_processed.shape[0]:
        from sklearn.model_selection import train_test_split
        X_processed, _, y_multi, _ = train_test_split(
            X_processed, y_multi,
            train_size=args.quick, stratify=y_multi, random_state=42,
        )
        print(f"  [quick] Stratified sample: {X_processed.shape[0]:,} rows")

    # ------------------------------------------------------------------
    # Phase 4 — Stratified 80/20 Holdout Split  (NO fitting here)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(X_processed, y_multi)
    del X_processed

    # ------------------------------------------------------------------
    # Phase 5 — Per-fold CV is handled inside each trainer:
    #   MI fit on fold train → transform fold train + val
    #   StandardScaler fit on fold train → transform fold train + val
    #   PCA fit on fold train → transform fold train + val
    #   K-means SMOTE on fold train only
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
    )
    print(f"  [main] LogReg completed in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Output Layer — Model Comparison
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
    )

    # --- Save preprocessing artifacts for each model ---
    for name, res in [('hgb', hgb_results), ('xgboost', xgb_results),
                      ('logistic_regression', lr_results)]:
        artifact_dir = f"artifacts/{name}"
        save_preprocessing_artifacts(
            selector=res['selector'],
            scaler=res['scaler'],
            pca=res['pca'],
            le=le,
            save_dir=artifact_dir,
        )

    # --- Save experiment config for each model ---
    configs = {
        'HGB': ('HGB', args),
        'XGBoost': ('XGBoost', args),
        'LogReg': ('LogReg', args),
    }
    for name, (model_name, a) in configs.items():
        cfg = build_model_config(
            model_name=model_name,
            mi_k=a.mi_k,
            pca_variance=a.pca_variance,
            n_splits=a.n_splits,
            balancer=a.balancer,
            k_neighbors=3,
            rus_cap=a.cap,
        )
        save_experiment_config(cfg, save_dir="results/corrected_pipeline")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
