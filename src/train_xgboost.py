"""
train_xgboost.py
================
Phase 8 + 9 — XGBoost Classifier

Phase 8: 5-fold cross-validation on balanced folds.
Phase 9: Final model trained on fold-0 balanced pool, evaluated on the
         blind 20 % holdout test set.

tree_method='hist' mirrors the XGBoost fast-path used in the notebook
and prevents RAM crashes on large datasets.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

from src.cross_validation import run_cv


def train_xgboost(
    balanced_folds: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normal_class_idx: int,
    class_names: list,
    num_class: int = 10,
    random_state: int = 42,
) -> tuple:
    """
    CV training (Phase 8) + blind-test evaluation (Phase 9).

    Parameters
    ----------
    balanced_folds   : list of fold dicts
    X_test           : float array  (0.2N, F)  untouched holdout
    y_test           : int array    (0.2N,)
    normal_class_idx : int
    class_names      : list of str
    num_class        : int   total number of output classes
    random_state     : int

    Returns
    -------
    cv_results_df    : pd.DataFrame  per-fold + mean CV metrics
    test_results_df  : pd.DataFrame  single-row blind test metrics
    final_model      : fitted XGBClassifier
    """
    print("=== Phase 8: XGBoost — 5-Fold Cross-Validation ===")

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_class,
        tree_method='hist',
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    cv_results_df, _ = run_cv(
        model=xgb_model,
        balanced_folds=balanced_folds,
        normal_class_idx=normal_class_idx,
        model_name="XGBoost",
    )

    print("\n  Cross-Validation Results (XGBoost):")
    print(cv_results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Phase 9 — Final model on blind test set
    # ------------------------------------------------------------------
    print("\n=== Phase 9: XGBoost — Final Blind-Test Evaluation ===")

    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_class,
        tree_method='hist',
        max_depth=3,
        n_estimators=30,
        subsample=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    X_tr_final = balanced_folds[0]['X_train_fold']
    y_tr_final = balanced_folds[0]['y_train_fold']
    print(f"  Training on balanced fold-0 pool "
          f"({X_tr_final.shape[0]:,} samples) ...")
    final_model.fit(X_tr_final, y_tr_final)

    y_pred_multi = final_model.predict(X_test)

    # Multi-class metrics
    test_multi_acc   = accuracy_score(y_test, y_pred_multi)
    test_macro_f1    = f1_score(y_test, y_pred_multi, average='macro',
                                 zero_division=0)
    test_weighted_f1 = f1_score(y_test, y_pred_multi, average='weighted',
                                 zero_division=0)

    # Binary metrics
    y_test_bin  = np.where(y_test == normal_class_idx, 0, 1)
    y_pred_bin  = np.where(y_pred_multi == normal_class_idx, 0, 1)
    test_bin_acc = accuracy_score(y_test_bin, y_pred_bin)
    test_bin_f1  = f1_score(y_test_bin, y_pred_bin, average='binary',
                             zero_division=0)

    test_results_df = pd.DataFrame([{
        'Model':       'XGBoost',
        'Binary Acc':  test_bin_acc,
        'Binary F1':   test_bin_f1,
        'Multi-Acc':   test_multi_acc,
        'Macro F1':    test_macro_f1,
        'Weighted F1': test_weighted_f1,
    }])

    print("\n  Blind Test Set Results (XGBoost):")
    print(test_results_df.to_string(index=False))
    print("\n  Per-class report:")
    print(classification_report(y_test, y_pred_multi,
                                 target_names=class_names, zero_division=0))

    return cv_results_df, test_results_df, final_model
