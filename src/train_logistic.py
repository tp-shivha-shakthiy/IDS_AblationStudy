"""
train_logistic.py
=================
Phase 10 — Logistic Regression (multinomial / saga)

'saga' solver handles large-scale multi-class data efficiently and
supports L1/L2 regularisation. n_jobs=-1 parallelises across all cores.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

from src.cross_validation import run_cv


def train_logistic(
    balanced_folds: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normal_class_idx: int,
    class_names: list,
    max_iter: int = 50,
    random_state: int = 42,
) -> tuple:
    """
    CV training + blind-test evaluation for Logistic Regression.

    Parameters
    ----------
    balanced_folds   : list of fold dicts
    X_test           : float array  holdout features
    y_test           : int array    holdout labels
    normal_class_idx : int
    class_names      : list of str
    max_iter         : int   saga iterations per fold
    random_state     : int

    Returns
    -------
    cv_results_df    : pd.DataFrame  per-fold + mean CV metrics
    test_results_df  : pd.DataFrame  blind test metrics
    final_model      : fitted LogisticRegression
    """
    print("=== Phase 10: Logistic Regression — 5-Fold Cross-Validation ===")

    lr_model = LogisticRegression(
        multi_class='multinomial',
        solver='saga',
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )

    cv_results_df, _ = run_cv(
        model=lr_model,
        balanced_folds=balanced_folds,
        normal_class_idx=normal_class_idx,
        model_name="LogReg",
    )

    print("\n  Cross-Validation Results (Logistic Regression):")
    print(cv_results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Final blind-test evaluation
    # ------------------------------------------------------------------
    print("\n  Training final LR model on fold-0 balanced pool ...")
    final_model = LogisticRegression(
        multi_class='multinomial',
        solver='saga',
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(
        balanced_folds[0]['X_train_fold'],
        balanced_folds[0]['y_train_fold'],
    )

    y_pred_multi = final_model.predict(X_test)

    test_multi_acc   = accuracy_score(y_test, y_pred_multi)
    test_macro_f1    = f1_score(y_test, y_pred_multi, average='macro',
                                 zero_division=0)
    test_weighted_f1 = f1_score(y_test, y_pred_multi, average='weighted',
                                 zero_division=0)

    y_test_bin  = np.where(y_test == normal_class_idx, 0, 1)
    y_pred_bin  = np.where(y_pred_multi == normal_class_idx, 0, 1)
    test_bin_acc = accuracy_score(y_test_bin, y_pred_bin)
    test_bin_f1  = f1_score(y_test_bin, y_pred_bin, average='binary',
                             zero_division=0)

    test_results_df = pd.DataFrame([{
        'Model':       'LogisticRegression',
        'Binary Acc':  test_bin_acc,
        'Binary F1':   test_bin_f1,
        'Multi-Acc':   test_multi_acc,
        'Macro F1':    test_macro_f1,
        'Weighted F1': test_weighted_f1,
    }])

    print("\n  Blind Test Set Results (Logistic Regression):")
    print(test_results_df.to_string(index=False))

    return cv_results_df, test_results_df, final_model
