"""
train_hgb.py
============
Phase 7 — HistGradientBoostingClassifier

HistGradientBoostingClassifier uses feature-binning (LightGBM-style)
making it 10x–100x faster than Random Forest on datasets with millions
of rows. Ideal as the fast ML baseline on UNSW-NB15.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.cross_validation import run_cv


def train_hgb(
    balanced_folds: list,
    normal_class_idx: int,
    class_names: list,
    max_iter: int = 30,
    random_state: int = 42,
) -> tuple:
    """
    Run 5-fold CV with HistGradientBoostingClassifier.

    Parameters
    ----------
    balanced_folds    : list of fold dicts from balancing.py
    normal_class_idx  : int
    class_names       : list of str   (le.classes_)
    max_iter          : int           boosting rounds
    random_state      : int

    Returns
    -------
    results_df  : pd.DataFrame   per-fold + mean metrics
    model       : fitted HGB estimator (last fold)
    """
    print("=== Phase 7: HistGradientBoostingClassifier ===")

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        random_state=random_state,
    )

    results_df, model = run_cv(
        model=clf,
        balanced_folds=balanced_folds,
        normal_class_idx=normal_class_idx,
        model_name="HGB",
    )

    print("\n  Cross-Validation Results (HistGradientBoosting):")
    print(results_df.to_string(index=False))

    # Detailed report on fold 1 validation split
    fold0 = balanced_folds[0]
    clf_fold1 = HistGradientBoostingClassifier(
        max_iter=max_iter, random_state=random_state
    )
    clf_fold1.fit(fold0['X_train_fold'], fold0['y_train_fold'])
    y_pred_f1 = clf_fold1.predict(fold0['X_val_fold'])
    print("\n  Detailed Classification Report (Fold 1 Validation):")
    print(classification_report(
        fold0['y_val_fold'], y_pred_f1, target_names=class_names,
        zero_division=0,
    ))

    return results_df, model
