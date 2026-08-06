"""
feature_selection.py
====================
Mutual Information Feature Selection — Shared by Tier 1 and Tier 2.

Single source of truth for all MI-based feature selection.

Provides:
  fit_mi_selector(X, y, k, sample_frac, random_state)
      Fit MI on the full data (optionally via stratified sampling).
      Used by Tier 1 trainers for final retrain and by Tier 2 scripts.

  apply_feature_selection(X, selector)
      Apply a fitted selector to transform new data without refitting.
"""

import numpy as np
import gc
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split


def fit_mi_selector(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 15,
    sample_frac: float = 0.0,
    random_state: int = 42,
) -> SelectKBest:
    """
    Fit a SelectKBest selector using Mutual Information scores.

    Parameters
    ----------
    X            : float32 array  (N, F)   preprocessed feature matrix
    y            : int array      (N,)     encoded labels
    k            : int            number of features to keep (default 15)
    sample_frac  : float          if > 0, fit MI on a stratified sample
                                 (faster for very large datasets)
    random_state : int

    Returns
    -------
    selector : SelectKBest  fitted on the (possibly sampled) data
    """
    if sample_frac > 0 and sample_frac < 1.0:
        X_fit, _, y_fit, _ = train_test_split(
            X, y, train_size=sample_frac, stratify=y,
            random_state=random_state,
        )
    else:
        X_fit, y_fit = X, y

    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_fit.shape[1]))
    selector.fit(X_fit, y_fit)

    if sample_frac > 0 and sample_frac < 1.0:
        del X_fit, y_fit
        gc.collect()

    return selector


def apply_feature_selection(
    X: np.ndarray,
    selector: SelectKBest,
) -> np.ndarray:
    """
    Apply a previously fitted selector to new data (e.g. test set).
    """
    return selector.transform(X)
