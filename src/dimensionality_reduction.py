"""
dimensionality_reduction.py
============================
Phase 4b + 5 -- Stratified 80/20 Holdout Split

Splits the preprocessed feature matrix into an 80 % training set and
a 20 % locked test set.  No transformers are fitted here; StandardScaler
and PCA are fitted inside the cross-validation loop and on the full
training set at final-retrain time.

The old code fitted StandardScaler and PCA on the *full* dataset before
splitting -- that was data leakage.  This module now performs only the
deterministic, leakage-free split.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """
    Stratified 80/20 train / test split.

    Parameters
    ----------
    X            : float array  (N, F)   preprocessed feature matrix
    y            : int array    (N,)     encoded labels
    test_size    : float        fraction held out for final evaluation
    random_state : int

    Returns
    -------
    X_train  : np.ndarray  (0.8N, F)
    X_test   : np.ndarray  (0.2N, F)
    y_train  : np.ndarray
    y_test   : np.ndarray
    """
    print("\n=== Phase 4b: Stratified 80/20 Holdout Split ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
