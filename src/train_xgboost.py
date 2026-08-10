"""
train_xgboost.py
================
XGBoost — thin wrapper around src/model_training.py.

Preserves the original API for main.py.
"""

from src.model_training import train_and_evaluate as _shared_train

MODEL_NAME = "XGBoost"


def train_and_evaluate(X_train, y_train, X_test, y_test, class_names,
                       n_splits=5, mi_k=15, pca_variance=0.95,
                       k_neighbors=3, random_state=42, rus_cap=0,
                       fold_cache=None, use_mi=True, use_pca=True,
                       use_balancing=True, experiment="mi_pca_balancing"):
    """Full XGBoost pipeline via shared infrastructure."""
    return _shared_train(
        MODEL_NAME,
        X_train, y_train, X_test, y_test, class_names,
        n_splits=n_splits, mi_k=mi_k, pca_variance=pca_variance,
        k_neighbors=k_neighbors, random_state=random_state,
        rus_cap=rus_cap, fold_cache=fold_cache,
        use_mi=use_mi, use_pca=use_pca, use_balancing=use_balancing,
        experiment=experiment,
    )
