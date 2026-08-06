"""
experiment_config.py
====================
Experiment metadata persistence for reproducible runs.

Generates and saves experiment_config.json containing:
  - Pipeline parameters (seed, split, CV folds, balancer, etc.)
  - Model-specific hyperparameters
  - Git commit hash
  - Timestamp
"""

import os
import json
import datetime
import subprocess


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unavailable'."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unavailable"


def build_experiment_config(
    model_name: str,
    model_params: dict = None,
    experiment_name: str = "mi_pca_balancing",
    preprocessing_mode: str = "mi_pca",
    use_mi: bool = True,
    use_pca: bool = True,
    use_balancing: bool = True,
    mi_k: int = 15,
    pca_variance: float = 0.95,
    n_splits: int = 5,
    balancer: str = "kmeans",
    k_neighbors: int = 3,
    random_state: int = 42,
    test_size: float = 0.20,
    tier: int = 1,
    dl_extra: dict = None,
) -> dict:
    """
    Build an experiment configuration dict with a unified schema.

    Works for both Tier 1 (classical ML) and Tier 2 (DL) models.

    Parameters
    ----------
    model_name    : str   e.g. "XGBoost", "DNN", "LSTM"
    model_params  : dict  model-specific hyperparameters
    mi_k          : int   MI top-k features
    pca_variance  : float cumulative PCA variance (Tier 1) or None (Tier 2)
    n_splits      : int   CV folds
    balancer      : str   balancing strategy
    k_neighbors   : int   SMOTE k_neighbors
    random_state  : int
    test_size     : float holdout fraction
    tier          : int   1 or 2
    dl_extra      : dict  DL-specific fields (architecture, epochs, lr, etc.)

    Returns
    -------
    dict ready for JSON serialization
    """
    config = {
        "model": model_name,
        "tier": tier,
        "seed": random_state,
        "train_test_split": f"{int((1 - test_size) * 100)}/{int(test_size * 100)}",
        "test_size": test_size,
        "cv_folds": n_splits,
        "balancer": balancer,
        "balancer_k_neighbors": k_neighbors,
        "experiment_name": experiment_name,
        "preprocessing_mode": preprocessing_mode,
        "use_mi": use_mi,
        "use_pca": use_pca,
        "use_balancing": use_balancing,
        "feature_selection": "mutual_information",
        "feature_selection_scope": "per_fold_training_data",
        "feature_selection_k": mi_k,
        "scaler": "StandardScaler",
        "scaler_scope": "per_fold_training_data",
        "test_set_locked": True,
        "final_retrain": "full_80_percent_training_set",
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
    }

    if pca_variance is not None:
        config["pca_variance"] = pca_variance
        config["pca_scope"] = "per_fold_training_data"

    if model_params:
        config["model_hyperparameters"] = model_params

    if dl_extra:
        config["dl_extra"] = dl_extra

    return config


def build_experiment_run_dir(
    base_dir: str,
    model_name: str,
    preprocessing_mode: str,
    timestamp: str = None,
) -> str:
    """Create a unique run directory under model/preprocessing mode."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(base_dir, model_name, preprocessing_mode, timestamp)


def save_experiment_config(config: dict, save_dir: str) -> str:
    """
    Save experiment config to JSON file.

    Parameters
    ----------
    config    : dict from build_experiment_config()
    save_dir  : str  directory to save into

    Returns
    -------
    str path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "experiment_config.json")
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Experiment config saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Model-specific parameter presets
# ---------------------------------------------------------------------------

XGBOOST_PARAMS = dict(
    n_estimators=30,
    subsample=0.1,
    max_depth=3,
    min_child_weight=20,
    gamma=0.2,
    learning_rate=0.05,
    colsample_bytree=0.1,
    reg_alpha=0.5,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42,
    verbosity=0,
)

HGB_PARAMS = dict(
    max_iter=30,
    learning_rate=0.05,
    max_depth=5,
    l2_regularization=1.0,
    random_state=42,
)

LOGREG_PARAMS = dict(
    solver='saga',
    max_iter=50,
    random_state=42,
    n_jobs=-1,
)


def build_model_config(model_name: str, **kwargs) -> dict:
    """Build config for a specific model with its actual hyperparameters."""
    param_map = {
        "XGBoost": XGBOOST_PARAMS,
        "HGB": HGB_PARAMS,
        "LogReg": LOGREG_PARAMS,
    }
    params = param_map.get(model_name, {})
    return build_experiment_config(model_name=model_name, model_params=params, **kwargs)
