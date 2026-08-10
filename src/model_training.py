"""
model_training.py
=================
Shared Tier 1 training pipeline for all classical ML models.

Provides:
  train_and_evaluate(model_name, X_train, y_train, X_test, y_test, class_names, ...)
      Full pipeline: CV → final retrain → test eval → save artifacts

  MODEL_REGISTRY
      Central dict mapping model names to (class, params, extra_artifacts).

Individual trainer scripts (train_hgb.py, train_xgboost.py, train_logistic.py)
thin-wrap this function, preserving their original API for main.py.
"""

import numpy as np
import gc
import os
import json
import warnings
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score)

from src.cross_validation import run_cv
from src.balancing import balance_full_train
from src.feature_selection import fit_mi_selector
from src.preprocessing import fit_categorical_encoder, transform_features
from src.evaluation import (plot_confusion_matrix, plot_roc_curve,
                            plot_feature_importance, compute_extended_metrics)
from src.experiment_config import build_experiment_config, save_experiment_config


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "HGB": {
        "model_class_path": "sklearn.ensemble.HistGradientBoostingClassifier",
        "model_class": None,  # lazy-imported below
        "params": dict(
            max_iter=30, learning_rate=0.05, max_depth=5,
            l2_regularization=1.0, random_state=42,
        ),
        "display_name": "HistGradientBoosting",
        "plot_feature_importance": False,
    },
    "XGBoost": {
        "model_class_path": "xgboost.XGBClassifier",
        "model_class": None,
        "params": dict(
            n_estimators=30, subsample=0.1, max_depth=3, min_child_weight=20,
            gamma=0.2, learning_rate=0.05, colsample_bytree=0.1, reg_alpha=0.5,
            use_label_encoder=False, eval_metric='mlogloss',
            tree_method='hist', random_state=42, verbosity=0,
        ),
        "display_name": "XGBoost",
        "plot_feature_importance": True,
    },
    "LogReg": {
        "model_class_path": "sklearn.linear_model.LogisticRegression",
        "model_class": None,
        "params": dict(
            solver='saga',
            max_iter=50, random_state=42, n_jobs=-1,
        ),
        "display_name": "Logistic Regression (saga / multinomial)",
        "plot_feature_importance": False,
    },
}


def _import_model_class(path: str):
    """Import a model class from its dotted path."""
    module_path, class_name = path.rsplit('.', 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _ensure_registry():
    """Lazy-import all model classes in the registry."""
    for key, entry in MODEL_REGISTRY.items():
        if entry["model_class"] is None:
            entry["model_class"] = _import_model_class(entry["model_class_path"])


# ---------------------------------------------------------------------------
# Single-model train + eval (used inside CV folds and final retrain)
# ---------------------------------------------------------------------------

def _train_and_evaluate(model_class, model_params, X_tr, y_tr, X_val, y_val):
    """Train a model on balanced data, return (model, metrics, y_pred)."""
    model = model_class(**model_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred,
                                     average='weighted', zero_division=0),
        'recall': recall_score(y_val, y_pred,
                               average='weighted', zero_division=0),
        'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0),
    }
    try:
        metrics['auc'] = roc_auc_score(y_val, model.predict_proba(X_val),
                                       multi_class='ovr', average='weighted')
    except Exception:
        metrics['auc'] = 0.0
    return model, metrics, y_pred


# ---------------------------------------------------------------------------
# Full pipeline: CV → final retrain → test eval → artifacts
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    n_splits: int = 5,
    mi_k: int = 15,
    pca_variance: float = 0.95,
    k_neighbors: int = 3,
    random_state: int = 42,
    rus_cap: int = 0,
    fold_cache: dict = None,
    use_mi: bool = True,
    use_pca: bool = True,
    use_balancing: bool = True,
    experiment: str = "mi_pca_balancing",
) -> dict:
    """
    Full Tier 1 pipeline for any model in MODEL_REGISTRY.

    Steps:
      1. Per-fold CV (MI -> Scaler -> PCA -> K-means SMOTE -> train -> eval)
      2. Full train -> MI -> Scaler -> PCA -> K-means SMOTE -> retrain
      3. Single evaluation on locked test set
      4. Save model, confusion matrix, ROC curve, feature importance

    Parameters
    ----------
    rus_cap       : int   if >0, cap each class to this many samples before
                          oversampling (speed/RAM saving)
    fold_cache    : dict  optional shared cache of preprocessed CV folds, so
                          multiple models don't redo MI/Scaler/PCA/SMOTE
    use_mi        : bool  apply mutual-information feature selection
    use_pca       : bool  apply PCA dimensionality reduction
    use_balancing : bool  apply K-means SMOTE balancing
    experiment    : str   ablation preset name (output directory identity)

    Returns
    -------
    dict with keys: model, cv_metrics, test_metrics, y_test_pred,
                    selector, scaler, pca, experiment, save_dir
    """
    _ensure_registry()
    entry = MODEL_REGISTRY[model_name]
    model_class = entry["model_class"]
    model_params = entry["params"]
    display = entry["display_name"]

    print(f"\n{'='*60}")
    print(f"  {model_name}  {display}")
    print(f"  Experiment: {experiment}  "
          f"(MI={'on' if use_mi else 'off'}, "
          f"PCA={'on' if use_pca else 'off'}, "
          f"KMeansSMOTE={'on' if use_balancing else 'off'})")
    print(f"{'='*60}")

    # --- Step 1: Per-fold CV ---
    print(f"\n  === Cross-Validation ({n_splits} folds) ===")
    cv_metrics, _, _, _ = run_cv(
        X_train, y_train,
        model_class=model_class,
        model_params=model_params,
        n_splits=n_splits,
        mi_k=mi_k,
        pca_variance=pca_variance,
        k_neighbors=k_neighbors,
        random_state=random_state,
        strategy="kmeans",
        rus_cap=rus_cap,
        fold_cache=fold_cache,
        use_mi=use_mi,
        use_pca=use_pca,
        use_balancing=use_balancing,
    )

    print(f"\n  CV Results ({model_name}):")
    for k, v in cv_metrics.items():
        print(f"    {k:>10s}: {np.mean(v):.4f} (+/- {np.std(v):.4f})")

    # --- Step 2: Full train → MI → Scaler → PCA → K-means SMOTE → retrain ---
    print(f"\n  === Final Retrain on Full Training Set ===")

    if hasattr(X_train, 'select_dtypes'):
        categorical_encoder = fit_categorical_encoder(X_train)
        X_train = transform_features(X_train, categorical_encoder)
        X_test = transform_features(X_test, categorical_encoder)
    else:
        categorical_encoder = None

    if use_mi:
        selector = fit_mi_selector(X_train, y_train, k=mi_k, random_state=random_state)
        X_train_mi = selector.transform(X_train)
        X_test_mi = selector.transform(X_test)
        print(f"    MI selected: {X_train_mi.shape[1]} features")
    else:
        selector = None
        X_train_mi = X_train
        X_test_mi = X_test

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_mi)

    if use_pca:
        pca = PCA(n_components=pca_variance, random_state=random_state)
        X_train_p = pca.fit_transform(X_train_s)
        print(f"    PCA components: {X_train_p.shape[1]}")
    else:
        pca = None
        X_train_p = X_train_s

    if use_balancing:
        X_train_b, y_train_b = balance_full_train(
            X_train_p, y_train, strategy="kmeans",
            k_neighbors=k_neighbors, random_state=random_state,
            rus_cap=rus_cap,
        )
    else:
        X_train_b, y_train_b = X_train_p, y_train

    if use_pca:
        X_test_p = pca.transform(scaler.transform(X_test_mi))
    else:
        X_test_p = scaler.transform(X_test_mi)

    model, test_metrics, y_test_pred = _train_and_evaluate(
        model_class, model_params,
        X_train_b, y_train_b, X_test_p, y_test,
    )

    # Enrich test metrics with binary + multiclass columns for the
    # ablation comparison tables (methodology unchanged — metrics only).
    try:
        y_proba = model.predict_proba(X_test_p)
    except Exception:
        y_proba = None
    extended = compute_extended_metrics(
        y_test, y_test_pred, y_proba=y_proba, normal_class_idx=0,
    )
    test_metrics.update(extended)

    print(f"\n  {model_name} Test Metrics:")
    for k, v in test_metrics.items():
        print(f"    {k:>10s}: {v:.4f}")

    # --- Step 3: Save artifacts (per experiment) ---
    save_dir = os.path.join("results", model_name, experiment)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{model_name.lower()}_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n  Model saved -> {model_path}")

    if selector is not None:
        joblib.dump(selector, os.path.join(save_dir, "mi_selector.joblib"))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    if pca is not None:
        joblib.dump(pca, os.path.join(save_dir, "pca.joblib"))

    config = build_experiment_config(
        model_name=model_name,
        model_params=model_params,
        experiment_name=experiment,
        preprocessing_mode=experiment,
        use_mi=use_mi,
        use_pca=use_pca,
        use_balancing=use_balancing,
        mi_k=mi_k,
        pca_variance=pca_variance,
        n_splits=n_splits,
        balancer="kmeans",
        k_neighbors=k_neighbors,
        random_state=random_state,
        rus_cap=rus_cap,
        tier=1,
    )
    save_experiment_config(config, save_dir)

    with open(os.path.join(save_dir, "test_metrics.json"), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  Test metrics saved -> {os.path.join(save_dir, 'test_metrics.json')}")

    cv_df = pd.DataFrame(cv_metrics)
    cv_df.to_csv(os.path.join(save_dir, "cv_metrics.csv"), index=False,
                 float_format='%.4f')
    print(f"  CV metrics saved -> {os.path.join(save_dir, 'cv_metrics.csv')}")

    plot_confusion_matrix(
        y_test, y_test_pred, class_names,
        normal_class_idx=0, save_dir=save_dir,
        prefix=f"{model_name.lower()}_",
    )
    plot_roc_curve(
        model, X_test_mi, y_test, class_names,
        scaler=scaler, pca=pca,
        title=f"{model_name} ROC Curve (Test Set)",
        save_dir=save_dir, prefix=f"{model_name.lower()}_",
    )
    if entry["plot_feature_importance"]:
        plot_feature_importance(
            model, n_components=X_train_p.shape[1], save_dir=save_dir,
        )

    results = {
        'model': model,
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics,
        'y_test_pred': y_test_pred,
        'selector': selector,
        'scaler': scaler,
        'pca': pca,
        'categorical_encoder': categorical_encoder,
        'experiment': experiment,
        'save_dir': save_dir,
    }

    del X_train_mi, X_test_mi, X_train_s, X_train_p
    del X_train_b, y_train_b; gc.collect()
    return results
