"""Presplit feature-ablation protocol.

This module is intentionally separate from the leakage-controlled pipeline.
Its supervised feature transforms are fitted on the complete prepared dataset
before the 80/20 holdout split, exactly as requested for this protocol.
"""

import json
import os
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.balancing import balance_full_train, balance_training_fold
from src.evaluation import compute_extended_metrics, resolve_normal_class_idx
from src.feature_selection import fit_mi_selector
from src.model_training import MODEL_REGISTRY, _ensure_registry
from src.preprocessing import feature_type_mask, fit_categorical_encoder, transform_features
from src.presplit_ablation_models import DL_MODELS, run_dl_presplit_experiment


PROTOCOL = "presplit_feature_ablation"
RESULTS_ROOT = os.path.join("results", "presplit_feature_ablation_2026_09")
EXPERIMENTS = ("E1_sequential_mi_pca", "E2_parallel_mi_pca", "E3_spearman")
SPEARMAN_SELECTION_RULES = ("top_k", "absolute_threshold")


class SpearmanSelectionCriterionRequired(ValueError):
    """Raised when E3 is requested without an explicit pre-registered rule."""


def validate_spearman_selection(rule, value):
    if rule not in SPEARMAN_SELECTION_RULES or value is None:
        raise SpearmanSelectionCriterionRequired(
            "E3_spearman requires --spearman-selection-rule and "
            "--spearman-selection-value; no calibrated repository criterion exists."
        )
    if rule == "top_k" and (int(value) != value or int(value) < 1):
        raise ValueError("Spearman top_k must be a positive integer.")
    if rule == "absolute_threshold" and not 0 <= float(value) <= 1:
        raise ValueError("Spearman absolute_threshold must be between 0 and 1.")


def apply_presplit_feature_processing(X, y, experiment, mi_k=15, pca_variance=0.95, random_state=42,
                                      spearman_selection_rule=None, spearman_selection_value=None):
    """Fit the requested experiment transform on the complete prepared dataset."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unsupported presplit experiment: {experiment}")
    if experiment == "E3_spearman":
        validate_spearman_selection(spearman_selection_rule, spearman_selection_value)

    start = time.perf_counter()
    if hasattr(X, "select_dtypes"):
        encoder = fit_categorical_encoder(X)
        feature_names = list(X.columns)
        X_prepared = transform_features(X, encoder, preserve_categorical_codes=True)
        discrete_mask = feature_type_mask(feature_names)
        if X_prepared.shape[1] != len(feature_names):
            raise RuntimeError("Presplit prepared features do not align with feature names.")
    else:
        encoder = None
        X_prepared = np.asarray(X, dtype=np.float32)
        feature_names = [f"feature_{index}" for index in range(X_prepared.shape[1])]
        discrete_mask = np.zeros(X_prepared.shape[1], dtype=bool)

    if experiment == "E3_spearman":
        from scipy.stats import spearmanr
        correlation = np.asarray(spearmanr(X_prepared, y, axis=0).statistic[:-1, -1], dtype=float)
        correlation = np.nan_to_num(correlation, nan=0.0)
        absolute = np.abs(correlation)
        ranking = np.argsort(-absolute, kind="stable")
        if spearman_selection_rule == "top_k":
            selected = ranking[:min(int(spearman_selection_value), X_prepared.shape[1])]
        else:
            selected = np.flatnonzero(absolute >= float(spearman_selection_value))
        if not len(selected):
            raise ValueError("The explicit Spearman rule selected zero features.")
        selected = np.sort(selected)
        selected_payload = {"method": "spearman", "selection_rule": spearman_selection_rule,
                            "selection_value": spearman_selection_value, "coefficients": correlation.tolist(),
                            "absolute_coefficients": absolute.tolist(), "ranking": ranking.tolist(),
                            "selected_indices": selected.tolist(), "selected_features": [feature_names[i] for i in selected]}
        return X_prepared[:, selected].astype(np.float32), {"prepared_dimension": int(X_prepared.shape[1]),
            "selected_dimension": int(len(selected)), "final_dimension": int(len(selected))}, {
            "categorical_encoder": encoder, "selected_features": selected_payload}, time.perf_counter() - start

    selector = fit_mi_selector(X_prepared, y, k=mi_k, random_state=random_state, discrete_features=discrete_mask)
    X_mi = selector.transform(X_prepared)

    if experiment == "E1_sequential_mi_pca":
        sequential_scaler = StandardScaler()
        pca = PCA(n_components=pca_variance, random_state=random_state)
        X_final = pca.fit_transform(sequential_scaler.fit_transform(X_mi))
        dimensions = {
            "prepared_dimension": int(X_prepared.shape[1]),
            "mi_dimension": int(X_mi.shape[1]),
            "pca_dimension": int(X_final.shape[1]),
            "final_dimension": int(X_final.shape[1]),
        }
        artifacts = {"categorical_encoder": encoder, "selector": selector,
                     "scaler": sequential_scaler, "pca": pca,
                     "selected_features": {"method": "mutual_information", "selected_indices": selector.get_support(indices=True).tolist(), "selected_features": [feature_names[i] for i in selector.get_support(indices=True)]}}
    else:
        scaler = StandardScaler()
        pca = PCA(n_components=pca_variance, random_state=random_state)
        X_scaled = scaler.fit_transform(X_prepared)
        X_pca = pca.fit_transform(X_scaled)
        X_final = np.concatenate([X_mi, X_pca], axis=1)
        dimensions = {
            "prepared_dimension": int(X_prepared.shape[1]),
            "mi_branch_dimension": int(X_mi.shape[1]),
            "pca_branch_dimension": int(X_pca.shape[1]),
            "concatenated_dimension": int(X_final.shape[1]),
            "final_dimension": int(X_final.shape[1]),
        }
        artifacts = {"categorical_encoder": encoder, "selector": selector,
                     "scaler": scaler, "pca": pca,
                     "selected_features": {"method": "mutual_information", "selected_indices": selector.get_support(indices=True).tolist(), "selected_features": [feature_names[i] for i in selector.get_support(indices=True)]}}
    return X_final.astype(np.float32), dimensions, artifacts, time.perf_counter() - start


def _metrics(y_true, y_pred, y_proba, normal_class_idx):
    metrics = {
        "multiclass_accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if normal_class_idx is not None:
        metrics.update(compute_extended_metrics(y_true, y_pred, y_proba, normal_class_idx))
        y_true_binary = (np.asarray(y_true) != normal_class_idx).astype(int)
        y_pred_binary = (np.asarray(y_pred) != normal_class_idx).astype(int)
        metrics["binary_precision"] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics["binary_recall"] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    return metrics


def _environment():
    import sklearn
    return {
        "cpu": platform.processor(), "gpu": None,
        "ram": None, "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    }


def _latency(model_path, X_test, training_seconds, balancing_seconds, feature_preprocessing_seconds,
             model_save_seconds):
    model_size_bytes = os.path.getsize(model_path)
    del_model = None
    del del_model
    load_start = time.perf_counter()
    model = joblib.load(model_path)
    model_load_seconds = time.perf_counter() - load_start
    cold_start = time.perf_counter()
    model.predict(X_test[:1])
    cold_start_prediction_ms = (time.perf_counter() - cold_start) * 1000
    model.predict(X_test[:1])
    batch_start = time.perf_counter()
    model.predict(X_test)
    batch_seconds = time.perf_counter() - batch_start
    samples = X_test[:1]
    single = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(samples)
        single.append((time.perf_counter() - start) * 1000)
    return {
        "feature_preprocessing_seconds": feature_preprocessing_seconds,
        "balancing_seconds": balancing_seconds,
        "training_seconds": training_seconds,
        "model_save_seconds": model_save_seconds,
        "model_load_seconds": model_load_seconds,
        "cold_start_prediction_ms": cold_start_prediction_ms,
        "cold_start_total_ms": model_load_seconds * 1000 + cold_start_prediction_ms,
        "warm_batch_prediction_seconds": batch_seconds,
        "warm_inference_ms_per_sample": 1000 * batch_seconds / len(X_test),
        "throughput_samples_per_second": len(X_test) / batch_seconds,
        "single_sample_mean_ms": float(np.mean(single)),
        "single_sample_median_ms": float(np.median(single)),
        "single_sample_p95_ms": float(np.percentile(single, 95)),
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_bytes / (1024 * 1024),
    }


def run_presplit_ablation_experiment(X, y, class_names, experiment, model_name="HGB", mi_k=15,
                             pca_variance=0.95, n_splits=5, k_neighbors=3, rus_cap=0,
                             random_state=42, results_root=RESULTS_ROOT,
                             spearman_selection_rule=None, spearman_selection_value=None):
    """Run one presplit experiment with one existing classical or DL architecture."""
    print("[1] Loading/preparing full dataset", flush=True)
    print("[2] Applying presplit feature processing", flush=True)
    X_final, dimensions, artifacts, preprocessing_seconds = apply_presplit_feature_processing(
        X, y, experiment, mi_k, pca_variance, random_state, spearman_selection_rule, spearman_selection_value
    )
    print(f"    dimensions: {dimensions}", flush=True)
    from src.dimensionality_reduction import split_data
    print("[3] Performing 80/20 stratified split", flush=True)
    X_train, X_test, y_train, y_test = split_data(X_final, y, random_state=random_state)
    print(f"    train/test shapes: {X_train.shape} / {X_test.shape}", flush=True)
    normal_class_idx = resolve_normal_class_idx(class_names)
    config = {"protocol": PROTOCOL, "result_warning": "Presplit feature-ablation protocol; not a leakage-free holdout estimate.",
              "experiment": experiment, "model": model_name, "random_state": random_state, "mi_k": mi_k,
              "pca_variance": pca_variance, "n_splits": n_splits, "k_neighbors": k_neighbors, "rus_cap": rus_cap,
              "spearman_selection_rule": spearman_selection_rule, "spearman_selection_value": spearman_selection_value}
    if model_name in DL_MODELS:
        return run_dl_presplit_experiment(X_train, X_test, y_train, y_test, class_names, normal_class_idx,
            experiment, model_name, dimensions, preprocessing_seconds, artifacts, results_root, n_splits,
            k_neighbors, rus_cap, random_state, config)
    _ensure_registry()
    entry = MODEL_REGISTRY[model_name]
    model_class, model_params = entry["model_class"], entry["params"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_metrics = []
    for fold_number, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        print(f"[CV {fold_number}/{n_splits}] balancing started", flush=True)
        balance_start = time.perf_counter()
        X_fold, y_fold = balance_training_fold(
            X_train[train_idx], y_train[train_idx], strategy="kmeans", k_neighbors=k_neighbors,
            random_state=random_state, rus_cap=rus_cap,
        )
        print(f"[CV {fold_number}/{n_splits}] balancing finished in "
              f"{time.perf_counter() - balance_start:.1f} seconds; "
              f"shapes {X_train[train_idx].shape} -> {X_fold.shape}", flush=True)
        print(f"[CV {fold_number}/{n_splits}] model training started", flush=True)
        training_start = time.perf_counter()
        fold_model = model_class(**model_params).fit(X_fold, y_fold)
        print(f"[CV {fold_number}/{n_splits}] model training finished in "
              f"{time.perf_counter() - training_start:.1f} seconds", flush=True)
        cv_metrics.append(accuracy_score(y_train[val_idx], fold_model.predict(X_train[val_idx])))
        print(f"[CV {fold_number}/{n_splits}] validation complete", flush=True)

    print("[FINAL] full 80% balancing started", flush=True)
    balance_start = time.perf_counter()
    X_balanced, y_balanced = balance_full_train(
        X_train, y_train, strategy="kmeans", k_neighbors=k_neighbors,
        random_state=random_state, rus_cap=rus_cap,
    )
    balancing_seconds = time.perf_counter() - balance_start
    print(f"[FINAL] full 80% balancing finished in {balancing_seconds:.1f} seconds", flush=True)
    model = model_class(**model_params)
    print("[FINAL] model training started", flush=True)
    training_start = time.perf_counter()
    model.fit(X_balanced, y_balanced)
    training_seconds = time.perf_counter() - training_start
    print(f"[FINAL] model training finished in {training_seconds:.1f} seconds", flush=True)
    print("[FINAL] locked-test evaluation started", flush=True)
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None
    metrics = _metrics(y_test, y_pred, y_proba, normal_class_idx)

    save_dir = os.path.join(results_root, experiment, model_name)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name.lower()}_model.joblib")
    save_start = time.perf_counter()
    joblib.dump(model, model_path)
    model_save_seconds = time.perf_counter() - save_start
    print("[FINAL] latency benchmark started", flush=True)
    latency = _latency(model_path, X_test, training_seconds, balancing_seconds,
                       preprocessing_seconds, model_save_seconds)
    for filename, payload in (("test_metrics.json", metrics), ("latency.json", latency),
                              ("feature_dimensions.json", dimensions), ("resolved_config.json", config),
                              ("environment.json", _environment())):
        with open(os.path.join(save_dir, filename), "w") as handle:
            json.dump(payload, handle, indent=2)
    if artifacts.get("selected_features") is not None:
        with open(os.path.join(save_dir, "selected_features.json"), "w") as handle:
            json.dump(artifacts["selected_features"], handle, indent=2)
    print(f"[FINAL] artifacts saved to {save_dir}", flush=True)
    return {"metrics": metrics, "latency": latency, "dimensions": dimensions,
            "cv_accuracy": cv_metrics, "save_dir": save_dir}
