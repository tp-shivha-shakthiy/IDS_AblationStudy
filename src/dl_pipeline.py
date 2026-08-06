"""
dl_pipeline.py
==============
Shared infrastructure for Tier 2 deep learning experiments.

Provides:
  - load_data()             preprocess via src/preprocessing.py + 80/20 split
  - preprocess_fold()       per-fold MI + Scaler + PCA + K-means SMOTE
  - preprocess_final()      full-train preprocessing for final retrain
  - compute_class_weights() inverse-frequency weights from training labels
  - evaluate_predictions()  binary + multiclass metrics
  - save_dl_artifacts()     model weights + metrics CSV + JSON + preprocessing artifacts
  - set_seeds()             deterministic seed control
  - get_device()            CUDA/CPU device selection

All DL model scripts should import from this module instead of
duplicating preprocessing, MI, PCA, balancing, or evaluation code.
"""

import os
import sys
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import MiniBatchKMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.feature_pipeline import PreprocessingConfig, fit_transform_fold, fit_transform_full


# ---------------------------------------------------------------------------
# Deterministic seeds
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42):
    """Set deterministic seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto"):
    """Return a verified Torch device, falling back to CPU for CUDA failures."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cpu":
        device = torch.device('cpu')
    elif not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but is not available to PyTorch.")
        device = torch.device('cpu')
    else:
        try:
            # Trigger the CUDA allocator before an expensive training run. This
            # catches common driver/NVML allocator failures and permits a safe
            # CPU fallback when device selection is automatic.
            probe = torch.empty(1, device='cuda')
            del probe
            torch.cuda.synchronize()
            device = torch.device('cuda')
        except RuntimeError as exc:
            if requested == "cuda":
                raise RuntimeError(
                    "CUDA initialization failed. Update the NVIDIA driver / "
                    "PyTorch CUDA build, or rerun with --device cpu."
                ) from exc
            print(f"CUDA health check failed ({exc}); falling back to CPU.")
            device = torch.device('cpu')
    print(f"Device: {device}")
    return device


def _balance_dl_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_clusters: int = 20,
    k_neighbors: int = 2,
    rus_cap: int = 15000,
    random_state: int = 42,
):
    """Apply the DL balancing policy on training data only."""
    class_counts = Counter(y_train)
    under_strategy = {c: min(cnt, rus_cap) for c, cnt in class_counts.items()}
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    actual_k = min(k_neighbors, min(Counter(y_train_rus).values()) - 1)
    kms = KMeansSMOTE(
        cluster_balance_threshold=0.0,
        k_neighbors=max(actual_k, 1),
        kmeans_estimator=MiniBatchKMeans(n_init='auto', random_state=random_state),
        random_state=random_state, n_jobs=1,
    )
    X_train_bal, y_train_bal = kms.fit_resample(X_train_rus, y_train_rus)
    del X_train_rus, y_train_rus, rus, kms; gc.collect()
    return X_train_bal, y_train_bal


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str = "data/raw"):
    """
    Load UNSW-NB15 through the shared preprocessing layer, then split.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test,
        class_names, num_classes, normal_class_idx, le
    """
    # Add project root to path so src/ imports work from models/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.preprocessing import load_and_preprocess
    from src.dimensionality_reduction import split_data

    X_processed, y_multi, le = load_and_preprocess(data_dir=data_dir)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    normal_class_idx = list(le.classes_).index('Normal')

    X_train, X_test, y_train, y_test = split_data(X_processed, y_multi)
    del X_processed; gc.collect()

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'class_names': class_names, 'num_classes': num_classes,
        'normal_class_idx': normal_class_idx, 'le': le,
    }


# ---------------------------------------------------------------------------
# Per-fold preprocessing (inside CV)
# ---------------------------------------------------------------------------

def preprocess_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mi_k: int = 30,
    pca_components: int = 15,
    n_clusters: int = 20,
    k_neighbors: int = 2,
    rus_cap: int = 15000,
    random_state: int = 42,
    use_mi: bool = True,
    use_pca: bool = True,
    use_balancing: bool = True,
):
    """
    Per-fold preprocessing: MI → StandardScaler → PCA → RUS + KMeansSMOTE.

    All transformers are fit on (X_tr, y_tr) only.
    (X_val, y_val) is never fitted — only transformed.

    Returns
    -------
    dict with:
        X_tr_final, y_tr_final  (balanced training data)
        X_val_final             (transformed validation, not balanced)
        y_val                   (unchanged)
        selector, scaler, pca   (fitted transformers, may be None)
    """
    config = PreprocessingConfig(
        use_mi=use_mi,
        use_pca=use_pca,
        mi_k=mi_k,
        pca_n_components=pca_components if use_pca else None,
        random_state=random_state,
    )

    if use_balancing:
        result = fit_transform_fold(
            X_tr, y_tr, X_val, y_val,
            config=config,
            balance_fn=_balance_dl_training,
            balance_kwargs=dict(
                n_clusters=n_clusters,
                k_neighbors=k_neighbors,
                rus_cap=rus_cap,
                random_state=random_state,
            ),
        )
    else:
        result = fit_transform_fold(
            X_tr, y_tr, X_val, y_val,
            config=config,
            balance_fn=None,
            balance_kwargs=None,
        )

    return {
        'X_tr': result['X_train'], 'y_tr': result['y_train'],
        'X_val': result['X_val'], 'y_val': result['y_val'],
        'selector': result['selector'], 'scaler': result['scaler'], 'pca': result['pca'],
    }


# ---------------------------------------------------------------------------
# Final retrain preprocessing (full training set)
# ---------------------------------------------------------------------------

def preprocess_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mi_k: int = 30,
    pca_components: int = 15,
    n_clusters: int = 20,
    k_neighbors: int = 2,
    rus_cap: int = 15000,
    random_state: int = 42,
    use_mi: bool = True,
    use_pca: bool = True,
    use_balancing: bool = True,
):
    """
    Full-train preprocessing for final model retraining.
    Test data is NEVER fitted — only transformed.

    Returns
    -------
    dict with:
        X_train_final, y_train_final  (balanced training data)
        X_test_final                  (transformed test data)
        y_test                        (unchanged)
        selector, scaler, pca         (fitted transformers)
    """
    config = PreprocessingConfig(
        use_mi=use_mi,
        use_pca=use_pca,
        mi_k=mi_k,
        pca_n_components=pca_components if use_pca else None,
        random_state=random_state,
    )

    if use_balancing:
        result = fit_transform_full(
            X_train, y_train, X_test, y_test,
            config=config,
            balance_fn=_balance_dl_training,
            balance_kwargs=dict(
                n_clusters=n_clusters,
                k_neighbors=k_neighbors,
                rus_cap=rus_cap,
                random_state=random_state,
            ),
        )
    else:
        result = fit_transform_full(
            X_train, y_train, X_test, y_test,
            config=config,
            balance_fn=None,
            balance_kwargs=None,
        )

    print(f"  Final train: {result['X_train'].shape} | Test: {result['X_test'].shape}")

    return {
        'X_train': result['X_train'], 'y_train': result['y_train'],
        'X_test': result['X_test'], 'y_test': result['y_test'],
        'selector': result['selector'], 'scaler': result['scaler'], 'pca': result['pca'],
    }


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights from training labels."""
    class_counts = np.bincount(y_train)
    num_classes = len(class_counts)
    total = len(y_train)
    weights = total / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normal_class_idx: int = 0,
) -> dict:
    """
    Compute binary + multiclass metrics.

    Returns dict with:
        binary_acc, binary_f1, multi_acc, macro_f1, weighted_f1,
        precision, recall, auc
    """
    y_true_bin = (y_true != normal_class_idx).astype(int)
    y_pred_bin = (y_pred != normal_class_idx).astype(int)

    metrics = {
        'binary_acc': accuracy_score(y_true_bin, y_pred_bin),
        'binary_f1': f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0),
        'multi_acc': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    try:
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_true)
        if len(classes) > 2:
            y_bin = label_binarize(y_true, classes=classes)
            # Need predict_proba for AUC — return 0.0 for argmax-only predictions
            metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
    except Exception:
        metrics['auc'] = 0.0
    return metrics


def evaluate_with_proba(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    normal_class_idx: int = 0,
) -> dict:
    """Compute metrics including AUC when probabilities are available."""
    metrics = evaluate_predictions(y_true, y_pred, normal_class_idx)
    try:
        classes = np.unique(y_true)
        if len(classes) == 2 and y_proba.shape[1] == 2:
            y_bin = (y_true != normal_class_idx).astype(int)
            p_attack = y_proba[:, 1] if normal_class_idx == 0 else y_proba[:, 0]
            metrics['auc'] = roc_auc_score(y_bin, p_attack)
        else:
            from sklearn.preprocessing import label_binarize
            n_classes = y_proba.shape[1]
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            metrics['auc'] = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        metrics['auc'] = 0.0
    return metrics


def get_probabilities(model, X_tensor, device, batch_size: int = 4096):
    """Run memory-bounded inference and return class probabilities.

    Validation and test partitions in UNSW-NB15 contain hundreds of thousands
    of rows.  Moving them to CUDA in one LSTM call can exhaust the caching
    allocator, even when training itself uses small DataLoader batches.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start:start + batch_size].to(device)
            logits = model(batch)
            # Multi-task models return (binary_logits, multiclass_logits).
            # Metrics in this helper use the multiclass prediction head.
            if isinstance(logits, tuple):
                logits = logits[-1]
            probabilities.append(torch.softmax(logits, dim=1).cpu())

    if not probabilities:
        return np.empty((0, 0), dtype=np.float32)
    return torch.cat(probabilities, dim=0).numpy()


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_dl_artifacts(
    model: nn.Module,
    model_name: str,
    cv_metrics: list,
    test_metrics: dict = None,
    save_dir: str = None,
    class_names: list = None,
    normal_class_idx: int = 0,
    y_test: np.ndarray = None,
    y_test_pred: np.ndarray = None,
    selector=None,
    scaler=None,
    pca=None,
    label_encoder=None,
    model_config: dict = None,
):
    """
    Save model weights, metrics, preprocessing, and architecture metadata.
    """
    import pandas as pd

    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, "models", "artifacts", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Model weights
    model_path = os.path.join(save_dir, f"{model_name.lower()}_model.pt")
    torch.save(model.state_dict(), model_path)
    metadata = {
        'model_name': model_name,
        'class_names': class_names,
        'normal_class_idx': normal_class_idx,
        'model_config': model_config or {},
    }
    metadata_path = os.path.join(save_dir, f"{model_name.lower()}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    for artifact, filename in (
        (selector, 'mi_selector.joblib'),
        (scaler, 'scaler.joblib'),
        (pca, 'pca.joblib'),
        (label_encoder, 'label_encoder.joblib'),
    ):
        if artifact is not None:
            joblib.dump(artifact, os.path.join(save_dir, filename))
    print(f"  Model saved -> {model_path}")

    # CV metrics CSV
    cv_df = pd.DataFrame(cv_metrics)
    cv_path = os.path.join(save_dir, f"{model_name.lower()}_cv_metrics.csv")
    cv_df.to_csv(cv_path, index=False, float_format='%.4f')
    print(f"  CV metrics saved -> {cv_path}")

    # Test metrics JSON
    if test_metrics:
        json_path = os.path.join(save_dir, f"{model_name.lower()}_test_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"  Test metrics saved -> {json_path}")

    # Confusion matrix
    if y_test is not None and y_test_pred is not None and class_names is not None:
        cm = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45
        )
        ax.set_title(f"{model_name} Confusion Matrix (Test Set)")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{model_name.lower()}_confusion_matrix.png"), dpi=150)
        plt.close(fig)
        print(f"  Confusion matrix saved -> {save_dir}")

    # Preprocessing artifacts
    return save_dir


def load_dl_artifacts(model: nn.Module, model_name: str, save_dir: str, device=None) -> dict:
    """Load weights and preprocessing for a caller-constructed DL model."""
    if device is None:
        device = torch.device('cpu')
    stem = model_name.lower()
    state = torch.load(os.path.join(save_dir, f"{stem}_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    def optional_joblib(filename):
        path = os.path.join(save_dir, filename)
        return joblib.load(path) if os.path.exists(path) else None

    with open(os.path.join(save_dir, f"{stem}_metadata.json")) as f:
        metadata = json.load(f)
    return {
        'model': model,
        'selector': optional_joblib('mi_selector.joblib'),
        'scaler': optional_joblib('scaler.joblib'),
        'pca': optional_joblib('pca.joblib'),
        'label_encoder': optional_joblib('label_encoder.joblib'),
        'metadata': metadata,
    }
