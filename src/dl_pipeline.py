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

Tier 2 models run the SAME seven-preset ablation as Tier 1
(raw, mi, mi_balancing, pca, pca_balancing, mi_pca, mi_pca_balancing)
with identical preprocessing hyperparameters (MI k=15, PCA 0.95 variance,
KMeansSMOTE k_neighbors=3, no undersampling).  Their configurations record
``ablation_scope=tier2`` so they can be aggregated into the same comparison
tables as the classical Tier 1 models.
"""

import os
import sys
import json
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)

from src.feature_selection import fit_mi_selector
from src.preprocessing import fit_categorical_encoder, transform_features
from src.balancing import balance_training_fold
from src.experiment_config import save_experiment_config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


TIER2_ABLATION_SCOPE = "tier2"


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


def get_device():
    """Return the best available torch device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    return device


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

    from src.preprocessing import load_and_prepare, fit_categorical_encoder, transform_features
    from src.dimensionality_reduction import split_data

    X_raw, y_multi, le = load_and_prepare(data_dir=data_dir)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    normal_class_idx = list(le.classes_).index('Normal')

    X_train, X_test, y_train, y_test = split_data(X_raw, y_multi)
    del X_raw; gc.collect()

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
    mi_k: int = 15,
    pca_variance: float = None,
    pca_components: int = None,
    n_clusters: int = 20,
    k_neighbors: int = 3,
    rus_cap: int = 0,
    random_state: int = 42,
    use_mi: bool = True,
    use_pca: bool = True,
    use_balancing: bool = True,
):
    """
    Per-fold preprocessing: MI → StandardScaler → PCA → RUS + KMeansSMOTE.

    All transformers are fit on (X_tr, y_tr) only.
    (X_val, y_val) is never fitted — only transformed.

    Mirrors the Tier 1 pipeline (src/cross_validation.py + src/balancing.py):
      - PCA supports either a target number of components (``pca_components``)
        or a cumulative-variance fraction (``pca_variance``, Tier 1 semantics).
      - Random undersampling is only applied when ``rus_cap > 0``; otherwise
        KMeansSMOTE oversamples the minority classes directly.

    Returns
    -------
    dict with:
        X_tr_final, y_tr_final  (balanced training data)
        X_val_final             (transformed validation, not balanced)
        y_val                   (unchanged)
        selector, scaler, pca   (fitted transformers, may be None)
    """
    selector, scaler, pca, categorical_encoder = None, None, None, None

    if hasattr(X_tr, 'select_dtypes'):
        categorical_encoder = fit_categorical_encoder(X_tr)
        X_tr = transform_features(X_tr, categorical_encoder)
        X_val = transform_features(X_val, categorical_encoder)

    # 1. MI Feature Selection (fit on fold train only)
    if use_mi and mi_k > 0:
        selector = fit_mi_selector(X_tr, y_tr, k=mi_k, random_state=random_state)
        X_tr = selector.transform(X_tr)
        X_val = selector.transform(X_val)

    # 2. StandardScaler (fit on fold train only)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # 3. PCA (fit on fold train only)
    if use_pca and (
        pca_variance is not None
        or (pca_components is not None and pca_components > 0)
    ):
        if pca_components is not None and pca_components > 0:
            n_comp = min(pca_components, X_tr.shape[1], X_tr.shape[0])
            pca = PCA(n_components=n_comp, random_state=random_state)
        else:
            pca = PCA(n_components=pca_variance, random_state=random_state)
        X_tr = pca.fit_transform(X_tr)
        X_val = pca.transform(X_val)

    # 4. Balancing (fold train only, never touch val)
    if use_balancing:
        X_tr, y_tr = balance_training_fold(
            X_tr, y_tr,
            strategy="kmeans",
            k_neighbors=k_neighbors,
            n_clusters=n_clusters,
            random_state=random_state,
            rus_cap=rus_cap,
            stage="fold",
        )
        gc.collect()

    return {
        'X_tr': X_tr, 'y_tr': y_tr,
        'X_val': X_val, 'y_val': y_val,
        'selector': selector, 'scaler': scaler, 'pca': pca,
        'categorical_encoder': categorical_encoder,
    }


# ---------------------------------------------------------------------------
# Final retrain preprocessing (full training set)
# ---------------------------------------------------------------------------

def preprocess_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mi_k: int = 15,
    pca_variance: float = None,
    pca_components: int = None,
    n_clusters: int = 20,
    k_neighbors: int = 3,
    rus_cap: int = 0,
    random_state: int = 42,
    use_mi: bool = True,
    use_pca: bool = True,
    use_balancing: bool = True,
):
    """
    Full-train preprocessing for final model retraining.
    Test data is NEVER fitted — only transformed.

    Mirrors the Tier 1 final-retrain path (src/model_training.py): PCA uses
    cumulative variance by default, and balancing skips undersampling when
    ``rus_cap == 0``.

    Returns
    -------
    dict with:
        X_train_final, y_train_final  (balanced training data)
        X_test_final                  (transformed test data)
        y_test                        (unchanged)
        selector, scaler, pca         (fitted transformers)
    """
    selector, scaler, pca, categorical_encoder = None, None, None, None

    if hasattr(X_train, 'select_dtypes'):
        categorical_encoder = fit_categorical_encoder(X_train)
        X_train = transform_features(X_train, categorical_encoder)
        X_test = transform_features(X_test, categorical_encoder)

    # 1. MI (fit on full training only)
    if use_mi and mi_k > 0:
        selector = fit_mi_selector(X_train, y_train, k=mi_k, random_state=random_state)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

    # 2. Scaler (fit on full training only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. PCA (fit on full training only)
    if use_pca and (
        pca_variance is not None
        or (pca_components is not None and pca_components > 0)
    ):
        if pca_components is not None and pca_components > 0:
            n_comp = min(pca_components, X_train.shape[1], X_train.shape[0])
            pca = PCA(n_components=n_comp, random_state=random_state)
        else:
            pca = PCA(n_components=pca_variance, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # 4. Balancing (full training only, never touch test)
    if use_balancing:
        X_train, y_train = balance_training_fold(
            X_train, y_train,
            strategy="kmeans",
            k_neighbors=k_neighbors,
            n_clusters=n_clusters,
            random_state=random_state,
            rus_cap=rus_cap,
            stage="final retrain",
        )
        gc.collect()

    print(f"  Final train: {X_train.shape} | Test: {X_test.shape}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'selector': selector, 'scaler': scaler, 'pca': pca,
        'categorical_encoder': categorical_encoder,
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
        precision, recall, auc, accuracy, f1, binary_auc
    """
    y_true_bin = (y_true != normal_class_idx).astype(int)
    y_pred_bin = (y_pred != normal_class_idx).astype(int)

    metrics = {
        'binary_acc': accuracy_score(y_true_bin, y_pred_bin),
        'binary_f1': f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0),
        'multi_acc': accuracy_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
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
    # AUC keys are only computable with probabilities; argmax-only callers get 0.0
    metrics['binary_auc'] = 0.0
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
        from sklearn.preprocessing import label_binarize
        classes = np.unique(np.concatenate([y_true, np.arange(y_proba.shape[1])]))
        y_bin = label_binarize(y_true, classes=list(range(y_proba.shape[1])))
        metrics['auc'] = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        metrics['auc'] = 0.0
    try:
        y_true_bin = (y_true != normal_class_idx).astype(int)
        p_attack = 1.0 - np.asarray(y_proba)[:, normal_class_idx]
        metrics['binary_auc'] = roc_auc_score(y_true_bin, p_attack)
    except Exception:
        metrics['binary_auc'] = 0.0
    return metrics


def get_probabilities(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Run a forward pass and return softmax probabilities as a numpy array.

    Parameters
    ----------
    model   : trained nn.Module whose forward() returns raw logits
    X       : array (N, F) — will be converted to a float32 tensor
    device  : torch device

    Returns
    -------
    np.ndarray (N, num_classes)  row-normalised probabilities
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_dl_artifacts(
    model: nn.Module,
    model_name: str,
    cv_metrics: list,
    test_metrics: dict = None,
    experiment: str = None,
    save_root: str = "results",
    class_names: list = None,
    normal_class_idx: int = 0,
    y_test: np.ndarray = None,
    y_test_pred: np.ndarray = None,
    selector=None,
    scaler=None,
    pca=None,
    le=None,
    config: dict = None,
):
    """
    Save model weights, metrics CSV, metrics JSON, confusion matrix,
    preprocessing artifacts (MI, Scaler, PCA, LabelEncoder), and config.json.

    When *experiment* is given the artifacts are written to the canonical
    Tier 1 ablation layout::

        results/<Model>/<experiment>/experiment_config.json
        results/<Model>/<experiment>/test_metrics.json
        results/<Model>/<experiment>/cv_metrics.csv
        results/<Model>/<experiment>/<model>_model.joblib
        results/<Model>/<experiment>/<model>_model.pt
        results/<Model>/<experiment>/<model>_confusion_matrix.png
        results/<Model>/<experiment>/{mi_selector,scaler,pca,label_encoder}.joblib

    so the DL model can be aggregated into the same ablation tables as the
    classical Tier 1 models.  Without *experiment* it falls back to the
    legacy ``models/artifacts/<model_name>/`` location.

    Parameters
    ----------
    model            : trained nn.Module
    model_name       : str  (used for file naming)
    cv_metrics       : list of per-fold metric dicts
    test_metrics     : dict  (final test metrics)
    experiment       : str   ablation preset name (None = legacy location)
    save_root        : str   parent of the results/<Model>/<experiment> layout
    class_names      : list  (for confusion matrix labels)
    normal_class_idx : int
    y_test           : ground-truth labels  (for confusion matrix)
    y_test_pred      : predicted labels     (for confusion matrix)
    selector         : fitted SelectKBest   (MI feature selector)
    scaler           : fitted StandardScaler
    pca              : fitted PCA
    le               : fitted LabelEncoder
    config           : dict of all experiment hyper-parameters
    """
    if experiment is not None:
        save_dir = os.path.join(save_root, model_name, experiment)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, "models", "artifacts", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Model weights (.pt) + full model object (.joblib, canonical ablation
    # artifact name that mirrors Tier 1's <model>_model.joblib).
    model_pt = os.path.join(save_dir, f"{model_name.lower()}_model.pt")
    torch.save(model.state_dict(), model_pt)
    print(f"  Model weights saved -> {model_pt}")

    model_path = os.path.join(save_dir, f"{model_name.lower()}_model.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved -> {model_path}")

    # CV metrics CSV (canonical name, columns compatible with the ablator)
    cv_df = pd.DataFrame(cv_metrics)
    cv_path = os.path.join(save_dir, "cv_metrics.csv")
    cv_df.to_csv(cv_path, index=False, float_format='%.4f')
    print(f"  CV metrics saved -> {cv_path}")

    # Test metrics JSON (canonical name)
    if test_metrics:
        json_path = os.path.join(save_dir, "test_metrics.json")
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

    # Preprocessing artifacts (for inference reproducibility)
    if selector is not None:
        joblib.dump(selector, os.path.join(save_dir, "mi_selector.joblib"))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    if pca is not None:
        joblib.dump(pca, os.path.join(save_dir, "pca.joblib"))
    if le is not None:
        joblib.dump(le, os.path.join(save_dir, "label_encoder.joblib"))

    # Config JSON (canonical name, written via the shared config writer)
    if config is None:
        config = {}
    config.setdefault("model_name", model_name)
    config.setdefault("class_names", class_names)
    config.setdefault("normal_class_idx", normal_class_idx)
    config.setdefault("num_classes", len(class_names) if class_names else None)
    save_experiment_config(config, save_dir)

    return save_dir
