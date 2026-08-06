"""
feature_pipeline.py
===================
Shared leakage-free preprocessing for classical and DL pipelines.

The same fit/transform sequence is reused everywhere:
  1. Optional Mutual Information feature selection on training data only
  2. StandardScaler fit on training data only
  3. Optional PCA fit on training data only
  4. Optional balancing applied to training data only

Validation/test data is never passed to any fit().
"""

from dataclasses import dataclass
from typing import Optional

import gc
import numpy as np

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configurable leakage-free preprocessing options."""

    use_mi: bool = True
    use_pca: bool = True
    use_balancing: bool = True
    mi_k: int = 15
    pca_n_components: Optional[float] = 0.95
    random_state: int = 42

    @property
    def mode_name(self) -> str:
        if self.use_mi and self.use_pca:
            return "mi_pca"
        if self.use_mi:
            return "mi"
        if self.use_pca:
            return "pca"
        return "raw"

    @property
    def experiment_name(self) -> str:
        parts = []
        if self.use_mi:
            parts.append("mi")
        if self.use_pca:
            parts.append("pca")
        if self.use_balancing:
            parts.append("balancing")
        if not parts:
            return "raw"
        return "_".join(parts)


def preprocessing_mode_from_string(mode: str, *, default_mi_k: int = 15,
                                   default_pca_n_components: Optional[float] = 0.95,
                                   random_state: int = 42) -> PreprocessingConfig:
    """Map a legacy CLI preprocessing string to a preprocessing configuration."""
    normalized = mode.strip().lower().replace("+", "_")
    aliases = {
        "raw": (False, False),
        "mi": (True, False),
        "pca": (False, True),
        "mi_pca": (True, True),
        "mipca": (True, True),
    }
    if normalized not in aliases:
        raise ValueError(
            "preprocessing must be one of: raw, mi, pca, mi+pca"
        )
    use_mi, use_pca = aliases[normalized]
    return PreprocessingConfig(
        use_mi=use_mi,
        use_pca=use_pca,
        use_balancing=True,
        mi_k=default_mi_k,
        pca_n_components=default_pca_n_components,
        random_state=random_state,
    )


def experiment_preset_from_string(preset: str, *, default_mi_k: int = 15,
                                  default_pca_n_components: Optional[float] = 0.95,
                                  random_state: int = 42) -> PreprocessingConfig:
    """Map an ablation experiment preset to a preprocessing configuration."""
    normalized = preset.strip().lower().replace("+", "_")
    presets = {
        "raw": (False, False, False),
        "mi": (True, False, False),
        "mi_balancing": (True, False, True),
        "pca": (False, True, False),
        "pca_balancing": (False, True, True),
        "mi_pca": (True, True, False),
        "mi_pca_balancing": (True, True, True),
        # legacy aliases kept for backward compatibility
        "balancing": (False, False, True),
        "mi_pca_legacy": (True, True, True),
    }
    if normalized not in presets:
        raise ValueError(
            "experiment must be one of: raw, mi, mi_balancing, pca, pca_balancing, mi_pca, mi_pca_balancing"
        )
    use_mi, use_pca, use_balancing = presets[normalized]
    return PreprocessingConfig(
        use_mi=use_mi,
        use_pca=use_pca,
        use_balancing=use_balancing,
        mi_k=default_mi_k,
        pca_n_components=default_pca_n_components,
        random_state=random_state,
    )


OFFICIAL_EXPERIMENTS = [
    'raw', 'mi', 'mi_balancing',
    'pca', 'pca_balancing',
    'mi_pca', 'mi_pca_balancing',
]

# The legacy --preprocessing CLI flag predates the independent balancing switch.
# To preserve its behaviour, every legacy mode keeps balancing at the default
# (ON) and only varies the MI/PCA feature transforms.
LEGACY_PREPROCESSING_ALIASES = {
    'raw': 'balancing',
    'mi': 'mi_balancing',
    'pca': 'pca_balancing',
    'mi+pca': 'mi_pca_balancing',
}

LEGACY_PREPROCESSING_MODES = ['raw', 'mi', 'pca', 'mi+pca']


def resolve_experiments(*, experiment: Optional[str] = None,
                        preprocessing: Optional[str] = None,
                        ablation: Optional[str] = None) -> list:
    """Resolve CLI flags into the ordered list of experiments to execute.

    Priority:
      1. A named ``--experiment`` preset runs alone.
      2. ``--ablation preprocessing`` runs all seven official presets.
      3. The legacy ``--preprocessing`` flag maps onto the corresponding
         presets (balancing stays at its default ON). ``all`` runs every
         legacy mode.
      4. Default: ``mi_pca_balancing`` (MI ON, PCA ON, Balancing ON).
    """
    if experiment is not None:
        return [experiment]
    if ablation == 'preprocessing':
        return list(OFFICIAL_EXPERIMENTS)
    if preprocessing is not None:
        if preprocessing == 'all':
            return [LEGACY_PREPROCESSING_ALIASES[m] for m in LEGACY_PREPROCESSING_MODES]
        return [LEGACY_PREPROCESSING_ALIASES[preprocessing]]
    return ['mi_pca_balancing']


def fit_transform_preprocessing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: Optional[np.ndarray] = None,
    *,
    config: PreprocessingConfig,
    balance_fn=None,
    balance_kwargs: Optional[dict] = None,
    balance_training: bool = False,
):
    """Fit preprocessing on X_train and optionally transform X_eval.

    Parameters
    ----------
    X_train, y_train
        Training data used for every fit.
    X_eval
        Optional validation/test matrix to transform with the fitted objects.
    config
        PreprocessingConfig describing MI/PCA usage.
    balance_fn
        Callable that balances the transformed training data only.
    balance_kwargs
        Extra kwargs forwarded to balance_fn.
    balance_training
        Whether to apply balancing after transformation.

    Returns
    -------
    dict with transformed arrays and fitted preprocessing objects.
    """
    balance_kwargs = balance_kwargs or {}
    selector = None
    scaler = None
    pca = None

    X_train_proc = X_train
    X_eval_proc = X_eval

    mi_enabled = bool(config.use_mi and config.mi_k is not None and config.mi_k > 0)
    pca_enabled = bool(
        config.use_pca and config.pca_n_components is not None and config.pca_n_components > 0
    )

    if mi_enabled:
        k = min(config.mi_k, X_train_proc.shape[1])
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X_train_proc, y_train)
        X_train_proc = selector.transform(X_train_proc)
        if X_eval_proc is not None:
            X_eval_proc = selector.transform(X_eval_proc)

    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train_proc)
    if X_eval_proc is not None:
        X_eval_proc = scaler.transform(X_eval_proc)

    if pca_enabled:
        if isinstance(config.pca_n_components, (int, np.integer)):
            n_components = min(
                int(config.pca_n_components),
                X_train_proc.shape[1],
                X_train_proc.shape[0],
            )
        else:
            n_components = config.pca_n_components
        pca = PCA(n_components=n_components, random_state=config.random_state)
        X_train_proc = pca.fit_transform(X_train_proc)
        if X_eval_proc is not None:
            X_eval_proc = pca.transform(X_eval_proc)

    y_train_proc = y_train
    if balance_training and balance_fn is not None:
        X_train_proc, y_train_proc = balance_fn(X_train_proc, y_train_proc, **balance_kwargs)

    result = {
        'X_train': X_train_proc,
        'y_train': y_train_proc,
        'selector': selector,
        'scaler': scaler,
        'pca': pca,
    }
    if X_eval is not None:
        result['X_eval'] = X_eval_proc

    del X_train_proc, X_eval_proc
    gc.collect()
    return result


def fit_transform_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: PreprocessingConfig,
    balance_fn=None,
    balance_kwargs: Optional[dict] = None,
):
    """Convenience wrapper for fold-level preprocessing."""
    result = fit_transform_preprocessing(
        X_train,
        y_train,
        X_val,
        config=config,
        balance_fn=balance_fn,
        balance_kwargs=balance_kwargs,
        balance_training=balance_fn is not None,
    )
    result['X_val'] = result.pop('X_eval')
    result['y_val'] = y_val
    return result


def fit_transform_full(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    config: PreprocessingConfig,
    balance_fn=None,
    balance_kwargs: Optional[dict] = None,
):
    """Convenience wrapper for the final train/test retrain path."""
    result = fit_transform_preprocessing(
        X_train,
        y_train,
        X_test,
        config=config,
        balance_fn=balance_fn,
        balance_kwargs=balance_kwargs,
        balance_training=balance_fn is not None,
    )
    result['X_test'] = result.pop('X_eval')
    result['y_test'] = y_test
    return result