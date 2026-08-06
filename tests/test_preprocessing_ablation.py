"""
test_preprocessing_ablation.py
==============================
Verify the preprocessing ablation framework:

  1. The seven official presets map to the correct (use_mi, use_pca, use_balancing)
  2. Legacy --preprocessing aliases keep balancing at its default (ON)
  3. CLI flag resolution (--experiment / --preprocessing / --ablation)
  4. Every preset's CV + final-retrain path reports the full metric set
     required for the publication-ready comparison tables (incl. macro_f1)
"""

import inspect

import numpy as np
import pytest

from src.feature_pipeline import (
    OFFICIAL_EXPERIMENTS,
    PreprocessingConfig,
    experiment_preset_from_string,
    resolve_experiments,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def split_dataset():
    """Small reproducible synthetic dataset for fast smoke tests."""
    from sklearn.model_selection import train_test_split
    rng = np.random.RandomState(7)
    X = rng.randn(1000, 20).astype(np.float32)
    y = rng.choice([0, 1, 2, 3, 4], size=1000, p=[0.5, 0.15, 0.1, 0.15, 0.1])
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=7)


# ---------------------------------------------------------------------------
# Test 1: Official presets
# ---------------------------------------------------------------------------

class TestOfficialPresets:
    @pytest.mark.parametrize(
        "name,use_mi,use_pca,use_balancing",
        [
            ("raw", False, False, False),
            ("mi", True, False, False),
            ("mi_balancing", True, False, True),
            ("pca", False, True, False),
            ("pca_balancing", False, True, True),
            ("mi_pca", True, True, False),
            ("mi_pca_balancing", True, True, True),
        ],
    )
    def test_preset_switches(self, name, use_mi, use_pca, use_balancing):
        cfg = experiment_preset_from_string(name)
        assert isinstance(cfg, PreprocessingConfig)
        assert cfg.use_mi == use_mi
        assert cfg.use_pca == use_pca
        assert cfg.use_balancing == use_balancing
        assert cfg.experiment_name == name

    def test_official_experiments_are_well_defined(self):
        assert OFFICIAL_EXPERIMENTS == [
            'raw', 'mi', 'mi_balancing',
            'pca', 'pca_balancing',
            'mi_pca', 'mi_pca_balancing',
        ]

    def test_unknown_preset_rejected(self):
        with pytest.raises(ValueError):
            experiment_preset_from_string("not_a_preset")


# ---------------------------------------------------------------------------
# Test 2: Legacy aliases
# ---------------------------------------------------------------------------

class TestLegacyAliases:
    @pytest.mark.parametrize(
        "alias,use_mi,use_pca,use_balancing",
        [
            ("balancing", False, False, True),
            ("mi_pca_legacy", True, True, True),
        ],
    )
    def test_legacy_aliases_keep_balancing_on(self, alias, use_mi, use_pca,
                                              use_balancing):
        cfg = experiment_preset_from_string(alias)
        assert cfg.use_mi == use_mi
        assert cfg.use_pca == use_pca
        assert cfg.use_balancing == use_balancing


# ---------------------------------------------------------------------------
# Test 3: CLI flag resolution
# ---------------------------------------------------------------------------

class TestCliResolution:
    def test_default_is_mi_pca_balancing(self):
        assert resolve_experiments() == ['mi_pca_balancing']

    def test_named_experiment_runs_alone(self):
        for name in OFFICIAL_EXPERIMENTS:
            assert resolve_experiments(experiment=name) == [name]

    def test_ablation_runs_all_seven(self):
        assert resolve_experiments(ablation='preprocessing') == OFFICIAL_EXPERIMENTS

    def test_legacy_preprocessing_modes(self):
        assert resolve_experiments(preprocessing='raw') == ['balancing']
        assert resolve_experiments(preprocessing='mi') == ['mi_balancing']
        assert resolve_experiments(preprocessing='pca') == ['pca_balancing']
        assert resolve_experiments(preprocessing='mi+pca') == ['mi_pca_balancing']

    def test_legacy_preprocessing_all(self):
        assert resolve_experiments(preprocessing='all') == [
            'balancing', 'mi_balancing', 'pca_balancing', 'mi_pca_balancing',
        ]

    def test_experiment_takes_priority_over_ablation(self):
        assert resolve_experiments(experiment='raw', ablation='preprocessing') == ['raw']

    def test_resolved_presets_build_valid_configs(self):
        for name in OFFICIAL_EXPERIMENTS:
            cfg = experiment_preset_from_string(name)
            assert cfg.experiment_name == name


# ---------------------------------------------------------------------------
# Test 4: Metric completeness for comparison tables
# ---------------------------------------------------------------------------

class TestComparisonTableMetrics:
    REQUIRED = {'accuracy', 'binary_accuracy', 'macro_f1', 'f1',
                'binary_f1', 'precision', 'recall', 'auc'}

    @pytest.mark.parametrize("preset", OFFICIAL_EXPERIMENTS)
    def test_cv_metrics_include_macro_f1(self, split_dataset, preset):
        """run_cv must report macro_f1 for every leakage-free preset."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from src.cross_validation import run_cv

        X_train, X_test, y_train, y_test = split_dataset
        cfg = experiment_preset_from_string(preset)

        cv_metrics, selector, scaler, pca = run_cv(
            X_train, y_train,
            model_class=HistGradientBoostingClassifier,
            model_params=dict(max_iter=5, random_state=7),
            n_splits=2, mi_k=8,
            pca_variance=0.95, k_neighbors=2,
            random_state=7, strategy="smote",
            use_mi=cfg.use_mi,
            use_pca=cfg.use_pca,
            use_balancing=cfg.use_balancing,
            normal_class_idx=0,
        )

        for metric in cv_metrics:
            assert len(cv_metrics[metric]) == 2
        assert 'macro_f1' in cv_metrics
        assert self.REQUIRED <= set(cv_metrics.keys())

    def test_classical_trainers_report_macro_f1(self, split_dataset, tmp_path):
        """Final test evaluation must include macro_f1 (publication tables)."""
        from src.train_hgb import train_and_evaluate as train_hgb

        X_train, X_test, y_train, y_test = split_dataset
        results = train_hgb(
            X_train, y_train, X_test, y_test,
            class_names=['a', 'b', 'c', 'd', 'e'],
            n_splits=2, mi_k=8, pca_variance=0.95, k_neighbors=2,
            random_state=7, balancer="smote", make_plots=False,
            normal_class_idx=0, use_mi=False, use_pca=False,
            use_balancing=False, experiment_name="raw",
            preprocessing_mode="raw",
            output_dir=str(tmp_path),
        )

        test_metrics = results['test_metrics']
        assert self.REQUIRED <= set(test_metrics.keys())
        assert 'macro_f1' in test_metrics
        assert 'macro_f1' in results['cv_metrics']

    def test_all_trainer_sources_compute_macro_f1(self):
        """All three classical trainers define macro_f1 in their metrics."""
        from src import train_hgb, train_logistic, train_xgboost

        for module in (train_hgb, train_xgboost, train_logistic):
            source = inspect.getsource(module)
            assert "average='macro'" in source
            assert "'macro_f1'" in source
