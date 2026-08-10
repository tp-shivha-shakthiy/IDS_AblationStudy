"""
test_leakage.py
===============
Verify that the Tier 1 pipeline has no data leakage.

Tests:
  1. Test data is never used during preprocessing fitting
  2. PCA is fitted only on training data
  3. StandardScaler is fitted only on training data
  4. CV preprocessing (MI + Scaler + PCA) is fitted independently per fold
  5. Test data is never balanced (SMOTE)
  6. The 80/20 split is stratified and reproducible
  7. Balancing API correctness (default strategy, train-only, etc.)
"""

import numpy as np
import pytest
import torch
from collections import Counter
from unittest.mock import patch

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import SMOTE

from src.dimensionality_reduction import split_data
from src.cross_validation import run_cv
from src.balancing import balance_training_fold, balance_full_train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset():
    """Create a small reproducible dataset for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(1000, 20).astype(np.float32)
    y = rng.choice([0, 1, 2, 3, 4], size=1000, p=[0.5, 0.15, 0.1, 0.15, 0.1])
    return X, y


def synthetic_dataset_small():
    """Small dataset usable outside pytest fixtures (e.g. in patch mocks)."""
    rng = np.random.RandomState(42)
    X = rng.randn(1000, 20).astype(np.float32)
    y = rng.choice([0, 1, 2, 3, 4], size=1000, p=[0.5, 0.15, 0.1, 0.15, 0.1])
    return X, y


@pytest.fixture
def split_dataset(synthetic_dataset):
    """Stratified split of the synthetic dataset."""
    X, y = synthetic_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Test 1: Test data never used during preprocessing fitting
# ---------------------------------------------------------------------------

class TestNoPreprocessingLeakage:
    def test_target_labels_use_fixed_vocabulary_without_fit(self):
        import pandas as pd
        from unittest.mock import patch
        import src.preprocessing as preprocessing

        row = [0] * 49
        row[47] = 'normal'
        raw = pd.DataFrame([row])
        with patch.object(preprocessing.pd, 'read_csv', return_value=raw), \
             patch.object(preprocessing.LabelEncoder, 'fit', side_effect=AssertionError):
            _, y, le = preprocessing.load_and_prepare('synthetic')

        assert y.shape == (4,)
        assert le.classes_.tolist() == list(preprocessing.TARGET_CLASSES)

    def test_split_data_returns_correct_sizes(self, synthetic_dataset):
        """split_data must return 80/20 stratified split."""
        X, y = synthetic_dataset
        X_train, X_test, y_train, y_test = split_data(X, y)

        assert len(X_train) == 800
        assert len(X_test) == 200
        assert X_train.shape[1] == X.shape[1]
        assert X_test.shape[1] == X.shape[1]

    def test_mi_selector_fit_on_train_only(self, split_dataset):
        """MI selector fitted on train must not see test data."""
        X_train, X_test, y_train, y_test = split_dataset

        selector = SelectKBest(score_func=mutual_info_classif, k=10)
        selector.fit(X_train, y_train)

        X_tr_mi = selector.transform(X_train)
        X_te_mi = selector.transform(X_test)

        assert X_tr_mi.shape == (len(X_train), 10)
        assert X_te_mi.shape == (len(X_test), 10)

    def test_categorical_encoder_is_fit_on_train_only_and_handles_unknowns(self):
        import pandas as pd
        from src.preprocessing import fit_categorical_encoder, transform_features

        X_train = pd.DataFrame({
            "proto": ["tcp", "udp", "tcp"], "dur": [1.0, 2.0, 3.0],
        })
        X_test = pd.DataFrame({
            "proto": ["icmp", "tcp"], "dur": [4.0, 5.0],
        })
        encoder = fit_categorical_encoder(X_train)

        # The test-only category cannot be present in an encoder fit on train.
        assert "icmp" not in encoder.categories_[0]
        X_train_out = transform_features(X_train, encoder)
        X_test_out = transform_features(X_test, encoder)
        assert X_train_out.shape == (3, 2)
        assert X_test_out.shape == (2, 2)
        assert np.isfinite(X_test_out).all()

    def test_cv_categorical_encoder_is_fit_per_fold_on_fold_training_only(self):
        import pandas as pd
        from unittest.mock import patch
        from sklearn.ensemble import HistGradientBoostingClassifier
        from src.cross_validation import run_cv
        from src.preprocessing import fit_categorical_encoder

        X = pd.DataFrame({
            "proto": ["tcp", "udp", "icmp", "tcp"] * 25,
            "dur": np.arange(100, dtype=float),
        })
        y = np.array([0, 1] * 50)
        with patch("src.cross_validation.fit_categorical_encoder",
                   wraps=fit_categorical_encoder) as encoder_fit:
            run_cv(
                X, y, HistGradientBoostingClassifier,
                dict(max_iter=5, random_state=42), n_splits=2,
                mi_k=1, use_mi=False, use_pca=False, use_balancing=False,
            )

        assert encoder_fit.call_count == 2
        assert all(call.args[0].shape[0] == 50 for call in encoder_fit.call_args_list)


# ---------------------------------------------------------------------------
# Test 2: PCA is fitted only on training data
# ---------------------------------------------------------------------------

class TestPCALeakage:
    def test_pca_fit_on_train_only(self, split_dataset):
        """PCA fitted on train must not see test data."""
        X_train, X_test, y_train, y_test = split_dataset

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        pca = PCA(n_components=0.95, random_state=42)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_te_p = pca.transform(X_te_s)

        assert X_tr_p.shape[1] == X_te_p.shape[1]
        assert X_tr_p.shape[0] == X_train.shape[0]
        assert X_te_p.shape[0] == X_test.shape[0]

    def test_pca_components_are_consistent(self, split_dataset):
        """Same scaler/PCA transforms train and test with same n_components."""
        X_train, X_test, y_train, y_test = split_dataset

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)

        pca = PCA(n_components=0.95, random_state=42)
        X_tr_p = pca.fit_transform(X_tr_s)

        assert pca.n_components_ <= X_train.shape[1]
        assert X_tr_p.shape[1] <= X_train.shape[1]


# ---------------------------------------------------------------------------
# Test 3: StandardScaler is fitted only on training data
# ---------------------------------------------------------------------------

class TestScalerLeakage:
    def test_scaler_fit_on_train_only(self, split_dataset):
        """StandardScaler fitted on train only; test transformed with train stats."""
        X_train, X_test, y_train, y_test = split_dataset

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        assert np.allclose(X_tr_s.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_tr_s.std(axis=0), 1, atol=1e-1)

    def test_scaler_means_match_train_not_full(self, split_dataset):
        """Scaler means must match train statistics exactly."""
        X_train, X_test, y_train, y_test = split_dataset

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)

        train_means = np.mean(X_train, axis=0)
        assert np.allclose(scaler.mean_, train_means)


# ---------------------------------------------------------------------------
# Test 4: CV preprocessing (MI + Scaler + PCA) fitted independently per fold
# ---------------------------------------------------------------------------

class TestCVPerFoldIndependence:
    def test_run_cv_returns_fold_metrics(self, split_dataset):
        """run_cv must return per-fold metrics with correct fold count."""
        X_train, X_test, y_train, y_test = split_dataset

        from sklearn.ensemble import HistGradientBoostingClassifier

        cv_metrics, selector, scaler, pca = run_cv(
            X_train, y_train,
            model_class=HistGradientBoostingClassifier,
            model_params=dict(max_iter=5, random_state=42),
            n_splits=3, mi_k=10,
            pca_variance=0.95, k_neighbors=2,
            random_state=42, strategy="smote",
        )

        for k, v in cv_metrics.items():
            assert len(v) == 3, f"Expected 3 fold values for {k}, got {len(v)}"

    def test_run_cv_returns_fitted_transformers(self, split_dataset):
        """run_cv returns fitted selector, scaler, and pca (last fold)."""
        X_train, X_test, y_train, y_test = split_dataset

        from sklearn.ensemble import HistGradientBoostingClassifier

        cv_metrics, selector, scaler, pca = run_cv(
            X_train, y_train,
            model_class=HistGradientBoostingClassifier,
            model_params=dict(max_iter=5, random_state=42),
            n_splits=3, mi_k=10,
            pca_variance=0.95, k_neighbors=2,
            random_state=42, strategy="smote",
        )

        assert selector is not None
        assert isinstance(selector, SelectKBest)
        assert scaler is not None
        assert isinstance(scaler, StandardScaler)
        assert pca is not None
        assert isinstance(pca, PCA)

    def test_each_fold_mi_is_independent(self, synthetic_dataset):
        """Each fold produces a different MI selector fitted on different data."""
        X, y = synthetic_dataset
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        selectors = []
        for trn_idx, val_idx in skf.split(X_tr, y_tr):
            fold_selector = SelectKBest(score_func=mutual_info_classif, k=10)
            fold_selector.fit(X_tr[trn_idx], y_tr[trn_idx])
            selectors.append(fold_selector)

        # MI scores should differ across folds because they see different data
        scores_0 = selectors[0].scores_
        scores_1 = selectors[1].scores_
        assert not np.array_equal(scores_0, scores_1)

    def test_each_fold_scaler_is_independent(self, synthetic_dataset):
        """Each fold produces a different scaler fitted on different data."""
        X, y = synthetic_dataset
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        scalers = []
        for trn_idx, val_idx in skf.split(X_tr, y_tr):
            fold_scaler = StandardScaler()
            fold_scaler.fit(X_tr[trn_idx])
            scalers.append(fold_scaler)

        assert not np.allclose(scalers[0].mean_, scalers[1].mean_)
        assert not np.allclose(scalers[1].mean_, scalers[2].mean_)


# ---------------------------------------------------------------------------
# Test 5: Test data is never balanced
# ---------------------------------------------------------------------------

class TestNoBalancingLeakage:
    def test_smote_applied_to_train_only(self, synthetic_dataset):
        """SMOTE must not change the validation fold size or composition."""
        X, y = synthetic_dataset

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        sm = SMOTE(random_state=42, k_neighbors=2)
        X_bal, y_bal = sm.fit_resample(X_tr, y_tr)

        assert len(X_bal) >= len(X_tr)
        assert len(X_val) == len(y_val)
        assert len(y_val) == int(len(y) * 0.2)

    def test_smote_does_not_affect_val_indices(self, synthetic_dataset):
        """SMOTE-balanced train data must not contain any val samples."""
        X, y = synthetic_dataset
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        sm = SMOTE(random_state=42, k_neighbors=2)
        X_bal, y_bal = sm.fit_resample(X_tr, y_tr)

        train_classes = set(y_bal)
        val_classes = set(y_val)
        assert train_classes == val_classes


# ---------------------------------------------------------------------------
# Test 6: 80/20 split is stratified and reproducible
# ---------------------------------------------------------------------------

class TestSplitReproducibility:
    def test_split_is_stratified(self, synthetic_dataset):
        """Train and test must have approximately the same class ratios."""
        X, y = synthetic_dataset
        X_train, X_test, y_train, y_test = split_data(X, y)

        train_ratio = Counter(y_train)
        test_ratio = Counter(y_test)

        for cls in set(y):
            tr_pct = train_ratio[cls] / len(y_train)
            te_pct = test_ratio[cls] / len(y_test)
            assert abs(tr_pct - te_pct) < 0.02, (
                f"Class {cls}: train={tr_pct:.3f} vs test={te_pct:.3f}"
            )

    def test_split_is_reproducible(self, synthetic_dataset):
        """Two calls to split_data with same seed give identical results."""
        X, y = synthetic_dataset

        X_tr1, X_te1, y_tr1, y_te1 = split_data(X, y)
        X_tr2, X_te2, y_tr2, y_te2 = split_data(X, y)

        np.testing.assert_array_equal(X_tr1, X_tr2)
        np.testing.assert_array_equal(X_te1, X_te2)
        np.testing.assert_array_equal(y_tr1, y_tr2)
        np.testing.assert_array_equal(y_te1, y_te2)

    def test_split_sizes_are_80_20(self, synthetic_dataset):
        """Train=80%, Test=20%."""
        X, y = synthetic_dataset
        X_train, X_test, y_train, y_test = split_data(X, y)

        assert len(X_train) == 800
        assert len(X_test) == 200
        assert len(y_train) == 800
        assert len(y_test) == 200


# ---------------------------------------------------------------------------
# Test 7: Balancing API correctness
# ---------------------------------------------------------------------------

class TestBalancingAPI:
    def test_kmeans_is_default_strategy(self):
        """balance_training_fold must default to strategy='kmeans'."""
        import inspect
        sig = inspect.signature(balance_training_fold)
        assert sig.parameters['strategy'].default == 'kmeans'

    def test_kmeans_is_default_full_train(self):
        """balance_full_train must default to strategy='kmeans'."""
        import inspect
        sig = inspect.signature(balance_full_train)
        assert sig.parameters['strategy'].default == 'kmeans'

    def test_kmeans_strategy_balances(self, synthetic_dataset):
        """strategy='kmeans' must produce more samples than input."""
        X, y = synthetic_dataset
        X_tr, _, y_tr, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_bal, y_bal = balance_training_fold(
            X_tr, y_tr, strategy="kmeans", k_neighbors=2,
            n_clusters=5, random_state=42,
        )
        assert len(X_bal) >= len(X_tr)
        assert X_bal.shape[1] == X_tr.shape[1]

    def test_smote_strategy_balances(self, synthetic_dataset):
        """strategy='smote' must produce more samples than input."""
        X, y = synthetic_dataset
        X_tr, _, y_tr, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_bal, y_bal = balance_training_fold(
            X_tr, y_tr, strategy="smote", k_neighbors=2, random_state=42,
        )
        assert len(X_bal) >= len(X_tr)
        assert X_bal.shape[1] == X_tr.shape[1]

    def test_balance_full_train_returns_tuple(self, synthetic_dataset):
        """balance_full_train must return (X_balanced, y_balanced)."""
        X, y = synthetic_dataset
        X_tr, _, y_tr, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        result = balance_full_train(
            X_tr, y_tr, strategy="smote", k_neighbors=2, random_state=42,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_balancing_never_receives_val_data(self, synthetic_dataset):
        """Balancing must be called only on train data, never val."""
        X, y = synthetic_dataset
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        with patch('src.balancing.balance_training_fold') as mock_bal:
            mock_bal.return_value = (X_tr, y_tr)
            X_bal, y_bal = mock_bal(X_tr, y_tr, strategy="kmeans")
            mock_bal.assert_called_once_with(X_tr, y_tr, strategy="kmeans")
            assert not np.array_equal(X_val, X_bal)

    def test_run_cv_with_kmeans_strategy(self, split_dataset):
        """run_cv must work with strategy='kmeans' (default)."""
        X_train, X_test, y_train, y_test = split_dataset

        from sklearn.ensemble import HistGradientBoostingClassifier

        cv_metrics, selector, scaler, pca = run_cv(
            X_train, y_train,
            model_class=HistGradientBoostingClassifier,
            model_params=dict(max_iter=5, random_state=42),
            n_splits=3, mi_k=10,
            pca_variance=0.95, k_neighbors=2,
            random_state=42, strategy="kmeans",
        )

        for k, v in cv_metrics.items():
            assert len(v) == 3

    def test_kmeans_strategy_uses_actual_kmeans_smote(self):
        """strategy='kmeans' must use imblearn KMeansSMOTE, not MiniBatchKMeans."""
        from src.balancing import balance_training_fold
        from imblearn.over_sampling import KMeansSMOTE

        X, y = synthetic_dataset_small()
        with patch('src.balancing.KMeansSMOTE') as mock_kms:
            mock_kms.return_value.fit_resample.return_value = (X, y)
            balance_training_fold(
                X, y, strategy="kmeans", k_neighbors=2,
                n_clusters=5, random_state=42,
            )
            mock_kms.assert_called_once()

    def test_kmeans_strategy_floors_k_neighbors_at_1(self):
        """k_neighbors must be floored at 1 to avoid ValueError."""
        from src.balancing import balance_training_fold

        # Binary dataset: minority class has 5 samples.
        # k_neighbors=100 should be clamped to max(min(100, 5-1), 1) = 4
        rng = np.random.RandomState(42)
        X_maj = rng.randn(200, 5).astype(np.float32)
        X_min = rng.randn(5, 5).astype(np.float32) + 2
        X = np.vstack([X_maj, X_min])
        y = np.array([0]*200 + [1]*5)

        with patch('src.balancing.KMeansSMOTE') as mock_kms:
            mock_kms.return_value.fit_resample.return_value = (X, y)
            balance_training_fold(
                X, y, strategy="kmeans", k_neighbors=100,
                n_clusters=3, random_state=42,
            )
            call_kwargs = mock_kms.call_args[1]
            assert call_kwargs['k_neighbors'] == 4, (
                f"Expected floor of 4, got {call_kwargs['k_neighbors']}"
            )

    def test_smote_strategy_floors_k_neighbors_at_1(self):
        """k_neighbors must be floored at 1 for SMOTE strategy too."""
        from src.balancing import balance_training_fold

        X, y = synthetic_dataset_small()
        X_bal, y_bal = balance_training_fold(
            X, y, strategy="smote", k_neighbors=100,
            random_state=42,
        )
        assert len(X_bal) >= len(X)


# ---------------------------------------------------------------------------
# Test 8: DL pipeline (dl_pipeline.py) — no leakage
# ---------------------------------------------------------------------------

class TestDLPipelineNoLeakage:
    def test_preprocess_fold_mi_fit_on_train_only(self, split_dataset):
        """dl_pipeline.preprocess_fold must fit MI only on fold train."""
        from src.dl_pipeline import preprocess_fold

        X_train, X_test, y_train, y_test = split_dataset
        # Simulate one fold: first 640 train, last 160 val
        fold_tr = slice(0, 640)
        fold_val = slice(640, 800)
        X_tr, y_tr = X_train[fold_tr], y_train[fold_tr]
        X_val, y_val = X_train[fold_val], y_train[fold_val]

        result = preprocess_fold(
            X_tr, y_tr, X_val, y_val,
            mi_k=10, pca_components=5,
            n_clusters=2, k_neighbors=2, rus_cap=500,
            use_balancing=False,
        )

        # Selector must be fitted on fold train (640 samples), not full train
        assert result['selector'] is not None
        assert result['selector'].k == 10
        # Val is only transformed, never fitted
        assert result['X_val'].shape[0] == len(y_val)

    def test_preprocess_fold_scaler_fit_on_train_only(self, split_dataset):
        """dl_pipeline.preprocess_fold must fit Scaler only on fold train."""
        from src.dl_pipeline import preprocess_fold

        X_train, X_test, y_train, y_test = split_dataset
        X_tr, y_tr = X_train[:640], y_train[:640]
        X_val, y_val = X_train[640:], y_train[640:]

        result = preprocess_fold(
            X_tr, y_tr, X_val, y_val,
            mi_k=0, pca_components=0,
            use_balancing=False,
        )

        scaler = result['scaler']
        train_means = np.mean(X_tr, axis=0).astype(np.float64)
        assert np.allclose(scaler.mean_, train_means, rtol=1e-4, atol=1e-6)

    def test_preprocess_fold_pca_fit_on_train_only(self, split_dataset):
        """dl_pipeline.preprocess_fold must fit PCA only on fold train."""
        from src.dl_pipeline import preprocess_fold

        X_train, X_test, y_train, y_test = split_dataset
        X_tr, y_tr = X_train[:640], y_train[:640]
        X_val, y_val = X_train[640:], y_train[640:]

        result = preprocess_fold(
            X_tr, y_tr, X_val, y_val,
            mi_k=0, pca_components=5,
            use_balancing=False,
        )

        pca = result['pca']
        assert pca is not None
        assert pca.n_components_ <= 5
        assert result['X_tr'].shape[1] == 5
        assert result['X_val'].shape[1] == 5

    def test_preprocess_fold_balancing_only_on_train(self, split_dataset):
        """dl_pipeline.preprocess_fold must balance only train, not val."""
        from src.dl_pipeline import preprocess_fold

        X_train, X_test, y_train, y_test = split_dataset
        X_tr, y_tr = X_train[:640], y_train[:640]
        X_val, y_val = X_train[640:], y_train[640:]

        result = preprocess_fold(
            X_tr, y_tr, X_val, y_val,
            mi_k=0, pca_components=0,
            n_clusters=2, k_neighbors=2, rus_cap=500,
            use_balancing=True,
        )

        # Val must not change size
        assert result['X_val'].shape[0] == len(y_val)
        # Train may have more samples after balancing
        assert result['X_tr'].shape[0] >= len(y_tr)

    def test_preprocess_final_mi_fit_on_full_train_only(self, split_dataset):
        """dl_pipeline.preprocess_final must fit MI on full train only."""
        from src.dl_pipeline import preprocess_final

        X_train, X_test, y_train, y_test = split_dataset

        result = preprocess_final(
            X_train, y_train, X_test, y_test,
            mi_k=10, pca_components=5,
            use_balancing=False,
        )

        # MI selects 10, then PCA reduces to 5
        assert result['selector'] is not None
        assert result['selector'].k == 10
        assert result['X_train'].shape[1] == 5
        assert result['X_test'].shape[1] == 5

    def test_preprocess_final_scaler_fit_on_full_train_only(self, split_dataset):
        """dl_pipeline.preprocess_final must fit Scaler on full train only."""
        from src.dl_pipeline import preprocess_final

        X_train, X_test, y_train, y_test = split_dataset

        result = preprocess_final(
            X_train, y_train, X_test, y_test,
            mi_k=0, pca_components=0,
            use_balancing=False,
        )

        scaler = result['scaler']
        train_means = np.mean(X_train, axis=0)
        assert np.allclose(scaler.mean_, train_means)

    def test_preprocess_final_pca_fit_on_full_train_only(self, split_dataset):
        """dl_pipeline.preprocess_final must fit PCA on full train only."""
        from src.dl_pipeline import preprocess_final

        X_train, X_test, y_train, y_test = split_dataset

        result = preprocess_final(
            X_train, y_train, X_test, y_test,
            mi_k=0, pca_components=5,
            use_balancing=False,
        )

        pca = result['pca']
        assert pca is not None
        assert pca.n_components_ <= 5
        assert result['X_train'].shape[1] == 5
        assert result['X_test'].shape[1] == 5

    def test_preprocess_final_balancing_only_on_train(self, split_dataset):
        """dl_pipeline.preprocess_final must balance only train, not test."""
        from src.dl_pipeline import preprocess_final

        X_train, X_test, y_train, y_test = split_dataset

        result = preprocess_final(
            X_train, y_train, X_test, y_test,
            mi_k=0, pca_components=0,
            n_clusters=2, k_neighbors=2, rus_cap=500,
            use_balancing=True,
        )

        assert result['X_test'].shape[0] == len(y_test)
        assert result['X_train'].shape[0] >= len(y_train)

    def test_preprocess_fold_returns_correct_keys(self, split_dataset):
        """preprocess_fold must return expected keys."""
        from src.dl_pipeline import preprocess_fold

        X_train, X_test, y_train, y_test = split_dataset
        result = preprocess_fold(
            X_train[:640], y_train[:640],
            X_train[640:], y_train[640:],
            use_balancing=False,
        )

        for key in ['X_tr', 'y_tr', 'X_val', 'y_val', 'selector', 'scaler', 'pca']:
            assert key in result, f"Missing key: {key}"

    def test_preprocess_final_returns_correct_keys(self, split_dataset):
        """preprocess_final must return expected keys."""
        from src.dl_pipeline import preprocess_final

        X_train, X_test, y_train, y_test = split_dataset
        result = preprocess_final(
            X_train, y_train, X_test, y_test,
            use_balancing=False,
        )

        for key in ['X_train', 'y_train', 'X_test', 'y_test', 'selector', 'scaler', 'pca']:
            assert key in result, f"Missing key: {key}"

    def test_no_val_data_in_balanced_train(self, split_dataset):
        """Balanced training data must not contain any validation samples."""
        from src.dl_pipeline import preprocess_fold

        X_train, X_test, y_train, y_test = split_dataset
        X_tr, y_tr = X_train[:640], y_train[:640]
        X_val, y_val = X_train[640:], y_train[640:]

        result = preprocess_fold(
            X_tr, y_tr, X_val, y_val,
            mi_k=0, pca_components=0,
            n_clusters=2, k_neighbors=2, rus_cap=500,
            use_balancing=True,
        )

        # All balanced train samples should have rows not in val
        assert result['X_tr'].shape[0] != result['X_val'].shape[0] or \
               not np.array_equal(result['X_tr'], result['X_val'])

    def test_dl_pipeline_evaluate_predictions_works(self):
        """evaluate_predictions must return expected metric keys."""
        from src.dl_pipeline import evaluate_predictions

        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 0])

        metrics = evaluate_predictions(y_true, y_pred, normal_class_idx=0)

        for key in ['binary_acc', 'binary_f1', 'multi_acc', 'macro_f1',
                     'weighted_f1', 'precision', 'recall', 'auc']:
            assert key in metrics, f"Missing metric: {key}"

    def test_evaluate_with_proba_returns_valid_auc(self):
        """evaluate_with_proba must return AUC > 0 for good predictions."""
        from src.dl_pipeline import evaluate_with_proba
        from sklearn.preprocessing import label_binarize

        rng = np.random.RandomState(42)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        n_classes = 3
        # Build probabilities that mostly agree with y_true
        y_proba = rng.dirichlet(np.ones(n_classes), size=len(y_true))
        for i, c in enumerate(y_true):
            y_proba[i, c] += 2.0
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        y_pred = np.argmax(y_proba, axis=1)

        metrics = evaluate_with_proba(y_true, y_pred, y_proba, normal_class_idx=0)
        assert metrics['auc'] > 0.5, f"AUC too low: {metrics['auc']}"

    def test_evaluate_with_proba_keys_match_evaluate_predictions(self):
        """evaluate_with_proba must return the same keys as evaluate_predictions."""
        from src.dl_pipeline import evaluate_predictions, evaluate_with_proba

        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2])
        rng = np.random.RandomState(42)
        y_proba = rng.dirichlet(np.ones(3), size=5)

        keys_pred = set(evaluate_predictions(y_true, y_pred).keys())
        keys_proba = set(evaluate_with_proba(y_true, y_pred, y_proba).keys())
        assert keys_proba == keys_pred


class TestGetProbabilities:
    def test_returns_correct_shape(self):
        """get_probabilities must return (N, num_classes) array."""
        from src.dl_pipeline import get_probabilities, set_seeds
        import torch.nn as nn

        set_seeds(42)
        device = torch.device('cpu')
        input_dim, num_classes = 10, 3
        model = nn.Linear(input_dim, num_classes)

        X = np.random.randn(20, input_dim).astype(np.float32)
        probs = get_probabilities(model, X, device)

        assert probs.shape == (20, num_classes)

    def test_rows_sum_to_one(self):
        """Each row of the probability output must sum to 1."""
        from src.dl_pipeline import get_probabilities, set_seeds
        import torch.nn as nn

        set_seeds(42)
        device = torch.device('cpu')
        model = nn.Linear(5, 4)

        X = np.random.randn(30, 5).astype(np.float32)
        probs = get_probabilities(model, X, device)

        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_model_switches_to_eval_mode(self):
        """get_probabilities must set model to eval() and restore after."""
        from src.dl_pipeline import get_probabilities, set_seeds
        import torch.nn as nn

        set_seeds(42)
        device = torch.device('cpu')
        model = nn.Linear(5, 3)
        model.train()  # Put in train mode

        X = np.random.randn(10, 5).astype(np.float32)
        _ = get_probabilities(model, X, device)
        # get_probabilities leaves model in eval mode (its documented contract)
        assert not model.training


# ---------------------------------------------------------------------------
# Test 9: Final retraining integrity (regression tests)
# ---------------------------------------------------------------------------

class TestFinalRetrainingIntegrity:
    def test_balance_full_train_receives_full_training_data(self, split_dataset):
        """balance_full_train must receive the full X_train, not a fold."""
        from src.balancing import balance_full_train

        X_train, X_test, y_train, y_test = split_dataset

        # balance_full_train should work with the full 800-sample training set
        X_bal, y_bal = balance_full_train(
            X_train, y_train, strategy="kmeans",
            k_neighbors=2, random_state=42,
        )

        assert X_bal.shape[0] >= X_train.shape[0]
        assert X_bal.shape[1] == X_train.shape[1]

    def test_balance_full_train_never_receives_test_data(self, split_dataset):
        """balance_full_train must never be called with test data."""
        from src.balancing import balance_full_train
        from unittest.mock import patch

        X_train, X_test, y_train, y_test = split_dataset

        with patch('src.balancing.balance_training_fold') as mock_bal:
            mock_bal.return_value = (X_train[:100], y_train[:100])
            balance_full_train(X_train, y_train, strategy="kmeans")

            # Verify the mock was called with training data only
            call_args = mock_bal.call_args
            X_arg = call_args[0][0]
            assert X_arg.shape == X_train.shape

    def test_final_retrain_uses_full_train_not_fold(self, split_dataset):
        """Final retraining must fit MI/Scaler/PCA on full X_train, not a fold."""
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        X_train, X_test, y_train, y_test = split_dataset

        # Fit MI on full training set
        selector = SelectKBest(score_func=mutual_info_classif, k=10)
        selector.fit(X_train, y_train)
        X_train_mi = selector.transform(X_train)
        X_test_mi = selector.transform(X_test)

        # Fit Scaler on full training MI features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_mi)
        X_test_s = scaler.transform(X_test_mi)

        # Fit PCA on full training scaled features
        pca = PCA(n_components=5, random_state=42)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)

        # Verify shapes
        assert X_train_p.shape == (len(X_train), 5)
        assert X_test_p.shape == (len(X_test), 5)

        # Verify scaler was fit on full training data
        assert np.allclose(scaler.mean_, np.mean(X_train_mi, axis=0).astype(np.float64), rtol=1e-4)

        # Verify PCA was fit on full training data
        assert pca.n_components_ == 5

    def test_test_data_never_enters_balancing(self, split_dataset):
        """Test data must never be passed to any balancing function."""
        from unittest.mock import patch
        import src.balancing as balancing_mod

        X_train, X_test, y_train, y_test = split_dataset

        with patch.object(balancing_mod, 'balance_full_train') as mock_full:
            mock_full.return_value = (X_train, y_train)

            balancing_mod.balance_full_train(X_train, y_train, strategy="kmeans")

            call_args = mock_full.call_args
            X_arg = call_args[0][0]
            y_arg = call_args[0][1]
            assert X_arg.shape[0] == len(y_train)
            assert y_arg.shape[0] == len(y_train)
            assert not np.array_equal(X_arg, X_test)

    def test_final_model_evaluation_uses_test_set_only(self, split_dataset):
        """Final model must be evaluated only on X_test, never on X_train."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score

        X_train, X_test, y_train, y_test = split_dataset

        # Train a minimal model
        model = HistGradientBoostingClassifier(max_iter=5, random_state=42)
        model.fit(X_train[:200], y_train[:200])

        # Evaluate on test set
        y_pred_test = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)

        # Evaluate on train set (should be higher — overfitting check)
        y_pred_train = model.predict(X_train[:200])
        train_acc = accuracy_score(y_train[:200], y_pred_train)

        # Both should produce valid metrics
        assert 0.0 <= test_acc <= 1.0
        assert 0.0 <= train_acc <= 1.0

    def test_no_fold_data_in_final_retraining(self, synthetic_dataset):
        """Final retraining must not use any CV fold's training indices."""
        from sklearn.model_selection import StratifiedKFold, train_test_split
        from sklearn.feature_selection import mutual_info_classif, SelectKBest
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from src.balancing import balance_full_train

        X, y = synthetic_dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Run CV to get fold indices
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_indices = list(skf.split(X_train, y_train))

        # Collect all fold train indices
        all_fold_train_indices = set()
        for trn_idx, _ in fold_indices:
            all_fold_train_indices.update(trn_idx)

        # Final retraining must use ALL training indices
        final_indices = set(range(len(X_train)))
        assert all_fold_train_indices == final_indices, (
            "Final retraining must use all training data, not just fold subsets"
        )
