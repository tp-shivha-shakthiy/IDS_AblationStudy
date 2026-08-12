"""
test_ablation.py
================
Tests for the faculty-requested 7-experiment ablation study:

  1. Preset registry -- exactly the 7 required experiments with correct flags
  2. resolve_experiment() -- flag resolution + unknown-name error
  3. run_cv() preprocessing toggles -- MI/PCA/balancing can be disabled
     without leaking (val/test still only transformed)
  4. train_and_evaluate() -- per-experiment outputs (test_metrics.json,
     cv_metrics.csv, experiment_config.json) under results/<Model>/<exp>/
  5. compute_extended_metrics() -- binary + multiclass metric keys
  6. save_model_ablation_tables() -- exactly 7 ordered rows per model
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split

from src.experiment_config import (
    ABLATION_PRESETS,
    ABLATION_ORDER,
    ABLATION_DISPLAY_NAMES,
    resolve_experiment,
    build_experiment_config,
)
from src.evaluation import compute_extended_metrics, build_model_ablation_rows


@pytest.fixture
def synthetic_dataset():
    rng = np.random.RandomState(42)
    X = rng.randn(500, 12).astype(np.float32)
    y = rng.choice([0, 1, 2, 3, 4], size=500, p=[0.4, 0.2, 0.15, 0.15, 0.1])
    return X, y


@pytest.fixture
def split_dataset(synthetic_dataset):
    X, y = synthetic_dataset
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# ---------------------------------------------------------------------------
# 1. Preset registry
# ---------------------------------------------------------------------------

class TestAblationPresets:
    def test_exactly_seven_presets(self):
        assert set(ABLATION_PRESETS.keys()) == {
            "raw", "mi", "mi_balancing", "pca", "pca_balancing",
            "mi_pca", "mi_pca_balancing",
        }
        assert len(ABLATION_PRESETS) == 7

    def test_flag_table_matches_spec(self):
        expected = {
            "raw":              (False, False, False),
            "mi":               (True,  False, False),
            "mi_balancing":     (True,  False, True),
            "pca":              (False, True,  False),
            "pca_balancing":    (False, True,  True),
            "mi_pca":           (True,  True,  False),
            "mi_pca_balancing": (True,  True,  True),
        }
        for name, (mi, pca, bal) in expected.items():
            p = ABLATION_PRESETS[name]
            assert (p["use_mi"], p["use_pca"], p["use_balancing"]) == (mi, pca, bal)

    def test_canonical_order_has_seven_rows(self):
        assert ABLATION_ORDER == [
            "raw", "mi", "mi_balancing", "pca", "pca_balancing",
            "mi_pca", "mi_pca_balancing",
        ]
        assert len(ABLATION_ORDER) == 7

    def test_display_labels(self):
        assert ABLATION_DISPLAY_NAMES["raw"] == "Raw"
        assert ABLATION_DISPLAY_NAMES["mi_balancing"] == "MI+KMeansSMOTE"
        assert ABLATION_DISPLAY_NAMES["mi_pca_balancing"] == "MI+PCA+KMeansSMOTE"


# ---------------------------------------------------------------------------
# 2. resolve_experiment()
# ---------------------------------------------------------------------------

class TestResolveExperiment:
    def test_resolve_default(self):
        assert resolve_experiment("mi_pca_balancing") == {
            "use_mi": True, "use_pca": True, "use_balancing": True
        }

    def test_resolve_raw(self):
        assert resolve_experiment("raw") == {
            "use_mi": False, "use_pca": False, "use_balancing": False
        }

    def test_unknown_experiment_raises(self):
        with pytest.raises(ValueError):
            resolve_experiment("not_a_preset")

    def test_config_contains_experiment_key(self):
        cfg = build_experiment_config(
            model_name="HGB",
            experiment_name="pca_balancing",
            use_mi=False, use_pca=True, use_balancing=True,
            tier=1,
        )
        assert cfg["experiment"] == "pca_balancing"
        assert cfg["use_mi"] is False
        assert cfg["use_pca"] is True
        assert cfg["use_balancing"] is True


# ---------------------------------------------------------------------------
# 3. run_cv() preprocessing toggles
# ---------------------------------------------------------------------------

class TestRunCvToggles:
    def _run_cv(self, X_train, y_train, **kwargs):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from src.cross_validation import run_cv
        return run_cv(
            X_train, y_train,
            model_class=HistGradientBoostingClassifier,
            model_params=dict(max_iter=5, random_state=42),
            n_splits=2, mi_k=8, pca_variance=0.95, k_neighbors=2,
            random_state=42, strategy="smote",
            **kwargs,
        )

    def test_raw_returns_no_transformers(self, split_dataset):
        X_train, X_test, y_train, y_test = split_dataset
        metrics, selector, scaler, pca = self._run_cv(
            X_train, y_train,
            use_mi=False, use_pca=False, use_balancing=False,
        )
        assert selector is None
        assert pca is None
        assert scaler is not None
        for k, v in metrics.items():
            assert len(v) == 2

    def test_mi_only(self, split_dataset):
        X_train, X_test, y_train, y_test = split_dataset
        _, selector, _, pca = self._run_cv(
            X_train, y_train,
            use_mi=True, use_pca=False, use_balancing=False,
        )
        assert selector is not None
        assert pca is None

    def test_pca_only(self, split_dataset):
        X_train, X_test, y_train, y_test = split_dataset
        _, selector, _, pca = self._run_cv(
            X_train, y_train,
            use_mi=False, use_pca=True, use_balancing=False,
        )
        assert selector is None
        assert pca is not None

    def test_full_pipeline_defaults(self, split_dataset):
        X_train, X_test, y_train, y_test = split_dataset
        _, selector, scaler, pca = self._run_cv(X_train, y_train)
        assert selector is not None
        assert scaler is not None
        assert pca is not None

    def test_balancing_off_skips_smote(self, split_dataset):
        from unittest.mock import patch
        X_train, X_test, y_train, y_test = split_dataset

        def _raise(*a, **k):
            raise AssertionError("balance_training_fold must not be called "
                                 "when use_balancing=False")

        with patch("src.cross_validation.balance_training_fold") as mock_bal:
            mock_bal.side_effect = _raise
            metrics, _, _, _ = self._run_cv(
                X_train, y_train,
                use_mi=False, use_pca=False, use_balancing=False,
            )
            for k, v in metrics.items():
                assert len(v) == 2

    def test_mi_selector_fitted_on_fold_train_only(self, split_dataset):
        from src.cross_validation import run_cv
        from sklearn.ensemble import HistGradientBoostingClassifier
        X_train, X_test, y_train, y_test = split_dataset

        _, selector, scaler, _ = run_cv(
            X_train, y_train,
            model_class=HistGradientBoostingClassifier,
            model_params=dict(max_iter=5, random_state=42),
            n_splits=2, mi_k=8, pca_variance=0.95, k_neighbors=2,
            random_state=42, strategy="smote",
            use_mi=True, use_pca=True, use_balancing=False,
        )
        assert len(selector.scores_) == X_train.shape[1]
        assert scaler.mean_.shape[0] == 8  # MI-selected feature count

    @pytest.mark.parametrize("experiment", ABLATION_ORDER)
    def test_every_preset_exercises_its_preprocessing_branches(
        self, split_dataset, experiment
    ):
        from unittest.mock import patch
        from src.cross_validation import run_cv
        from src.feature_selection import fit_mi_selector
        from src.balancing import balance_training_fold
        from sklearn.ensemble import HistGradientBoostingClassifier

        X_train, _, y_train, _ = split_dataset
        flags = ABLATION_PRESETS[experiment]
        with patch("src.cross_validation.fit_mi_selector", wraps=fit_mi_selector) as mi_mock, \
             patch("src.cross_validation.balance_training_fold", wraps=balance_training_fold) as balance_mock:
            _, selector, _, pca = run_cv(
                X_train, y_train,
                model_class=HistGradientBoostingClassifier,
                model_params=dict(max_iter=5, random_state=42),
                n_splits=2, mi_k=8, pca_variance=0.95, k_neighbors=2,
                random_state=42, strategy="smote",
                use_mi=flags["use_mi"], use_pca=flags["use_pca"],
                use_balancing=flags["use_balancing"],
            )

        assert (mi_mock.call_count > 0) is flags["use_mi"]
        assert (selector is not None) is flags["use_mi"]
        assert (pca is not None) is flags["use_pca"]
        assert (balance_mock.call_count > 0) is flags["use_balancing"]
        for call in balance_mock.call_args_list:
            # Each fold train is 200 samples; validation is the other 200.
            assert call.args[0].shape[0] == 200


# ---------------------------------------------------------------------------
# 4. train_and_evaluate() per-experiment outputs
# ---------------------------------------------------------------------------

class TestTrainAndEvaluateExperiment:
    def test_raw_experiment_outputs(self, split_dataset, tmp_path, monkeypatch):
        from src.model_training import train_and_evaluate

        X_train, X_test, y_train, y_test = split_dataset
        monkeypatch.chdir(tmp_path)

        res = train_and_evaluate(
            "HGB", X_train, y_train, X_test, y_test,
            class_names=["c0", "c1", "c2", "c3", "c4"],
            n_splits=2, mi_k=8, pca_variance=0.95, k_neighbors=2,
            use_mi=False, use_pca=False, use_balancing=False,
            experiment="raw",
        )

        exp_dir = os.path.join("results", "HGB", "raw")
        assert res["experiment"] == "raw"
        assert res["selector"] is None
        assert res["pca"] is None
        assert os.path.isfile(os.path.join(exp_dir, "test_metrics.json"))
        assert os.path.isfile(os.path.join(exp_dir, "cv_metrics.csv"))
        assert os.path.isfile(os.path.join(exp_dir, "experiment_config.json"))
        assert os.path.isfile(os.path.join(exp_dir, "hgb_model.joblib"))

        with open(os.path.join(exp_dir, "test_metrics.json")) as f:
            tm = json.load(f)
        for key in ["accuracy", "macro_f1", "weighted_f1",
                    "binary_acc", "binary_f1", "binary_auc", "auc"]:
            assert key in tm, f"Missing extended metric: {key}"

        with open(os.path.join(exp_dir, "experiment_config.json")) as f:
            cfg = json.load(f)
        assert cfg["experiment"] == "raw"
        assert cfg["use_mi"] is False
        assert cfg["use_pca"] is False
        assert cfg["use_balancing"] is False

        cv_df = pd.read_csv(os.path.join(exp_dir, "cv_metrics.csv"))
        assert "accuracy" in cv_df.columns
        assert len(cv_df) == 2  # 2 CV folds

    def test_mi_pca_balancing_default_outputs(self, split_dataset, tmp_path, monkeypatch):
        from src.model_training import train_and_evaluate

        X_train, X_test, y_train, y_test = split_dataset
        monkeypatch.chdir(tmp_path)

        res = train_and_evaluate(
            "HGB", X_train, y_train, X_test, y_test,
            class_names=["c0", "c1", "c2", "c3", "c4"],
            n_splits=2, mi_k=8, pca_variance=0.95, k_neighbors=2,
            use_mi=True, use_pca=True, use_balancing=True,
            experiment="mi_pca_balancing",
        )
        exp_dir = os.path.join("results", "HGB", "mi_pca_balancing")
        assert res["selector"] is not None
        assert res["pca"] is not None
        assert os.path.isfile(os.path.join(exp_dir, "experiment_config.json"))


# ---------------------------------------------------------------------------
# 5. compute_extended_metrics()
# ---------------------------------------------------------------------------

class TestComputeExtendedMetrics:
    def test_keys_present(self):
        y_true = np.array([0, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 2])
        rng = np.random.RandomState(42)
        y_proba = rng.dirichlet(np.ones(3), size=6)
        for i, c in enumerate(y_true):
            y_proba[i, c] += 2.0
        y_proba /= y_proba.sum(axis=1, keepdims=True)

        m = compute_extended_metrics(y_true, y_pred, y_proba=y_proba)
        for key in ["accuracy", "precision", "recall", "f1", "auc",
                    "multi_acc", "macro_f1", "weighted_f1",
                    "binary_acc", "binary_f1", "binary_auc"]:
            assert key in m, f"Missing key: {key}"
        assert 0.0 <= m["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# 6. Ablation comparison tables
# ---------------------------------------------------------------------------

class TestAblationTables:
    def _fabricate_experiments(self, root, model="HGB"):
        for exp in ABLATION_ORDER:
            exp_dir = os.path.join(root, model, exp)
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "test_metrics.json"), 'w') as f:
                json.dump({
                    "accuracy": 0.9, "precision": 0.88, "recall": 0.87,
                    "f1": 0.88, "auc": 0.95, "multi_acc": 0.9,
                    "macro_f1": 0.85, "weighted_f1": 0.88,
                    "binary_acc": 0.93, "binary_f1": 0.92, "binary_auc": 0.97,
                }, f)
            pd.DataFrame([
                {"accuracy": 0.9, "precision": 0.88, "recall": 0.87,
                 "f1": 0.88, "auc": 0.95},
                {"accuracy": 0.91, "precision": 0.89, "recall": 0.88,
                 "f1": 0.89, "auc": 0.96},
            ]).to_csv(os.path.join(exp_dir, "cv_metrics.csv"), index=False)
            with open(os.path.join(exp_dir, "experiment_config.json"), 'w') as f:
                json.dump({
                    "experiment": exp, "experiment_name": exp,
                    "tier": 1, "ablation_scope": "tier1",
                    **ABLATION_PRESETS[exp], "seed": 42,
                    "feature_selection_k": 15, "pca_variance": 0.95,
                    "balancer": "kmeans", "cv_folds": 5,
                    "balancer_k_neighbors": 3, "balancer_rus_cap": 0,
                }, f)
            open(os.path.join(exp_dir, f"{model.lower()}_model.joblib"), 'wb').close()

    def test_table_has_seven_rows_in_order(self, tmp_path):
        from src.evaluation import save_model_ablation_tables

        self._fabricate_experiments(str(tmp_path))
        save_model_ablation_tables("HGB", results_root=str(tmp_path))

        df = pd.read_csv(os.path.join(str(tmp_path), "HGB", "ablation_test_metrics.csv"))
        assert len(df) == 7
        assert df["Preprocessing"].tolist() == [
            "Raw", "MI", "MI+KMeansSMOTE", "PCA",
            "PCA+KMeansSMOTE", "MI+PCA", "MI+PCA+KMeansSMOTE",
        ]

        cv_df = pd.read_csv(os.path.join(str(tmp_path), "HGB", "ablation_cv_metrics.csv"))
        assert len(cv_df) == 7
        assert cv_df["Preprocessing"].tolist() == df["Preprocessing"].tolist()

    def test_build_rows_rejects_missing_experiments(self, tmp_path):
        self._fabricate_experiments(str(tmp_path))
        exp = ABLATION_ORDER[3]
        os.remove(os.path.join(str(tmp_path), "HGB", exp, "test_metrics.json"))

        with pytest.raises(ValueError, match="pca.*test_metrics.json"):
            build_model_ablation_rows("HGB", str(tmp_path))

    def test_build_rows_rejects_configuration_mismatch(self, tmp_path):
        self._fabricate_experiments(str(tmp_path))
        path = os.path.join(str(tmp_path), "HGB", "raw", "experiment_config.json")
        with open(path) as f:
            config = json.load(f)
        config["use_balancing"] = True
        with open(path, 'w') as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="use_balancing"):
            build_model_ablation_rows("HGB", str(tmp_path))

    def test_build_rows_rejects_noncanonical_cv_metadata(self, tmp_path):
        self._fabricate_experiments(str(tmp_path))
        path = os.path.join(str(tmp_path), "HGB", "raw", "experiment_config.json")
        with open(path) as f:
            config = json.load(f)
        config["cv_folds"] = 2
        with open(path, 'w') as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="cv_folds=5"):
            build_model_ablation_rows("HGB", str(tmp_path))

    def test_build_rows_accepts_tier2_ablation_scope(self, tmp_path):
        """DL models with ablation_scope='tier2' must aggregate into 7 rows."""
        model = "DNN"
        for exp in ABLATION_ORDER:
            exp_dir = os.path.join(str(tmp_path), model, exp)
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "test_metrics.json"), 'w') as f:
                json.dump({"accuracy": 0.9, "f1": 0.88, "auc": 0.95}, f)
            pd.DataFrame([
                {"accuracy": 0.9, "precision": 0.88, "recall": 0.87,
                 "f1": 0.88, "auc": 0.95},
            ]).to_csv(os.path.join(exp_dir, "cv_metrics.csv"), index=False)
            with open(os.path.join(exp_dir, "experiment_config.json"), 'w') as f:
                json.dump({
                    "experiment": exp, "experiment_name": exp,
                    "tier": 2, "ablation_scope": "tier2",
                    **ABLATION_PRESETS[exp], "seed": 42,
                    "feature_selection_k": 15, "pca_variance": 0.95,
                    "balancer": "kmeans", "cv_folds": 5,
                    "balancer_k_neighbors": 3, "balancer_rus_cap": 0,
                }, f)
            open(os.path.join(exp_dir, "dnn_model.joblib"), 'wb').close()

        summary_rows, cv_rows = build_model_ablation_rows(model, str(tmp_path))
        assert len(summary_rows) == 7
        assert len(cv_rows) == 7
        assert summary_rows[0]["Preprocessing"] == "Raw"
        assert summary_rows[6]["Preprocessing"] == "MI+PCA+KMeansSMOTE"

    def test_build_rows_rejects_excluded_tier2(self, tmp_path):
        """Non-ablation tier-2 artifacts must still be rejected."""
        self._fabricate_experiments(str(tmp_path))
        path = os.path.join(str(tmp_path), "HGB", "raw", "experiment_config.json")
        with open(path) as f:
            config = json.load(f)
        config["ablation_scope"] = "excluded_tier2"
        with open(path, 'w') as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="not a Tier 1 or Tier 2"):
            build_model_ablation_rows("HGB", str(tmp_path))

    def test_synthetic_seven_experiment_kmeans_run_aggregates(self, split_dataset, tmp_path, monkeypatch):
        from src.model_training import train_and_evaluate
        from src.evaluation import save_model_ablation_tables
        import src.model_training as training

        X_train, X_test, y_train, y_test = split_dataset
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(training, "plot_confusion_matrix", lambda *args, **kwargs: None)
        monkeypatch.setattr(training, "plot_roc_curve", lambda *args, **kwargs: None)
        monkeypatch.setattr(training, "plot_feature_importance", lambda *args, **kwargs: None)

        for experiment in ABLATION_ORDER:
            flags = ABLATION_PRESETS[experiment]
            train_and_evaluate(
                "HGB", X_train, y_train, X_test, y_test,
                class_names=["c0", "c1", "c2", "c3", "c4"],
                n_splits=5, mi_k=15, pca_variance=0.95, k_neighbors=3,
                use_mi=flags["use_mi"], use_pca=flags["use_pca"],
                use_balancing=flags["use_balancing"], experiment=experiment,
            )

        save_model_ablation_tables("HGB")
        table = pd.read_csv("results/HGB/ablation_test_metrics.csv")
        assert len(table) == 7
        assert table["Preprocessing"].tolist() == [
            ABLATION_DISPLAY_NAMES[experiment] for experiment in ABLATION_ORDER
        ]
