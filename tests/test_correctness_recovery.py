import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src import balancing, cross_validation
from src.evaluation import (
    attack_probabilities,
    compute_extended_metrics,
    normal_attack_labels,
    resolve_normal_class_idx,
)
from src.feature_selection import fit_mi_selector
from src.preprocessing import (
    CATEGORICAL_COLUMNS,
    EXPECTED_RAW_COLUMNS,
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    feature_metadata,
    feature_type_mask,
    fit_categorical_encoder,
    transform_features,
)


def test_canonical_schema_and_feature_metadata():
    expected = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
        "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
        "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb",
        "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
        "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt",
        "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
        "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
        "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
    ]
    metadata = feature_metadata()
    assert len(RAW_COLUMNS) == EXPECTED_RAW_COLUMNS == 49
    assert len(set(RAW_COLUMNS)) == len(RAW_COLUMNS)
    assert RAW_COLUMNS == expected
    assert RAW_COLUMNS[28] == "stime"
    assert RAW_COLUMNS[29] == "ltime"
    assert RAW_COLUMNS[46:] == ["ct_dst_src_ltm", "attack_cat", "label"]
    assert "ct_dst_host_ltm" not in RAW_COLUMNS
    assert FEATURE_COLUMNS[0] == "sport"
    assert FEATURE_COLUMNS[-1] == "ct_dst_src_ltm"
    assert len(metadata["feature_names"]) == len(FEATURE_COLUMNS)
    assert len(metadata["discrete_mask"]) == len(FEATURE_COLUMNS)
    assert [FEATURE_COLUMNS[i] for i in metadata["discrete_indices"]] == list(CATEGORICAL_COLUMNS)


def test_transform_requires_canonical_feature_order_and_handles_unknown_category():
    train = pd.DataFrame({name: [1, 2] for name in FEATURE_COLUMNS})
    for name in CATEGORICAL_COLUMNS:
        train[name] = ["known", "other"]
    encoder = fit_categorical_encoder(train)
    deployment = train.iloc[[0]].copy()
    deployment["proto"] = "unseen"
    transformed = transform_features(deployment, encoder)
    assert transformed.shape == (1, len(FEATURE_COLUMNS))
    assert np.isfinite(transformed).all()


def test_mi_is_discrete_aware_deterministic_and_uses_requested_k():
    rng = np.random.default_rng(42)
    X = np.column_stack([
        rng.integers(0, 3, size=80),
        rng.normal(size=80),
        rng.normal(size=80),
    ]).astype(np.float32)
    y = (X[:, 0] == 1).astype(int)
    mask = np.array([True, False, False])
    first = fit_mi_selector(X, y, k=2, random_state=42, discrete_features=mask)
    second = fit_mi_selector(X, y, k=2, random_state=42, discrete_features=mask)
    assert first.k == 2
    assert first.feature_type_mask_.tolist() == mask.tolist()
    assert first.mi_random_state_ == 42
    assert np.allclose(first.scores_, second.scores_, equal_nan=True)
    assert np.array_equal(first.get_support(), second.get_support())


def test_semantic_normal_mapping_and_attack_probability():
    classes = ["Analysis", "Backdoor", "Normal", "Generic"]
    normal_idx = resolve_normal_class_idx(classes)
    assert normal_idx == 2
    assert normal_attack_labels(np.array([0, 2, 3]), normal_idx).tolist() == [1, 0, 1]
    probabilities = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.8, 0.0]])
    assert np.allclose(attack_probabilities(probabilities, normal_idx), [0.9, 0.2])
    metrics = compute_extended_metrics(
        np.array([0, 2]), np.array([0, 2]), probabilities, normal_idx,
    )
    assert metrics["binary_acc"] == 1.0


@pytest.mark.parametrize("n_clusters", [8, 20])
def test_kmeans_cluster_count_reaches_estimator(monkeypatch, n_clusters):
    captured = {}

    class FakeKMeans:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeSampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_resample(self, X, y):
            return X, y

    monkeypatch.setattr(balancing, "MiniBatchKMeans", FakeKMeans)
    monkeypatch.setattr(balancing, "KMeansSMOTE", FakeSampler)
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.array([0] * 6 + [1] * 6)
    balancing.balance_training_fold(X, y, n_clusters=n_clusters, k_neighbors=2)
    assert captured["n_clusters"] == n_clusters


def test_cv_fits_selector_on_fold_train_and_never_balances_validation(monkeypatch):
    X = pd.DataFrame({name: np.arange(20) for name in FEATURE_COLUMNS})
    for name in CATEGORICAL_COLUMNS:
        X[name] = ["tcp" if i % 2 else "udp" for i in range(20)]
    y = np.array([0, 1] * 10)
    fit_sizes = []
    balance_sizes = []

    def fake_mi(X_fit, y_fit, **kwargs):
        fit_sizes.append(len(X_fit))
        return fit_mi_selector(
            X_fit, y_fit, k=1, random_state=42,
            discrete_features=kwargs["discrete_features"],
        )

    def fake_balance(X_train, y_train, **kwargs):
        balance_sizes.append(len(X_train))
        return X_train, y_train

    class TinyModel:
        def fit(self, X_fit, y_fit):
            return self

        def predict(self, X_eval):
            return np.zeros(len(X_eval), dtype=int)

        def predict_proba(self, X_eval):
            return np.tile([0.5, 0.5], (len(X_eval), 1))

    monkeypatch.setattr(cross_validation, "fit_mi_selector", fake_mi)
    monkeypatch.setattr(cross_validation, "balance_training_fold", fake_balance)
    metrics, *_ = cross_validation.run_cv(
        X, y, TinyModel, {}, n_splits=2, mi_k=1, use_pca=False, use_balancing=True,
    )
    assert len(metrics["accuracy"]) == 2
    assert fit_sizes == [10, 10]
    assert balance_sizes == [10, 10]


def test_manifest_contains_raw_inference_metadata(tmp_path):
    from src.dl_pipeline import save_dl_artifacts
    import torch

    model = torch.nn.Linear(2, 2)
    save_dl_artifacts(
        model, "RecoveryTest", cv_metrics=[], experiment="raw", save_root=str(tmp_path),
        scaler=StandardScaler().fit([[0, 0], [1, 1]]), class_names=["Analysis", "Normal"],
        normal_class_idx=1,
    )
    manifest = json.loads((tmp_path / "RecoveryTest" / "raw" / "preprocessing_manifest.json").read_text())
    required = {"schema_version", "raw_columns", "feature_names", "dropped_columns",
                "categorical_columns", "numeric_columns", "discrete_mask", "unknown_category_policy"}
    assert required.issubset(manifest)
