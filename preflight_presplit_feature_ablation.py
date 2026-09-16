"""No-model preflight for the presplit feature-ablation protocol."""
import ast
import time
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.balancing import balance_full_train, balance_training_fold
from src.preprocessing import load_and_prepare
from src.presplit_feature_ablation_protocol import apply_presplit_feature_processing


def counts(y):
    return dict(sorted(Counter(np.asarray(y)).items()))


def main():
    X, y, _ = load_and_prepare("data/raw")
    print("Quick stratified subsamples are unsuitable for KMeansSMOTE validation because rare UNSW-NB15 attack classes become underrepresented.")
    X_e1, e1_dimensions, _, _ = apply_presplit_feature_processing(X, y, "E1_sequential_mi_pca")
    X_e2, e2_dimensions, _, _ = apply_presplit_feature_processing(X, y, "E2_parallel_mi_pca")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    e1_train, e1_test = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    e2_train, e2_test = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    print("COMPLETE", counts(y))
    print("TRAIN", counts(y[train_idx]))
    print("TEST", counts(y[test_idx]))
    print("E1/E2 OUTER SPLIT IDENTICAL:", np.array_equal(e1_train, e2_train) and np.array_equal(e1_test, e2_test))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(train_idx, y[train_idx]))
    e2_folds = list(cv.split(train_idx, y[train_idx]))
    print("E1/E2 CV FOLDS IDENTICAL:", all(np.array_equal(a, c) and np.array_equal(b, d) for (a,b),(c,d) in zip(folds,e2_folds)))
    for number, (fold_train, fold_validation) in enumerate(folds, 1):
        X_fold, y_fold = X_e1[train_idx][fold_train], y[train_idx][fold_train]
        y_val = y[train_idx][fold_validation]
        print(f"FOLD {number} TRAIN", counts(y_fold), "VALIDATION", counts(y_val),
              "MIN_TRAIN", min(counts(y_fold).values()), "MIN_VALIDATION", min(counts(y_val).values()))
        start = time.perf_counter()
        try:
            X_out, y_out = balance_training_fold(X_fold, y_fold, strategy="kmeans", k_neighbors=3, random_state=42, rus_cap=0)
            print(f"BALANCING PREFLIGHT FOLD {number}: PASS", X_fold.shape, counts(y_fold), X_out.shape, counts(y_out), time.perf_counter() - start)
        except Exception as exc:
            print(f"BALANCING PREFLIGHT FOLD {number}: FAIL", repr(exc), time.perf_counter() - start)
    X_train, y_train = X_e1[train_idx], y[train_idx]
    start = time.perf_counter()
    try:
        X_out, y_out = balance_full_train(X_train, y_train, strategy="kmeans", k_neighbors=3, random_state=42, rus_cap=0)
        print("FULL-TRAIN BALANCING: PASS", X_train.shape, counts(y_train), X_out.shape, counts(y_out), time.perf_counter() - start)
    except Exception as exc:
        print("FULL-TRAIN BALANCING: FAIL", repr(exc), time.perf_counter() - start)
    source = open("src/presplit_feature_ablation_protocol.py", encoding="utf-8").read()
    tree = ast.parse(source)
    latency = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_latency")
    keys = {key.value for node in ast.walk(latency) if isinstance(node, ast.Dict) for key in node.keys if isinstance(key, ast.Constant)}
    required = {"training_seconds", "model_save_seconds", "model_load_seconds", "cold_start_prediction_ms", "warm_batch_prediction_seconds", "warm_inference_ms_per_sample", "throughput_samples_per_second", "single_sample_mean_ms", "single_sample_median_ms", "single_sample_p95_ms", "model_size_mb"}
    print("LATENCY INSTRUMENTATION CODE:", required <= keys)
    print("E1", e1_dimensions)
    print("E2", e2_dimensions)


if __name__ == "__main__":
    main()
