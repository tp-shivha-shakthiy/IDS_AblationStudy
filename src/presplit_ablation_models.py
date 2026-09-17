"""DL adapters for the presplit ablation protocol; they never preprocess data."""

import json
import os
import platform
import time

import numpy as np

from src.balancing import balance_full_train, balance_training_fold
from src.evaluation import compute_extended_metrics
from src.presplit_cv_checkpoints import (
    build_cv_config_fingerprint, load_compatible_fold_checkpoint,
    save_fold_checkpoint,
)


DL_MODELS = ("DNN", "LSTM", "BiLSTM", "BiLSTMSharedFeature")


def _torch():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, optim, DataLoader, TensorDataset


def _spec(name, input_dim, classes, device):
    if name == "DNN":
        from models.train_dnn import DeepNeuralNetwork
        return DeepNeuralNetwork(input_dim, classes).to(device), 5, 1024, 0.01, 1e-4, True, False
    if name == "LSTM":
        from models.train_lstm import BiLSTMNetwork
        return BiLSTMNetwork(input_dim, classes).to(device), 5, 512, 0.005, 1e-4, False, False
    if name == "BiLSTM":
        from models.train_bilstm import WeightedBiLSTM
        return WeightedBiLSTM(input_dim, classes).to(device), 5, 512, 0.005, 0.0, True, False
    if name == "BiLSTMSharedFeature":
        from models.train_bilstm_shared_feature import MultiTaskHierarchicalDNN
        return MultiTaskHierarchicalDNN(input_dim, classes).to(device), 8, 512, 0.005, 1e-4, False, True
    raise ValueError(f"Unsupported DL model: {name}")


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fit(model_name, X, y, classes, normal_index, device):
    torch, nn, optim, DataLoader, TensorDataset = _torch()
    model, epochs, batch, learning_rate, weight_decay, weighted, shared = _spec(
        model_name, X.shape[1], classes, device
    )
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    if shared:
        binary = torch.tensor((np.asarray(y) != normal_index).astype(np.int64))
        loader = DataLoader(TensorDataset(x_tensor, y_tensor, binary), batch_size=batch, shuffle=True, drop_last=True)
    else:
        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch, shuffle=True, drop_last=True)
    weights = None
    if weighted:
        counts = np.bincount(y, minlength=classes)
        weights = torch.tensor(len(y) / (classes * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for batch_data in loader:
            optimizer.zero_grad()
            if shared:
                bx, by, binary = (item.to(device) for item in batch_data)
                binary_logits, logits = model(bx)
                loss = 0.4 * nn.CrossEntropyLoss()(binary_logits, binary) + 0.6 * criterion(logits, by)
            else:
                bx, by = (item.to(device) for item in batch_data)
                loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return model


def _predict(model, X, device):
    torch, *_ = _torch()
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(X, dtype=torch.float32, device=device))
        logits = output[1] if isinstance(output, tuple) else output
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities.argmax(axis=1), probabilities


def _latency(model_path, model_name, input_dim, classes, X_test, device, training_seconds,
             balancing_seconds, preprocessing_seconds, save_seconds, normal_index):
    torch, *_ = _torch()
    start = time.perf_counter()
    payload = torch.load(model_path, map_location=device, weights_only=True)
    model, *_ = _spec(model_name, input_dim, classes, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _sync(torch, device)
    load_seconds = time.perf_counter() - start
    def infer(values):
        with torch.no_grad():
            output = model(torch.tensor(values, dtype=torch.float32, device=device))
            return output[1] if isinstance(output, tuple) else output
    _sync(torch, device); start = time.perf_counter(); infer(X_test[:1]); _sync(torch, device)
    cold_ms = (time.perf_counter() - start) * 1000
    infer(X_test[:1]); _sync(torch, device); start = time.perf_counter(); infer(X_test); _sync(torch, device)
    batch_seconds = time.perf_counter() - start
    samples = []
    for _ in range(100):
        _sync(torch, device); start = time.perf_counter(); infer(X_test[:1]); _sync(torch, device)
        samples.append((time.perf_counter() - start) * 1000)
    return {"feature_preprocessing_seconds": preprocessing_seconds, "balancing_seconds": balancing_seconds,
            "training_seconds": training_seconds, "model_save_seconds": save_seconds,
            "model_load_seconds": load_seconds, "cold_start_prediction_ms": cold_ms,
            "warm_inference_ms_per_sample": 1000 * batch_seconds / len(X_test),
            "throughput_samples_per_second": len(X_test) / batch_seconds,
            "single_sample_mean_ms": float(np.mean(samples)), "single_sample_median_ms": float(np.median(samples)),
            "single_sample_p95_ms": float(np.percentile(samples, 95)),
            "model_size_mb": os.path.getsize(model_path) / (1024 * 1024)}


def run_dl_presplit_experiment(X_train, X_test, y_train, y_test, class_names, normal_index,
                               experiment, model_name, dimensions, preprocessing_seconds, artifacts,
                               results_root, n_splits, k_neighbors, rus_cap, random_state, config,
                               resume=False):
    torch, *_ = _torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = len(class_names)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    cv_accuracy = []
    balancing_audit = []
    save_dir = os.path.join(results_root, experiment, model_name)
    fingerprint = build_cv_config_fingerprint(
        config, dimensions, artifacts.get("selected_features"),
    )
    config["cv_checkpoint_fingerprint"] = fingerprint
    for fold_number, (train_index, validation_index) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X_train, y_train), 1):
        if resume:
            checkpoint = load_compatible_fold_checkpoint(
                save_dir, fold_number, fingerprint, config, dimensions,
            )
            if checkpoint is not None:
                cv_accuracy.append(checkpoint["validation_metrics"]["accuracy"])
                balancing_audit.append(checkpoint["balancing_audit"])
                continue
        balance_start = time.perf_counter()
        X_fold, y_fold, fold_audit = balance_training_fold(X_train[train_index], y_train[train_index], strategy="kmeans", k_neighbors=k_neighbors, random_state=random_state, rus_cap=rus_cap, return_audit=True)
        fold_balancing_seconds = time.perf_counter() - balance_start
        fold_audit.update({"phase": "cross_validation", "fold": fold_number})
        balancing_audit.append(fold_audit)
        training_start = time.perf_counter()
        model = _fit(model_name, X_fold, y_fold, classes, normal_index, device)
        fold_training_seconds = time.perf_counter() - training_start
        prediction, _ = _predict(model, X_train[validation_index], device)
        validation_metrics = {"accuracy": float(accuracy_score(y_train[validation_index], prediction))}
        cv_accuracy.append(validation_metrics["accuracy"])
        save_fold_checkpoint(save_dir, fold_number, validation_metrics, fold_audit,
                             fold_balancing_seconds, fold_training_seconds, config,
                             dimensions, fingerprint)
    start = time.perf_counter()
    X_balanced, y_balanced, final_audit = balance_full_train(X_train, y_train, strategy="kmeans", k_neighbors=k_neighbors, random_state=random_state, rus_cap=rus_cap, return_audit=True)
    final_audit["phase"] = "final_retrain"
    balancing_audit.append(final_audit)
    balancing_seconds = time.perf_counter() - start
    start = time.perf_counter(); model = _fit(model_name, X_balanced, y_balanced, classes, normal_index, device)
    training_seconds = time.perf_counter() - start
    prediction, probabilities = _predict(model, X_test, device)
    metrics = compute_extended_metrics(y_test, prediction, probabilities, normal_index)
    metrics["multiclass_accuracy"] = metrics["multi_acc"]
    from sklearn.metrics import recall_score, precision_score
    metrics["macro_recall"] = recall_score(y_test, prediction, average="macro", zero_division=0)
    binary_true, binary_pred = y_test != normal_index, prediction != normal_index
    metrics["binary_precision"] = precision_score(binary_true, binary_pred, zero_division=0)
    metrics["binary_recall"] = recall_score(binary_true, binary_pred, zero_division=0)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model.pt"); start = time.perf_counter()
    torch.save({"state_dict": model.state_dict(), "input_dimension": int(X_train.shape[1]), "num_classes": classes}, path)
    save_seconds = time.perf_counter() - start
    latency = _latency(path, model_name, X_train.shape[1], classes, X_test, device, training_seconds, balancing_seconds, preprocessing_seconds, save_seconds, normal_index)
    environment = {"cpu": platform.processor(), "gpu": str(device), "python": platform.python_version()}
    for filename, payload in (("test_metrics.json", metrics), ("latency.json", latency), ("feature_dimensions.json", dimensions), ("resolved_config.json", config), ("environment.json", environment), ("balancing_audit.json", {"events": balancing_audit})):
        with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as handle: json.dump(payload, handle, indent=2)
    if artifacts.get("selected_features") is not None:
        with open(os.path.join(save_dir, "selected_features.json"), "w", encoding="utf-8") as handle: json.dump(artifacts["selected_features"], handle, indent=2)
    return {"metrics": metrics, "latency": latency, "dimensions": dimensions, "cv_accuracy": cv_accuracy, "save_dir": save_dir}
