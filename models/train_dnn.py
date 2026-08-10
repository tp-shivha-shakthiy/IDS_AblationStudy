"""
train_dnn.py
============
Deep Neural Network — Baseline (no MI/PCA/balancing, class-weight loss)

Uses shared infrastructure from src/dl_pipeline.py.
Architecture preserved: 2 hidden layers (64→32), BatchNorm, Dropout(0.1).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to cp1252; reconfigure so Unicode output never crashes.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

from src.dl_pipeline import (
    set_seeds, get_device, load_data,
    compute_class_weights, evaluate_with_proba, get_probabilities,
    save_dl_artifacts,
)
from src.experiment_config import build_experiment_config

set_seeds(42)
device = get_device()
MODEL_NAME = "DNN"


# ======================================================================
# Model Architecture (preserved from original)
# ======================================================================

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.network(x)


# ======================================================================
# Pipeline
# ======================================================================

def main(data_dir="data/raw"):
    data = load_data(data_dir)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    num_classes = data['num_classes']
    normal_class_idx = data['normal_class_idx']
    class_names = data['class_names']

    # --- Per-fold CV (Scaler only, no MI/PCA/balancing for this model) ---
    print(f"\n{'='*60}")
    print(f"  {MODEL_NAME} - Baseline DNN (class-weight loss)")
    print(f"{'='*60}")
    print(f"\n  Cross-Validation (5 folds)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = []

    class_weights = compute_class_weights(y_train, device)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n  Fold {fold}/5")
        X_tr, y_tr = X_train.iloc[trn_idx], y_train[trn_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

        # Scaler fitted on fold train only
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t), batch_size=1024, shuffle=True,
            drop_last=True,
        )

        model = DeepNeuralNetwork(X_tr.shape[1], num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

        model.train()
        for epoch in range(5):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val_t.to(device)), dim=1).cpu().numpy()

        val_proba = get_probabilities(model, X_val_s, device)
        metrics = evaluate_with_proba(y_val, preds, val_proba, normal_class_idx)
        metrics['fold'] = fold
        cv_metrics.append(metrics)
        print(f"    Acc={metrics['multi_acc']:.4f}  F1={metrics['weighted_f1']:.4f}")

    # --- Final retrain on full training set ---
    print(f"\n  === Final Retrain ===")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=1024, shuffle=True,
        drop_last=True,
    )

    final_model = DeepNeuralNetwork(X_train.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(final_model.parameters(), lr=0.01, weight_decay=1e-4)

    final_model.train()
    for epoch in range(5):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(final_model(bx), by)
            loss.backward()
            optimizer.step()

    # --- Test evaluation ---
    final_model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(final_model(X_te_t.to(device)), dim=1).cpu().numpy()

    test_proba = get_probabilities(final_model, X_test_s, device)
    test_metrics = evaluate_with_proba(y_test, test_preds, test_proba, normal_class_idx)

    print(f"\n  {MODEL_NAME} Test Metrics:")
    for k, v in test_metrics.items():
        if k != 'fold':
            print(f"    {k:>12s}: {v:.4f}")

    # --- Save artifacts ---
    save_dl_artifacts(
        final_model, MODEL_NAME,
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        class_names=class_names,
        normal_class_idx=normal_class_idx,
        y_test=y_test, y_test_pred=test_preds,
        scaler=scaler,
        le=data['le'],
        config=build_experiment_config(
            model_name=MODEL_NAME,
            model_params={"layers": [64, 32], "dropout": 0.1,
                          "lr": 0.01, "weight_decay": 1e-4,
                          "epochs": 5, "batch_size": 1024},
            mi_k=0, pca_variance=None, tier=2,
            dl_extra={"preprocessing": ["StandardScaler"],
                      "balance_strategy": "class_weights"},
        ),
    )

    return final_model, cv_metrics, test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    main(data_dir=args.data_dir)
