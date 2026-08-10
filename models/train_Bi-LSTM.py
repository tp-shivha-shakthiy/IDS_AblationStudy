"""
train_Bi-LSTM.py
================
Weighted Bidirectional LSTM with MI + PCA + KMeansSMOTE.

Uses shared infrastructure from src/dl_pipeline.py.
Architecture preserved: 1-layer BiLSTM(32) + FC(64→32→out), no Dropout.

Note: The original script also ran an XGBoost pipeline. That has been
removed since XGBoost is already covered by Tier 1 (src/train_xgboost.py).
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
from collections import Counter

from src.dl_pipeline import (
    set_seeds, get_device, load_data,
    preprocess_fold, preprocess_final,
    evaluate_with_proba, get_probabilities, save_dl_artifacts,
)
from src.experiment_config import build_experiment_config

set_seeds(42)
device = get_device()
MODEL_NAME = "BiLSTM"


# ======================================================================
# Model Architecture (preserved from original)
# ======================================================================

class WeightedBiLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.classifier(last)


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

    print(f"\n{'='*60}")
    print(f"  {MODEL_NAME} - Weighted BiLSTM + MI(30) + PCA(15) + KMeansSMOTE")
    print(f"{'='*60}")
    print(f"\n  Cross-Validation (5 folds)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n  Fold {fold}/5")
        X_tr, y_tr = X_train.iloc[trn_idx], y_train[trn_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

        fold_data = preprocess_fold(
            X_tr, y_tr, X_val, y_val,
            mi_k=30, pca_components=15,
            n_clusters=20, k_neighbors=2, rus_cap=15000,
        )

        # Class weights computed from balanced fold training data
        fold_counts = Counter(fold_data['y_tr'])
        total = sum(fold_counts.values())
        n_cls = len(fold_counts)
        weights = torch.tensor(
            [total / (n_cls * fold_counts[c]) for c in range(n_cls)],
            dtype=torch.float32,
        ).to(device)

        X_tr_t = torch.tensor(fold_data['X_tr'], dtype=torch.float32)
        y_tr_t = torch.tensor(fold_data['y_tr'], dtype=torch.long)
        X_val_t = torch.tensor(fold_data['X_val'], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True,
            drop_last=True,
        )

        model = WeightedBiLSTM(fold_data['X_tr'].shape[1], num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.005)

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

        val_proba = get_probabilities(model, fold_data['X_val'], device)
        metrics = evaluate_with_proba(y_val, preds, val_proba, normal_class_idx)
        metrics['fold'] = fold
        cv_metrics.append(metrics)
        print(f"    Acc={metrics['multi_acc']:.4f}  F1={metrics['weighted_f1']:.4f}")

    # --- Final retrain ---
    print(f"\n  === Final Retrain ===")
    final_data = preprocess_final(
        X_train, y_train, X_test, y_test,
        mi_k=30, pca_components=15,
        n_clusters=20, k_neighbors=2, rus_cap=15000,
    )

    final_counts = Counter(final_data['y_train'])
    total = sum(final_counts.values())
    n_cls = len(final_counts)
    weights = torch.tensor(
        [total / (n_cls * final_counts[c]) for c in range(n_cls)],
        dtype=torch.float32,
    ).to(device)

    X_tr_t = torch.tensor(final_data['X_train'], dtype=torch.float32)
    y_tr_t = torch.tensor(final_data['y_train'], dtype=torch.long)
    X_te_t = torch.tensor(final_data['X_test'], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True,
        drop_last=True,
    )

    final_model = WeightedBiLSTM(final_data['X_train'].shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(final_model.parameters(), lr=0.005)

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

    test_proba = get_probabilities(final_model, final_data['X_test'], device)
    test_metrics = evaluate_with_proba(y_test, test_preds, test_proba, normal_class_idx)

    print(f"\n  {MODEL_NAME} Test Metrics:")
    for k, v in test_metrics.items():
        if k != 'fold':
            print(f"    {k:>12s}: {v:.4f}")

    save_dl_artifacts(
        final_model, MODEL_NAME,
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        class_names=class_names,
        normal_class_idx=normal_class_idx,
        y_test=y_test, y_test_pred=test_preds,
        selector=final_data['selector'],
        scaler=final_data['scaler'],
        pca=final_data['pca'],
        le=data['le'],
        config=build_experiment_config(
            model_name=MODEL_NAME,
            model_params={"hidden_dim": 32, "dropout": 0.0,
                          "lr": 0.005, "epochs": 5, "batch_size": 512},
            mi_k=30, pca_variance=None, tier=2,
            dl_extra={"preprocessing": ["MI_k30", "StandardScaler", "PCA_15", "KMeansSMOTE"],
                      "balance_strategy": "kmeans + class_weights"},
        ),
    )

    return final_model, cv_metrics, test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    main(data_dir=args.data_dir)
