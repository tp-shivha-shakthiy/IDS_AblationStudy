"""
train_Bi-LSTM_shared-feature-extractor.py
==========================================
Multi-Task Hierarchical DNN with shared feature extractor.

Uses shared infrastructure from src/dl_pipeline.py.
Architecture preserved:
  Shared base: Linear(in→128)→LN→ReLU→Drop(0.2)→Linear(128→64)→LN→ReLU→Drop(0.2)
  Binary head: Linear(64→2)
  Multi head:  Linear(64→32)→ReLU→Linear(32→num_classes)
  Loss: 0.4 * CE(binary) + 0.6 * CE(multi)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

from src.dl_pipeline import (
    set_seeds, get_device, load_data,
    preprocess_fold, preprocess_final,
    evaluate_with_proba, save_dl_artifacts,
)

set_seeds(42)
device = get_device()
MODEL_NAME = "BiLSTM_SharedFE"


# ======================================================================
# Model Architecture (preserved from original)
# ======================================================================

class MultiTaskHierarchicalDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Binary head (Normal vs Attack)
        self.binary_head = nn.Linear(64, 2)
        # Multi-class head (9 attack categories)
        self.multi_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        features = self.shared(x)
        binary_out = self.binary_head(features)
        multi_out = self.multi_head(features)
        return binary_out, multi_out


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
    print(f"  {MODEL_NAME} — Multi-Task Hierarchical DNN")
    print(f"{'='*60}")
    print(f"\n  Cross-Validation (5 folds)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n  Fold {fold}/5")
        X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        fold_data = preprocess_fold(
            X_tr, y_tr, X_val, y_val,
            mi_k=30, pca_components=15,
            n_clusters=20, k_neighbors=2, rus_cap=15000,
        )

        y_tr_binary = (fold_data['y_tr'] != normal_class_idx).astype(int)
        y_val_binary = (fold_data['y_val'] != normal_class_idx).astype(int)

        X_tr_t = torch.tensor(fold_data['X_tr'], dtype=torch.float32)
        y_tr_t = torch.tensor(fold_data['y_tr'], dtype=torch.long)
        y_tr_bin_t = torch.tensor(y_tr_binary, dtype=torch.long)
        X_val_t = torch.tensor(fold_data['X_val'], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t, y_tr_bin_t),
            batch_size=512, shuffle=True, drop_last=True,
        )

        model = MultiTaskHierarchicalDNN(
            fold_data['X_tr'].shape[1], num_classes,
        ).to(device)

        ce_multi = nn.CrossEntropyLoss()
        ce_binary = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

        model.train()
        for epoch in range(8):
            for bx, by_multi, by_bin in train_loader:
                bx = bx.to(device)
                by_multi = by_multi.to(device)
                by_bin = by_bin.to(device)
                optimizer.zero_grad()
                bin_out, multi_out = model(bx)
                loss = 0.4 * ce_binary(bin_out, by_bin) + 0.6 * ce_multi(multi_out, by_multi)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            _, multi_out = model(X_val_t.to(device))
            y_proba = torch.softmax(multi_out, dim=1).cpu().numpy()
        preds_multi = np.argmax(y_proba, axis=1)

        metrics = evaluate_with_proba(y_val, preds_multi, y_proba, normal_class_idx)
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

    y_tr_binary = (final_data['y_train'] != normal_class_idx).astype(int)

    X_tr_t = torch.tensor(final_data['X_train'], dtype=torch.float32)
    y_tr_t = torch.tensor(final_data['y_train'], dtype=torch.long)
    y_tr_bin_t = torch.tensor(y_tr_binary, dtype=torch.long)
    X_te_t = torch.tensor(final_data['X_test'], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t, y_tr_bin_t),
        batch_size=512, shuffle=True, drop_last=True,
    )

    final_model = MultiTaskHierarchicalDNN(
        final_data['X_train'].shape[1], num_classes,
    ).to(device)

    ce_multi = nn.CrossEntropyLoss()
    ce_binary = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=0.005, weight_decay=1e-4)

    final_model.train()
    for epoch in range(8):
        for bx, by_multi, by_bin in train_loader:
            bx = bx.to(device)
            by_multi = by_multi.to(device)
            by_bin = by_bin.to(device)
            optimizer.zero_grad()
            bin_out, multi_out = final_model(bx)
            loss = 0.4 * ce_binary(bin_out, by_bin) + 0.6 * ce_multi(multi_out, by_multi)
            loss.backward()
            optimizer.step()

    # --- Test evaluation ---
    final_model.eval()
    with torch.no_grad():
        _, multi_out = final_model(X_te_t.to(device))
        y_proba = torch.softmax(multi_out, dim=1).cpu().numpy()
    test_preds = np.argmax(y_proba, axis=1)

    test_metrics = evaluate_with_proba(y_test, test_preds, y_proba, normal_class_idx)

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
        selector=final_data['selector'], scaler=final_data['scaler'],
        pca=final_data['pca'], label_encoder=data['le'],
        model_config={'input_dim': final_data['X_train'].shape[1], 'num_classes': num_classes},
    )

    return final_model, cv_metrics, test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    main(data_dir=args.data_dir)
