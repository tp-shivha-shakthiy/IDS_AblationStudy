"""
train_dnn.py
============
Deep Neural Network — participates in the seven-preset ablation.

Architecture preserved: 2 hidden layers (64→32), BatchNorm, Dropout(0.1),
class-weight loss.  The MI / PCA / KMeansSMOTE preprocessing is driven by
the ``--experiment`` ablation preset (identical hyperparameters to Tier 1:
MI k=15, PCA 0.95 variance, KMeansSMOTE k_neighbors=3, no undersampling).
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
    save_dl_artifacts, preprocess_fold, preprocess_final,
)
from src.experiment_config import build_experiment_config, resolve_experiment, OFFICIAL_RUS_CAP

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

def main(data_dir="data/raw", experiment="mi_pca_balancing", cap=OFFICIAL_RUS_CAP):
    flags = resolve_experiment(experiment)
    use_mi, use_pca, use_balancing = (
        flags["use_mi"], flags["use_pca"], flags["use_balancing"],
    )
    data = load_data(data_dir)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    num_classes = data['num_classes']
    normal_class_idx = data['normal_class_idx']
    class_names = data['class_names']

    # --- Per-fold CV (preprocessing driven by the ablation preset) ---
    print(f"\n{'='*60}")
    print(f"  {MODEL_NAME} - DNN (class-weight loss)")
    print(f"  Experiment: {experiment}  "
          f"(MI={'on' if use_mi else 'off'}, "
          f"PCA={'on' if use_pca else 'off'}, "
          f"KMeansSMOTE={'on' if use_balancing else 'off'})")
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
            mi_k=15, pca_variance=0.95,
            n_clusters=20, k_neighbors=3, rus_cap=cap,
            use_mi=use_mi, use_pca=use_pca, use_balancing=use_balancing,
        )
        X_tr_s, X_val_s = fold_data['X_tr'], fold_data['X_val']
        y_tr_bal = fold_data['y_tr']
        fold_weights = compute_class_weights(y_tr_bal, device)

        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_bal, dtype=torch.long)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t), batch_size=1024, shuffle=True,
            drop_last=True,
        )

        model = DeepNeuralNetwork(X_tr_s.shape[1], num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=fold_weights)
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
    final_data = preprocess_final(
        X_train, y_train, X_test, y_test,
        mi_k=15, pca_variance=0.95,
        n_clusters=20, k_neighbors=3, rus_cap=cap,
        use_mi=use_mi, use_pca=use_pca, use_balancing=use_balancing,
    )
    X_train_s, X_test_s = final_data['X_train'], final_data['X_test']
    final_weights = compute_class_weights(final_data['y_train'], device)

    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_tr_t = torch.tensor(final_data['y_train'], dtype=torch.long)
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=1024, shuffle=True,
        drop_last=True,
    )

    final_model = DeepNeuralNetwork(X_train_s.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=final_weights)
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
        experiment=experiment,
        class_names=class_names,
        normal_class_idx=normal_class_idx,
        y_test=y_test, y_test_pred=test_preds,
        scaler=final_data['scaler'],
        le=data['le'],
        config=build_experiment_config(
            model_name=MODEL_NAME,
            model_params={"layers": [64, 32], "dropout": 0.1,
                          "lr": 0.01, "weight_decay": 1e-4,
                          "epochs": 5, "batch_size": 1024},
            experiment_name=experiment,
            preprocessing_mode=experiment,
            use_mi=use_mi, use_pca=use_pca, use_balancing=use_balancing,
            mi_k=15, pca_variance=0.95,
            n_splits=5, balancer="kmeans",
            k_neighbors=3, rus_cap=cap,
            tier=2, ablation_scope="tier2",
            dl_extra={"preprocessing": ["StandardScaler"],
                      "balance_strategy": "class_weights"},
        ),
    )

    return final_model, cv_metrics, test_metrics


if __name__ == "__main__":
    import argparse
    from src.experiment_config import ABLATION_PRESETS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--experiment", choices=list(ABLATION_PRESETS.keys()),
        default="mi_pca_balancing",
        help="Ablation preset (default: mi_pca_balancing). "
             "Use 'raw' for the scaler-only baseline.",
    )
    parser.add_argument(
        "--cap", type=int, default=OFFICIAL_RUS_CAP,
        help="Per-class sample cap before KMeansSMOTE oversampling "
             "(0 = no cap). Official full-data protocol: 15000.",
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir, experiment=args.experiment, cap=args.cap)
