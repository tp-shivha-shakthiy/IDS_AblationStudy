# Intrusion Detection System — UNSW-NB15

Multi-model network intrusion detection pipeline trained on the UNSW-NB15 dataset. Produces both **binary** (Normal / Attack) and **multi-class** (10 attack categories) predictions using an ensemble of classical ML and deep learning approaches — fully cross-validated with strict data leakage prevention.

Implements the framework described in:

> **A novel intrusion detection system for class imbalance datasets using hybrid sampling with deep learning techniques** — Kasina et al., *Information Sciences* 741 (2026)

---

## Problem Statement

Network intrusion detection systems (NIDS) must classify network traffic as benign or malicious (binary) and identify the specific attack type (multi-class, 10 categories). The core challenge is **severe class imbalance** — attack types like Worms (~130 samples) are dwarfed by benign traffic (~1.2M samples), causing models to ignore rare but dangerous attacks.

This repo implements an end-to-end pipeline that addresses imbalance through hybrid sampling (SMOTE, KMeansSMOTE), feature engineering (Mutual Information, PCA), and multi-model comparison (3 classical ML + 5 deep learning architectures), with rigorous cross-validation and data leakage prevention.

All results below are from the **corrected, leakage-free pipeline**. See [Data Leakage Fix](#data-leakage-fix).

---

## Key Results (Corrected Pipeline)

### Deep Learning Models

| Model | Binary Acc | Binary F1 | Multi Acc | Macro F1 | Weighted F1 | AUC |
|---|---|---|---|---|---|---|
| **BiLSTM_SharedFE** | **98.70%** | **0.9510** | 96.61% | 0.4884 | 0.9727 | **0.9992** |
| DNN_MI_PCA_KMeans | 98.69% | 0.9505 | **96.66%** | 0.4992 | **0.9729** | 0.9992 |
| BiLSTM | 98.65% | 0.9492 | 96.65% | **0.5022** | 0.9728 | 0.9990 |
| LSTM | 98.58% | 0.9469 | 96.55% | 0.4897 | 0.9721 | 0.9989 |
| DNN | 98.57% | 0.9465 | 96.65% | 0.4887 | 0.9723 | 0.9990 |

**Best in bold.** Results from `models/artifacts/*/*_test_metrics.json`.

### Classical ML Models (sklearn)

| Model | Multi Acc | Macro F1 | Weighted F1 | AUC |
|---|---|---|---|---|
| **HGB** | **96.28%** | **0.4639** | **0.9703** | **0.9975** |
| Logistic Regression | 95.45% | 0.3786 | 0.9640 | 0.9922 |
| XGBoost | 91.22% | 0.3648 | 0.9275 | 0.9834 |

Results from `results/model_comparison.csv` and per-class reports.

### Key Takeaways

- **BiLSTM_SharedFE** achieves the best binary accuracy (98.70%) and binary F1 (0.9510)
- **BiLSTM** achieves the best Macro F1 (0.5022) — strongest minority class detection
- **DNN_MI_PCA_KMeans** achieves the best weighted F1 (0.9729) and multi-class accuracy (96.66%)
- **HGB** is the best classical model (96.28% multi acc), competitive with DL
- **XGBoost underperforms** in the corrected pipeline (91.22%) — the previous high results (98.70%) were inflated by data leakage
- All DL models achieve AUC > 0.998, showing excellent ranking ability
- The corrected results are ~0.3–1.5% lower across the board than the leaky versions, reflecting realistic generalization

### Comparison with Published Paper

Kasina et al. (2026) report 99.95% binary F1 and 97.92% weighted F1 using SMOTE-ENN + DNN. **These figures likely reflect data leakage** as they are significantly higher than our leakage-free results (best binary F1: 0.9510, best weighted F1: 0.9729). Our implementation prioritizes methodological rigor — all preprocessing is scoped per CV fold, and the test set is never touched until final evaluation.

The multi-task hierarchical DNN (BiLSTM_SharedFE) with shared feature extractor is an architectural addition not present in the original paper.

---

## Architecture — Two Tiers

### Tier 1: Classical ML Pipeline (`main.py` + `src/`)

Orchestrates 3 sklearn models (HistGradientBoosting, XGBoost, Logistic Regression) through a shared pipeline:

```
data/raw/UNSW-NB15_{1..4}.csv
         │
         ▼
[Phase 3] src/preprocessing.py
  - Load 4 CSV files, concatenate
  - Clean attack_cat target → LabelEncoder
  - Drop metadata columns (id, label, stime, ltime, srcip, dstip)
  - LabelEncode categorical features
  - Log1p normalization (clip → log1p → fillna → float32)
          │
          ▼
[Phase 4] src/dimensionality_reduction.py
  - Stratified 80/20 train/test split (X_test is LOCKED)
  - StandardScaler and PCA applied inside CV folds only
         │
         ▼
[Phase 6] src/balancing.py + src/cross_validation.py
  - StratifiedKFold (5 folds)
  - Per fold: MI fit on fold train → transform fold train+val
  - Per fold: Scaler fit on fold train → transform fold train+val
  - Per fold: PCA fit on fold train → transform fold train+val
  - Per fold: SMOTE or KMeansSMOTE on fold train only
  - Train model → evaluate on validation fold
         │
     ┌───┼───────────┐
     ▼   ▼           ▼
[Phase 7]  [Phase 8+9]  [Phase 10]
train_hgb  train_xgboost  train_logistic
 (HGB)      (XGBoost)      (LogReg)
         │   │           │
         └───┼───────────┘
             ▼
[Output] src/evaluation.py
  - Confusion matrices (binary + multi-class PNGs)
  - ROC curves (per-class + weighted)
  - Feature importance (XGBoost)
  - model_comparison.csv, metrics.csv, per-class reports
  - Artifact persistence (.joblib files)
```

### Tier 2: Deep Learning Pipeline (`models/` + `src/dl_pipeline.py`)

Five self-contained PyTorch scripts share common infrastructure from `src/dl_pipeline.py`:

```
load_data() → preprocessing + stratified split
    │
    ▼
5-fold StratifiedKFold
    │
    ▼ per fold:
preprocess_fold()
  - MI SelectKBest(k=30) fit on fold train only
  - StandardScaler fit on fold train only
  - PCA(n_components=15) fit on fold train only
  - RandomUnderSampler(cap=15,000) + KMeansSMOTE(k=2) on fold train only
    │
    ▼
PyTorch model training (5–10 epochs)
    │
    ▼
Final retrain on full training set → test evaluation
    │
    ▼
save_dl_artifacts()
  - Model weights (.pt)
  - Metadata JSON
  - CV metrics CSV
  - Test metrics JSON
  - Confusion matrix PNG
```

---

## Data Leakage Fix — Critical Correctness Improvement

An earlier version of the pipeline had **critical data leakage**: StandardScaler and PCA were fitted on the **entire dataset** before the train/test split. This caused test set statistics to influence training, inflating metrics by ~0.3–7.5% (XGBoost was particularly affected — dropping from 98.70% to 91.22% after correction).

**Fix applied — all transforms fitted per fold:**

| Transform | Before (Leaky) | After (Correct) |
|---|---|---|
| StandardScaler | Fit on ALL data | Fit on fold train only |
| PCA | Fit on ALL data | Fit on fold train only |
| MI SelectKBest | Fit on ALL data | Fit on fold train only |
| SMOTE / KMeansSMOTE | Applied pre-split | Applied on fold train only |
| Final model | Trained on fold-0 only | Trained on full 80% training set |

The locked 20% holdout set is never touched until final evaluation. See `ARCHITECTURE_GAP.md` for the full audit and `tests/test_leakage.py` for 35+ regression tests that enforce leakage-free pipelines.

---

## Project Structure

```
INTRUSION-DETECTION-SYSTEM/
│
├── main.py                              Tier 1 orchestrator (sklearn models)
├── requirements.txt                     Python dependencies
├── README.md                            This file
├── ARCHITECTURE_GAP.md                  Architecture gap analysis & leakage audit
├── AUDIT.md                             Full codebase audit (duplication, bugs, fixes)
├── License                              All rights reserved
├── .gitignore                           Ignores data/raw/*.csv, __pycache__
│
├── src/                                 Shared pipeline modules + sklearn trainers
│   ├── preprocessing.py                 Load, clean, encode, log1p normalization
│   ├── dimensionality_reduction.py      Stratified 80/20 split
│   ├── balancing.py                     SMOTE / KMeansSMOTE per fold
│   ├── cross_validation.py              StratifiedKFold CV runner
│   ├── evaluation.py                    CM, ROC, feature importance, CSV, artifacts
│   ├── experiment_config.py             Experiment metadata + JSON persistence
│   ├── train_hgb.py                     HistGradientBoosting wrapper
│   ├── train_xgboost.py                 XGBoost wrapper
│   ├── train_logistic.py                Logistic Regression wrapper
│   └── dl_pipeline.py                   Shared DL infrastructure (Tier 2 backbone)
│
├── models/                              Self-contained deep learning training scripts
│   ├── __init__.py
│   ├── train_dnn.py                     3-layer DNN, weighted cross-entropy loss
│   ├── train_LSTM.py                    BiLSTM + MI(30) + PCA(15) + KMeansSMOTE
│   ├── train_Bi-LSTM.py                 Weighted BiLSTM + MI(30) + PCA(15) + KMeansSMOTE
│   ├── train_Bi-LSTM_shared-feature-extractor.py  Multi-task DNN (shared backbone, binary + multi heads)
│   ├── train_dnn_mi_pca_kmeans.py       4-layer DNN + MI + PCA + KMeansSMOTE
│   └── artifacts/                       Saved model weights, metadata, plots
│       ├── DNN/
│       ├── LSTM/
│       ├── BiLSTM/
│       ├── BiLSTM_SharedFE/
│       ├── DNN_MI_PCA_KMeans/
│       └── XGBoost/
│
├── tests/                               Pytest test suite (70+ tests)
│   ├── test_leakage.py                  Data leakage regression tests
│   ├── test_dl_pipeline.py              DL pipeline correctness tests
│   └── test_audit_fixes.py              Audit finding regression tests
│
├── notebooks/
│   └── Intrusion_Detection.ipynb        Exploratory Jupyter notebook
│
├── data/
│   ├── raw/                             Place UNSW-NB15_1..4.csv here
│   └── processed/                       Reserved for cached intermediate arrays
│
├── assets/                              Generated plots and architecture diagram
│   ├── Architecture.jpeg                Pipeline architecture diagram
│   ├── feature_importance.png
│   ├── xgboost_binary_cm.png
│   ├── xgboost_multiclass_cm.png
│   ├── logreg_binary_cm.png
│   └── logreg_multiclass_cm.png
│
├── artifacts/                           Saved sklearn preprocessing artifacts
│   ├── hgb/
│   ├── xgboost/
│   └── logistic_regression/
│
└── results/                             Output metrics and reports
    ├── model_comparison.csv             Blind holdout metrics (sklearn models)
    ├── metrics.csv                      Per-fold CV metrics
    ├── hgb_per_class_report.csv
    ├── xgboost_per_class_report.csv
    ├── logreg_per_class_report.csv
    └── corrected_pipeline/
        └── experiment_config.json       Run metadata + hyperparameters
```

---

## Detailed Module Reference

### Core Pipeline Modules (`src/`)

| Module | Purpose | Key Details |
|---|---|---|
| `preprocessing.py` | Data loading + cleaning | Reads 4 CSVs (47/49 col variants), maps attack categories, drops metadata, LabelEncodes objects, applies log1p normalization. Returns float32 arrays. |
| `dimensionality_reduction.py` | Train/test split | Stratified 80/20 split only. Scaler/PCA moved inside CV loop to prevent leakage. |
| `balancing.py` | Class imbalance handling | Two strategies: `"kmeans"` (KMeansSMOTE + MiniBatchKMeans) and `"smote"` (plain SMOTE). Applied only to training data. Handles edge cases (minority_count < 2). |
| `cross_validation.py` | Stratified CV runner | Per fold: fits MI selector, StandardScaler, PCA, and balancer on fold train. Returns per-fold metrics + last fold's transformers for test evaluation. |
| `evaluation.py` | Output + visualization | Matplotlib (Agg backend). Saves confusion matrices (binary + multi-class), ROC curves, feature importance plots, CSVs, and .joblib artifacts. |
| `experiment_config.py` | Experiment metadata | Records seed, split ratio, CV folds, balancer, MI k, PCA variance, timestamps, git commit hash. Saves as JSON. |
| `dl_pipeline.py` | Shared DL infrastructure | CUDA fallback, per-fold preprocessing (MI→Scaler→PCA→RUS→KMeansSMOTE), batch inference for large validation sets, probability computation, artifact persistence. |

### Sklearn Model Trainers (`src/`)

| Module | Model | Hyperparameters | CV | Test Eval |
|---|---|---|---|---|
| `train_hgb.py` | `HistGradientBoostingClassifier` | max_iter=30, lr=0.05, max_depth=5, l2_reg=1.0 | 5-fold | Full retrain → blind holdout |
| `train_xgboost.py` | `XGBClassifier` | n_estimators=30, subsample=0.1, max_depth=3, min_child_weight=20, gamma=0.2, lr=0.05, colsample_bytree=0.1, reg_alpha=0.5, tree_method='hist' | 5-fold | Full retrain → blind holdout |
| `train_logistic.py` | `LogisticRegression` | multi_class='multinomial', solver='saga', max_iter=50, n_jobs=-1 | 5-fold | Full retrain → blind holdout |

### Deep Learning Models (`models/`)

| Script | Architecture | Preprocessing | Balancing | Epochs |
|---|---|---|---|---|
| `train_dnn.py` | 3-layer DNN (64→32→n), LayerNorm, Dropout | Log1p only | Weighted CE loss | 5 |
| `train_LSTM.py` | BiLSTM(hidden=32) → Linear(64→32→n) | MI(30) → PCA(15) | RUS(15k) + KMeansSMOTE | 5 |
| `train_Bi-LSTM.py` | Weighted BiLSTM(hidden=32) → Linear(64→32→n) | MI(30) → PCA(15) | RUS(15k) + KMeansSMOTE | 5 |
| `train_Bi-LSTM_shared-feature-extractor.py` | Multi-task DNN: shared(128→64), binary + multi heads, joint 40/60 loss | MI(30) → PCA(15) | None (weighted heads) | 8 |
| `train_dnn_mi_pca_kmeans.py` | 4-layer DNN (128→64→32→n), LayerNorm, Dropout | MI(30) → PCA(15) | RUS(15k) + KMeansSMOTE | 10 |

---

## Models Summary

### Classical ML (sklearn) — Tier 1
1. **HistGradientBoosting** — Gradient boosted trees (no missing value imputation). Best classical model: 96.28% multi acc.
2. **XGBoost** — Regularized gradient boosting. Underperforms in corrected pipeline (91.22%) — heavily affected by the leakage fix.
3. **Logistic Regression** — Multinomial logistic with saga solver. Solid baseline at 95.45% multi acc.

### Deep Learning (PyTorch) — Tier 2
4. **DNN (3-layer)** — Feedforward network with LayerNorm, Dropout, weighted cross-entropy loss. 98.57% binary acc.
5. **Bidirectional LSTM** — Feature vector through BiLSTM layer. Strong all-rounder at 98.65% binary acc.
6. **Weighted BiLSTM** — BiLSTM with class-weighted loss. Best macro F1 (0.5022) — top minority class detection.
7. **Multi-Task Hierarchical DNN (BiLSTM_SharedFE)** — Shared backbone with two classification heads (binary + multi-class), 40/60 joint loss. Best binary acc (98.70%) and binary F1 (0.9510).
8. **DNN (4-layer, MI+PCA+KMeansSMOTE)** — Deeper DNN with full preprocessing. Best multi acc (96.66%) and weighted F1 (0.9729).

---

## Dataset — UNSW-NB15

**Source:** Moustafa & Slay (2015)

- **~2.5M rows**, 47 features (after dropping metadata columns like IP addresses, timestamps, IDs)
- **10 attack categories** + Normal (benign) traffic
- **Severely imbalanced:**

| Class | Samples | % of Total |
|---|---|---|
| Normal | ~1,200,000 | ~48% |
| Generic | ~188,000 | ~7.5% |
| Exploits | ~111,000 | ~4.4% |
| Fuzzers | ~24,000 | ~1.0% |
| DoS | ~16,000 | ~0.6% |
| Reconnaissance | ~14,000 | ~0.6% |
| Analysis | ~2,700 | ~0.1% |
| Backdoor | ~2,300 | ~0.1% |
| Shellcode | ~1,500 | ~0.06% |
| Worms | ~130 | ~0.005% |

Expected location: `data/raw/UNSW-NB15_1.csv` through `UNSW-NB15_4.csv`

---

## Class Imbalance Handling — Four Techniques

| Technique | Where Applied | How It Works |
|---|---|---|
| **SMOTE** | sklearn models | Synthetic minority oversampling with k_neighbors=3, per fold only |
| **KMeansSMOTE** | sklearn + DL models | MiniBatchKMeans clustering + SMOTE in safe zones only |
| **RandomUnderSampler** | DL models | Cap majority classes at 15,000 per fold before SMOTE |
| **Class-weighted loss** | DNN, Weighted BiLSTM | Inverse frequency weighting in PyTorch CrossEntropyLoss |

All resampling applied **per fold on training data only**.

---

## Evaluation Metrics

| Metric | Description | What It Measures |
|---|---|---|
| Binary Accuracy | Normal vs Attack | Overall attack detection correctness |
| Binary F1 | Harm. mean of precision/recall (binary) | Binary detection quality |
| Multi-class Accuracy | Accuracy across all 10 classes | Attack type classification correctness |
| Macro F1 | Unweighted mean F1 across classes | Equality-focused — penalizes poor minority performance |
| Weighted F1 | Support-weighted mean F1 | Real-world performance weighted by class frequency |

### Output Classes (Multi-class)

| Label | Description |
|---|---|
| Normal | Benign traffic |
| Fuzzers | Automated fuzzing attacks |
| Analysis | Port scanning / network probing |
| Backdoor | Unauthorized remote access |
| DoS | Denial of Service |
| Exploits | Exploit-based attacks |
| Generic | Protocol-level attacks |
| Reconnaissance | Information gathering |
| Shellcode | Code injection |
| Worms | Self-propagating malware |

---

## Test Suite — 70+ Regression Tests

| File | Tests | What It Verifies |
|---|---|---|
| `tests/test_leakage.py` | ~35 | Split sizes, per-fold fitting, SMOTE on train only, fold independence, DL no-leakage |
| `tests/test_dl_pipeline.py` | ~30 | AUC validity, LayerNorm (not BatchNorm), probability arrays, k_neighbors floor |
| `tests/test_audit_fixes.py` | 4 | No hardcoded class indices, runtime options exposed, real KMeansSMOTE |

```bash
python -m pytest tests/
```

---

## Technologies

| Technology | Usage |
|---|---|
| Python 3.8+ | Primary language |
| PyTorch 2.0+ | Deep learning (DNN, LSTM, BiLSTM, Multi-Task DNN) |
| scikit-learn 1.3+ | Preprocessing, feature selection, PCA, models, metrics |
| XGBoost 1.7+ | Gradient boosted trees |
| imbalanced-learn 0.11+ | SMOTE, KMeansSMOTE, RandomUnderSampler |
| Pandas / NumPy | Data manipulation |
| Matplotlib | Visualization (Agg backend) |
| Joblib | Model/artifact serialization |
| Pytest | Test framework |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Place UNSW-NB15_1..4.csv from the [official source](https://research.unsw.edu.au/projects/unsw-nb15-dataset) into `data/raw/`.

### 3. Run sklearn pipeline
```bash
python main.py
```

### 4. Run deep learning model
```bash
python models/train_dnn.py
python models/train_LSTM.py
python models/train_Bi-LSTM.py
python models/train_Bi-LSTM_shared-feature-extractor.py
python models/train_dnn_mi_pca_kmeans.py
```

For CPU: `python models/train_LSTM.py --device cpu`

### 5. Pipeline options (`main.py`)

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/raw` | Path to raw CSV files |
| `--balancer` | `kmeans` | `kmeans` or `smote` |
| `--n-splits` | `5` | CV folds |
| `--mi-k` | `15` | Top MI features |
| `--pca-variance` | `0.95` | PCA variance to retain |
| `--skip-plots` | off | Skip saving PNGs |

---

## Results Outputs

| Path | Description |
|---|---|
| `results/model_comparison.csv` | Blind holdout metrics (sklearn models) |
| `results/metrics.csv` | Per-fold CV metrics (sklearn models) |
| `results/*_per_class_report.csv` | Per-class precision/recall/F1 for each sklearn model |
| `results/corrected_pipeline/experiment_config.json` | Run configuration |
| `models/artifacts/<model>/*_test_metrics.json` | Blind holdout metrics (DL models) |
| `models/artifacts/<model>/*_cv_metrics.csv` | Per-fold CV metrics (DL models) |
| `models/artifacts/<model>/*_confusion_matrix.png` | Confusion matrix plots |
| `models/artifacts/<model>/*_model.pt` | Trained model weights |

---

## For Presentation — Faculty Meeting Talking Points

1. **Problem**: Network intrusion detection on severely imbalanced traffic (Worms: ~130 vs Normal: ~1.2M). Standard ML ignores rare but dangerous attacks.

2. **Approach**: Two-tier multi-model system — 3 classical ML + 5 deep learning models. All use 5-fold stratified CV with strict per-fold preprocessing to prevent data leakage. Four imbalance techniques: SMOTE, KMeansSMOTE, RandomUnderSampler, weighted loss.

3. **Critical Fix — Data Leakage**: Original pipeline fitted StandardScaler/PCA on the full dataset before splitting. This inflated results (e.g., XGBoost dropped from 98.70% → 91.22% after correction). All results in this README are from the corrected leakage-free pipeline. The `ARCHITECTURE_GAP.md` and `tests/test_leakage.py` document this thoroughly.

4. **Best Models**:
   - **BiLSTM_SharedFE** — best binary accuracy (98.70%) and binary F1 (0.9510)
   - **BiLSTM** — best Macro F1 (0.5022), strongest minority class detection
   - **DNN_MI_PCA_KMeans** — best multi-class accuracy (96.66%) and weighted F1 (0.9729)
   - **HGB** — best classical model (96.28%), competitive with DL
   - All 5 DL models perform similarly (98.57–98.70% binary acc, ~96.6% multi acc)

5. **Rigor**: 70+ regression tests prevent data leakage, verify pipeline correctness, and catch known bug patterns. Experiment configs record git commit + all hyperparameters for reproducibility.

---

## Citation

```
Kasina et al. (2026). A novel intrusion detection system for class imbalance datasets
using hybrid sampling with deep learning techniques. Information Sciences, 741.
```

Dataset:
```
Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network
intrusion detection systems. Military Communications and Information Systems
Conference (MilCIS), IEEE.
```
