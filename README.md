# Intrusion Detection System — UNSW-NB15

A multi-model network intrusion detection pipeline trained on the UNSW-NB15 dataset. Produces both binary (Normal / Attack) and multi-class (10 attack categories) predictions.

Implements the framework described in:

> **A novel intrusion detection system for class imbalance datasets using hybrid sampling with deep learning techniques** — Kasina et al., *Information Sciences* 741 (2026)

---

## Corrected Results (Leakage-Free Pipeline)

After eliminating data leakage (Scaler/PCA fit on full data, fold-0 retraining), corrected metrics on the locked 20% test set:

| Model | Accuracy | Precision | Recall | Weighted F1 | AUC |
|---|---|---|---|---|---|
| **HGB** | **0.9628** | **0.9831** | **0.9628** | **0.9703** | **0.9975** |
| XGBoost | 0.9122 | 0.9498 | 0.9122 | 0.9275 | 0.9834 |
| Logistic Regression | 0.9545 | 0.9763 | 0.9545 | 0.9640 | 0.9922 |

### Legacy Results (Pre-Correction)

The previous pipeline had known methodological issues (data leakage, fold-0 retraining). Preserved at `results/legacy_pipeline/` for reference only.

| Model | Binary Acc | Binary F1 | Multi-class Acc | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| XGBoost | 0.9848 | 0.9431 | 0.9624 | 0.4519 | 0.9701 |
| Logistic Regression | 0.9821 | 0.9337 | 0.9535 | 0.3705 | 0.9626 |

**Do not interpret accuracy decreases as model failures.** The legacy pipeline's inflated metrics were caused by preprocessing leakage.

---

## Project Structure

```
INTRUSION-DETECTION-SYSTEM/
│
├── main.py                              Tier 1 orchestrator (sklearn models)
├── requirements.txt
│
├── src/                                 Shared pipeline modules
│   ├── preprocessing.py                 Load, clean, encode, log1p normalisation
│   ├── feature_selection.py             Mutual Information (SelectKBest)
│   ├── dimensionality_reduction.py      Stratified 80/20 split (no fitting)
│   ├── balancing.py                     SMOTE / MiniBatchKMeans+SMOTE per fold
│   ├── cross_validation.py              Shared stratified CV loop
│   ├── dl_pipeline.py                   Shared DL infrastructure
│   ├── evaluation.py                    Confusion matrices, CSV results, plots
│   ├── experiment_config.py             Experiment metadata persistence
│   ├── train_hgb.py                     HistGradientBoostingClassifier
│   ├── train_xgboost.py                 XGBoost
│   └── train_logistic.py                Logistic Regression
│
├── models/                              Deep learning training scripts (Tier 2)
│   ├── train_dnn.py                     DNN baseline (class-weight loss)
│   ├── train_dnn_mi_pca_kmeans.py       DNN + MI + PCA + KMeansSMOTE
│   ├── train_LSTM.py                    Bi-LSTM + MI + PCA + KMeansSMOTE
│   ├── train_Bi-LSTM.py                 Weighted Bi-LSTM + class-weight loss
│   └── train_Bi-LSTM_shared-feature-extractor.py  Multi-task DNN (binary + multi-class heads)
│
├── tests/
│   └── test_leakage.py                  40 tests: leakage verification + regression
│
├── notebooks/
│   └── Intrusion_Detection.ipynb        Exploratory notebook
│
├── data/
│   └── raw/                             Place UNSW-NB15_1.csv … UNSW-NB15_4.csv here
│
├── assets/
│   └── Architecture.jpeg                Pipeline architecture diagram
│
├── results/
│   ├── legacy_pipeline/                 Historical benchmarks (pre-correction)
│   ├── corrected_pipeline/              Current results + experiment configs
│   └── comparison/                      Legacy vs corrected side-by-side
│
└── artifacts/                           Model + preprocessing artifacts (joblib)
    ├── hgb/
    ├── xgboost/
    └── logistic_regression/
```

---

## Pipeline Flow

### Tier 1: Classical ML (`main.py`)

```
data/raw/UNSW-NB15_1..4.csv
         │
         ▼
┌──────────────────────────────┐
│ Preprocessing                │  src/preprocessing.py
│ - Load 4 CSVs, concatenate   │
│ - Clean targets → LabelEncode│
│ - Drop metadata cols          │
│ - Encode categorical features │
│ - Log1p normalisation         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Stratified 80/20 Split       │  src/dimensionality_reduction.py
│ - NO transformers fitted     │
│ - Locked test set created    │
└──────────────┬───────────────┘
               │
     ┌─────────┴──────────┐
     ▼                    ▼
┌──────────────┐   ┌──────────────┐
│ Per-Fold CV  │   │ Final Retrain│
│ (5 folds)    │   │ (full 80%)   │
│              │   │              │
│ MI → fit     │   │ MI → fit     │
│ Scaler → fit │   │ Scaler → fit │
│ PCA → fit    │   │ PCA → fit    │
│ SMOTE only   │   │ SMOTE only   │
│ on train     │   │ on train     │
│ → train      │   │ → retrain    │
│ → eval val   │   │ → eval test  │
└──────┬───────┘   └──────┬───────┘
       │                  │
       └────────┬─────────┘
                ▼
     ┌─────────────────────┐
     │ Single test eval    │  locked 20% set
     │ on locked test set  │
     └─────────────────────┘
```

**Critical invariants (no data leakage):**
- MI selection is fitted on fold-train / full-train only
- StandardScaler is fitted on fold-train / full-train only
- PCA is fitted on fold-train / full-train only
- K-means SMOTE is applied to fold-train / full-train only — never val or test
- Test set is locked and never touched until final evaluation
- Final model is retrained on the COMPLETE 80% training set, not a CV fold

### Tier 2: Deep Learning (`models/*.py`)

DL scripts use shared infrastructure from `src/dl_pipeline.py`:

```
src/dl_pipeline.py
├── load_data()              preprocessing + 80/20 split
├── preprocess_fold()        per-fold MI → Scaler → PCA → KMeansSMOTE
├── preprocess_final()       full-train preprocessing for final retrain
├── evaluate_predictions()   binary + multiclass metrics
├── save_dl_artifacts()      model weights + metrics + confusion matrix
└── set_seeds() / get_device()  reproducibility
```

Each DL script follows the same leakage-free protocol as Tier 1.

---

## Models

### Tier 1 — Classical ML (`src/`)

| Model | File | Hyperparameters |
|---|---|---|
| HistGradientBoosting | `src/train_hgb.py` | max_iter=30, lr=0.05, max_depth=5, l2=1.0 |
| XGBoost | `src/train_xgboost.py` | n_estimators=30, subsample=0.1, max_depth=3, colsample=0.1 |
| Logistic Regression | `src/train_logistic.py` | solver=saga, multi_class=multinomial, max_iter=50 |

### Tier 2 — Deep Learning (`models/`)

| Script | Architecture | Preprocessing |
|---|---|---|
| `train_dnn.py` | 2-layer DNN (64→32) + BatchNorm + Dropout(0.1) | Scaler only, class-weight loss |
| `train_dnn_mi_pca_kmeans.py` | 3-layer DNN (128→64→32) + BatchNorm + Dropout(0.2) | MI(30) → PCA(15) → RUS + KMeansSMOTE |
| `train_LSTM.py` | Bi-LSTM (hidden=32, 1 layer) + FC(32→out) | MI(30) → PCA(15) → RUS + KMeansSMOTE |
| `train_Bi-LSTM.py` | Weighted Bi-LSTM (hidden=32) + FC(32→out) | MI(30) → PCA(15) → RUS + KMeansSMOTE, class weights |
| `train_Bi-LSTM_shared-feature-extractor.py` | Multi-task DNN: shared backbone (128→64), binary head + multi-class head, joint loss (0.4/0.6) | MI(30) → PCA(15) → RUS + KMeansSMOTE |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install torch    # required for DL models
```

### 2. Download the dataset

Download UNSW-NB15 from the [official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place all four CSV files in `data/raw/`:

```
data/raw/UNSW-NB15_1.csv
data/raw/UNSW-NB15_2.csv
data/raw/UNSW-NB15_3.csv
data/raw/UNSW-NB15_4.csv
```

### 3. Run Tier 1 (sklearn models)
```bash
python main.py
```

### 4. Run Tier 2 (DL models)

Each model runs independently:

```bash
python models/train_dnn.py
python models/train_dnn_mi_pca_kmeans.py
python models/train_LSTM.py
python "models/train_Bi-LSTM.py"
python "models/train_Bi-LSTM_shared-feature-extractor.py"
```

All DL scripts accept `--data-dir` for custom data paths:

```bash
python models/train_dnn.py --data-dir /path/to/data/raw
```

### 5. Run the test suite
```bash
python -m pytest tests/test_leakage.py -v
```

---

## Pipeline Options (`main.py`)

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/raw` | Path to raw CSV files |
| `--balancer` | `kmeans` | Balancing strategy: `kmeans` or `smote` |
| `--n-splits` | `5` | Number of CV folds |
| `--mi-k` | `15` | Top-k MI features to retain per fold |
| `--pca-variance` | `0.95` | Cumulative PCA variance to retain |
| `--skip-plots` | off | Skip saving confusion matrix PNGs |

```bash
python main.py --balancer smote --mi-k 20 --skip-plots
```

---

## Output Classes

**Binary:** Normal / Attack

**Multi-class (10 attack categories):**

| Label | Description |
|---|---|
| Normal | Benign traffic |
| Fuzzers | Fuzzing attempts |
| Analysis | Network scanning / analysis |
| Backdoor | Backdoor access |
| DoS | Denial of Service |
| Exploits | Exploit-based attacks |
| Generic | Generic protocol attacks |
| Reconnaissance | Reconnaissance sweeps |
| Shellcode | Shellcode injection |
| Worms | Worm propagation |

---

## Class Imbalance

The dataset is heavily imbalanced — the Worms class has only **~111 samples** out of approximately **~2.5M rows**. The pipeline handles this with:

- **K-means SMOTE (default):** MiniBatchKMeans cluster pre-processing + SMOTE, applied inside each CV fold or on the full training set for final retrain.
- **Standard SMOTE:** `--balancer smote` flag, `k_neighbors=3`.
- **DL models:** RandomUnderSampler (cap 15,000) + KMeansSMOTE (`k_neighbors=2`), or class-weighted loss functions.

Balancing is applied **only to training data** — validation and test sets are never balanced.

---

## Evaluation

### Metrics

All models report:

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| Precision | Support-weighted precision |
| Recall | Support-weighted recall |
| Weighted F1 | Support-weighted F1 across classes |
| AUC | One-vs-rest weighted AUC |

Per-class precision, recall, and F1 are saved to `results/corrected_pipeline/*_per_class_report.csv`.

### Output Files

| Path | Description |
|---|---|
| `results/corrected_pipeline/model_comparison.csv` | Blind test metrics for all models |
| `results/corrected_pipeline/metrics.csv` | Per-fold CV metrics |
| `results/corrected_pipeline/experiment_config.json` | Pipeline parameters + timestamps |
| `results/corrected_pipeline/*_per_class_report.csv` | Per-class classification reports |
| `results/legacy_pipeline/` | Historical benchmarks |
| `results/comparison/legacy_vs_corrected.csv` | Side-by-side comparison |
| `artifacts/*/model.joblib` | Trained model files |
| `artifacts/*/scaler.joblib` | Fitted StandardScaler |
| `artifacts/*/pca.joblib` | Fitted PCA |
| `artifacts/*/mi_selector.joblib` | Fitted MI selector |

---

## Testing

The test suite (`tests/test_leakage.py`) contains 40 tests verifying:

- StandardScaler fitted on training data only
- PCA fitted on training data only
- MI feature selection fitted on training data only
- K-means SMOTE applied to training data only
- Test data never used during preprocessing fitting
- 80/20 split is stratified and reproducible
- Per-fold CV transformers are independently fitted
- DL pipeline (`dl_pipeline.py`) preprocessing is leakage-free
- Final retraining uses the full 80% training set (not a CV fold)
- Test data never enters any balancing function

```bash
python -m pytest tests/test_leakage.py -v
```

---

## Citation

```
Kasina et al. (2026). A novel intrusion detection system for class imbalance datasets
using hybrid sampling with deep learning techniques.
Information Sciences, 741.
```

Dataset:
```
Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network
intrusion detection systems. Military Communications and Information Systems
Conference (MilCIS), IEEE.
```
