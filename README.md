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

## Ablation Study (Faculty-Requested 7-Experiment Sequence)

The ablation study varies exactly **three factors** while keeping the leakage-free methodology, the model hyperparameters, and the 80/20 split untouched:

- **MI** — mutual-information feature selection (SelectKBest, k=15, fitted on fold/full train only)
- **PCA** — dimensionality reduction (95% variance, fitted on fold/full train only)
- **KMeansSMOTE** — K-means SMOTE class balancing (train data only, never val/test)

StandardScaler is **not** an ablation factor — it stays on in every experiment, exactly as the current methodology requires.

| Experiment       | MI | PCA | KMeansSMOTE |
|------------------|----|-----|-------------|
| `raw`            | ✗  | ✗   | ✗           |
| `mi`             | ✓  | ✗   | ✗           |
| `mi_balancing`   | ✓  | ✗   | ✓           |
| `pca`            | ✗  | ✓   | ✗           |
| `pca_balancing`  | ✗  | ✓   | ✓           |
| `mi_pca`         | ✓  | ✓   | ✗           |
| `mi_pca_balancing` | ✓  | ✓  | ✓           |

`mi_pca_balancing` is the default and is byte-for-byte the current (corrected) pipeline.

### Running the ablation

```bash
python main.py --experiment raw            # raw features, no MI/PCA/SMOTE
python main.py --experiment mi             # MI only
python main.py --experiment mi_balancing   # MI + KMeansSMOTE
python main.py --experiment pca            # PCA only
python main.py --experiment pca_balancing  # PCA + KMeansSMOTE
python main.py --experiment mi_pca         # MI + PCA
python main.py --experiment mi_pca_balancing   # full pipeline (default)
```

Presets are the single source of truth for preprocessing: using `--experiment` together with `--balancer` / `--mi-k` / `--pca-variance` overrides raises a `ValueError`.

Per-experiment outputs are written under `results/<Model>/<experiment>/`. After each run, the per-model comparison tables in `results/<Model>/` are regenerated; once all seven experiments have been run, `ablation_test_metrics.csv` / `ablation_cv_metrics.csv` contain exactly seven rows in the order **Raw, MI, MI+KMeansSMOTE, PCA, PCA+KMeansSMOTE, MI+PCA, MI+PCA+KMeansSMOTE**.

### Leakage invariants — unchanged for every experiment

- MI / Scaler / PCA are fitted on fold-train or full-train only; val/test are only transformed
- K-means SMOTE is applied to train data only — never val or test
- The 20% test set is locked and never touched until final evaluation

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
│   ├── test_leakage.py                  Leakage verification + regression
│   ├── test_dl_pipeline.py              DL architectures + config schema
│   └── test_ablation.py                 Ablation presets, toggles, tables (84 total)
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
│   ├── corrected_pipeline/              Legacy corrected-run CSVs (reference)
│   ├── <experiment>/                    Cross-model comparison per experiment
│   ├── HGB/
│   │   ├── raw/ … mi_pca_balancing/     Per-experiment: model, metrics, config, plots
│   │   └── ablation_*.csv               Per-model 7-row ablation tables
│   ├── XGBoost/
│   │   ├── raw/ … mi_pca_balancing/
│   │   └── ablation_*.csv
│   └── LogReg/
│       ├── raw/ … mi_pca_balancing/
│       └── ablation_*.csv
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
| `--experiment` | `mi_pca_balancing` | Ablation preset: `raw`, `mi`, `mi_balancing`, `pca`, `pca_balancing`, `mi_pca`, `mi_pca_balancing` |
| `--balancer` | `kmeans` | Balancing strategy: `kmeans` or `smote` |
| `--n-splits` | `5` | Number of CV folds |
| `--mi-k` | `15` | Top-k MI features to retain per fold |
| `--pca-variance` | `0.95` | Cumulative PCA variance to retain |
| `--cap` | `0` | Cap each class to N samples before oversampling (speed/RAM, e.g. `--cap 15000`) |
| `--quick` | `0` | Run on a stratified sample of N rows (pipeline smoke test) |
| `--skip-plots` | off | Skip saving confusion matrix PNGs |

```bash
python main.py --experiment mi --balancer smote --skip-plots
python main.py --cap 15000 --quick 200000        # fast verification run
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
| `results/<Model>/<experiment>/test_metrics.json` | Blind-test metrics (incl. binary + multiclass) per experiment |
| `results/<Model>/<experiment>/cv_metrics.csv` | Per-fold CV metrics per experiment |
| `results/<Model>/<experiment>/experiment_config.json` | Pipeline params + experiment identity (`experiment`, `use_mi`, `use_pca`, `use_balancing`) |
| `results/<Model>/<experiment>/*_model.joblib` | Trained model |
| `results/<Model>/<experiment>/{scaler,pca,mi_selector,label_encoder}.joblib` | Fitted transformers |
| `results/<Model>/<experiment>/*_cm.png`, `*_roc_curve.png` | Confusion matrices + ROC |
| `results/<Model>/ablation_test_metrics.csv` | 7-row ablation table (test metrics), one per experiment |
| `results/<Model>/ablation_cv_metrics.csv` | 7-row ablation table (mean CV metrics) |
| `results/<Model>/ablation_<metric>.csv` | Pivoted metric tables (Model × experiment) |
| `results/<experiment>/model_comparison.csv` | Blind-test comparison across models for one experiment |
| `results/<experiment>/metrics.csv` | CV metrics across models for one experiment |
| `results/legacy_pipeline/` | Historical benchmarks |
| `artifacts/*/model.joblib` | Trained model files |

---

## Testing

The test suite contains **84 tests** across `tests/test_leakage.py`, `tests/test_dl_pipeline.py`, and `tests/test_ablation.py`:

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
- Ablation presets: exactly 7 experiments, correct MI/PCA/balancing flag combos
- `run_cv()` toggles: MI/PCA/balancing can be disabled without leaking
- Per-experiment outputs (`test_metrics.json`, `cv_metrics.csv`, `experiment_config.json`)
- Ablation tables: exactly 7 rows in the required order

```bash
python -m pytest tests/ -v
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
