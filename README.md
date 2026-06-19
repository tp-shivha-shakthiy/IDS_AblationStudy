# Intrusion Detection System — UNSW-NB15

A multi-model network intrusion detection pipeline trained on the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Produces both **binary** (Normal / Attack) and **multi-class** (10 attack categories) predictions.

Implements the framework described in:
> *A novel intrusion detection system for class imbalance datasets using hybrid sampling with deep learning techniques* — Kasina et al., Information Sciences 741 (2026)

---

## Project Structure

```
INTRUSION-DETECTION-SYSTEM/
│
├── main.py                              Pipeline orchestrator (sklearn models)
├── requirements.txt
│
├── src/                                 Shared pipeline modules
│   ├── preprocessing.py                 Load, clean, encode, log1p normalisation
│   ├── feature_selection.py             Mutual Information (SelectKBest)
│   ├── dimensionality_reduction.py      StandardScaler → PCA → 80/20 split
│   ├── balancing.py                     SMOTE / MiniBatchKMeans+SMOTE per fold
│   ├── cross_validation.py              Shared stratified CV loop
│   └── evaluation.py                    Confusion matrices, CSV results, plots
│
├── models/                              Model training scripts
│   ├── train_hgb.py                     HistGradientBoostingClassifier (CV)
│   ├── train_xgboost.py                 XGBoost (CV + blind holdout test)
│   ├── train_logistic.py                Logistic Regression (multinomial/saga)
│   ├── train_dnn.py                     PyTorch DNN with weighted cross-entropy
│   ├── train_LSTM.py                    Bi-LSTM + MI + PCA + KMeansSMOTE
│   ├── train_Bi-LSTM.py                  Weighted Bi-LSTM vs XGBoost dual pipeline
│   ├── train_Bi-LSTM_shared-feature-extractor.py  Multi-task DNN (shared backbone, binary + multi-class heads)
│   └── train_dnn_mi_pca_kmeans.py        4-layer DNN with MI, PCA, and KMeansSMOTE
│
├── notebooks/
│   └── Intrusion_Detection.ipynb        Exploratory notebook with full pipeline & results
│
├── data/
│   ├── raw/                             Place UNSW-NB15_1.csv … UNSW-NB15_4.csv here
│   └── processed/                       Reserved for cached intermediate arrays
│
├── assets/
│   └── Architecture.jpeg                Pipeline architecture diagram
│
└── results/
    └── model_comparison.xlsx            Blind holdout metrics (pre-generated)
```

---

## Pipeline Flow

```
data/raw/UNSW-NB15_1..4.csv
         │
         ▼
┌──────────────────────────────┐
│ Phase 3: Preprocessing       │  src/preprocessing.py
│ - Load 4 CSVs, concatenate   │
│ - Clean targets → LabelEncode│
│ - Drop metadata cols          │
│ - Encode categorical features │
│ - Log1p normalization         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Phase 4a: MI Feature Sel.    │  src/feature_selection.py
│ - 5% stratified sample       │
│ - SelectKBest (top-k)        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Phase 4b+5: PCA + Split      │  src/dimensionality_reduction.py
│ - StandardScaler             │
│ - PCA (n components)         │
│ - 80/20 stratified split     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Phase 6: CV + Balancing      │  src/balancing.py
│ - StratifiedKFold (5 folds)  │
│ - SMOTE or KMeans+SMOTE      │
└──────────────┬───────────────┘
               │
        ┌──────┴────────┐
        ▼                ▼           ▼
┌──────────────┐ ┌────────────┐ ┌──────────────┐
│ HGB (CV)     │ │ XGBoost    │ │ LogisticReg  │
│ Phase 7      │ │ CV+Test    │ │ CV+Test      │
│              │ │ Phase 8+9  │ │ Phase 10     │
└──────┬───────┘ └──────┬─────┘ └──────┬───────┘
       └────────────────┼──────────────┘
                        ▼
               ┌──────────────────┐
               │ Evaluation       │  src/evaluation.py
               │ - Confusion mats │
               │ - Feature import │
               │ - CSV results    │
               └──────────────────┘
```

The sklearn models (HGB, XGBoost, Logistic Regression) run through the shared pipeline via `main.py`. The deep learning models are self-contained scripts with their own preprocessing, feature selection, and balancing built in.

---

## Models

### Sklearn models (`main.py`)

| Model | File | Notes |
|-------|------|-------|
| HistGradientBoostingClassifier | `models/train_hgb.py` | LightGBM-style binning, 5-fold CV |
| XGBoost | `models/train_xgboost.py` | CV + blind 20% holdout evaluation |
| Logistic Regression | `models/train_logistic.py` | Multinomial, saga solver, CV + test |

### Deep learning models (self-contained PyTorch scripts)

| Script | Architecture | Preprocessing | Balancing |
|--------|-------------|---------------|-----------|
| `train_dnn.py` | 3-layer DNN (64→32→n) with BatchNorm, Dropout | Log1p only | Weighted loss per fold |
| `train_LSTM.py` | Bidirectional LSTM (hidden=32) | MI top-30 → PCA 15 | RandomUnderSampler + KMeansSMOTE |
| `train_Bi-LSTM.py` | Weighted Bi-LSTM + XGBoost dual pipeline | MI top-30 → PCA 15 | KMeansSMOTE |
| `train_Bi-LSTM_shared-feature-extractor.py` | Multi-task DNN: shared backbone (128→64), binary + multi-class heads, joint 40/60 loss | MI top-30 → PCA 15 | None |
| `train_dnn_mi_pca_kmeans.py` | 4-layer DNN (128→64→32→n) with BatchNorm, Dropout | MI top-30 → PCA 15 | RandomUnderSampler + KMeansSMOTE |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> The DL scripts require PyTorch (`torch`), which is not in `requirements.txt`. Install it separately:
> ```bash
> pip install torch
> ```

### 2. Download the dataset

Download UNSW-NB15 from the [official page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place all four CSV files in `data/raw/`:

```
data/raw/UNSW-NB15_1.csv
data/raw/UNSW-NB15_2.csv
data/raw/UNSW-NB15_3.csv
data/raw/UNSW-NB15_4.csv
```

### 3. Run the sklearn pipeline

```bash
python main.py
```

### 4. Run a deep learning model

Each DL script is self-contained:

```bash
python models/train_dnn.py
python models/train_LSTM.py
python models/train_Bi-LSTM_shared-feature-extractor.py
```

### 5. Pipeline options (`main.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/raw` | Path to raw CSV files |
| `--balancer` | `smote` | Balancing strategy: `smote` or `kmeans` |
| `--n-splits` | `5` | Number of CV folds |
| `--mi-k` | `15` | Top-k MI features to retain |
| `--pca-components` | `10` | PCA output dimensions |
| `--skip-plots` | off | Skip saving confusion matrix PNGs |

```bash
python main.py --balancer kmeans --pca-components 12 --skip-plots
```

---

## Output Classes

**Binary:** `Normal` / `Attack`

**Multi-class (10 attack categories + Normal):**

| Label | Description |
|-------|-------------|
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

The dataset is heavily imbalanced — the `Worms` class has only ~111 samples out of ~2.5M rows. The pipeline handles this with:

- **Default (SMOTE):** Applied inside each CV fold with `k_neighbors=3` to avoid data leakage. Handles ultra-minority classes safely.
- **KMeans+SMOTE:** Uses `MiniBatchKMeans` for safe-zone discovery + SMOTE. May raise `RuntimeError` on extreme minority classes like Worms.

Pass `--balancer kmeans` to use the KMeans hybrid variant.

DL models use class-weighted loss functions or `RandomUnderSampler` + `KMeansSMOTE` combinations.

---

## Evaluation Metrics

All models report 5 metrics for both CV folds and blind holdout:

1. **Binary Accuracy** — Normal vs Attack (derived from multi-class predictions)
2. **Binary F1** — F1-score for binary detection
3. **Multi-class Accuracy** — Accuracy across all 10 attack categories
4. **Macro F1** — Unweighted mean F1 across classes (sensitive to minorities)
5. **Weighted F1** — Support-weighted mean F1 across classes

---

## Results

Generated outputs:

| Path | Description |
|------|-------------|
| `results/model_comparison.csv` | Blind holdout metrics for all sklearn models |
| `results/metrics.csv` | Per-fold CV metrics for all sklearn models |
| `results/model_comparison.xlsx` | Pre-generated blind holdout metrics (git-committed) |
| `assets/` | Confusion matrix PNGs + XGBoost feature importance plot |
---

## Comparison with Kasina et al. (2026)

The paper reports 99.95% binary F1 and 97.92% weighted F1 on UNSW-NB15 using SMOTE-ENN + DNN. This implementation matches the weighted F1 (0.9792, Bi-LSTM) and improves Macro F1 through the Weighted Bi-LSTM configuration (0.4932 vs unreported in paper), reflecting stronger minority class detection on rare attack types (Worms, Shellcode, Analysis). The multi-task hierarchical DNN with shared feature extractor is an architectural addition not present in the original paper.


