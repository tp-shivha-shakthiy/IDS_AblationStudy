# Repository Audit — Intrusion Detection System

## 1. Repository Structure

```
INTRUSION-DETECTION-SYSTEM/
│
├── main.py                              Pipeline orchestrator (sklearn models only)
├── requirements.txt                     Dependencies (pandas, numpy, sklearn, imblearn, xgboost, torch, matplotlib)
├── README.md                            Project documentation
├── License
│
├── src/                                 Shared pipeline modules + sklearn trainers
│   ├── __init__.py                      Package marker
│   ├── preprocessing.py                 CSV loading, target cleaning, LabelEncoder, log1p
│   ├── feature_selection.py             Mutual Information (SelectKBest, top-k)
│   ├── dimensionality_reduction.py      StandardScaler -> PCA -> 80/20 split
│   ├── balancing.py                     SMOTE / MiniBatchKMeans+SMOTE per fold
│   ├── cross_validation.py              Generic stratified CV runner for sklearn estimators
│   ├── evaluation.py                    Confusion matrices, feature importance, CSV output
│   ├── train_hgb.py                     HistGradientBoostingClassifier wrapper
│   ├── train_xgboost.py                 XGBoost wrapper (CV + blind test eval)
│   └── train_logistic.py               LogisticRegression wrapper (CV + blind test eval)
│
├── models/                              Self-contained deep learning scripts
│   ├── __init__.py                      Package marker
│   ├── train_dnn.py                     3-layer DNN, log1p preprocessing, weighted cross-entropy
│   ├── train_LSTM.py                    Bi-LSTM + MI + PCA + KMeansSMOTE
│   ├── train_Bi-LSTM.py                 Weighted Bi-LSTM + XGBoost dual pipeline
│   ├── train_Bi-LSTM_shared-feature-extractor.py   Multi-task DNN (binary + multi-class heads)
│   └── train_dnn_mi_pca_kmeans.py       4-layer DNN + MI + PCA + KMeansSMOTE
│
├── notebooks/
│   └── Intrusion_Detection.ipynb        Exploratory notebook
│
├── data/
│   ├── raw/                             UNSW-NB15 CSV files (not committed)
│   └── processed/                       Reserved for cached arrays
│
├── assets/
│   └── Architecture.jpeg                Pipeline architecture diagram
│
└── results/
    └── model_comparison.xlsx            Pre-generated blind holdout metrics
```

---

## 2. Important Files and Their Purposes

### Entry Points

| File | Executable | Purpose |
|---|---|---|
| `main.py` | `python main.py` | Orchestrates full sklearn pipeline: preprocess -> MI -> PCA -> split -> CV -> train HGB/XGBoost/LR -> evaluate |
| `models/train_dnn.py` | `python models/train_dnn.py` | Self-contained DNN training with its own preprocessing, no oversampling |
| `models/train_LSTM.py` | `python models/train_LSTM.py` | Self-contained Bi-LSTM with MI+PCA+KMeansSMOTE |
| `models/train_Bi-LSTM.py` | `python models/train_Bi-LSTM.py` | Dual-pipeline: Weighted Bi-LSTM + XGBoost comparison |
| `models/train_Bi-LSTM_shared-feature-extractor.py` | `python models/train_Bi-LSTM_shared-feature-extractor.py` | Multi-task hierarchical DNN (shared backbone, dual heads) |
| `models/train_dnn_mi_pca_kmeans.py` | `python models/train_dnn_mi_pca_kmeans.py` | 4-layer DNN with MI+PCA+KMeansSMOTE |

### Shared Modules (`src/`)

| File | Purpose |
|---|---|
| `preprocessing.py` | Loads UNSW-NB15 CSVs, cleans `attack_cat` target, LabelEncodes categorical features, applies log1p normalization |
| `feature_selection.py` | MI-based feature selection using SelectKBest on a 5% stratified sample |
| `dimensionality_reduction.py` | StandardScaler -> PCA -> stratified 80/20 train/test split |
| `balancing.py` | Two SMOTE strategies applied inside StratifiedKFold: plain SMOTE and MiniBatchKMeans+SMOTE |
| `cross_validation.py` | Generic CV runner: takes balanced folds + sklearn estimator, returns per-fold + mean metrics DataFrame |
| `evaluation.py` | Binary + multi-class confusion matrices, XGBoost feature importance plot, CSV results persistence, console summary |
| `train_hgb.py` | HistGradientBoostingClassifier training via `cross_validation.run_cv` |
| `train_xgboost.py` | XGBoost CV + blind holdout test evaluation via `cross_validation.run_cv` |
| `train_logistic.py` | LogisticRegression CV + blind holdout test evaluation via `cross_validation.run_cv` |

---

## 3. Current Execution Flow

### Pipeline A: `main.py` (sklearn models)

```
Phase 3: src/preprocessing.py::load_and_preprocess()
    Input:  data/raw/UNSW-NB15_{1..4}.csv
    Action: Load CSVs, clean attack_cat, LabelEncode all categorical cols, log1p
    Output: X_processed (N, F) float32, y_multi (N,) int, le (LabelEncoder)

Phase 4a: src/feature_selection.py::select_features()
    Input:  X_processed, y_multi
    Action: MI scores on 5% stratified sample, SelectKBest top-k
    Output: X_mi (N, k), selector (fitted SelectKBest)

Phase 4b+5: src/dimensionality_reduction.py::reduce_and_split()
    Input:  X_mi, y_multi
    Action: StandardScaler (fit on ALL data) -> PCA (fit on ALL data) -> stratified 80/20 split
    Output: X_train, X_test, y_train, y_test, scaler, pca

Phase 6: src/balancing.py::smote_folds() or kmeans_smote_folds()
    Input:  X_train, y_train
    Action: StratifiedKFold -> SMOTE or KMeans+SMOTE on each training fold
    Output: balanced_folds (list of dicts with X_train_fold, y_train_fold, X_val_fold, y_val_fold)

Phase 7: src/train_hgb.py::train_hgb()
    Input:  balanced_folds, normal_class_idx, class_names
    Action: 5-fold CV via cross_validation.run_cv() (no test set eval)

Phase 8+9: src/train_xgboost.py::train_xgboost()
    Input:  balanced_folds, X_test, y_test, normal_class_idx, class_names
    Action: 5-fold CV via cross_validation.run_cv() + final model on fold-0 data evaluated on X_test

Phase 10: src/train_logistic.py::train_logistic()
    Input:  balanced_folds, X_test, y_test, normal_class_idx, class_names
    Action: 5-fold CV via cross_validation.run_cv() + final model on fold-0 data evaluated on X_test

Output: src/evaluation.py::print_final_summary(), save_results(), plot_confusion_matrix(), plot_feature_importance()
```

### Pipeline B: `models/` (self-contained DL scripts)

Each DL script is fully independent — duplicates all preprocessing, feature selection, PCA, and balancing inline at module level, then defines a PyTorch model class and runs StratifiedKFold CV.

---

## 4. All Model Implementations

### 4.1 Classical ML (via `main.py` + `src/`)

| Model | File | Architecture | CV | Test Eval | Loss |
|---|---|---|---|---|---|
| HistGradientBoostingClassifier | `src/train_hgb.py` | sklearn default (max_iter=30) | Yes (5-fold) | No | N/A |
| XGBoost | `src/train_xgboost.py` | XGBClassifier (multi:softprob, hist, n_estimators=30, max_depth=3, subsample=0.1) | Yes (5-fold) | Yes (fold-0 retrain) | N/A |
| LogisticRegression | `src/train_logistic.py` | multinomial/saga, max_iter=50 | Yes (5-fold) | Yes (fold-0 retrain) | N/A |

### 4.2 Deep Learning (via `models/`)

| Script | Model Class | Architecture | Loss | Epochs |
|---|---|---|---|---|
| `train_dnn.py` | `DeepNeuralNetwork` | Linear(64) -> BN -> ReLU -> Dropout(0.1) -> Linear(32) -> BN -> ReLU -> Linear(n) | Weighted CrossEntropy | 5 |
| `train_LSTM.py` | `BidirectionalLSTMNetwork` | BiLSTM(input, 32) -> Linear(64, 32) -> Dropout(0.2) -> Linear(32, n) | CrossEntropy (unweighted) | 5 |
| `train_Bi-LSTM.py` (Pipeline 1) | `WeightedBidirectionalLSTM` | BiLSTM(input, 32) -> Linear(64, 32) -> ReLU -> Linear(32, n) | Weighted CrossEntropy (per-resampled-distribution) | 5 |
| `train_Bi-LSTM.py` (Pipeline 2) | XGBClassifier | n_estimators=40, max_depth=6, lr=0.1, hist | N/A | N/A |
| `train_Bi-LSTM_shared-feature-extractor.py` | `MultiTaskHierarchicalDNN` | Shared: Linear(128)->BN->ReLU->Drop(0.2)->Linear(64)->BN->ReLU->Drop(0.2). Binary head: Linear(64,2). Multi head: Linear(64,32)->ReLU->Linear(32,n) | Joint loss: 0.4*CE_bin + 0.6*CE_multi (unweighted) | 8 |
| `train_dnn_mi_pca_kmeans.py` | `DeepNeuralNetwork` | Linear(128)->BN->ReLU->Drop(0.2)->Linear(64)->BN->ReLU->Drop(0.2)->Linear(32)->ReLU->Linear(n) | CrossEntropy (unweighted) | 10 |

---

## 5. All Preprocessing Implementations

### 5.1 `src/preprocessing.py` (used by `main.py`)

| Step | Description |
|---|---|
| CSV Loading | Reads UNSW-NB15_{1..4}.csv with `pd.read_csv(header=None)`, handles 47-col and 49-col variants |
| Target Cleaning | `fillna('Normal')` -> lowercase -> map to standardized names via `CATEGORY_MAPPING` |
| Target Encoding | `LabelEncoder.fit_transform()` on `attack_cat` |
| Column Dropping | Drops: id, label, stime, ltime, srcip, dstip |
| Feature Encoding | `LabelEncoder.fit_transform()` per categorical column (proto, state, service, etc.) |
| Normalization | `np.log1p(X_raw.clip(lower=0))` -> `fillna(0)` -> float32 |

### 5.2 DL scripts (inline preprocessing, all in `models/`)

| Step | Description | Variation |
|---|---|---|
| CSV Loading | Same as `src/preprocessing.py` but reads from CWD (`f'UNSW-NB15_{i}.csv'`) | All DL scripts load from CWD, not `data/raw/` |
| Target Cleaning | Same logic as `src/preprocessing.py` | Identical |
| Target Encoding | `LabelEncoder.fit_transform()` | Identical |
| Column Dropping | Drops: id, label, stime, ltime, srcip, dstip | Identical |
| Feature Splitting | Separates continuous (log1p), categorical (LabelEncoder), binary (numeric cast) | Different from `src/preprocessing.py` which LabelEncodes all object columns uniformly |
| Normalization | Continuous: `np.log1p(clip(lower=0))` -> `fillna(0)`. Categorical: LabelEncoder. Binary: `pd.to_numeric` -> `fillna(0)` | Structurally different: 3-way split vs flat encoding |

**Key difference**: `src/preprocessing.py` applies LabelEncoder uniformly to all object columns. DL scripts separate features into continuous/categorical/binary groups with different treatment. The DL preprocessing produces a **different feature matrix** than the `src/` pipeline.

---

## 6. All Balancing Implementations

### 6.1 `src/balancing.py` (used by `main.py`)

**Strategy 1 — `smote_folds()`** (default)
- StratifiedKFold(n_splits=5) on X_train
- Per fold: `SMOTE(k_neighbors=3)` on training portion only
- Validation fold is untouched

**Strategy 2 — `kmeans_smote_folds()`**
- StratifiedKFold(n_splits=5) on X_train
- Per fold: `MiniBatchKMeans.fit_predict()` -> keeps ALL samples (cluster filtering is a no-op, lines 146-148) -> `SMOTE(k_neighbors=2)` on all fold data
- **Bug**: KMeans clustering has no effect; the `clean_indices` is always `range(len(X_tr_raw))`

### 6.2 DL scripts (inline balancing)

| Script | Strategy | Scope | Leakage? |
|---|---|---|---|
| `train_dnn.py` | None (uses weighted loss only) | Per-fold StandardScaler only | No |
| `train_LSTM.py` | `RandomUnderSampler(max=15000)` + `KMeansSMOTE(k=2)` | **Global** on full dataset before CV | **YES** |
| `train_Bi-LSTM.py` (Pipe 1) | `RandomUnderSampler(max=15000)` + `KMeansSMOTE(k=2)` | **Global** on full dataset before CV | **YES** |
| `train_Bi-LSTM.py` (Pipe 2) | `RandomUnderSampler(max=8000)` + `KMeansSMOTE(k=2)` | **Global** on full dataset before CV | **YES** |
| `train_Bi-LSTM_shared-feature-extractor.py` | None | N/A | No |
| `train_dnn_mi_pca_kmeans.py` | `RandomUnderSampler(max=15000)` + `KMeansSMOTE(k=2)` | **Global** on full dataset before CV | **YES** |

---

## 7. All Train/Test/CV Split Logic

| Location | Split Type | Details |
|---|---|---|
| `src/feature_selection.py:46` | Stratified subsample | 5% of data for MI score estimation (discarded after fit) |
| `src/dimensionality_reduction.py:65` | Stratified 80/20 holdout | `train_test_split(stratify=y, test_size=0.20)` — used by `main.py` |
| `src/balancing.py:58` | Stratified 5-fold CV | `StratifiedKFold(n_splits=5)` on training set only |
| `models/train_dnn.py:189` | Stratified 5-fold CV | On **full preprocessed data** (no train/test split) |
| `models/train_LSTM.py:191` | Stratified 5-fold CV | On **globally oversampled** data |
| `models/train_Bi-LSTM.py:183` | Stratified 5-fold CV | On **globally oversampled** data (both pipelines) |
| `models/train_Bi-LSTM_shared-feature-extractor.py:160` | Stratified 5-fold CV | On full PCA data (no oversampling, no holdout) |
| `models/train_dnn_mi_pca_kmeans.py:176` | Stratified 5-fold CV | On **globally oversampled** data |

---

## 8. All Oversampling Locations

| File | Line(s) | Type | Scope |
|---|---|---|---|
| `src/balancing.py:70-71` | `sm.fit_resample(X_tr, y_tr)` | SMOTE (k=3) | Per-fold, training only |
| `src/balancing.py:150-152` | `smote_engine.fit_resample(X_tr_clean, y_tr_clean)` | SMOTE (k=2) | Per-fold, training only (after no-op KMeans) |
| `models/train_LSTM.py:136` | `rus.fit_resample(X_pca, y_all)` | RandomUnderSampler | Global |
| `models/train_LSTM.py:139` | `kms.fit_resample(X_rus, y_rus)` | KMeansSMOTE | Global |
| `models/train_Bi-LSTM.py:140` | `RandomUnderSampler(...).fit_resample(X_pca, y_all)` | RandomUnderSampler | Global |
| `models/train_Bi-LSTM.py:141` | `KMeansSMOTE(...).fit_resample(X_rus_1, y_rus_1)` | KMeansSMOTE | Global |
| `models/train_Bi-LSTM.py:154` | `RandomUnderSampler(...).fit_resample(X_processed, y_all)` | RandomUnderSampler | Global |
| `models/train_Bi-LSTM.py:155` | `KMeansSMOTE(...).fit_resample(X_rus_2, y_rus_2)` | KMeansSMOTE | Global |
| `models/train_dnn_mi_pca_kmeans.py:135-136` | `rus.fit_resample(X_pca, y_all)` | RandomUnderSampler | Global |
| `models/train_dnn_mi_pca_kmeans.py:138-139` | `kms.fit_resample(X_rus, y_rus)` | KMeansSMOTE | Global |

---

## 9. Evaluation Implementations

### 9.1 `src/evaluation.py` (used by `main.py`)

| Function | Description |
|---|---|
| `plot_confusion_matrix()` | Binary CM (Normal/Attack) + Multi-class CM (10 classes), saved as PNG |
| `plot_feature_importance()` | XGBoost `feature_importances_` bar chart for PCA components |
| `save_results()` | Writes `results/model_comparison.csv` (blind test) and `results/metrics.csv` (CV) |
| `print_final_summary()` | Formatted console table of blind test metrics |

### 9.2 `src/cross_validation.py` (used by sklearn trainers)

- Per-fold: multi-class accuracy, macro F1, weighted F1, binary accuracy, binary F1
- Appends mean row to results DataFrame

### 9.3 DL scripts (inline)

All DL scripts compute the same 5 metrics: binary acc, binary F1, multi-class acc, macro F1, weighted F1. Results are printed to console only; no CSV/PNG persistence.

### 9.4 Model Saving/Loading

**None.** No model is saved to disk anywhere in the codebase. All trained models are lost when the script finishes.

---

## 10. Duplicated or Conflicting Implementations

### 10.1 Preprocessing Duplication

The following preprocessing block is copy-pasted across **all 5 DL scripts** in `models/`:

- CSV loading (col_names, file reading, column assignment)
- Target cleaning & mapping
- LabelEncoder on target
- Column dropping
- Feature splitting (continuous/categorical/binary)
- log1p normalization
- LabelEncoder on categorical features

`src/preprocessing.py` implements the same logic but with a **different feature encoding strategy** (flat LabelEncoder on all object columns vs. the 3-way split in DL scripts).

### 10.2 MI Feature Selection Duplication

MI-based feature selection is implemented independently in:
- `src/feature_selection.py` (uses `SelectKBest`, top-k default=15)
- `models/train_LSTM.py` (manual argsort, top-30)
- `models/train_Bi-LSTM.py` (manual argsort, top-30)
- `models/train_Bi-LSTM_shared-feature-extractor.py` (manual argsort, top-30)
- `models/train_dnn_mi_pca_kmeans.py` (manual argsort, top-30)

The `src/` version uses `SelectKBest`; the DL scripts use raw `np.argsort` on MI scores. Different k values (15 vs 30).

### 10.3 PCA Duplication

PCA is implemented independently in:
- `src/dimensionality_reduction.py` (n_components=10 default)
- `models/train_LSTM.py` (n_components=15)
- `models/train_Bi-LSTM.py` (n_components=15)
- `models/train_Bi-LSTM_shared-feature-extractor.py` (n_components=15)
- `models/train_dnn_mi_pca_kmeans.py` (n_components=15)

### 10.4 Balancing Duplication

- `src/balancing.py` implements SMOTE per fold (correct)
- Each DL script implements its own global oversampling (incorrect for evaluation)

### 10.5 DNN Architecture Duplication

- `DeepNeuralNetwork` class is defined in both `train_dnn.py` (64->32) and `train_dnn_mi_pca_kmeans.py` (128->64->32). Different architectures, same class name.

### 10.6 `train_Bi-LSTM.py` contains XGBoost

`models/train_Bi-LSTM.py` trains **both** a Weighted Bi-LSTM and an XGBoost classifier (Pipeline 2). This XGBoost implementation is a **duplicate** of the one in `src/train_xgboost.py` with different hyperparameters and no shared code.

---

## 11. Broken Imports and Executable Entry Points

### 11.1 Import Analysis

| File | Imports from `src.*` | Status |
|---|---|---|
| `main.py` | All src modules | OK (runs from project root) |
| `src/train_hgb.py` | `src.cross_validation` | OK |
| `src/train_xgboost.py` | `src.cross_validation` | OK |
| `src/train_logistic.py` | `src.cross_validation` | OK |
| `models/train_dnn.py` | None (self-contained) | OK |
| `models/train_LSTM.py` | None (self-contained) | OK |
| `models/train_Bi-LSTM.py` | None (self-contained) | OK |
| `models/train_Bi-LSTM_shared-feature-extractor.py` | None (self-contained) | OK |
| `models/train_dnn_mi_pca_kmeans.py` | None (self-contained) | OK |

### 11.2 Data Path Issue in DL Scripts

All 5 DL scripts load CSVs from the **current working directory**:
```python
files = [f'UNSW-NB15_{i}.csv' for i in range(1, 4 + 1)]
```

This means they must be run **from inside `data/raw/`** or the CSVs must be symlinked/copied to CWD. The README says `python models/train_dnn.py` (from project root), which would fail unless the CSVs are at project root level — contradicting the stated `data/raw/` location.

### 11.3 `train_Bi-LSTM.py` Filename

Contains a hyphen, making it non-importable as a Python module (`import train_Bi-LSTM` would fail). Must be run as a script.

### 11.4 Executable Entry Points (confirmed working)

| Entry Point | Run Command | Working Dir Required |
|---|---|---|
| `main.py` | `python main.py` | Project root |
| `models/train_dnn.py` | `python models/train_dnn.py` | Must have CSVs in CWD |
| `models/train_LSTM.py` | `python models/train_LSTM.py` | Must have CSVs in CWD |
| `models/train_Bi-LSTM.py` | `python models/train_Bi-LSTM.py` | Must have CSVs in CWD |
| `models/train_Bi-LSTM_shared-feature-extractor.py` | `python models/train_Bi-LSTM_shared-feature-extractor.py` | Must have CSVs in CWD |
| `models/train_dnn_mi_pca_kmeans.py` | `python models/train_dnn_mi_pca_kmeans.py` | Must have CSVs in CWD |
