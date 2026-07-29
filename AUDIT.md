# Repository Audit — Intrusion Detection System (Post-Refactoring)

## 1. Repository Structure

```
INTRUSION-DETECTION-SYSTEM/
│
├── main.py                              Pipeline orchestrator (sklearn models only)
├── requirements.txt                     Dependencies (pandas, numpy, sklearn, imblearn, xgboost, torch, matplotlib)
├── README.md                            Project documentation (leakage-free pipeline)
├── ARCHITECTURE_GAP.md                  Architecture gap analysis (resolved)
├── AUDIT.md                             This file — repository audit
├── License
│
├── src/                                 Shared pipeline modules + trainers
│   ├── __init__.py                      Package marker
│   ├── preprocessing.py                 CSV loading, target cleaning, LabelEncoder, log1p
│   ├── dimensionality_reduction.py      split_data() only — scaler/PCA moved to per-fold CV
│   ├── balancing.py                     KMeansSMOTE / SMOTE per fold
│   ├── cross_validation.py              Generic stratified CV with per-fold MI→Scaler→PCA→balancing
│   ├── evaluation.py                    Confusion matrices, feature importance, CSV output
│   ├── experiment_config.py             Experiment metadata + model hyperparameter presets
│   ├── dl_pipeline.py                   Shared DL infrastructure (load, preprocess, evaluate, save)
│   ├── train_hgb.py                     HistGradientBoostingClassifier (CV + final retrain + test eval)
│   ├── train_xgboost.py                 XGBoost (CV + final retrain + test eval)
│   └── train_logistic.py               LogisticRegression (CV + final retrain + test eval)
│
├── models/                              Self-contained deep learning scripts
│   ├── __init__.py                      Package marker
│   ├── train_dnn.py                     DNN + shared DL pipeline (no balancing)
│   ├── train_LSTM.py                    Bi-LSTM + shared DL pipeline (MI+PCA+KMeansSMOTE per fold)
│   ├── train_Bi-LSTM.py                 Bi-LSTM + shared DL pipeline (weighted loss)
│   ├── train_Bi-LSTM_shared-feature-extractor.py   Multi-task DNN (binary + multi-class heads)
│   └── train_dnn_mi_pca_kmeans.py       4-layer DNN + shared DL pipeline (MI+PCA+KMeansSMOTE per fold)
│   └── artifacts/                       Per-model subdirectories with saved models, metrics, plots
│       ├── DNN/                         dnn_model.pt, test_metrics.json, confusion matrix, etc.
│       ├── LSTM/                        lstm_model.pt, lstm_test_metrics.json, etc.
│       ├── BiLSTM/                      bilstm_model.pt, bilstm_test_metrics.json, etc.
│       ├── BiLSTM_SharedFE/             ... shared feature extractor artifacts
│       ├── DNN_MI_PCA_KMeans/           ... DNN with MI+PCA+KMeansSMOTE artifacts
│       ├── HGB/                         hgb_model.joblib, test_metrics.json, confusion matrices
│       ├── XGBoost/                     xgboost_model.joblib, test_metrics.json, feature importance
│       └── LogReg/                      logreg_model.joblib, test_metrics.json, confusion matrices
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
├── artifacts/                           Sklearn preprocessing artifacts (scaler, pca, mi_selector, le)
│   ├── hgb/
│   ├── xgboost/
│   └── logistic_regression/
│
├── results/
│   ├── model_comparison.csv             Blind holdout metrics per model
│   └── metrics.csv                      Per-fold CV metrics
│
└── tests/                               Unit tests
    ├── test_leakage.py                  ~35 tests: per-fold independence, no leakage
    ├── test_dl_pipeline.py              ~30 tests: architecture, metrics, drop-last, batchnorm
    └── test_audit_fixes.py              4 tests: balancer options, KMeansSMOTE, preprocessing
```

---

## 2. Important Files and Their Purposes

### Entry Points

| File | Executable | Purpose |
|---|---|---|
| `main.py` | `python main.py` | Orchestrates full sklearn pipeline: preprocess → MI → PCA → split → CV → train HGB/XGBoost/LR → evaluate |
| `models/train_dnn.py` | `python models/train_dnn.py --data-dir data/raw` | DNN training via shared dl_pipeline (no balancing, weighted loss only) |
| `models/train_LSTM.py` | `python models/train_LSTM.py --data-dir data/raw` | Bi-LSTM with MI+PCA+KMeansSMOTE per fold |
| `models/train_Bi-LSTM.py` | `python models/train_Bi-LSTM.py --data-dir data/raw` | Weighted Bi-LSTM with per-fold preprocessing |
| `models/train_Bi-LSTM_shared-feature-extractor.py` | `python models/train_Bi-LSTM_shared-feature-extractor.py --data-dir data/raw` | Multi-task hierarchical DNN |
| `models/train_dnn_mi_pca_kmeans.py` | `python models/train_dnn_mi_pca_kmeans.py --data-dir data/raw` | 4-layer DNN with MI+PCA+KMeansSMOTE per fold |

### Shared Modules (`src/`)

| File | Purpose |
|---|---|
| `preprocessing.py` | Loads UNSW-NB15 CSVs, cleans `attack_cat` target, LabelEncodes categorical features, applies log1p normalization |
| `dimensionality_reduction.py` | `split_data()` — stratified 80/20 train/test split on preprocessed+MI-selected data |
| `balancing.py` | `balance_training_fold()` (per-fold) and `balance_full_train()` (full retrain) using KMeansSMOTE or SMOTE |
| `cross_validation.py` | Generic CV runner: per-fold MI→Scaler→PCA→balancing→train→eval, returns metrics + transformers |
| `evaluation.py` | Binary + multi-class confusion matrices, XGBoost feature importance plot, CSV results persistence, console summary |
| `experiment_config.py` | Experiment config builder with git hash, timestamp, hyperparameters, JSON persistence |
| `dl_pipeline.py` | Shared DL infrastructure: `load_data()`, `preprocess_fold()`, `preprocess_final()`, `evaluate_predictions()`, `save_dl_artifacts()`, model classes |
| `train_hgb.py` | HistGradientBoostingClassifier: per-fold CV via `run_cv()` + full-train retrain + blind test eval + model/artifact persistence |
| `train_xgboost.py` | XGBoost: same pipeline as HGB + feature importance plot |
| `train_logistic.py` | LogisticRegression: same pipeline as HGB/XGBoost |

---

## 3. Current Execution Flow (Leakage-Free Pipeline)

### Pipeline A: `main.py` (sklearn models)

```
Phase 1: src/preprocessing.py::load_and_preprocess()
    Input:  data/raw/UNSW-NB15_{1..4}.csv
    Action: Load CSVs, clean attack_cat, LabelEncode all categorical cols, log1p
    Output: X_processed (N, F) float32, y_multi (N,) int, le (LabelEncoder)

Phase 2a: src/dimensionality_reduction.py::split_data()
    Input:  X_mi, y_multi
    Action: stratified 80/20 train/test split (NO scaler/PCA fitting)
    Output: X_train, X_test, y_train, y_test

Phase 3: src/cross_validation.py::run_cv()  [called by each trainer]
    Input:  X_train, y_train
    Action: StratifiedKFold -> per fold: MI fit -> Scaler fit -> PCA fit -> 
            KMeansSMOTE -> train estimator -> evaluate
    Output: cv_metrics, fold_models, transformers

Phase 4: Each trainer performs final retrain
    Action: MI fit on full X_train -> Scaler fit -> PCA fit -> 
            KMeansSMOTE -> retrain model -> evaluate on X_test
    Output: test_metrics, saved model, test_metrics.json, plots

Output: src/evaluation.py::print_final_summary(), save_results(), 
        plot_confusion_matrix(), plot_feature_importance()
```

### Pipeline B: `models/` (DL scripts via shared `src/dl_pipeline.py`)

All 5 DL scripts follow the same pattern:

```
1. load_data(data_dir) -> train_df, test_df, le, X_columns
   (calls src.preprocessing.load_and_preprocess() internally)

2. StratifiedKFold CV loop:
   preprocess_fold(train_fold_idx, val_fold_idx, train_df, ...)
     -> MI fit on train only -> Scaler fit -> PCA fit -> KMeansSMOTE on train only
     -> returns X_tr, y_tr, X_val, y_val + transformers
   Train model on X_tr, evaluate on X_val
   Save fold metrics

3. Final retrain + test evaluation:
   preprocess_final(train_df, ...)
     -> MI fit on full train -> Scaler fit -> PCA fit -> KMeansSMOTE on full train
     -> returns X_train_b, y_train_b + transformers
   Transform test_df with train-fitted transformers
   Train model on balanced train, evaluate on test
   Save test_metrics.json, model.pt, confusion matrix, metadata
```

---

## 4. All Model Implementations

### 4.1 Classical ML (via `main.py` + `src/`)

| Model | File | Architecture | CV | Test Eval | Persistence |
|---|---|---|---|---|---|
| HistGradientBoostingClassifier | `src/train_hgb.py` | max_iter=30, lr=0.05, max_depth=5, l2=1.0 | Yes (5-fold) | Yes (blind test) | Model + test_metrics.json + plots |
| XGBoost | `src/train_xgboost.py` | n_estimators=30, subsample=0.1, max_depth=3, hist | Yes (5-fold) | Yes (blind test) | Model + test_metrics.json + plots |
| LogisticRegression | `src/train_logistic.py` | multinomial/saga, max_iter=50 | Yes (5-fold) | Yes (blind test) | Model + test_metrics.json + plots |

### 4.2 Deep Learning (via `models/`)

| Script | Model Class | Architecture | Loss | Epochs | Test Eval |
|---|---|---|---|---|---|
| `train_dnn.py` | `DeepNeuralNetwork` | Linear(64) -> BN -> ReLU -> Drop(0.1) -> Linear(32) -> BN -> ReLU -> Linear(n) | Weighted CrossEntropy | 5 | Yes |
| `train_LSTM.py` | `BidirectionalLSTMNetwork` | BiLSTM(input, 32) -> Linear(64, 32) -> Drop(0.2) -> Linear(32, n) | CE (unweighted) | 5 | Yes |
| `train_Bi-LSTM.py` | `WeightedBidirectionalLSTM` | BiLSTM(input, 32) -> Linear(64, 32) -> ReLU -> Linear(32, n) | Weighted CE | 5 | Yes |
| `train_Bi-LSTM_shared-feature-extractor.py` | `MultiTaskHierarchicalDNN` | Shared: Linear(128)->BN->ReLU->Drop(0.2)->Linear(64)->BN->ReLU->Drop(0.2). Binary head: Linear(64,2). Multi head: Linear(64,32)->ReLU->Linear(32,n) | 0.4*CE_bin + 0.6*CE_multi | 8 | Yes |
| `train_dnn_mi_pca_kmeans.py` | `DeepNeuralNetwork` | Linear(128)->BN->ReLU->Drop(0.2)->Linear(64)->BN->ReLU->Drop(0.2)->Linear(32)->ReLU->Linear(n) | CE (unweighted) | 10 | Yes |

---

## 5. Preprocessing — Unified

### 5.1 `src/preprocessing.py` (used by ALL pipelines)

| Step | Description |
|---|---|
| CSV Loading | Reads UNSW-NB15_{1..4}.csv with `pd.read_csv(header=None)`, handles 47-col and 49-col variants |
| Target Cleaning | `fillna('Normal')` -> lowercase -> map via `CATEGORY_MAPPING` |
| Target Encoding | `LabelEncoder.fit_transform()` on `attack_cat` |
| Column Dropping | Drops: id, label, stime, ltime, srcip, dstip |
| Feature Encoding | `LabelEncoder.fit_transform()` per categorical column |
| Normalization | `np.log1p(X_raw.clip(lower=0))` -> `fillna(0)` -> float32 |

### 5.2 DL scripts

All 5 DL scripts import `load_data()` from `src.dl_pipeline`, which calls `src.preprocessing.load_and_preprocess()`. The preprocessing is **identical** between sklearn and DL pipelines.

---

## 6. Balancing — Per-Fold Only

### 6.1 `src/balancing.py` (used by sklearn and DL pipelines)

**Strategy 1 — KMeansSMOTE** (default, `--balancer kmeans`)
- Applied inside CV loop on each training fold only
- Uses `KMeansSMOTE(k_neighbors=2)` for per-fold, `KMeansSMOTE(k_neighbors=3)` for full retrain
- Proper cluster-based synthetic sample generation

**Strategy 2 — SMOTE** (`--balancer smote`)
- `SMOTE(k_neighbors=3)` on each training fold only
- No cluster-based filtering

### 6.2 No Global Oversampling

**No script applies oversampling before CV split.** All oversampling (KMeansSMOTE or SMOTE) happens inside the CV loop on training-fold data only, or on the full training set for the final retrain.

---

## 7. Train/Test/CV Split Logic

| Location | Split Type | Details |
|---|---|---|
| `src/dimensionality_reduction.py:65` | Stratified 80/20 holdout | `train_test_split(stratify=y, test_size=0.20)` — used by `main.py` |
| `src/dl_pipeline.py:load_data()` | Stratified 80/20 holdout | Same split used by DL scripts |
| `src/balancing.py` | Stratified 5-fold CV | `StratifiedKFold(n_splits=5)` on training set only |
| `src/cross_validation.py` | Stratified 5-fold CV | Per-fold MI→Scaler→PCA→balancing→train→eval |
| `src/dl_pipeline.py:preprocess_fold()` | Stratified 5-fold CV | Same per-fold pipeline for DL scripts |

---

## 8. Oversampling Locations — Per-Fold Only, No Leakage

| File | Line(s) | Type | Scope |
|---|---|---|---|
| `src/balancing.py` | `balance_training_fold()` | KMeansSMOTE/SMOTE | Per-fold training only |
| `src/balancing.py` | `balance_full_train()` | KMeansSMOTE/SMOTE | Full 80% training set (final retrain) |
| `src/dl_pipeline.py` | `preprocess_fold()` | KMeansSMOTE | Per-fold training only |
| `src/dl_pipeline.py` | `preprocess_final()` | KMeansSMOTE | Full 80% training set (final retrain) |

---

## 9. Evaluation Implementations

### 9.1 `src/evaluation.py` (used by `main.py`)

| Function | Description |
|---|---|
| `plot_confusion_matrix()` | Binary CM (Normal/Attack) + Multi-class CM (10 classes), saved as PNG |
| `plot_feature_importance()` | XGBoost `feature_importances_` bar chart |
| `save_results()` | Writes `results/model_comparison.csv` and `results/metrics.csv` |
| `print_final_summary()` | Formatted console table of blind test metrics |

### 9.2 `src/cross_validation.py`

- Per-fold: multi-class accuracy, macro F1, weighted F1, binary accuracy, binary F1
- Returns dict of metric lists + mean row

### 9.3 `src/dl_pipeline.py:evaluate_predictions()`

- Returns: binary_acc, binary_f1, multi_acc, macro_f1, weighted_f1, precision, recall, auc
- Saved to `test_metrics.json` and `*_cv_metrics.csv`

### 9.4 Model Persistence

| Model Type | Saved To | Format |
|---|---|---|
| HGB | `models/artifacts/HGB/hgb_model.joblib` | joblib |
| XGBoost | `models/artifacts/XGBoost/xgboost_model.joblib` | joblib |
| LogReg | `models/artifacts/LogReg/logreg_model.joblib` | joblib |
| DNN | `models/artifacts/DNN/dnn_model.pt` | torch.save |
| LSTM | `models/artifacts/LSTM/lstm_model.pt` | torch.save |
| BiLSTM | `models/artifacts/BiLSTM/bilstm_model.pt` | torch.save |
| BiLSTM_SharedFE | `models/artifacts/BiLSTM_SharedFE/bilstm_sharedfe_model.pt` | torch.save |
| DNN_MI_PCA_KMeans | `models/artifacts/DNN_MI_PCA_KMeans/dnn_mi_pca_kmeans_model.pt` | torch.save |
| Preprocessing artifacts | `artifacts/{model}/` or `models/artifacts/{model}/` | joblib |

---

## 10. Duplications — Minimized

| Area | Current State |
|---|---|
| **Preprocessing** | Single `src/preprocessing.py` used by both pipelines via `src.dl_pipeline.load_data()` |
| **MI Feature Selection** | Inline in `run_cv()` for sklearn (k=15), `src/dl_pipeline.preprocess_fold()` for DL (k=30) — intentional difference |
| **PCA** | `src/dimensionality_reduction.py` for sklearn (variance=0.95), `src/dl_pipeline` for DL (components=15) — intentional difference |
| **Balancing** | Single `src/balancing.py` used by both pipelines |
| **DNN Architecture** | Two `DeepNeuralNetwork` classes (train_dnn.py vs train_dnn_mi_pca_kmeans.py) — different architectures, kept separate |
| **XGBoost in Bi-LSTM script** | `train_Bi-LSTM.py` previously contained XGBoost Pipeline 2 — removed in refactoring |

---

## 11. Import and Entry Point Status

### 11.1 Import Analysis

| File | Imports from `src.*` | Status |
|---|---|---|
| `main.py` | All src modules | OK |
| `src/train_hgb.py` | `src.cross_validation`, `src.balancing`, `src.evaluation` | OK |
| `src/train_xgboost.py` | `src.cross_validation`, `src.balancing`, `src.evaluation` | OK |
| `src/train_logistic.py` | `src.cross_validation`, `src.balancing`, `src.evaluation` | OK |
| `models/train_dnn.py` | `src.dl_pipeline` | OK |
| `models/train_LSTM.py` | `src.dl_pipeline` | OK |
| `models/train_Bi-LSTM.py` | `src.dl_pipeline` | OK |
| `models/train_Bi-LSTM_shared-feature-extractor.py` | `src.dl_pipeline` | OK |
| `models/train_dnn_mi_pca_kmeans.py` | `src.dl_pipeline` | OK |

### 11.2 Data Path

All DL scripts accept `--data-dir` argument (default `data/raw/`). They use `src.dl_pipeline.load_data(data_dir)` which loads from `data/raw/UNSW-NB15_{1..4}.csv`.

### 11.3 Filename Notes

- `train_Bi-LSTM.py` contains a hyphen — cannot be imported as a module, must be run as a script. This is acceptable for a research experiment script.

### 11.4 Executable Entry Points

| Entry Point | Run Command |
|---|---|
| `main.py` | `python main.py` |
| `models/train_dnn.py` | `python models/train_dnn.py --data-dir data/raw` |
| `models/train_LSTM.py` | `python models/train_LSTM.py --data-dir data/raw` |
| `models/train_Bi-LSTM.py` | `python models/train_Bi-LSTM.py --data-dir data/raw` |
| `models/train_Bi-LSTM_shared-feature-extractor.py` | `python models/train_Bi-LSTM_shared-feature-extractor.py --data-dir data/raw` |
| `models/train_dnn_mi_pca_kmeans.py` | `python models/train_dnn_mi_pca_kmeans.py --data-dir data/raw` |
| `tests/` | `python -m pytest tests/` (77 tests) |

---

## 12. Leakage Verification

| Test | File | What It Verifies |
|---|---|---|
| `test_mi_selector_fit_on_train_only` | `test_leakage.py` | MI fitted on training data only |
| `test_pca_fit_on_train_only` | `test_leakage.py` | PCA fitted on training data only |
| `test_scaler_fit_on_train_only` | `test_leakage.py` | Scaler fitted on training data only |
| `test_scaler_means_match_train_not_full` | `test_leakage.py` | Scaler means = training means != full data means |
| `test_each_fold_mi_is_independent` | `test_leakage.py` | Each fold has its own MI selector |
| `test_each_fold_scaler_is_independent` | `test_leakage.py` | Each fold has its own Scaler |
| `test_smote_applied_to_train_only` | `test_leakage.py` | SMOTE only sees training fold data |
| `test_smote_does_not_affect_val_indices` | `test_leakage.py` | Validation indices unchanged by SMOTE |
| `test_preprocess_fold_balancing_only_on_train` | `test_leakage.py` | DL preprocessing: balancing on train only |
| `test_preprocess_final_balancing_only_on_train` | `test_leakage.py` | DL final preprocessing: balancing on train only |
| `test_final_model_evaluation_uses_test_set_only` | `test_leakage.py` | Final model evaluated on test set only |
| `test_balance_full_train_never_receives_test_data` | `test_leakage.py` | Full train balancing doesn't see test data |
| `test_no_val_data_in_balanced_train` | `test_leakage.py` | No validation leak into balanced training set |

All **77 tests pass** with no leakage.
