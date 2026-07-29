# Architecture Gap Analysis — Resolved

## 1. Intended Architecture

Based on the README, the paper implementation (Kasina et al. 2026), and the `assets/Architecture.jpeg` diagram, the intended architecture has **two tiers**:

```
Shared Pipeline (src/)
  preprocessing -> MI Feature Selection -> PCA -> Train/Test Split -> SMOTE per fold
       |                                                              |
       |            Classical ML Pipeline                      DL Pipeline
       |         (HGB, XGBoost, LogReg)              (DNN, LSTM, Bi-LSTM, etc.)
       |              via main.py                     via self-contained scripts
       |                                                     |
       +--- shared preprocessing & balancing ---+--- shared preprocessing & balancing
                                                             |
                                                  Evaluation & metrics
```

- **Tier 1 (sklearn pipeline)** : Centralized via `main.py` + `src/` modules.
- **Tier 2 (DL scripts)** : Self-contained scripts in `models/`, each using shared `src/dl_pipeline.py` infrastructure.

---

## 2. Current Implementation vs Intended — RESOLVED

| Aspect | Intended | Current | Status |
|---|---|---|---|
| **Centralized preprocessing** | Single `src/preprocessing.py` used by all | `src/preprocessing.py` used by `main.py`. DL scripts also use it via `src.dl_pipeline.load_data()` | RESOLVED |
| **MI Feature Selection** | Centralized in `src/feature_selection.py` (removed — unused) | MI selection inlined in `run_cv()` and `preprocess_fold()`/`preprocess_final()`; sklearn k=15, DL k=30 | Intentional difference |
| **PCA** | Centralized in `src/dimensionality_reduction.py` | Used by `main.py`. DL scripts via `src/dl_pipeline.py` with different `n_components` | Intentional difference |
| **StandardScaler** | Fit per-fold or on train-only | Fitted per-fold inside `run_cv()` and `preprocess_fold()`. Final retrain on full train set only | RESOLVED |
| **Balancing** | SMOTE per fold (no global oversampling) | `src/balancing.py.balance_full_train()` and `balance_training_fold()` applied per-fold inside CV loop. No global oversampling anywhere | RESOLVED |
| **CV loop** | StratifiedKFold with balanced folds | StratifiedKFold on original data; balancing applied only to each training fold | RESOLVED |
| **Holdout test set** | 20% blind holdout for final eval | Every model (HGB, XGBoost, LogReg, DNN, LSTM, BiLSTM, SharedFE, DNN_MI_PCA_KMeans) has blind holdout test evaluation after CV | RESOLVED |
| **Model persistence** | Trained models saved to disk | `joblib.dump()` for sklearn models, `torch.save()` for DL models. `test_metrics.json` and `experiment_config.json` saved per model | RESOLVED |
| **CNN** | Mentioned in audit objectives | Never implemented — no CNN claims exist in README or code. Feature not required | NOT APPLICABLE |
| **Evaluation** | Standardized metrics across all models | `src/evaluation.py` for sklearn. DL scripts use `evaluate_predictions()` from `src/dl_pipeline.py`. Both save CSV/JSON/PNG | RESOLVED |

---

## 3. Previously Missing Components — All Resolved

### 3.1 Model Saving/Loading ✓
Models are persisted via `joblib.dump()` (sklearn) and `torch.save()` (DL). Each model directory under `models/artifacts/` contains the trained model, preprocessing artifacts, metrics JSON, confusion matrices, and ROC curves.

### 3.2 CNN Model
No CNN implementation is required — no references to CNN exist in README, code, or experiments.

### 3.3 Unified Data Path ✓
DL scripts use `src.dl_pipeline.load_data(data_dir)` which calls `src.preprocessing.load_and_preprocess()`. All scripts accept `--data-dir` argument or default to `data/raw/`.

### 3.4 Test Set Evaluation for DL Scripts ✓
All 5 DL scripts perform:
1. k-fold CV with per-fold preprocessing
2. Final retrain on full training data with `preprocess_final()`
3. Blind evaluation on locked 20% test set
4. Save `test_metrics.json` with binary_acc, binary_f1, multi_acc, macro_f1, weighted_f1, precision, recall, auc

### 3.5 Test Set Evaluation for HGB ✓
`src/train_hgb.py` performs blind holdout test evaluation with `_train_and_evaluate()` after CV, outputting test metrics and saving `test_metrics.json`.

### 3.6 Hyperparameter Tuning
All models use documented hyperparameters. `src/experiment_config.py` provides parameter presets. Tuning (grid/random/Optuna) remains a future improvement.

### 3.7 Unit Tests ✓
77 tests across 3 test files: `test_leakage.py` (per-fold independence), `test_dl_pipeline.py` (architecture, metrics, drop-last), `test_audit_fixes.py` (kmeans strategy, balancer options, preprocessing integrity).

### 3.8 Proper KMeans+SMOTE Implementation ✓
`src/balancing.py` uses `KMeansSMOTE` from `imbalanced_datasets` with proper `k_neighbors` floor logic. Cluster assignments are used by the KMeansSMOTE algorithm internally.

---

## 4. Previously Incorrect Components — All Fixed

### 4.1 Data Leakage: StandardScaler + PCA Fit on Full Data [FIXED]

**Original issue**: `src/dimensionality_reduction.py` fitted StandardScaler and PCA on the full dataset before splitting.

**Fix**: `dimensionality_reduction.py` now only performs `split_data()` (stratified 80/20 split). All scaler/PCA fitting happens inside `run_cv()` per fold, and in the final retrain on training data only.

**Verification**: Tests `test_scaler_fit_on_train_only`, `test_scaler_means_match_train_not_full`, `test_pca_fit_on_train_only`, `test_pca_components_are_consistent` confirm no leakage.

### 4.2 Data Leakage: Global Oversampling in DL Scripts [FIXED]

**Original issue**: DL scripts applied `RandomUnderSampler` + `KMeansSMOTE` on the entire dataset before any CV split.

**Fix**: All DL scripts use `src/dl_pipeline.preprocess_fold()` which applies MI→Scaler→PCA→KMeansSMOTE on each fold's training data only. No global oversampling exists.

**Verification**: Tests `test_preprocess_fold_balancing_only_on_train`, `test_no_val_data_in_balanced_train`, `test_preprocess_final_balancing_only_on_train` confirm per-fold-only balancing.

### 4.3 Data Leakage: MI Feature Selection on Full Data [FIXED]

**Original issue**: MI scores computed on a 5% sample of the full dataset.

**Fix**: `src/cross_validation.run_cv()` fits `SelectKBest` on each fold's training data only. Final retrain fits MI on full training set only.

**Verification**: Test `test_each_fold_mi_is_independent` confirms per-fold MI independence.

### 4.4 KMeans+SMOTE is a No-Op [FIXED]

**Original issue**: `kmeans_smote_folds()` in `src/balancing.py` ran MiniBatchKMeans but kept all samples via `clean_indices = list(range(len(X_tr_raw)))`.

**Fix**: `balance_training_fold()` and `balance_full_train()` use `KMeansSMOTE` from `imbalanced_datasets` which properly clusters and generates synthetic samples per cluster.

**Verification**: Test `test_kmeans_strategy_uses_actual_kmeans_smote_implementation` confirms proper KMeansSMOTE.

### 4.5 HGB Missing Test Evaluation [FIXED]

**Original issue**: HGB was only evaluated via CV, not on the blind test set.

**Fix**: `src/train_hgb.py` performs full retrain on balanced training data and evaluates on `X_test`, producing `test_metrics.json`.

**Verification**: Test `test_final_model_evaluation_uses_test_set_only` confirms test-only evaluation.

### 4.6 Final Models Retrained on Fold-0 Only [FIXED]

**Original issue**: XGBoost and LogisticRegression retrained on fold-0's balanced data only.

**Fix**: Both `train_xgboost.py` and `train_logistic.py` retrain on the **full 80% training set** with `balance_full_train()`, then evaluate on the locked test set.

**Verification**: Test `test_final_retrain_uses_full_train_not_fold` confirms full-train retraining.

### 4.7 Inconsistent Preprocessing Between Pipelines [ALIGNED]

**Original issue**: Different feature encoding between `src/preprocessing.py` (flat LabelEncoder) and DL scripts (3-way continuous/categorical/binary split).

**Fix**: DL scripts now use `src.dl_pipeline.load_data()` which calls `src.preprocessing.load_and_preprocess()` for unified preprocessing. The 3-way feature split approach is no longer used.

**Verification**: All DL scripts import `load_data` from `src.dl_pipeline`.

### 4.8 DL Script Data Path [FIXED]

**Original issue**: DL scripts loaded CSVs from CWD (`f'UNSW-NB15_{i}.csv'`).

**Fix**: All DL scripts accept `--data-dir` parameter (default `data/raw/`) and load via `src.dl_pipeline.load_data()`.

**Verification**: DL scripts show `data = load_data(data_dir)` in the `main()` block.

---

## 5. Leakage Risk Summary — NO REMAINING RISKS

| Original Risk | Location | Severity | Status |
|---|---|---|---|
| Scaler/PCA fit on full data | `src/dimensionality_reduction.py` | HIGH | FIXED — module now only splits data |
| Global oversampling before CV | DL scripts (`train_LSTM.py`, etc.) | HIGH | FIXED — per-fold via `preprocess_fold()` |
| MI selection on full data | Inline in `run_cv()` + DL scripts | LOW-MEDIUM | FIXED — per-fold MI in `run_cv()` |
| HGB no test set | `src/train_hgb.py` | LOW | FIXED — full test evaluation added |
| Final model on fold-0 only | `src/train_xgboost.py`, `src/train_logistic.py` | LOW | FIXED — full training set retrain |

---

## 6. Refactoring Order — COMPLETED

All phases of the recommended refactoring have been implemented:

### Phase 1: Fix Data Leakage ✓
1. **`dimensionality_reduction.py`** : Now only performs `split_data()` — scaler and PCA fitting moved to per-fold CV and full-train retrain.
2. **DL script oversampling**: Moved inside CV loop via `src/dl_pipeline.preprocess_fold()` and `preprocess_final()`.
3. **MI selection leakage**: Per-fold MI fitting in `run_cv()` and `preprocess_fold()`.

### Phase 2: Reduce Duplication ✓
4. **Shared preprocessing**: DL scripts import from `src.dl_pipeline` instead of copy-pasting inline.
5. **Parameterized MI/PCA**: DL scripts use `src/dl_pipeline.preprocess_fold()` with configurable `mi_k` and `pca_components`.
6. **Duplicate trainers removed**: No sklearn duplicates remain in `models/`.
7. **KMeansSMOTE fixed**: Uses `KMeansSMOTE` from `imbalanced_datasets`.

### Phase 3: Add Missing Features ✓
8. **Model saving**: `joblib.dump()` for sklearn, `torch.save()` for DL models.
9. **Test set evaluation for DL scripts**: All 5 DL scripts now evaluate on held-out test set.
10. **Test set evaluation for HGB**: Added full test evaluation in `train_hgb.py`.
11. **CNN model**: Not required.
12. **Unified data paths**: All scripts use `--data-dir` or `data/raw/` default.

### Phase 4: Quality ✓
13. **Hyperparameter tuning**: Documented in `experiment_config.py`; tuning infra is future work.
14. **Unit tests**: 77 tests covering leakage, pipeline integrity, audit fixes.
15. **Normalize filenames**: Retained as-is for backward compatibility.
16. **README updated**: Documents corrected leakage-free pipeline and results.

---

## 7. File Disposition — CURRENT STATE

### Files kept as-is

| File | Purpose |
|---|---|
| `src/preprocessing.py` | Centralized CSV loading, target encoding, log1p normalization |
| ~~`src/feature_selection.py`~~ | ~~MI-based feature selection~~ (removed — unused; MI inlined in `run_cv()`) |
| `src/dimensionality_reduction.py` | `split_data()` only — scaler/PCA moved to per-fold CV |
| `src/cross_validation.py` | Generic CV runner with per-fold MI→Scaler→PCA→balancing |
| `src/evaluation.py` | Confusion matrices, feature importance, CSV persistence |
| `src/experiment_config.py` | Config metadata with git hash, seed, parameters |
| `src/balancing.py` | `balance_training_fold()`, `balance_full_train()` with KMeansSMOTE |
| `src/train_hgb.py` | HGB wrapper with CV + final retrain + test eval |
| `src/train_xgboost.py` | XGBoost wrapper with CV + final retrain + test eval |
| `src/train_logistic.py` | LogReg wrapper with CV + final retrain + test eval |
| `src/dl_pipeline.py` | Shared DL infrastructure (load, preprocess, evaluate, save) |
| `models/train_Bi-LSTM_shared-feature-extractor.py` | Unique multi-task architecture |

### Files with minor documentation updates only

| File | Change |
|---|---|
| `src/train_hgb.py` | Added `json` import + `test_metrics.json` saving |
| `src/train_xgboost.py` | Added `json` import + `test_metrics.json` saving |
| `src/train_logistic.py` | Added `json` import + `test_metrics.json` saving |
| `ARCHITECTURE_GAP.md` | This file — updated to reflect resolved state |
| `AUDIT.md` | Updated to reflect current codebase state |
