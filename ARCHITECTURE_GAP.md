# Architecture Gap Analysis

## 1. Intended Architecture

Based on the README, the paper implementation (Kasina et al. 2026), and the `assets/Architecture.jpeg` diagram, the intended architecture is:

```
Shared Pipeline (src/)
  preprocessing -> MI Feature Selection -> PCA -> Train/Test Split -> SMOTE per fold
       |                                                              |
       |              Classical ML Pipeline                    DL Pipeline
       |         (HGB, XGBoost, LogReg)              (DNN, LSTM, Bi-LSTM, etc.)
       |              via main.py                     via self-contained scripts
       |                                                     |
       +--- shared preprocessing & balancing ---+--- duplicated preprocessing & balancing
                                                                    |
                                                         Evaluation & metrics
```

The architecture has **two tiers**:

- **Tier 1 (sklearn pipeline)**: Centralized via `main.py` + `src/` modules. Shared preprocessing, feature selection, PCA, balancing, and evaluation. Classical ML models trained via the shared CV runner.

- **Tier 2 (DL scripts)**: Self-contained scripts in `models/`. Each script handles its own end-to-end pipeline (load -> preprocess -> MI -> PCA -> oversample -> train -> evaluate). These are research-grade experiment scripts, not production modules.

---

## 2. Current Implementation vs Intended

| Aspect | Intended | Current | Gap |
|---|---|---|---|
| **Centralized preprocessing** | Single `src/preprocessing.py` used by all | `src/preprocessing.py` used by `main.py` only. 5 DL scripts duplicate preprocessing with **different feature encoding** | Partial |
| **MI Feature Selection** | Centralized in `src/feature_selection.py` | Used by `main.py` (k=15). DL scripts each implement their own (k=30) | Partial |
| **PCA** | Centralized in `src/dimensionality_reduction.py` | Used by `main.py` (n=10). DL scripts each implement their own (n=15) | Partial |
| **StandardScaler** | Should be fit per-fold for DL, or on train-only for sklearn | `main.py`: fit on **full data before split** (LEAK). DL scripts: fit per-fold (correct) | Incorrect in `main.py` |
| **Balancing** | SMOTE per fold (no global oversampling) | `src/balancing.py`: per-fold (correct). DL scripts: **global oversampling before CV** (WRONG) | Broken in DL scripts |
| **CV loop** | StratifiedKFold with balanced folds | `src/cross_validation.py` correct. DL scripts: StratifiedKFold on **already-oversampled** data | Broken in DL scripts |
| **Holdout test set** | 20% blind holdout for final eval | `main.py`: exists for XGBoost/LR but not HGB. DL scripts: **no holdout** | Partial |
| **Model persistence** | Trained models saved to disk | **No model saving anywhere** | Missing |
| **CNN** | Mentioned in audit objectives | **Not implemented** | Missing |
| **Evaluation** | Standardized metrics across all models | `src/evaluation.py` for sklearn. DL scripts: console-only, no CSV/PNG output | Partial |

---

## 3. Missing Components

### 3.1 Model Saving/Loading (CRITICAL)
No trained model is persisted. There is no `torch.save()`, `joblib.dump()`, or `pickle` anywhere. All training results are lost on script exit. For any production use or reproducibility, this must be added.

### 3.2 CNN Model
No CNN (1D or 2D) implementation exists anywhere in the codebase. If the architecture spec requires one, it needs to be built from scratch.

### 3.3 Unified Data Path
DL scripts hardcode `f'UNSW-NB15_{i}.csv'` (CWD-relative), while `src/preprocessing.py` uses `data_dir` parameter. The DL scripts cannot be run from project root without copying/symlinking data files.

### 3.4 Test Set Evaluation for DL Scripts
None of the DL scripts hold out a blind test set. All evaluation is CV-only. This makes their reported metrics incomparable with the sklearn pipeline's holdout results.

### 3.5 Test Set Evaluation for HGB
`train_hgb.py` runs CV only — it has no blind holdout test evaluation. The `main.py` orchestrator does not evaluate HGB on `X_test`. HGB results appear only in `cv_results` dict, not in `all_test_results`.

### 3.6 Hyperparameter Tuning
All models use hardcoded hyperparameters. No grid search, random search, or Optuna integration.

### 3.7 Unit Tests
No test suite exists.

### 3.8 Proper KMeans+SMOTE Implementation
`kmeans_smote_folds()` in `src/balancing.py` runs MiniBatchKMeans but never uses the cluster assignments — `clean_indices = list(range(len(X_tr_raw)))` on line 146 means all samples are kept regardless. The KMeans step is a no-op.

---

## 4. Incorrect Components

### 4.1 Data Leakage: StandardScaler + PCA Fit on Full Data [HIGH RISK]

**File**: `src/dimensionality_reduction.py:53-57`
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mi)   # fit on ENTIRE dataset
pca = PCA(n_components=n_components, random_state=random_state)
X_pca = pca.fit_transform(X_scaled)     # fit on ENTIRE dataset
```

StandardScaler and PCA are fit on the **full dataset** (including the 20% that becomes the test set). The test set's statistical properties leak into the training pipeline. For a proper pipeline:
1. Split first
2. Fit scaler/PCA on train only
3. Transform test with train-fitted scaler/PCA

**Severity**: HIGH — inflates all test set metrics. The docstring acknowledges this ("matches the original notebook approach") but it's still incorrect.

### 4.2 Data Leakage: Global Oversampling in DL Scripts [HIGH RISK]

**Files**: `models/train_LSTM.py:131-139`, `models/train_Bi-LSTM.py:140-141,154-155`, `models/train_dnn_mi_pca_kmeans.py:133-139`

All DL scripts that use KMeansSMOTE apply `RandomUnderSampler` + `KMeansSMOTE` on the **entire dataset** before any CV split. This means:
- Synthetic samples generated from test-fold data influence training-fold samples
- SMOTE interpolation uses neighbors that may belong to the held-out fold
- CV metrics are inflated because the "validation" fold has seen synthetic neighbors of its samples in training

**Severity**: HIGH — DL script metrics are unreliable.

### 4.3 Data Leakage: MI Feature Selection on Full Data [MEDIUM RISK]

**File**: `src/feature_selection.py:46-57` (and all DL scripts)

MI scores are computed using a stratified sample of the **full dataset** (including test data). The selected features are then applied to the test set. This is a mild form of selection leakage.

**Severity**: MEDIUM — the effect is small with MI on a 5% sample, but technically the feature selection has seen test-set label distributions.

### 4.4 KMeans+SMOTE is a No-Op [BUG]

**File**: `src/balancing.py:146-148`
```python
clean_indices = list(range(len(X_tr_raw)))
X_tr_clean = X_tr_raw[clean_indices]
y_tr_clean = y_tr_raw[clean_indices]
```

The MiniBatchKMeans clustering is performed but its output (`cluster_labels`) is never used. All samples pass through unchanged. The `--balancer kmeans` option in `main.py` behaves identically to `--balancer smote` except for the `k_neighbors=2` vs `k_neighbors=3` difference.

### 4.5 HGB Missing Test Evaluation [INCOMPLETE]

`main.py:128-130` trains HGB via CV but never evaluates it on `X_test`. The HGB model is not included in `all_test_results` (line 152), so it does not appear in the final comparison table or CSV output.

### 4.6 Final Models Retrained on Fold-0 Only [CONCERN]

Both `train_xgboost.py:88-92` and `train_logistic.py:77-80` retrain a "final" model on **only fold-0's balanced training data** for the blind test evaluation. This is a subset of the training data. A more robust approach would be to retrain on the **full training set** with a single SMOTE pass, or use the last fold's model from CV.

### 4.7 Inconsistent Preprocessing Between Pipelines [CONFLICT]

`src/preprocessing.py` applies `LabelEncoder` uniformly to all object columns. DL scripts separate features into continuous/categorical/binary groups with explicit handling. The resulting feature matrices have different shapes, column orders, and encoding semantics. Metrics from the two pipelines are not directly comparable.

### 4.8 DL Script Data Path [BROKEN]

All DL scripts load CSVs from CWD (`f'UNSW-NB15_{i}.csv'`), not from `data/raw/`. Running `python models/train_dnn.py` from project root will fail with FileNotFoundError unless CSVs happen to be at project root.

---

## 5. Leakage Risk Summary

| Risk | Location | Severity | Impact |
|---|---|---|---|
| Scaler/PCA fit on full data | `src/dimensionality_reduction.py:53-57` | HIGH | Inflated test metrics for all sklearn models |
| Global oversampling before CV | `models/train_LSTM.py`, `train_Bi-LSTM.py`, `train_dnn_mi_pca_kmeans.py` | HIGH | Inflated CV metrics for DL models |
| MI selection on full data | `src/feature_selection.py:46-57` + all DL scripts | LOW-MEDIUM | Mild feature selection leakage |
| HGB no test set | `main.py:128-130` | LOW | Missing evaluation (not leakage) |
| Final model on fold-0 only | `src/train_xgboost.py:88`, `src/train_logistic.py:77` | LOW | Suboptimal final model (not leakage) |

---

## 6. Recommended Refactoring Order

The refactoring should address leakage risks first (most impactful), then reduce duplication, then add missing features.

### Phase 1: Fix Data Leakage (Critical)

1. **Fix `dimensionality_reduction.py`**: Split data first, then fit StandardScaler and PCA on training set only, transform test set with fitted objects.
2. **Fix DL script oversampling**: Move KMeansSMOTE inside the CV loop so oversampling happens only on training folds, not globally.
3. **Fix MI selection leakage**: Either accept the minor leakage (common in practice) or fit MI on training folds only.

### Phase 2: Reduce Duplication (High)

4. **Extract shared preprocessing**: Make DL scripts import from `src/preprocessing.py` (or a variant that supports the 3-way feature split) instead of copy-pasting inline.
5. **Parameterize MI/PCA**: Make DL scripts use `src/feature_selection.py` and `src/dimensionality_reduction.py` with configurable k and n_components.
6. **Remove duplicate trainers from `models/`**: Delete any sklearn training scripts that were previously moved to `src/`.
7. **Fix `kmeans_smote_folds()`**: Either implement actual cluster-based filtering or remove the dead code.

### Phase 3: Add Missing Features (Medium)

8. **Add model saving**: `torch.save()` for DL models, `joblib.dump()` for sklearn models.
9. **Add test set evaluation for DL scripts**: Implement a held-out test set split in each DL script.
10. **Add test set evaluation for HGB**: Add holdout eval to `train_hgb.py` and include it in `main.py`'s output.
11. **Add CNN model**: Implement 1D CNN if required by architecture spec.
12. **Unify data paths**: All scripts should accept a `--data-dir` argument or use a shared config.

### Phase 4: Quality (Low)

13. **Add hyperparameter tuning**: Grid/random search or Optuna integration.
14. **Add unit tests**: At least for preprocessing and evaluation functions.
15. **Normalize DL script filenames**: Replace hyphens with underscores.
16. **Update README**: Fix model file location references.

---

## 7. File Disposition Recommendations

### Files to DELETE (duplicates, superseded by `src/`)

| File | Reason |
|---|---|
| None currently | All `models/` scripts are DL-specific; no sklearn duplicates remain in `models/` |

### Files to MERGE (reduce duplication)

| Files | Action |
|---|---|
| `models/train_dnn.py` + `models/train_dnn_mi_pca_kmeans.py` | Both define `DeepNeuralNetwork` class with slightly different architectures. Merge into a single parameterized DNN script. |
| `models/train_LSTM.py` + `models/train_Bi-LSTM.py` (Pipeline 1) | `train_LSTM.py` IS a Bi-LSTM (bidirectional=True). `train_Bi-LSTM.py` Pipeline 1 is essentially the same with weighted loss. Merge into one configurable Bi-LSTM script. |
| `models/train_Bi-LSTM.py` (Pipeline 2 XGBoost) | This XGBoost implementation duplicates `src/train_xgboost.py`. Remove Pipeline 2 from `train_Bi-LSTM.py` — run XGBoost via `main.py` instead. |
| Preprocessing code across all 5 DL scripts | Extract to a shared module (e.g., `src/preprocessing_dl.py` or make `src/preprocessing.py` parameterizable) |

### Files to DEPRECATE

| File | Reason |
|---|---|
| `src/balancing.py` → `kmeans_smote_folds()` | The KMeans step is a no-op. Either fix it or remove it. The `--balancer kmeans` option currently does nothing meaningful. |
| `models/train_LSTM.py` | Name says "LSTM" but it IS a Bi-LSTM. Rename or merge with `train_Bi-LSTM.py`. |

### Files to KEEP as-is

| File | Reason |
|---|---|
| `src/preprocessing.py` | Correct implementation for the sklearn pipeline |
| `src/feature_selection.py` | Clean MI selection implementation |
| `src/cross_validation.py` | Generic, reusable, well-structured |
| `src/evaluation.py` | Well-structured evaluation layer |
| `src/train_hgb.py` | Clean wrapper, minor addition needed (test eval) |
| `src/train_xgboost.py` | Functional, minor fix needed (full-train retrain) |
| `src/train_logistic.py` | Functional, minor fix needed (full-train retrain) |
| `models/train_Bi-LSTM_shared-feature-extractor.py` | Unique architecture (multi-task DNN), no duplication |
