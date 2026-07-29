# Complete Repository Visualization — Intrusion Detection System

## Overview
Multi-model network intrusion detection on UNSW-NB15 dataset. 3 classical ML + 5 deep learning models. Leakage-free per-fold preprocessing pipeline. 77 regression tests. All artifacts persisted.

**Paper**: Kasina et al., *Information Sciences* 741 (2026)

---

## 1. Repository Tree

```
INTRUSION-DETECTION-SYSTEM/
├── main.py                           # Tier 1 orchestrator (223 lines)
├── requirements.txt                  # 7 dependencies
├── .gitignore                        # 4 entries
├── License                           # All rights reserved
├── README.md                         # 477 lines, full docs
├── ARCHITECTURE_GAP.md               # Architecture gap analysis (resolved)
├── AUDIT.md                          # Full codebase audit (post-refactor)
├── REFACTORING_REPORT.md             # Engineering report (this task)
├── COMPLETE_VISUALIZATION.md         # This file
│
├── src/                              # Shared pipeline modules
│   ├── __init__.py                   # "# src package"
│   ├── preprocessing.py              # 152 lines — load_and_preprocess()
│   ├── dimensionality_reduction.py   # 53 lines — split_data()
│   ├── balancing.py                  # 123 lines — balance_training_fold(), balance_full_train()
│   ├── cross_validation.py           # 212 lines — train_and_score_fold(), run_cv()
│   ├── evaluation.py                 # 303 lines — 6 functions
│   ├── experiment_config.py          # 156 lines — build/save config + param presets
│   ├── dl_pipeline.py                # 495 lines — shared DL infrastructure
│   ├── train_hgb.py                  # 185 lines — HGB wrapper
│   ├── train_xgboost.py              # 193 lines — XGBoost wrapper
│   └── train_logistic.py             # 185 lines — LogReg wrapper
│
├── models/                           # DL training scripts
│   ├── __init__.py                   # "# models package"
│   ├── train_dnn.py                  # 178 lines — DNN baseline
│   ├── train_LSTM.py                 # 186 lines — BiLSTM
│   ├── train_Bi-LSTM.py              # 200 lines — Weighted BiLSTM
│   ├── train_Bi-LSTM_shared-feature-extractor.py  # 213 lines — Multi-task DNN
│   └── train_dnn_mi_pca_kmeans.py    # 182 lines — DNN + MI+PCA+KMeansSMOTE
│
├── tests/                            # 77 tests total
│   ├── test_leakage.py               # 748 lines, 9 classes, ~35 tests
│   ├── test_dl_pipeline.py           # 480 lines, 9 classes, ~30 tests
│   └── test_audit_fixes.py           # 37 lines, 4 tests
│
├── notebooks/
│   └── Intrusion_Detection.ipynb
│
├── data/
│   ├── raw/                          # UNSW-NB15_{1..4}.csv (not committed)
│   └── processed/                    # (empty, reserved for cache)
│
├── assets/
│   └── Architecture.jpeg             # Pipeline diagram
│
├── artifacts/                        # sklearn preprocessing artifacts
│   ├── hgb/                          # mi_selector, scaler, pca, label_encoder .joblib
│   ├── xgboost/                      # (same)
│   └── logistic_regression/          # (same)
│
├── models/artifacts/                 # Per-model saved artifacts
│   ├── DNN/                          # dnn_model.pt, dnn_test_metrics.json, confusion matrix, cv_metrics.csv
│   ├── LSTM/                         # lstm_model.pt, lstm_test_metrics.json, mi_selector, scaler, pca
│   ├── BiLSTM/                       # bilstm_model.pt, bilstm_test_metrics.json, etc.
│   ├── BiLSTM_SharedFE/              # bilstm_sharedfe_model.pt, etc.
│   ├── DNN_MI_PCA_KMeans/            # dnn_mi_pca_kmeans_model.pt, etc.
│   ├── HGB/                          # hgb_model.joblib, test_metrics.json, confusion matrices, ROC
│   ├── XGBoost/                      # xgboost_model.joblib, test_metrics.json, feature importance, ROC
│   └── LogReg/                       # logreg_model.joblib, test_metrics.json, confusion matrices, ROC
│
└── results/
    ├── model_comparison.csv          # HGB: 0.9628, XGB: 0.9122, LogReg: 0.9545

    ├── metrics.csv                   # Per-fold CV metrics (15 rows)
    ├── hgb_per_class_report.csv
    ├── xgboost_per_class_report.csv
    ├── logreg_per_class_report.csv
    └── corrected_pipeline/
        └── experiment_config.json    # Run metadata + hyperparameters
```

---

## 2. Root-Level Files

### 2.1 `requirements.txt`
```
pandas>=1.5
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost>=1.7
torch>=2.0
matplotlib>=3.7
```

### 2.2 `.gitignore`
```
data/raw/*.csv
__pycache__/
.pytest_cache/
*.pyc
```

### 2.3 `License`
Copyright (c) 2026 T P SHIVHA SHAKTHIY — All rights reserved.

### 2.4 `main.py` — Tier 1 Orchestrator (223 lines)
**CLI** (via argparse):
- `--data-dir` (default: `data/raw`)
- `--balancer` (choices: `kmeans`, `smote`; default: `kmeans`)
- `--n-splits` (int, default: 5)
- `--mi-k` (int, default: 15)
- `--pca-variance` (float, default: 0.95)
- `--skip-plots` (store_true)

**Pipeline** (`main()`):
1. `load_and_preprocess(data_dir)` → `X_processed, y_multi, le`
2. `split_data(X_processed, y_multi)` → `X_train, X_test, y_train, y_test`
3. Validation: 0 < pca_variance ≤ 1, 1 ≤ mi_k ≤ X_train.shape[1], 2 ≤ n_splits ≤ smallest class count
4. `normal_class_idx = np.where(le.classes_ == 'Normal')[0][0]`
5. `train_hgb(X_train, y_train, X_test, y_test, class_names, n_splits, mi_k, pca_variance, balancer, make_plots, normal_class_idx)` → results dict
6. `train_xgboost(...)` → results dict
7. `train_logistic(...)` → results dict
8. Build `all_test_results` (list of 1-row DataFrames), `cv_results` (dict of DataFrames), `y_pred_dict`
9. `print_final_summary(all_test_results)`
10. `save_results(all_test_results, cv_results, y_true=y_test, y_pred_dict, class_names)`
11. `save_preprocessing_artifacts(selector, scaler, pca, le)` for each model to `artifacts/{model}/`
12. `build_model_config(model_name, ...)` + `save_experiment_config()` to `results/corrected_pipeline/{model}/`

### 2.5 `results/model_comparison.csv`
```
Model,accuracy,precision,recall,f1,auc
HGB,0.9628,0.9831,0.9628,0.9703,0.9975
XGBoost,0.9122,0.9498,0.9122,0.9275,0.9834
LogReg,0.9545,0.9763,0.9545,0.9640,0.9922
```

### 2.6 `results/metrics.csv` (per-fold CV, 5 folds per model × 3 models = 15 rows)
HGB: accuracy 0.9644-0.9652, AUC 0.9976 (all folds)
XGBoost: accuracy 0.9074-0.9120, AUC 0.9832-0.9836
LogReg: accuracy 0.9545-0.9550, AUC 0.9920-0.9923

---

## 3. `src/` Package — Full Module Reference

### 3.1 `src/preprocessing.py` (152 lines)
**Constants:**
- `COL_NAMES`: list of 49 column names (srcip, sport, dstip, dsport, proto, state, dur, sbytes, dbytes, sttl, dttl, sloss, dloss, service, sload, dload, spkts, dpkts, swin, dwin, stcpb, dtcpb, smeansz, dmeansz, trans_depth, res_bdy_len, sjit, djit, sintpkt, dintpkt, tcprtt, synack, ackdat, is_sm_ips_ports, ct_src_ltm, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm_d, ct_srv_dst, ct_state_ttl, ct_src_user_ltm, ct_src_zone_ltm, ct_dst_host_ltm, ct_srv_src, attack_cat, label)
- `CATEGORY_MAPPING`: dict mapping 10 lowercase class names to title case
- `DROP_COLS`: `['id', 'label', 'stime', 'ltime', 'srcip', 'dstip']`
- `TARGET_COL`: `'attack_cat'`

**Function:**
```python
def load_and_preprocess(data_dir: str = "data/raw") -> tuple:
    """Load UNSW-NB15 CSV files, clean, preprocess.
    
    Returns:
        X_processed: float32 (N, F)
        y_multi: int (N,)
        le: LabelEncoder fitted on attack_cat
    """
```
Implementation:
1. Load `data_dir/UNSW-NB15_{1..4}.csv` with `pd.read_csv(header=None, low_memory=False)`
2. Handle 47/49 column variants
3. `pd.concat()` all files
4. Clean target: `fillna('Normal')` → `astype(str)` → `str.strip()` → `str.lower()` → `.map(CATEGORY_MAPPING)` → `fillna('Normal')`
5. `LabelEncoder().fit_transform(df['attack_cat'])`
6. Drop DROP_COLS + TARGET_COL
7. LabelEncode all `object` type columns with per-column `LabelEncoder().fit_transform(col.astype(str))`
8. `np.log1p(X_raw.clip(lower=0))` → `fillna(0)` → `astype('float32')` → `.values`
9. Return `(X_processed, y_multi, le)`

### 3.2 `src/dimensionality_reduction.py` (53 lines)
**Function:**
```python
def split_data(
    X: np.ndarray, y: np.ndarray,
    test_size: float = 0.20, random_state: int = 42,
) -> tuple:
    """Stratified 80/20 train/test split. No transformers fitted here.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
```
Implementation:
- `train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)`
- Returns the 4-tuple

### 3.3 `src/balancing.py` (123 lines)
**Functions:**
```python
def balance_training_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    strategy: str = "kmeans", k_neighbors: int = 3,
    n_clusters: int = 20, random_state: int = 42,
) -> tuple:
    """Balance a single training fold. Must receive ONLY training data.
    
    Returns:
        X_balanced, y_balanced
    """
```
Implementation:
1. Check `minority_count = min(Counter(y_train).values())` ≥ 2, else raise ValueError
2. `actual_k = min(k_neighbors, minority_count - 1)`
3. If `strategy == "kmeans"`: `KMeansSMOTE(k_neighbors=max(actual_k,1), kmeans_estimator=MiniBatchKMeans(n_clusters=min(n_clusters, len(X_train)), batch_size=2048, random_state=random_state, n_init='auto'), cluster_balance_threshold=0.0, random_state=random_state, n_jobs=1)`
4. If `strategy == "smote"`: `SMOTE(random_state=random_state, k_neighbors=max(actual_k,1))`
5. `sm.fit_resample(X_train, y_train)` → return

```python
def balance_full_train(
    X_train: np.ndarray, y_train: np.ndarray,
    strategy: str = "kmeans", k_neighbors: int = 3,
    n_clusters: int = 20, random_state: int = 42,
) -> tuple:
    """Balance full training set for final retrain. Must never receive val/test data.
    
    Returns:
        X_balanced, y_balanced
    """
```
Implementation:
- Delegates to `balance_training_fold()`, prints balanced count message

### 3.4 `src/cross_validation.py` (212 lines)
**Functions:**
```python
def train_and_score_fold(
    model, X_tr, y_tr, X_val, y_val,
    use_sample_weight: bool = False, normal_class_idx=None,
) -> dict:
    """Fit model on (X_tr, y_tr), score on (X_val, y_val).
    
    Returns dict: accuracy, precision, recall, f1, auc, plus binary_accuracy, binary_f1 if normal_class_idx specified.
    """
```
Implementation:
1. Optionally compute inverse-frequency `sample_weight` from y_tr
2. `model.fit(X_tr, y_tr)` with optional sample_weight, `warnings.simplefilter("ignore")`
3. Compute: `accuracy_score`, `precision_score(average='weighted')`, `recall_score(average='weighted')`, `f1_score(average='weighted')`, `roc_auc_score(multi_class='ovr', average='weighted')` from `predict_proba`
4. If `normal_class_idx`: binary labels via `(y != normal_class_idx).astype(int)`, compute `binary_accuracy`, `binary_f1`

```python
def run_cv(
    X_train: np.ndarray, y_train: np.ndarray,
    model_class, model_params: dict,
    n_splits: int = 5, mi_k: int = 15, pca_variance: float = 0.95,
    k_neighbors: int = 3, random_state: int = 42,
    use_sample_weight: bool = False, strategy: str = "kmeans",
    n_clusters: int = 20, normal_class_idx=None,
) -> tuple:
    """Run Stratified K-Fold CV with per-fold leakage-free preprocessing.
    
    For each fold:
      1. MI SelectKBest fit on fold train → transform fold train + val
      2. StandardScaler fit on fold train → transform fold train + val
      3. PCA fit on scaled fold train → transform fold train + val
      4. K-means SMOTE on fold train only
      5. Train model → evaluate on val
    
    Returns:
        metrics: dict {metric_name: [fold_values]}
        selector: last fold's SelectKBest
        scaler: last fold's StandardScaler
        pca: last fold's PCA
    """
```
Implementation:
1. `StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)`
2. Init metrics dict with keys: accuracy, precision, recall, f1, auc (+ binary_accuracy, binary_f1 if normal_class_idx)
3. For each fold:
   a. `X_tr, X_val = X_train[trn_idx], X_train[val_idx]`
   b. `y_tr, y_val = y_train[trn_idx], y_train[val_idx]`
   c. MI: `SelectKBest(mutual_info_classif, k=mi_k).fit(X_tr, y_tr)` → transform both
   d. Scaler: `StandardScaler().fit_transform(X_tr_mi)` → transform X_val
   e. PCA: `PCA(n_components=pca_variance).fit_transform(X_tr_s)` → transform X_val_s
   f. Balancing: `balance_training_fold(X_tr_p, y_tr, strategy=strategy, k_neighbors=k_neighbors, n_clusters=n_clusters, random_state=random_state)`
   g. Train: `model_class(**model_params).fit(X_tr_b, y_tr_b)`
   h. Eval: `train_and_score_fold(model, X_tr_b, y_tr_b, X_val_p, y_val, normal_class_idx=normal_class_idx)`
   i. Append metrics, store last fold's transformers
4. Return `(metrics, selector, scaler, pca)`

### 3.5 `src/evaluation.py` (303 lines)
**Functions:**
```python
def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: list, normal_class_idx: int = 0,
    save_dir: str = "assets", prefix: str = "",
) -> None:
```
- Binary CM (Normal vs Attack, 5×4" figure, Blues colormap)
- Multi-class CM (10 classes, 10×8" figure, Blues colormap, rotated x-labels)
- Both saved to `save_dir/{prefix}binary_cm.png` and `{prefix}multiclass_cm.png`

```python
def plot_roc_curve(
    model, X_test: np.ndarray, y_test: np.ndarray,
    class_names: list, scaler=None, pca=None,
    title: str = "ROC Curve", save_dir: str = "assets",
    save_path: str = None, prefix: str = "",
) -> None:
```
- Optionally transforms X_test through scaler + pca
- `label_binarize` y_test, `predict_proba` → per-class ROC curves
- Weighted AUC annotation, 8×6" figure, tab10 colors
- Saved to `save_dir/{prefix}roc_curve.png`

```python
def plot_feature_importance(model, n_components: int = 10, save_dir: str = "assets") -> None:
```
- Bar chart of `model.feature_importances_` (if available) labeled PC1-PCn
- 9×4" figure, saved to `save_dir/feature_importance.png`

```python
def save_preprocessing_artifacts(
    selector=None, scaler=None, pca=None, le=None, save_dir: str = "artifacts",
) -> None:
```
- `joblib.dump` each non-None artifact to `save_dir/{name}.joblib`

```python
def save_results(
    all_test_results: list, cv_results: dict,
    y_true: np.ndarray = None, y_pred_dict: dict = None,
    class_names: list = None, results_dir: str = "results",
) -> None:
```
- `pd.concat(all_test_results)` → `results/model_comparison.csv`
- Per-model CV rows → `results/metrics.csv`
- `classification_report(y_true, y_pred, target_names=class_names, output_dict=True)` → per-model per-class CSV

```python
def print_final_summary(all_test_results: list) -> None:
```
- `pd.concat(all_test_results)` → formatted console table with 4-decimal float formatting

### 3.6 `src/experiment_config.py` (156 lines)
**Functions + Constants:**
```python
def get_git_commit() -> str:
    """subprocess.run(['git', 'rev-parse', 'HEAD']) or 'unavailable'"""
```

```python
def build_experiment_config(
    model_name: str, model_params: dict = None,
    mi_k: int = 15, pca_variance: float = 0.95,
    n_splits: int = 5, balancer: str = "kmeans",
    k_neighbors: int = 3, random_state: int = 42,
    test_size: float = 0.20,
) -> dict:
```
Returns dict with keys:
- model, seed, train_test_split (e.g. "80/20"), test_size, cv_folds, balancer, balancer_k_neighbors
- feature_selection: "mutual_information", feature_selection_scope: "per_fold_training_data", feature_selection_k
- pca_variance, scaler: "StandardScaler", scaler_scope: "per_fold_training_data"
- pca_scope: "per_fold_training_data", test_set_locked: True, final_retrain: "full_80_percent_training_set"
- timestamp (ISO), git_commit, model_hyperparameters (if provided)

```python
def save_experiment_config(config: dict, save_dir: str) -> str:
    """json.dump(config, open(save_dir/experiment_config.json, 'w'), indent=2)"""
```

**Hyperparameter presets:**
- `XGBOOST_PARAMS`: n_estimators=30, subsample=0.1, max_depth=3, min_child_weight=20, gamma=0.2, learning_rate=0.05, colsample_bytree=0.1, reg_alpha=0.5, eval_metric='mlogloss', tree_method='hist', random_state=42, verbosity=0
- `HGB_PARAMS`: max_iter=30, learning_rate=0.05, max_depth=5, l2_regularization=1.0, random_state=42
- `LOGREG_PARAMS`: multi_class='multinomial', solver='saga', max_iter=50, random_state=42, n_jobs=-1

```python
def build_model_config(model_name: str, **kwargs) -> dict:
    """Look up model in param_map, call build_experiment_config with those params."""
```

### 3.7 `src/dl_pipeline.py` — Shared DL Infrastructure (495 lines)
**Seed/Device Functions:**
```python
def set_seeds(seed: int = 42):
    """torch.manual_seed, np.random.seed, cuda manual_seed_all, deterministic=True, benchmark=False"""

def get_device(requested: str = "auto") -> torch.device:
    """'auto': try CUDA with health check probe, fall back to CPU.
       'cpu': force CPU. 'cuda': raise if unavailable.
    """
```

**Data Loading:**
```python
def load_data(data_dir: str = "data/raw") -> dict:
    """Load via src.preprocessing.load_and_preprocess() + src.dimensionality_reduction.split_data().
    
    Adds project root to sys.path for src imports.
    
    Returns dict:
        X_train, X_test, y_train, y_test,
        class_names, num_classes, normal_class_idx, le
    """
```

**Per-Fold Preprocessing:**
```python
def preprocess_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    mi_k: int = 30, pca_components: int = 15,
    n_clusters: int = 20, k_neighbors: int = 2,
    rus_cap: int = 15000, random_state: int = 42,
    use_mi: bool = True, use_pca: bool = True, use_balancing: bool = True,
) -> dict:
    """Per-fold preprocessing: MI → StandardScaler → PCA → RUS + KMeansSMOTE.
    All transformers fit on (X_tr, y_tr) only.
    
    Returns dict:
        X_tr, y_tr (balanced training)
        X_val, y_val (transformed val, untouched)
        selector, scaler, pca (fitted transformers)
    """
```
Implementation:
1. If `use_mi` and `mi_k > 0`: `SelectKBest(mutual_info_classif, k=min(mi_k, X_tr.shape[1])).fit(X_tr, y_tr)` → transform both
2. `StandardScaler().fit_transform(X_tr)` → transform X_val
3. If `use_pca` and `pca_components > 0`: `PCA(n_components=min(pca_components, X_tr.shape[1], X_tr.shape[0])).fit_transform(X_tr)` → transform X_val
4. If `use_balancing`:
   a. `RandomUnderSampler(sampling_strategy={c: min(cnt, rus_cap)})` → cap each class at rus_cap
   b. `KMeansSMOTE(cluster_balance_threshold=0.0, k_neighbors=max(actual_k,1), kmeans_estimator=MiniBatchKMeans(n_init='auto'))` → oversample to equalize

**Final Preprocessing:**
```python
def preprocess_final(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    mi_k: int = 30, pca_components: int = 15,
    n_clusters: int = 20, k_neighbors: int = 2,
    rus_cap: int = 15000, random_state: int = 42,
    use_mi: bool = True, use_pca: bool = True, use_balancing: bool = True,
) -> dict:
    """Full-train preprocessing for final model retraining.
    Test data is NEVER fitted — only transformed.
    
    Returns dict:
        X_train, y_train (balanced)
        X_test, y_test (transformed)
        selector, scaler, pca (fitted transformers)
    """
```
Implementation: Same steps as `preprocess_fold()` but on full X_train/X_test with no val set.

**Class Weights:**
```python
def compute_class_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights: total / (num_classes * class_counts)"""
```

**Evaluation:**
```python
def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray,
    normal_class_idx: int = 0,
) -> dict:
    """Return dict: binary_acc, binary_f1, multi_acc, macro_f1, weighted_f1,
       precision, recall, auc (always 0.0 — no proba)."""
```

```python
def evaluate_with_proba(
    y_true: np.ndarray, y_pred: np.ndarray,
    y_proba: np.ndarray, normal_class_idx: int = 0,
) -> dict:
    """Like evaluate_predictions but computes real AUC from probabilities.
    For binary: roc_auc_score on attack-vs-normal.
    For multi-class: label_binarize + roc_auc_score(ovr, weighted)."""
```

```python
def get_probabilities(
    model, X_tensor, device, batch_size: int = 4096,
) -> np.ndarray:
    """Memory-bounded batch inference. Handles multi-task models (tuple output).
    Returns (N, num_classes) softmax probabilities."""
```

**Artifact Persistence:**
```python
def save_dl_artifacts(
    model: nn.Module, model_name: str,
    cv_metrics: list, test_metrics: dict = None,
    save_dir: str = None, class_names: list = None,
    normal_class_idx: int = 0,
    y_test: np.ndarray = None, y_test_pred: np.ndarray = None,
    selector=None, scaler=None, pca=None,
    label_encoder=None, model_config: dict = None,
):
    """Save to models/artifacts/{model_name}/:
    - {name}_model.pt (state_dict)
    - {name}_metadata.json (model_name, class_names, normal_class_idx, model_config)
    - Preprocessing .joblib files
    - {name}_cv_metrics.csv
    - {name}_test_metrics.json
    - {name}_confusion_matrix.png
    """
```

```python
def load_dl_artifacts(
    model: nn.Module, model_name: str, save_dir: str, device=None,
) -> dict:
    """Load model state_dict, preprocessing joblibs, metadata JSON.
    Returns: model, selector, scaler, pca, label_encoder, metadata.
    """
```

### 3.8-10 Sklearn Trainers (identical structure, ~185 lines each)

All three trainers (`train_hgb.py`, `train_xgboost.py`, `train_logistic.py`) share the same structure:

**MODEL_NAME** = "HGB" / "XGBoost" / "LogReg"

**Private function:**
```python
def _train_and_evaluate(X_tr, y_tr, X_val, y_val, random_state=42):
    """Train model on balanced data, return (model, metrics_dict, y_pred)."""
```
- Instantiate model with hardcoded hyperparams
- `model.fit(X_tr, y_tr)` with `warnings.simplefilter("ignore")`
- Predict on X_val
- Compute: accuracy, precision(weighted), recall(weighted), f1(weighted), auc (from predict_proba)
- Return `(model, metrics, y_pred)`

**Public function:**
```python
def train_and_evaluate(
    X_train, y_train, X_test, y_test,
    class_names: list,
    n_splits: int = 5, mi_k: int = 15, pca_variance: float = 0.95,
    k_neighbors: int = 3, random_state: int = 42,
    balancer: str = "kmeans", make_plots: bool = True,
    normal_class_idx: int = 0,
) -> dict:
    """Full pipeline: CV → final retrain → test eval.
    
    Returns: {model, cv_metrics, test_metrics, y_test_pred, selector, scaler, pca}
    """
```

Pipeline:
1. Print banner with MODEL_NAME
2. **Step 1 — Per-fold CV**: `run_cv(X_train, y_train, model_class, model_params, n_splits, mi_k, pca_variance, k_neighbors, random_state, strategy=balancer, normal_class_idx=normal_class_idx)` → `cv_metrics, selector, scaler, pca`
3. Print CV results (mean ± std)
4. **Step 2 — Final retrain**: 
   a. `SelectKBest(mutual_info_classif, k=mi_k).fit(X_train, y_train)` → transform both
   b. `StandardScaler().fit_transform(X_train_mi)` → transform X_test
   c. `PCA(n_components=pca_variance).fit_transform(X_train_s)` → transform X_test
   d. `balance_full_train(X_train_p, y_train, strategy, k_neighbors, random_state)`
   e. `_train_and_evaluate(X_train_b, y_train_b, X_test_p, y_test)` → `model, test_metrics, y_test_pred`
5. Add binary_accuracy and binary_f1 to test_metrics
6. Print test metrics
7. **Step 3 — Save artifacts**:
   a. `os.path.join("models", "artifacts", MODEL_NAME)`
   b. `joblib.dump(model, save_dir / {model_name}_model.joblib)`
   c. **NEW**: `json.dump(test_metrics, open(save_dir/test_metrics.json, 'w'))`
   d. If make_plots: `plot_confusion_matrix(..., prefix)` + `plot_roc_curve(..., scaler, pca)`
   e. XGBoost additionally: `plot_feature_importance(model, n_components)`
8. Return results dict

**HGB-specific hyperparams**: max_iter=30, learning_rate=0.05, max_depth=5, l2_regularization=1.0, random_state=42
**XGBoost-specific hyperparams**: n_estimators=30, subsample=0.1, max_depth=3, min_child_weight=20, gamma=0.2, learning_rate=0.05, colsample_bytree=0.1, reg_alpha=0.5, use_label_encoder=False, eval_metric='mlogloss', tree_method='hist', random_state=42, verbosity=0
**LogReg-specific hyperparams**: multi_class='multinomial', solver='saga', max_iter=50, random_state=42, n_jobs=-1

---

## 4. `models/` Package — DL Training Scripts

### 4.1 `train_dnn.py` (178 lines) — Baseline DNN
**Imports from `src.dl_pipeline`**: `set_seeds`, `get_device`, `load_data`, `compute_class_weights`, `evaluate_with_proba`, `get_probabilities`, `save_dl_artifacts`

**Model class:**
```python
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        # Linear(input_dim, 64) → LayerNorm(64) → ReLU → Dropout(0.1)
        # → Linear(64, 32) → LayerNorm(32) → ReLU → Linear(32, output_dim)
    
    def forward(self, x): return self.network(x)
```

**Pipeline** (`main(data_dir="data/raw")`):
1. `load_data(data_dir)` → data dict
2. 5-fold StratifiedKFold CV:
   - Per fold: `StandardScaler.fit_transform(X_tr)` + transform X_val (NO MI/PCA/balancing)
   - `compute_class_weights(y_tr)` → weighted `CrossEntropyLoss`
   - `AdamW(lr=0.01, weight_decay=1e-4)`
   - Train 5 epochs, batch_size=1024, `drop_last=True`
   - Evaluate via `get_probabilities` → `evaluate_with_proba`
3. Final retrain on full training set (same architecture, 5 epochs)
4. Evaluate on test set
5. `save_dl_artifacts(model, "DNN", cv_metrics, test_metrics, class_names, normal_class_idx, y_test, test_preds, scaler, label_encoder, model_config)`

**CLI**: `--data-dir` (default: `data/raw`)

### 4.2 `train_LSTM.py` (186 lines) — BiLSTM
**Imports from `src.dl_pipeline`**: `set_seeds`, `get_device`, `load_data`, `preprocess_fold`, `preprocess_final`, `evaluate_with_proba`, `get_probabilities`, `save_dl_artifacts`

**Model class:**
```python
class BiLSTMNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        # LSTM(input_dim, hidden_dim=32, batch_first=True, bidirectional=True)
        # → Linear(hidden_dim*2, 32) → ReLU → Dropout(0.2) → Linear(32, output_dim)
    
    def forward(self, x):
        # If 2D: unsqueeze(1); LSTM → take last timestep → classifier
```

**Pipeline** (`main(data_dir="data/raw", device_name="auto")`):
1. `load_data(data_dir)` → data dict
2. 5-fold CV: per fold → `preprocess_fold(X_tr, y_tr, X_val, y_val, mi_k=30, pca_components=15, n_clusters=20, k_neighbors=2, rus_cap=15000)` → balanced data
3. `CrossEntropyLoss()` (unweighted), `AdamW(lr=0.005, weight_decay=1e-4)`, 5 epochs, batch_size=512, `drop_last=True`
4. Final: `preprocess_final(X_train, y_train, X_test, y_test, mi_k=30, pca_components=15, n_clusters=20, k_neighbors=2, rus_cap=15000)` → retrain 5 epochs → test eval
5. `save_dl_artifacts(model, "LSTM", cv_metrics, test_metrics, selector=..., scaler=..., pca=..., label_encoder=..., model_config=...)`

**CLI**: `--data-dir` (default: `data/raw`), `--device` (choices: auto/cpu/cuda, default: auto)

### 4.3 `train_Bi-LSTM.py` (200 lines) — Weighted BiLSTM
**Model class:**
```python
class WeightedBiLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        # LSTM(input_dim, hidden_dim=32, batch_first=True, bidirectional=True)
        # → Linear(hidden_dim*2, 32) → ReLU → Linear(32, output_dim)  [NO Dropout]
```
Same forward as BiLSTMNetwork but without dropout in classifier.

**Pipeline**: Same structure as LSTM but with:
- `nn.CrossEntropyLoss(weight=inverse_freq_weights)` — class weights computed per fold
- `AdamW(lr=0.005)` (no weight_decay)
- 5 epochs, batch_size=512

**Note**: XGBoost Pipeline 2 that existed in the original script has been removed — XGBoost is now covered by `src/train_xgboost.py`.

### 4.4 `train_Bi-LSTM_shared-feature-extractor.py` (213 lines) — Multi-Task DNN
**Model class:**
```python
class MultiTaskHierarchicalDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        # Shared: Linear(input_dim, 128) → LayerNorm(128) → ReLU → Dropout(0.2)
        #         → Linear(128, 64) → LayerNorm(64) → ReLU → Dropout(0.2)
        # Binary head: Linear(64, 2)
        # Multi head: Linear(64, 32) → ReLU → Linear(32, num_classes)
    
    def forward(self, x):
        # features = shared(x)
        # return binary_head(features), multi_head(features)  [tuple]
```

**Pipeline** (note: no balancing needed — weighted heads handle imbalance):
1. `preprocess_fold` with `use_balancing=True` (the model gets balanced data)
2. Joint loss: `0.4 * CrossEntropyLoss(binary) + 0.6 * CrossEntropyLoss(multi)`
3. `AdamW(lr=0.005, weight_decay=1e-4)`, 8 epochs, batch_size=512
4. `get_probabilities` extracts multi-class head for evaluation

**CLI**: `--data-dir` (default: `data/raw`)

### 4.5 `train_dnn_mi_pca_kmeans.py` (182 lines) — Deep DNN with Preprocessing
**Model class:**
```python
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        # Linear(input_dim, 128) → LayerNorm(128) → ReLU → Dropout(0.2)
        # → Linear(128, 64) → LayerNorm(64) → ReLU → Dropout(0.2)
        # → Linear(64, 32) → LayerNorm(32) → ReLU → Linear(32, output_dim)
```

**Pipeline** (same structure as LSTM):
1. `preprocess_fold` / `preprocess_final` with mi_k=30, pca_components=15, k_neighbors=2, rus_cap=15000
2. `CrossEntropyLoss()` (unweighted), `AdamW(lr=0.005, weight_decay=1e-4)`, 10 epochs, batch_size=512
3. Assert `bx.shape[0] > 1` for LayerNorm safety

**CLI**: `--data-dir` (default: `data/raw`)

---

## 5. Tests

### 5.1 `tests/test_leakage.py` (748 lines, 9 classes)
**Fixtures:**
- `synthetic_dataset()`: rng.randn(1000, 20) float32, 5-class random (p=[0.5, 0.15, 0.1, 0.15, 0.1])
- `split_dataset(synthetic_dataset)`: stratified 80/20 split

**Test Classes:**

1. `TestNoPreprocessingLeakage`:
   - `test_split_data_returns_correct_sizes()`: split → 800/200 samples
   - `test_mi_selector_fit_on_train_only()`: MI transform → correct shapes

2. `TestPCALeakage`:
   - `test_pca_fit_on_train_only()`: fit on train → consistent n_components
   - `test_pca_components_are_consistent()`: n_components_ ≤ n_features

3. `TestScalerLeakage`:
   - `test_scaler_fit_on_train_only()`: mean=0, std=1 on train
   - `test_scaler_means_match_train_not_full()`: scaler.mean_ == train means

4. `TestCVPerFoldIndependence`:
   - `test_run_cv_returns_fold_metrics()`: 3 folds, all metrics have 3 values
   - `test_run_cv_returns_fitted_transformers()`: selector/scaler/pca are not None and correct types
   - `test_each_fold_mi_is_independent()`: MI scores differ across folds
   - `test_each_fold_scaler_is_independent()`: scaler means differ across folds

5. `TestNoBalancingLeakage`:
   - `test_smote_applied_to_train_only()`: balanced train ≥ original, val unchanged
   - `test_smote_does_not_affect_val_indices()`: balanced train doesn't contain val samples

6. `TestSplitReproducibility`:
   - `test_split_is_stratified()`: train/test class ratios within 2%
   - `test_split_is_reproducible()`: same seed → identical splits
   - `test_split_sizes_are_80_20()`: exact counts

7. `TestBalancingAPI`:
   - `test_kmeans_is_default_strategy()`: inspect default param
   - `test_kmeans_is_default_full_train()`: inspect default param
   - `test_kmeans_strategy_balances()`: output ≥ input size
   - `test_smote_strategy_balances()`: output ≥ input size
   - `test_balance_full_train_returns_tuple()`: isinstance(tuple) and len==2
   - `test_balancing_never_receives_val_data()`: mock verifies call with X_tr only
   - `test_run_cv_with_kmeans_strategy()`: 3 folds with kmeans strategy

8. `TestDLPipelineNoLeakage`:
   - `test_preprocess_fold_mi_fit_on_train_only()`: selector fitted on 640 samples, k=10
   - `test_preprocess_fold_scaler_fit_on_train_only()`: scaler.mean_ ≈ X_tr[:640].mean
   - `test_preprocess_fold_pca_fit_on_train_only()`: n_components_ ≤ 5, shapes correct
   - `test_preprocess_fold_balancing_only_on_train()`: val unchanged, train grows
   - `test_preprocess_final_mi_fit_on_full_train_only()`: MI→PCA shapes correct
   - `test_preprocess_final_scaler_fit_on_full_train_only()`: scaler.mean_ matches full train
   - `test_preprocess_final_pca_fit_on_full_train_only()`: n_components_ ≤ 5
   - `test_preprocess_final_balancing_only_on_train()`: test unchanged, train grows
   - `test_preprocess_fold_returns_correct_keys()`: exact key set
   - `test_preprocess_final_returns_correct_keys()`: exact key set
   - `test_no_val_data_in_balanced_train()`: balanced ≠ val
   - `test_dl_pipeline_evaluate_predictions_works()`: all 8 metric keys present

9. `TestFinalRetrainingIntegrity`:
   - `test_balance_full_train_receives_full_training_data()`: output ≥ input, same shape[1]
   - `test_balance_full_train_never_receives_test_data()`: mock verifies X_arg.shape == X_train.shape
   - `test_final_retrain_uses_full_train_not_fold()`: MI→Scaler→PCA on full 800 train → correct shapes
   - `test_test_data_never_enters_balancing()`: mock verifies X_arg is train, not test
   - `test_final_model_evaluation_uses_test_set_only()`: HGB eval on test, sanity check
   - `test_no_fold_data_in_final_retraining()`: verifies fold indices cover all training data

### 5.2 `tests/test_dl_pipeline.py` (480 lines, 9 classes)
**Fixtures:**
- `synthetic_tensors()`: torch.randn(100, 20), torch.randint(0, 5, (100,))
- `synthetic_numpy()`: same as test_leakage.py fixture

**Test Classes:**

1. `TestDataLoaderDropLast`:
   - `test_training_loader_has_drop_last()`: assert loader.drop_last is True
   - `test_training_loader_drops_last_batch()`: 10 samples, batch=8, drop_last → 8 samples
   - `test_val_loader_no_drop_last()`: 10 samples, batch=8, no drop → 10 samples

2. `TestMetricsJsonAUC`:
   - `test_auc_in_valid_range()`: evaluate_with_proba → auc ∈ [0,1]
   - `test_auc_not_always_zero()`: near-perfect predictions → auc > 0.5
   - `test_saved_json_auc_finite()`: JSON round-trip → finite auc ∈ [0,1]

3. `TestArchitectureLayerNorm`:
   - `test_dnn_mi_pca_kmeans_uses_layernorm()`: has LayerNorm, no BatchNorm1d
   - `test_dnn_forward_pass_small_batch()`: batch=1 → valid output
   - `test_dnn_baseline_uses_layernorm()`: has LayerNorm, no BatchNorm1d
   - `test_dnn_baseline_forward_pass_small_batch()`: batch=1 → valid output
   - `test_shared_fe_uses_layernorm()`: model import, has LayerNorm, no BatchNorm1d

4. `TestGetProbabilities`:
   - `test_returns_valid_probabilities()`: (50,5), sum≈1, ∈[0,1], finite
   - `test_single_sample_probabilities()`: (1,5), sum≈1

5. `TestEvaluateWithProba`:
   - `test_auc_nonzero_with_good_predictions()`: auc > 0.9
   - `test_evaluate_predictions_returns_zero_auc()`: evaluate_predictions → auc=0.0 (no proba)
   - `test_binary_auc_nonzero()`: binary case → auc > 0.9

6. `TestKNeighborsFloor`:
   - `test_dl_pipeline_preprocess_fold_k_floor()`: k=100 → floored to 7
   - `test_dl_pipeline_preprocess_fold_runs_without_error()`: synthetic binary → valid output
   - `test_balancing_floor_k_neighbors()`: SMOTE with large k → still works

7. `TestNoHardcodedAUC`:
   - `test_train_dnn_uses_evaluate_with_proba()`: source audit
   - `test_train_lstm_uses_evaluate_with_proba()`: source audit
   - `test_train_bilstm_uses_evaluate_with_proba()`: source audit
   - `test_train_bilstm_sharedfe_uses_evaluate_with_proba()`: source audit
   - `test_train_dnn_mi_pca_kmeans_uses_evaluate_with_proba()`: source audit
   - `test_no_script_has_auc_zero_literal_in_final_eval()`: all 5 scripts checked line-by-line

8. `TestTrainingDropLastAudit`:
   - Parses all DataLoader() calls across all 5 scripts, asserts `drop_last=True` on each
   - One test per script (5 total)

9. `TestNoBatchNorm`:
   - `test_train_dnn_no_batchnorm()`: source audit
   - `test_train_dnn_mi_pca_kmeans_no_batchnorm()`: source audit
   - `test_train_bilstm_sharedfe_no_batchnorm()`: source audit

### 5.3 `tests/test_audit_fixes.py` (37 lines, 4 tests)
1. `test_normal_class_index_is_derived_from_encoder_not_assumed_zero()`: LabelEncoder(['Normal', 'Analysis', 'Worms']) → index != 0
2. `test_classical_trainers_expose_runtime_balancer_and_plot_options()`: HGB/XGBoost/LogReg all have 'balancer', 'make_plots', 'normal_class_idx' params
3. `test_kmeans_strategy_uses_actual_kmeans_smote_implementation()`: balance_training_fold source has 'KMeansSMOTE(' and no 'fit_predict'
4. `test_final_dl_preprocessing_keeps_the_bounded_resampling_policy()`: preprocess_final source has 'RandomUnderSampler' and 'rus_cap'

---

## 6. Dataset Statistics (UNSW-NB15)
- ~2.5M rows, 47 features after dropping metadata
- 10 attack categories + Normal
- Class distribution: Normal (~48%), Generic (~7.5%), Exploits (~4.4%), Fuzzers (~1%), DoS (~0.6%), Reconnaissance (~0.6%), Analysis (~0.1%), Backdoor (~0.1%), Shellcode (~0.06%), Worms (~0.005%)

## 7. Key Results (leakage-free)
**HGB**: Acc=0.9628, F1=0.9703, AUC=0.9975  
**XGBoost**: Acc=0.9122, F1=0.9275, AUC=0.9834  
**LogReg**: Acc=0.9545, F1=0.9640, AUC=0.9922  
**BiLSTM_SharedFE**: Binary Acc=0.9870, Binary F1=0.9510, Multi Acc=0.9661, Weighted F1=0.9727, AUC=0.9992  
**DNN_MI_PCA_KMeans**: Binary Acc=0.9869, Binary F1=0.9505, Multi Acc=0.9666, Weighted F1=0.9729, AUC=0.9992  
**BiLSTM**: Binary Acc=0.9865, Binary F1=0.9492, Multi Acc=0.9665, Weighted F1=0.9728, AUC=0.9990  
**LSTM**: Binary Acc=0.9858, Binary F1=0.9469, Multi Acc=0.9655, Weighted F1=0.9721, AUC=0.9989  
**DNN**: Binary Acc=0.9857, Binary F1=0.9465, Multi Acc=0.9665, Weighted F1=0.9723, AUC=0.9990
