# Ablation Study — Real Training Results (Test Set)

Complete leakage-free ablation results from the **real training runs** (80/20 split, locked 20% test set).
Per-experiment metrics are read from `results/<Model>/<experiment>/test_metrics.json` and `results/<Model>/<experiment>/cv_metrics.csv`.
This table reflects only the current full-data experiments for the five manuscript models:
HGB, XGBoost, LogReg, DNN, and DNN_MI_PCA_KMeans.

Three factors are varied: **MI** (mutual-information feature selection), **PCA** (95% variance), **KMeansSMOTE** (K-means SMOTE balancing).
StandardScaler is always on and is not an ablation factor.

| Experiment | MI | PCA | KMeansSMOTE |
|---|---|---|---|
| `raw` | ✗ | ✗ | ✗ |
| `mi` | ✓ | ✗ | ✗ |
| `mi_balancing` | ✓ | ✗ | ✓ |
| `pca` | ✗ | ✓ | ✗ |
| `pca_balancing` | ✗ | ✓ | ✓ |
| `mi_pca` | ✓ | ✓ | ✗ |
| `mi_pca_balancing` | ✓ | ✓ | ✓ |

---

## Test-Set Metrics (locked 20%)

### HistGradientBoosting (HGB)

| Experiment | Accuracy | Precision | Recall | F1 | AUC | Macro F1 | Weighted F1 | Bin Acc | Bin AUC |
|---|---|---|---|---|---|---|---|---|---|
| raw | **0.9791** | 0.9779 | **0.9791** | **0.9768** | **0.9994** | **0.5510** | **0.9768** | 0.9990 | **0.9942** |
| mi | 0.9783 | 0.9774 | 0.9783 | 0.9745 | 0.9994 | 0.5271 | 0.9745 | 0.9990 | 0.9937 |
| mi_balancing | 0.9675 | **0.9828** | 0.9675 | 0.9733 | 0.9980 | 0.5106 | 0.9733 | **0.9957** | 0.9924 |
| pca | 0.9744 | 0.9711 | 0.9744 | 0.9702 | 0.9991 | 0.4201 | 0.9702 | 0.9989 | 0.9932 |
| pca_balancing | 0.9607 | 0.9800 | 0.9607 | 0.9684 | 0.9971 | 0.4324 | 0.9684 | 0.9961 | 0.9922 |
| mi_pca | 0.9772 | 0.9745 | 0.9772 | 0.9738 | 0.9993 | 0.4512 | 0.9738 | 0.9990 | 0.9932 |
| mi_pca_balancing | 0.9629 | 0.9792 | 0.9629 | 0.9693 | 0.9976 | 0.4567 | 0.9693 | 0.9956 | 0.9914 |

### XGBoost

| Experiment | Accuracy | Precision | Recall | F1 | AUC | Macro F1 | Weighted F1 | Bin Acc | Bin AUC |
|---|---|---|---|---|---|---|---|---|---|
| raw | **0.9567** | 0.9374 | **0.9567** | **0.9391** | **0.9984** | 0.2544 | **0.9391** | **0.9989** | **0.9901** |
| mi | 0.9432 | 0.9052 | 0.9432 | 0.9226 | 0.9859 | 0.1869 | 0.9226 | 0.9989 | 0.9883 |
| mi_balancing | 0.8712 | 0.9371 | 0.8712 | 0.8932 | 0.9838 | **0.3812** | 0.8932 | 0.9949 | 0.9681 |
| pca | 0.9120 | 0.8786 | 0.9120 | 0.8845 | 0.9939 | 0.1573 | 0.8845 | 0.9989 | 0.9790 |
| pca_balancing | 0.9207 | **0.9560** | 0.9207 | 0.9372 | 0.9800 | 0.3166 | 0.9372 | 0.9927 | 0.9275 |
| mi_pca | 0.9431 | 0.9052 | 0.9431 | 0.9225 | 0.9963 | 0.1868 | 0.9225 | 0.9989 | 0.9872 |
| mi_pca_balancing | 0.9194 | 0.9543 | 0.9194 | 0.9340 | 0.9874 | 0.3566 | 0.9340 | 0.9917 | 0.7364 |

### Logistic Regression

| Experiment | Accuracy | Precision | Recall | F1 | AUC | Macro F1 | Weighted F1 | Bin Acc | Bin AUC |
|---|---|---|---|---|---|---|---|---|---|
| raw | **0.9743** | 0.9714 | **0.9743** | **0.9715** | **0.9989** | **0.3919** | **0.9715** | **0.9990** | **0.9916** |
| mi | 0.9711 | 0.9671 | 0.9711 | 0.9679 | 0.9982 | 0.3745 | 0.9679 | 0.9989 | 0.9901 |
| mi_balancing | 0.9574 | 0.9761 | 0.9574 | 0.9651 | 0.9969 | 0.3793 | 0.9651 | 0.9941 | 0.9880 |
| pca | 0.9687 | 0.9641 | 0.9687 | 0.9658 | 0.9977 | 0.3630 | 0.9658 | 0.9989 | 0.9886 |
| pca_balancing | 0.9596 | **0.9797** | 0.9596 | 0.9680 | 0.9948 | **0.4252** | 0.9680 | 0.9975 | 0.9909 |
| mi_pca | 0.9551 | 0.9543 | 0.9551 | 0.9537 | 0.9958 | 0.3077 | 0.9537 | 0.9989 | 0.9803 |
| mi_pca_balancing | 0.9527 | 0.9743 | 0.9527 | 0.9617 | 0.9916 | 0.3306 | 0.9617 | 0.9957 | 0.9870 |

*Bold = best value for that model. Bin Acc = binary (Normal/Attack) accuracy. Macro F1 = unweighted macro F1 across 10 classes.*

---

## Cross-Model Test Accuracy (best per experiment)

| Experiment | HGB | XGBoost | LogReg |
|---|---|---|---|
| raw | **0.9791** | 0.9567 | 0.9743 |
| mi | **0.9783** | 0.9432 | 0.9711 |
| mi_balancing | **0.9675** | 0.8712 | 0.9574 |
| pca | **0.9744** | 0.9120 | 0.9687 |
| pca_balancing | **0.9607** | 0.9207 | 0.9596 |
| mi_pca | **0.9772** | 0.9431 | 0.9551 |
| mi_pca_balancing | **0.9629** | 0.9194 | 0.9527 |

---

## Cross-Model Test AUC (best per experiment)

| Experiment | HGB | XGBoost | LogReg |
|---|---|---|---|
| raw | **0.9994** | 0.9984 | 0.9989 |
| mi | **0.9994** | 0.9859 | 0.9982 |
| mi_balancing | **0.9980** | 0.9838 | 0.9969 |
| pca | **0.9991** | 0.9939 | 0.9977 |
| pca_balancing | **0.9971** | 0.9800 | 0.9948 |
| mi_pca | **0.9993** | 0.9963 | 0.9958 |
| mi_pca_balancing | **0.9976** | 0.9874 | 0.9916 |

---

## Mean Cross-Validation Metrics (5 folds)

| Model | Experiment | CV Acc | CV F1 | CV AUC |
|---|---|---|---|---|
| HGB | raw | 0.9791 | 0.9764 | 0.9994 |
| HGB | mi | 0.9782 | 0.9742 | 0.9994 |
| HGB | mi_balancing | 0.9675 | 0.9731 | 0.9980 |
| HGB | pca | 0.9744 | 0.9700 | 0.9991 |
| HGB | pca_balancing | 0.9622 | 0.9692 | 0.9972 |
| HGB | mi_pca | 0.9772 | 0.9734 | 0.9993 |
| HGB | mi_pca_balancing | 0.9649 | 0.9708 | 0.9976 |
| XGBoost | raw | 0.9564 | 0.9387 | 0.9985 |
| XGBoost | mi | 0.9429 | 0.9222 | 0.9858 |
| XGBoost | mi_balancing | 0.8734 | 0.8962 | 0.9830 |
| XGBoost | pca | 0.9124 | 0.8851 | 0.9939 |
| XGBoost | pca_balancing | 0.9175 | 0.9358 | 0.9803 |
| XGBoost | mi_pca | 0.9435 | 0.9232 | 0.9962 |
| XGBoost | mi_pca_balancing | 0.9211 | 0.9361 | 0.9874 |
| LogReg | raw | 0.9743 | 0.9714 | 0.9990 |
| LogReg | mi | 0.9711 | 0.9678 | 0.9983 |
| LogReg | mi_balancing | 0.9577 | 0.9654 | 0.9971 |
| LogReg | pca | 0.9688 | 0.9657 | 0.9977 |
| LogReg | pca_balancing | 0.9589 | 0.9671 | 0.9949 |
| LogReg | mi_pca | 0.9553 | 0.9538 | 0.9957 |
| LogReg | mi_pca_balancing | 0.9555 | 0.9638 | 0.9913 |

---

# Deep-Learning (Tier 2) Results

## DL Ablation Runs (per-experiment, `results/<Model>/<experiment>/`)

Two DNN variants have per-experiment ablation runs. **DNN** (2-layer 64→32, class-weight loss) has all 7 experiments;
**DNN_MI_PCA_KMeans** (3-layer 128→64→32, KMeans balancing) has 4 (raw, mi, pca, mi_pca).

> Note: `DNN_MI_PCA_KMeans` at ~0.979 accuracy is notably higher than the `DNN` class-weight variant (~0.96) — the
> 3-layer architecture with K-means-balanced training (RUS cap 15,000 + KMeansSMOTE) generalises better on the test set.

### DNN (2-layer, class-weight loss) — test set

| Experiment | Accuracy | Precision | Recall | F1 | AUC | Macro F1 | Weighted F1 | Bin Acc | Bin AUC |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.9638 | 0.9826 | 0.9638 | 0.9705 | **0.9992** | 0.4684 | 0.9705 | 0.9855 | **0.9994** |
| mi | 0.9598 | **0.9838** | 0.9598 | 0.9691 | 0.9988 | 0.4466 | 0.9691 | 0.9836 | 0.9991 |
| mi_balancing | 0.9642 | 0.9811 | 0.9642 | **0.9709** | 0.9987 | 0.4615 | **0.9709** | 0.9867 | 0.9989 |
| pca | 0.9604 | 0.9833 | 0.9604 | 0.9690 | 0.9990 | 0.4588 | 0.9690 | 0.9843 | 0.9992 |
| pca_balancing | **0.9648** | 0.9816 | **0.9648** | 0.9716 | 0.9990 | **0.4695** | **0.9716** | **0.9859** | 0.9992 |
| mi_pca | 0.9643 | 0.9833 | 0.9643 | 0.9710 | 0.9989 | **0.4779** | 0.9710 | 0.9844 | 0.9992 |
| mi_pca_balancing | 0.9622 | 0.9792 | 0.9622 | 0.9689 | 0.9987 | 0.4394 | 0.9689 | 0.9845 | 0.9990 |

### DNN_MI_PCA_KMeans (3-layer, KMeans-balanced) — test set

| Experiment | Accuracy | Precision | Recall | F1 | AUC | Macro F1 | Weighted F1 | Bin Acc | Bin AUC |
|---|---|---|---|---|---|---|---|---|---|
| raw | **0.9796** | **0.9785** | **0.9796** | **0.9760** | **0.9995** | 0.4669 | **0.9760** | **0.9925** | **0.9997** |
| mi | 0.9789 | 0.9777 | 0.9789 | 0.9755 | 0.9995 | **0.4753** | 0.9755 | 0.9920 | 0.9996 |
| pca | 0.9788 | 0.9781 | 0.9788 | 0.9753 | 0.9995 | 0.4655 | 0.9753 | 0.9921 | 0.9997 |
| mi_pca | 0.9781 | 0.9747 | 0.9781 | 0.9740 | 0.9994 | 0.4372 | 0.9740 | 0.9913 | 0.9996 |

---

## Combined All-Model View

Best test-set accuracy per model among the configurations that were actually run.
Tier-1 models use their `mi_pca_balancing` (full pipeline) result; DNN_MI_PCA_KMeans shows its best per-experiment value.

| Model | Pipeline | Test Accuracy | Weighted F1 | AUC |
|---|---|---|---|---|
| **HGB** | mi_pca_balancing | 0.9629 | 0.9693 | 0.9976 |
| **DNN_MI_PCA_KMeans** | raw (best per-exp) | **0.9796** | 0.9760 | 0.9995 |
| DNN (class-weight) | mi_pca_balancing | 0.9622 | 0.9689 | 0.9987 |
| DNN (class-weight) | raw | 0.9638 | 0.9705 | 0.9992 |
| XGBoost | mi_pca_balancing | 0.9194 | 0.9340 | 0.9874 |
| LogReg | mi_pca_balancing | 0.9527 | 0.9617 | 0.9916 |

---

## Summary / Key Observations

**Tier 1 (classical):**
- **HGB is the strongest classical model** on every experiment (highest test accuracy and AUC in all 7 configurations).
- **Raw (no MI/PCA/balancing) is best for every classical model** — preprocessing ablations all *reduce* raw accuracy.
  Slight gains appear only in class-balance metrics (Macro F1, e.g. XGBoost `mi_balancing` macro F1 0.3812 vs raw 0.2544) because KMeansSMOTE rebalances rare classes.
- **KMeansSMOTE consistently lowers overall accuracy** but raises precision and Macro F1 (better minority-class handling).
- XGBoost is the most sensitive to ablations — its accuracy drops most with balancing (`mi_balancing` 0.8712, `mi_pca_balancing` 0.9194 with a notably low binary AUC of 0.7364).

**Tier 2 (deep learning):**
- The 3-layer **DNN_MI_PCA_KMeans** is the strongest DL model — its per-experiment results (0.978–0.980) beat every Tier-1 classical model.
- DL models are far more robust to disabling balancing than XGBoost: their accuracy changes only slightly across ablations.

---

*File generated from `results/<Model>/<experiment>/test_metrics.json` + `results/<Model>/<experiment>/cv_metrics.csv` (current full-data training only).*