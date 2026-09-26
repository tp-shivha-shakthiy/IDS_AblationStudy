| Experiment | Model | Final Dim | Multiclass Accuracy (%) | Macro-F1 (%) | Macro Recall (%) | Weighted-F1 (%) | Binary Accuracy (%) | Binary Precision (%) | Binary Recall (%) | Binary F1 (%) | Binary AUC (%) | Balancing |
| ---------- | ----- | --------: | ----------------------: | -----------: | ---------------: | --------------: | ------------------: | -------------------: | ----------------: | ------------: | -------------: | --------- |
| E1_sequential_mi_pca | HGB | 7 | 96.55% | 49.61% | 62.74% | 97.07% | 98.37% | 88.57% | 100.00% | 93.94% | 99.70% | KMeansSMOTE |
| E1_sequential_mi_pca | LogReg | 7 | 95.74% | 41.46% | 56.71% | 96.41% | 98.09% | 86.93% | 99.90% | 92.97% | 98.98% | KMeansSMOTE |
| E1_sequential_mi_pca | LSTM | 7 | 96.63% | 52.20% | 66.89% | 97.18% | 98.32% | 88.23% | 100.00% | 93.75% | 99.90% | KMeansSMOTE |
| E1_sequential_mi_pca | BiLSTM | 7 | 96.85% | 53.28% | 64.94% | 97.33% | 98.53% | 89.57% | 99.98% | 94.49% | 99.92% | KMeansSMOTE |
| E2_parallel_mi_pca | HGB | 30 | 97.05% | 54.95% | 72.31% | 97.49% | 98.72% | 90.81% | 99.99% | 95.18% | 99.74% | KMeansSMOTE |
| E2_parallel_mi_pca | LogReg | 30 | 96.36% | 47.40% | 65.50% | 96.95% | 98.45% | 89.08% | 99.97% | 94.21% | 99.62% | KMeansSMOTE |
| E2_parallel_mi_pca | LSTM | 30 | 96.88% | 54.42% | 70.06% | 97.34% | 98.59% | 89.96% | 99.99% | 94.71% | 99.91% | KMeansSMOTE |
| E2_parallel_mi_pca | BiLSTM | 30 | 96.98% | 53.89% | 70.33% | 97.43% | 98.65% | 90.35% | 99.99% | 94.93% | 99.94% | KMeansSMOTE |
| E3_spearman | HGB | 4 | 93.92% | 27.61% | 44.53% | 95.41% | 98.25% | 87.85% | 99.99% | 93.53% | 99.72% | KMeansSMOTE → SMOTE fallback |
| E3_spearman | LogReg | 4 | 94.00% | 27.30% | 42.40% | 95.44% | 98.18% | 87.42% | 99.97% | 93.27% | 99.44% | KMeansSMOTE → SMOTE fallback |
| E3_spearman | LSTM | 4 | 93.81% | 26.11% | 42.79% | 95.29% | 98.20% | 87.56% | 99.99% | 93.36% | 99.78% | KMeansSMOTE → SMOTE fallback |
| E3_spearman | BiLSTM | 4 | 94.04% | 26.93% | 43.30% | 95.39% | 98.20% | 87.55% | 99.99% | 93.36% | 99.71% | KMeansSMOTE → SMOTE fallback |

| Experiment | Model | Preprocessing (s) | Training (s) | Median Latency (ms) | Throughput (samples/s) | Model Size (MB) |
| ---------- | ----- | ----------------: | -----------: | ------------------: | ---------------------: | --------------: |
| E1_sequential_mi_pca | HGB | 1651.18 | 3.81 | 14.18230 | 106,865.32 | 0.83419 |
| E1_sequential_mi_pca | LogReg | 1541.50 | 11.65 | 0.11665 | 9,136,344.67 | 0.00119 |
| E1_sequential_mi_pca | LSTM | 1794.65 | 22.89 | 0.72870 | 963,648.04 | 0.05307 |
| E1_sequential_mi_pca | BiLSTM | 1959.10 | 25.83 | 1.32050 | 645,849.29 | 0.05307 |
| E2_parallel_mi_pca | HGB | 2083.59 | 6.40 | 11.74135 | 56,255.98 | 0.84451 |
| E2_parallel_mi_pca | LogReg | 2181.68 | 20.18 | 0.12015 | 2,700,339.45 | 0.00206 |
| E2_parallel_mi_pca | LSTM | 3204.42 | 35.84 | 0.85275 | 627,420.94 | 0.07553 |
| E2_parallel_mi_pca | BiLSTM | 1909.28 | 36.51 | 1.15125 | 515,599.42 | 0.07553 |
| E3_spearman | HGB | 32.72 | 2.59 | 8.59310 | 153,956.75 | 0.54775 |
| E3_spearman | LogReg | 122.81 | 8.55 | 0.09585 | 9,450,012.65 | 0.00107 |
| E3_spearman | LSTM | 47.54 | 32.57 | 0.60755 | 306,175.81 | 0.05014 |
| E3_spearman | BiLSTM | 38.66 | 21.98 | 0.75655 | 975,285.07 | 0.05014 |

1. E2_parallel_mi_pca with HGB is the top multiclass run: 97.05% accuracy, 54.95% macro-F1, and 72.31% macro recall.
2. E2 improves mean macro-F1 from 49.14% (E1) to 52.67% (+3.53 percentage points) and mean accuracy from 96.44% to 96.82% (+0.37 points) across the four models.
3. E3_spearman reduces the representation to 4 dimensions versus 7 for E1 and 30 for E2, but its mean macro-F1 is 26.99%, 25.68 points below E2's 52.67%.
4. Accuracy overstates multiclass balance: E2 HGB records 97.05% accuracy and 97.49% weighted-F1 but only 54.95% macro-F1.
5. E2 HGB has the best binary F1 at 95.18%; E2 BiLSTM has the best binary AUC at 99.94%.
6. All 4 E3 runs recorded KMeansSMOTE → SMOTE fallback in all 5 CV folds and final retraining (6 fallback events per run), while all 8 E1/E2 runs recorded KMeansSMOTE without fallback.

BEST OVERALL: E2_parallel_mi_pca / HGB — 97.05% multiclass accuracy, 54.95% macro-F1, 95.18% binary F1.

BEST MULTICLASS: E2_parallel_mi_pca / HGB — 97.05% accuracy.

BEST MACRO-F1: E2_parallel_mi_pca / HGB — 54.95%.

BEST BINARY: E2_parallel_mi_pca / HGB — 95.18% binary F1; E2_parallel_mi_pca / BiLSTM — 99.94% binary AUC.

MOST COMPACT: E3_spearman — 4 final dimensions; E3_spearman / LogReg — 0.00107 MB model.

FASTEST INFERENCE: E3_spearman / LogReg — 0.09585 ms median latency and 9,450,012.65 samples/s throughput.
