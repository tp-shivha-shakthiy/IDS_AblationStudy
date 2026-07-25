# Legacy Pipeline Results

These are **historical benchmark** results produced by the original (uncorrected) pipeline.

**Known methodological issues in the legacy pipeline:**
- StandardScaler and PCA were fitted on the **full dataset** before the 80/20 split (data leakage)
- MI feature selection was fitted on a stratified sample of the full dataset (including test data)
- Final model retraining used only fold-0 balanced training data (incomplete retraining)
- KMeans+SMOTE had a no-op clustering step (cluster filtering was dead code)

These results are preserved for comparison purposes only. They should **not** be used to evaluate model quality.

## Source

- **Pipeline**: Legacy (pre-refactoring)
- **Date**: Pre-refactoring historical run
- **Classification**: Legacy pipeline / historical benchmark
