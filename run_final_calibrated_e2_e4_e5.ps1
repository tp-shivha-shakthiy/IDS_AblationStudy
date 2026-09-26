$ErrorActionPreference = "Stop"

$dataDir = "data/raw"
$manifest = "configs/presplit_feature_calibration.json"
$resultsRoot = "results/presplit_feature_ablation_calibrated_2026_09"
$cap = 15000
$experiments = @("E2_parallel_mi_pca", "E4_mi_only", "E5_pca_only")
$models = @("HGB", "LogReg", "XGBoost", "DNN", "LSTM", "BiLSTM")

foreach ($experiment in $experiments) {
    foreach ($model in $models) {
        & python run_presplit_feature_ablation.py `
            --data-dir $dataDir `
            --experiment $experiment `
            --model $model `
            --cap $cap `
            --results-root $resultsRoot `
            --calibration-manifest $manifest `
            --resume
        if ($LASTEXITCODE -ne 0) {
            throw "Failed: $experiment / $model"
        }
    }
}
