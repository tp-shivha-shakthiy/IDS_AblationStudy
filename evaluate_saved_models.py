"""Re-evaluate saved Tier 1 models on the deterministic locked 20% holdout.

This is a test-only command. It loads the final model and every fitted
preprocessing artifact from a completed run, reconstructs the existing
stratified 80/20 split, and transforms only the held-out rows. It never fits
transformers, balances data, performs cross-validation, or retrains a model.

Deep-learning artifacts are deliberately not evaluated here. The currently
saved DNN artifacts do not persist the training-fitted categorical encoder,
and the DNN baseline also omits MI/PCA artifacts when those stages are used.
Using a newly fitted encoder would not be a faithful artifact-only evaluation.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.dimensionality_reduction import split_data
from src.evaluation import compute_extended_metrics
from src.experiment_config import ABLATION_ORDER, ABLATION_PRESETS
from src.preprocessing import load_and_prepare, transform_features


MODEL_NAMES = ("HGB", "XGBoost", "LogReg", "DNN", "DNN_MI_PCA_KMeans")
CLASSICAL_MODELS = frozenset(("HGB", "XGBoost", "LogReg"))


def model_artifact_name(model_name: str) -> str:
    """Return the canonical final-model filename written by the trainers."""
    return f"{model_name.lower()}_model.joblib"


def artifact_status(results_root: Path, model_name: str, experiment: str) -> dict:
    """Report whether an experiment has the artifacts needed for test-only use."""
    experiment_dir = results_root / model_name / experiment
    config_path = experiment_dir / "experiment_config.json"
    expected = ABLATION_PRESETS[experiment]
    required = [
        config_path,
        experiment_dir / model_artifact_name(model_name),
        experiment_dir / "label_encoder.joblib",
        experiment_dir / "categorical_encoder.joblib",
        experiment_dir / "scaler.joblib",
    ]
    if expected["use_mi"]:
        required.append(experiment_dir / "mi_selector.joblib")
    if expected["use_pca"]:
        required.append(experiment_dir / "pca.joblib")

    missing = [path.name for path in required if not path.is_file()]
    supported = model_name in CLASSICAL_MODELS
    return {
        "directory": experiment_dir,
        "model_exists": (experiment_dir / model_artifact_name(model_name)).is_file(),
        "preprocessing_exists": not any(
            name != model_artifact_name(model_name) for name in missing
        ),
        "metrics_exists": (experiment_dir / "test_metrics.json").is_file(),
        "missing": missing,
        "supported": supported,
        "can_evaluate": supported and not missing,
    }


def load_and_validate_config(experiment_dir: Path, model_name: str, experiment: str) -> dict:
    """Load config metadata and reject artifacts that do not match the request."""
    with (experiment_dir / "experiment_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if config.get("model") != model_name:
        raise ValueError(
            f"{experiment_dir}: config model is {config.get('model')!r}, "
            f"not {model_name!r}."
        )
    if config.get("experiment") != experiment:
        raise ValueError(
            f"{experiment_dir}: config experiment is {config.get('experiment')!r}, "
            f"not {experiment!r}."
        )
    for key, value in ABLATION_PRESETS[experiment].items():
        if config.get(key) is not value:
            raise ValueError(
                f"{experiment_dir}: config {key}={config.get(key)!r}, expected {value!r}."
            )
    return config


def transform_locked_test_set(X_test, experiment_dir: Path, config: dict) -> np.ndarray:
    """Apply only training-fitted transforms to the untouched locked test rows."""
    categorical_encoder = joblib.load(experiment_dir / "categorical_encoder.joblib")
    X_transformed = transform_features(X_test, categorical_encoder)

    if config["use_mi"]:
        selector = joblib.load(experiment_dir / "mi_selector.joblib")
        X_transformed = selector.transform(X_transformed)

    scaler = joblib.load(experiment_dir / "scaler.joblib")
    X_transformed = scaler.transform(X_transformed)

    if config["use_pca"]:
        pca = joblib.load(experiment_dir / "pca.joblib")
        X_transformed = pca.transform(X_transformed)

    return X_transformed


def aligned_probabilities(model, X_test: np.ndarray, num_classes: int) -> np.ndarray:
    """Align estimator probability columns with the persisted label encoding."""
    probabilities = np.asarray(model.predict_proba(X_test))
    model_classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))
    if probabilities.shape[1] != len(model_classes):
        raise ValueError("Model predict_proba output does not match model.classes_.")
    if np.array_equal(model_classes, np.arange(num_classes)):
        return probabilities
    if np.any(model_classes < 0) or np.any(model_classes >= num_classes):
        raise ValueError("Model classes are incompatible with the saved label encoder.")

    aligned = np.zeros((probabilities.shape[0], num_classes), dtype=probabilities.dtype)
    aligned[:, model_classes.astype(int)] = probabilities
    return aligned


def evaluate_classical_model(
    model_name: str,
    experiment: str,
    results_root: Path,
    X_test,
    y_test: np.ndarray,
    source_label_encoder,
) -> dict:
    """Evaluate one persisted Tier 1 final model without any fitting."""
    status = artifact_status(results_root, model_name, experiment)
    if status["missing"]:
        raise FileNotFoundError(
            f"{status['directory']} is missing: {', '.join(status['missing'])}"
        )
    config = load_and_validate_config(status["directory"], model_name, experiment)
    saved_label_encoder = joblib.load(status["directory"] / "label_encoder.joblib")
    if list(saved_label_encoder.classes_) != list(source_label_encoder.classes_):
        raise ValueError(
            f"{status['directory']}: saved label encoder does not match the loaded dataset."
        )
    if "Normal" not in saved_label_encoder.classes_:
        raise ValueError(f"{status['directory']}: saved label encoder has no 'Normal' class.")

    X_test_final = transform_locked_test_set(X_test, status["directory"], config)
    model = joblib.load(status["directory"] / model_artifact_name(model_name))
    y_pred = np.asarray(model.predict(X_test_final))
    y_proba = aligned_probabilities(model, X_test_final, len(saved_label_encoder.classes_))
    normal_class_idx = list(saved_label_encoder.classes_).index("Normal")
    metrics = compute_extended_metrics(
        y_test, y_pred, y_proba=y_proba, normal_class_idx=normal_class_idx,
    )
    return {key: float(value) for key, value in metrics.items()}


def selected_values(value: str, all_values: tuple[str, ...]) -> tuple[str, ...]:
    """Expand the CLI's all sentinel while preserving canonical ordering."""
    return all_values if value == "all" else (value,)


def print_artifact_report(results_root: Path, models: tuple[str, ...], experiments: tuple[str, ...]) -> None:
    """Print a no-data-load inventory of the persisted artifact interface."""
    print("Model                  Experiment          Model  Preprocessing  Metrics  Test-only")
    print("-" * 88)
    for model_name in models:
        for experiment in experiments:
            status = artifact_status(results_root, model_name, experiment)
            model_ok = "yes" if status["model_exists"] else "no"
            preprocessing_ok = "yes" if status["preprocessing_exists"] else "no"
            metrics_ok = "yes" if status["metrics_exists"] else "no"
            if status["can_evaluate"]:
                evaluation = "yes"
            elif not status["supported"]:
                evaluation = "no (DL interface)"
            else:
                evaluation = "no"
            print(
                f"{model_name:<22} {experiment:<19} {model_ok:<6} "
                f"{preprocessing_ok:<14} {metrics_ok:<8} {evaluation}"
            )
            if status["missing"]:
                print(f"  missing: {', '.join(status['missing'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="UNSW-NB15 CSV directory")
    parser.add_argument("--results-root", default="results", help="Saved experiment root")
    parser.add_argument("--model", choices=("all", *MODEL_NAMES), default="all")
    parser.add_argument("--experiment", choices=("all", *ABLATION_ORDER), default="all")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing test_metrics.json after test-only evaluation.",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Only report artifact availability; do not load data or evaluate models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    models = selected_values(args.model, MODEL_NAMES)
    experiments = selected_values(args.experiment, tuple(ABLATION_ORDER))

    if args.inspect:
        print_artifact_report(results_root, models, experiments)
        return

    ready, skipped = [], []
    for model_name in models:
        for experiment in experiments:
            status = artifact_status(results_root, model_name, experiment)
            if not status["supported"]:
                skipped.append(
                    f"{model_name}/{experiment}: DL artifacts need a dedicated complete interface"
                )
            elif status["missing"]:
                skipped.append(
                    f"{model_name}/{experiment}: missing {', '.join(status['missing'])}"
                )
            elif (status["directory"] / "test_metrics.json").exists() and not args.overwrite:
                skipped.append(
                    f"{model_name}/{experiment}: test_metrics.json exists (use --overwrite)"
                )
            else:
                ready.append((model_name, experiment))

    if not ready:
        details = "\n".join(f"  - {message}" for message in skipped)
        raise SystemExit(f"No models were eligible for test-only evaluation:\n{details}")

    # Reconstruct the same split used by training before applying any fitted artifact.
    X_raw, y_all, source_label_encoder = load_and_prepare(args.data_dir)
    _, X_test, _, y_test = split_data(X_raw, y_all, test_size=0.20, random_state=42)
    del X_raw

    for model_name, experiment in ready:
        metrics = evaluate_classical_model(
            model_name, experiment, results_root, X_test, y_test, source_label_encoder,
        )
        output_path = results_root / model_name / experiment / "test_metrics.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Saved locked-test metrics: {output_path}")

    if skipped:
        print("\nNot evaluated:")
        for message in skipped:
            print(f"  - {message}")


if __name__ == "__main__":
    main()
