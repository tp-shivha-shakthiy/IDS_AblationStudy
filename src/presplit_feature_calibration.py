"""Classifier-free, full-dataset calibration for the presplit ablation.

The professor's presplit protocol deliberately fits feature analysis before
the outer holdout split.  This module therefore calibrates feature-method
parameters from the complete prepared dataset and never imports a classifier
or reads experiment metrics.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.feature_selection import fit_mi_selector
from src.preprocessing import (
    feature_type_mask,
    fit_categorical_encoder,
    transform_features,
)


CALIBRATION_PROTOCOL = "presplit_feature_parameter_calibration_v1"
KNEE_METHOD = "maximum_distance_to_endpoint_chord_normalized"
RANDOM_STATE = 42


class CalibrationError(RuntimeError):
    """Raised when a profile has no defensible, deterministic knee."""


def _manifest_sha256(path):
    """Return the identity hash for the exact frozen manifest bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_presplit_calibration_features(X):
    """Prepare the exact full-data representation used by the presplit runner."""
    if hasattr(X, "select_dtypes"):
        encoder = fit_categorical_encoder(X)
        feature_names = list(X.columns)
        prepared = transform_features(
            X,
            encoder,
            preserve_categorical_codes=True,
        )
        discrete_mask = feature_type_mask(feature_names)
        if prepared.shape[1] != len(feature_names):
            raise CalibrationError(
                "Prepared calibration features do not align with feature names."
            )
    else:
        encoder = None
        prepared = np.asarray(X, dtype=np.float32)
        feature_names = [f"feature_{index}" for index in range(prepared.shape[1])]
        discrete_mask = np.zeros(prepared.shape[1], dtype=bool)

    if prepared.ndim != 2 or prepared.shape[1] < 3:
        raise CalibrationError("Calibration requires at least three prepared features.")
    return prepared, feature_names, discrete_mask, encoder


def _normalized_knee(values, *, method_name):
    """Select a knee from descending values using normalized chord distance.

    Endpoints cannot be selected.  Tied distances intentionally resolve to the
    smaller rank, making the cutoff deterministic and conservative.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3:
        raise CalibrationError(f"{method_name} needs at least three values.")
    if not np.isfinite(values).all():
        raise CalibrationError(f"{method_name} contains non-finite values.")

    high, low = float(values.max()), float(values.min())
    if np.isclose(high, low):
        raise CalibrationError(f"{method_name} has no non-flat knee.")

    x = np.linspace(0.0, 1.0, len(values))
    y = (values - low) / (high - low)
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    chord = end - start
    chord_norm = float(np.hypot(*chord))
    if np.isclose(chord_norm, 0.0):
        raise CalibrationError(f"{method_name} endpoint chord is degenerate.")

    distances = np.abs(
        chord[0] * (start[1] - y) - (start[0] - x) * chord[1]
    ) / chord_norm
    distances[[0, -1]] = 0.0
    max_distance = float(distances[1:-1].max())
    if max_distance <= np.finfo(float).eps:
        raise CalibrationError(f"{method_name} has no positive interior knee.")

    tolerance = max(1e-12, max_distance * 1e-10)
    candidates = np.flatnonzero(
        np.isclose(distances, max_distance, rtol=0.0, atol=tolerance)
    )
    candidates = candidates[(candidates > 0) & (candidates < len(values) - 1)]
    if not len(candidates):
        raise CalibrationError(f"{method_name} has no valid interior knee.")
    selected_index = int(candidates[0])
    curve = [
        {
            "rank": int(index + 1),
            "normalized_rank": float(x[index]),
            "normalized_value": float(y[index]),
            "distance_to_chord": float(distances[index]),
        }
        for index in range(len(values))
    ]
    return selected_index, curve, max_distance


def _rank_profile(values, feature_names, *, method_name):
    """Rank feature scores descending and apply the common knee detector."""
    values = np.asarray(values, dtype=float)
    if len(values) != len(feature_names):
        raise CalibrationError(f"{method_name} scores do not align with features.")
    if not np.isfinite(values).all():
        raise CalibrationError(f"{method_name} contains non-finite feature scores.")

    ranking = np.argsort(-values, kind="stable")
    ranked_values = values[ranking]
    knee_index, curve, max_distance = _normalized_knee(
        ranked_values,
        method_name=method_name,
    )
    ranked = [
        {
            "rank": int(rank + 1),
            "feature_index": int(index),
            "feature": feature_names[index],
            "score": float(values[index]),
        }
        for rank, index in enumerate(ranking)
    ]
    return {
        "selection_method": KNEE_METHOD,
        "maximum_distance": max_distance,
        "ranking": ranked,
        "normalized_curve": curve,
        "selected_k": int(knee_index + 1),
        "selected_indices_rank_order": ranking[: knee_index + 1].tolist(),
        "selected_features_rank_order": [
            feature_names[index] for index in ranking[: knee_index + 1]
        ],
    }


def calibrate_mutual_information(X_prepared, y, feature_names, discrete_mask):
    """Score every feature with the existing MI implementation before cutoff."""
    selector = fit_mi_selector(
        X_prepared,
        y,
        k=X_prepared.shape[1],
        random_state=RANDOM_STATE,
        discrete_features=discrete_mask,
    )
    scores = np.asarray(selector.scores_, dtype=float)
    profile = _rank_profile(scores, feature_names, method_name="mutual information")
    profile.update(
        {
            "score_method": "existing_fit_mi_selector_all_features",
            "random_state": RANDOM_STATE,
            "n_neighbors": int(selector.mi_n_neighbors_),
            "discrete_feature_mask": np.asarray(discrete_mask, dtype=bool).tolist(),
            "all_feature_scores": scores.tolist(),
        }
    )
    return profile


def calibrate_pca(X_prepared):
    """Derive a scree-curve knee without supplying a variance cutoff."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_prepared)
    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(X_scaled)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
    if not np.isfinite(explained).all():
        raise CalibrationError("PCA explained variance contains non-finite values.")
    knee_index, curve, max_distance = _normalized_knee(
        explained,
        method_name="PCA explained-variance spectrum",
    )
    cumulative = np.cumsum(explained)
    return {
        "selection_method": KNEE_METHOD,
        "input_branch": "E2/E5 independent PCA branch: StandardScaler -> PCA",
        "random_state": RANDOM_STATE,
        "maximum_distance": max_distance,
        "selected_component_count": int(knee_index + 1),
        "selected_cumulative_variance": float(cumulative[knee_index]),
        "recommended_frozen_parameterization": "component_count",
        "recommendation_reason": (
            "A fixed component count keeps the PCA contribution comparable "
            "between E1's post-MI PCA and E2/E5's independent PCA branch; "
            "the final protocol must reject a count exceeding E1's frozen MI width."
        ),
        "components": [
            {
                "component": int(index + 1),
                "explained_variance_ratio": float(explained[index]),
                "cumulative_explained_variance_ratio": float(cumulative[index]),
            }
            for index in range(len(explained))
        ],
        "normalized_curve": curve,
    }


def calibrate_spearman(X_prepared, y, feature_names):
    """Recompute E3's absolute feature-target Spearman profile."""
    correlation = np.asarray(
        spearmanr(X_prepared, y, axis=0).statistic[:-1, -1],
        dtype=float,
    )
    correlation = np.nan_to_num(correlation, nan=0.0)
    absolute = np.abs(correlation)
    profile = _rank_profile(absolute, feature_names, method_name="Spearman")
    profile.update(
        {
            "score_method": "existing_E3_absolute_feature_target_spearman",
            "coefficients": correlation.tolist(),
            "absolute_coefficients": absolute.tolist(),
        }
    )
    return profile


def _fingerprint_prepared_data(X_prepared, y, feature_names):
    """Hash schema, prepared features, and labels without making a second copy."""
    digest = hashlib.sha256()
    digest.update("\n".join(feature_names).encode("utf-8"))
    digest.update(np.asarray(y, dtype=np.int64).tobytes())
    for start in range(0, len(X_prepared), 65_536):
        digest.update(np.ascontiguousarray(X_prepared[start:start + 65_536]).tobytes())
    return digest.hexdigest()


def load_and_validate_frozen_calibration_manifest(path, X, y, class_names):
    """Load the final calibration manifest and bind it to the supplied dataset.

    The returned prepared context is deliberately reused by the ablation runner,
    so validation and feature fitting observe exactly one full-data categorical
    preparation pass.  This validates provenance only; it does not fit a model
    or select replacement features.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise CalibrationError(
            f"Frozen calibration manifest does not exist: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(
            f"Frozen calibration manifest is malformed: {manifest_path}"
        ) from exc

    if not isinstance(manifest, dict):
        raise CalibrationError("Frozen calibration manifest must be a JSON object.")
    if manifest.get("protocol") != CALIBRATION_PROTOCOL:
        raise CalibrationError("Frozen calibration manifest protocol is incompatible.")

    try:
        prepared_dimension = int(manifest["prepared_dimension"])
        feature_names = manifest["feature_names"]
        manifest_class_names = manifest["class_names"]
        dataset = manifest["dataset"]
        mi = manifest["mi"]
        pca = manifest["pca"]
        spearman = manifest["spearman"]
        mi_k = int(mi["selected_k"])
        pca_components = int(pca["selected_component_count"])
        spearman_k = int(spearman["selected_k"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(
            "Frozen calibration manifest is missing required selected parameters."
        ) from exc

    if prepared_dimension != 43:
        raise CalibrationError(
            "Frozen calibration manifest must describe the 43-feature UNSW schema."
        )
    if not isinstance(feature_names, list) or len(feature_names) != prepared_dimension:
        raise CalibrationError("Frozen calibration manifest feature schema is malformed.")
    if not isinstance(manifest_class_names, list) or list(class_names) != manifest_class_names:
        raise CalibrationError("Frozen calibration manifest class ordering is incompatible.")
    feature_order_hash = hashlib.sha256(
        "\n".join(feature_names).encode("utf-8")
    ).hexdigest()
    if manifest.get("feature_name_order_hash") != feature_order_hash:
        raise CalibrationError("Frozen calibration manifest feature-order hash is invalid.")
    if not isinstance(dataset, dict) or not dataset.get("prepared_feature_fingerprint"):
        raise CalibrationError("Frozen calibration manifest dataset fingerprint is missing.")
    if pca.get("recommended_frozen_parameterization") != "component_count":
        raise CalibrationError("Frozen calibration PCA is not a component-count calibration.")
    if not spearman.get("selection_method"):
        raise CalibrationError("Frozen calibration manifest Spearman selection rule is missing.")
    if not (0 < mi_k <= prepared_dimension):
        raise CalibrationError("Frozen calibration MI selected_k is invalid.")
    if not (0 < pca_components <= mi_k):
        raise CalibrationError("Frozen calibration PCA component count is invalid.")
    if not (0 < spearman_k <= prepared_dimension):
        raise CalibrationError("Frozen calibration Spearman selected_k is invalid.")

    prepared, current_names, discrete_mask, encoder = (
        prepare_presplit_calibration_features(X)
    )
    if prepared.shape[1] != prepared_dimension or current_names != feature_names:
        raise CalibrationError("Frozen calibration manifest feature schema is incompatible.")
    current_fingerprint = _fingerprint_prepared_data(prepared, y, current_names)
    if current_fingerprint != dataset["prepared_feature_fingerprint"]:
        raise CalibrationError("Frozen calibration manifest dataset fingerprint is incompatible.")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "protocol": manifest["protocol"],
        "dataset_fingerprint": current_fingerprint,
        "prepared_dimension": prepared_dimension,
        "mi_k": mi_k,
        "mi_selected_features_rank_order": list(
            mi.get("selected_features_rank_order", [])
        ),
        "pca_parameterization": "component_count",
        "pca_components": pca_components,
        # A calibrated count is consumed by the existing E3 top-k selector.
        # The selection-method field remains provenance for how that count arose.
        "spearman_selection_rule": "top_k",
        "spearman_calibration_method": spearman["selection_method"],
        "spearman_k": spearman_k,
        "spearman_selected_features_rank_order": list(
            spearman.get("selected_features_rank_order", [])
        ),
        "prepared_context": (prepared, current_names, discrete_mask, encoder),
    }


def _versions():
    import scipy
    import sklearn

    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def _historical_spearman_k(path):
    if not path or not Path(path).is_file():
        return None
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("detected_knee")


def calibrate_presplit_feature_parameters(
    X,
    y,
    class_names,
    *,
    dataset_files=(),
    historical_spearman_path=None,
):
    """Calibrate all three feature methods from one complete dataset snapshot."""
    prepared, feature_names, discrete_mask, _ = prepare_presplit_calibration_features(X)
    mi = calibrate_mutual_information(prepared, y, feature_names, discrete_mask)
    pca = calibrate_pca(prepared)
    spearman = calibrate_spearman(prepared, y, feature_names)
    historical_k = _historical_spearman_k(historical_spearman_path)
    pca["component_count_within_selected_mi_width"] = (
        pca["selected_component_count"] <= mi["selected_k"]
    )
    spearman["historical_selected_k"] = historical_k
    spearman["historical_result_reproduced"] = (
        historical_k is not None and spearman["selected_k"] == historical_k
    )

    return {
        "protocol": CALIBRATION_PROTOCOL,
        "calibration_algorithm": {
            "version": 1,
            "knee_method": KNEE_METHOD,
            "tie_break": "smaller_k_when_distances_are_effectively_equal",
            "failure_policy": "raise_CalibrationError_no_historical_default_fallback",
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "files": list(dataset_files),
            "row_count": int(prepared.shape[0]),
            "prepared_feature_fingerprint": _fingerprint_prepared_data(
                prepared,
                y,
                feature_names,
            ),
        },
        "prepared_dimension": int(prepared.shape[1]),
        "feature_names": feature_names,
        "feature_name_order_hash": hashlib.sha256(
            "\n".join(feature_names).encode("utf-8")
        ).hexdigest(),
        "class_names": list(class_names),
        "random_state": RANDOM_STATE,
        "environment": _versions(),
        "mi": mi,
        "pca": pca,
        "spearman": spearman,
    }


def _knee_window(items, selected_k, key):
    start = max(0, selected_k - 4)
    end = min(len(items), selected_k + 3)
    return items[start:end], key


def _write_report(manifest, destination):
    mi = manifest["mi"]
    pca = manifest["pca"]
    spearman = manifest["spearman"]
    lines = [
        "# Presplit Feature-Parameter Calibration",
        "",
        "This report contains feature-only calibration evidence. No classifier or "
        "experiment metric was used.",
        "",
        "## Mutual Information",
        "",
        f"Selected k: `{mi['selected_k']}`  ",
        f"Knee method: `{mi['selection_method']}`  ",
        f"Maximum chord distance: `{mi['maximum_distance']:.12g}`",
        "",
        "Selected features: " + ", ".join(mi["selected_features_rank_order"]),
        "",
        "| Rank | Feature | MI score |",
        "|---:|---|---:|",
    ]
    for item in mi["ranking"][max(0, mi["selected_k"] - 4):mi["selected_k"] + 3]:
        lines.append(f"| {item['rank']} | {item['feature']} | {item['score']:.12g} |")

    lines.extend([
        "",
        "## PCA",
        "",
        f"Selected components: `{pca['selected_component_count']}`  ",
        f"Cumulative variance at selection: `{pca['selected_cumulative_variance']:.12g}`  ",
        "Frozen representation recommendation: `component_count`.",
        "",
        "| Component | Individual variance | Cumulative variance |",
        "|---:|---:|---:|",
    ])
    for item in pca["components"][max(0, pca["selected_component_count"] - 4):pca["selected_component_count"] + 3]:
        lines.append(
            f"| {item['component']} | {item['explained_variance_ratio']:.12g} | "
            f"{item['cumulative_explained_variance_ratio']:.12g} |"
        )

    lines.extend([
        "",
        "## Spearman",
        "",
        f"Selected k: `{spearman['selected_k']}`  ",
        f"Historical k: `{spearman['historical_selected_k']}`  ",
        f"Historical result reproduced: `{spearman['historical_result_reproduced']}`",
        "",
        "Selected features: " + ", ".join(spearman["selected_features_rank_order"]),
        "",
        "| Rank | Feature | Absolute correlation |",
        "|---:|---|---:|",
    ])
    for item in spearman["ranking"][max(0, spearman["selected_k"] - 4):spearman["selected_k"] + 3]:
        lines.append(f"| {item['rank']} | {item['feature']} | {item['score']:.12g} |")
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(manifest, results_dir):
    """Write evidence plots when matplotlib is installed; never select from them."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths = []
    profiles = (
        ("mi", "MI score", manifest["mi"]["ranking"], manifest["mi"]["selected_k"]),
        ("spearman", "Absolute Spearman correlation", manifest["spearman"]["ranking"], manifest["spearman"]["selected_k"]),
    )
    for name, ylabel, ranking, selected_k in profiles:
        ranks = [entry["rank"] for entry in ranking]
        scores = [entry["score"] for entry in ranking]
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(ranks, scores, marker="o", markersize=3)
        axis.axvline(selected_k, color="crimson", linestyle="--", label=f"k={selected_k}")
        axis.set(xlabel="Rank", ylabel=ylabel, title=f"{name.upper()} calibration")
        axis.legend()
        figure.tight_layout()
        path = Path(results_dir) / f"{name}_knee.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(str(path))

    components = manifest["pca"]["components"]
    selected = manifest["pca"]["selected_component_count"]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        [entry["component"] for entry in components],
        [entry["explained_variance_ratio"] for entry in components],
        marker="o",
        markersize=3,
        label="individual explained variance",
    )
    axis.axvline(selected, color="crimson", linestyle="--", label=f"k={selected}")
    axis.set(xlabel="Component", ylabel="Explained variance ratio", title="PCA scree calibration")
    axis.legend()
    figure.tight_layout()
    path = Path(results_dir) / "pca_knee.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))
    return paths


def write_calibration_artifacts(manifest, manifest_path, results_dir):
    """Persist one machine-readable manifest plus report and optional plots."""
    manifest_path = Path(manifest_path)
    results_dir = Path(results_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    _write_report(manifest, results_dir / "calibration_report.md")
    return _write_plots(manifest, results_dir)
