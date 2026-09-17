"""Atomic, configuration-bound CV fold checkpoints for presplit ablations."""

import hashlib
import json
import os
import tempfile


CHECKPOINT_SCHEMA_VERSION = 1


def build_cv_config_fingerprint(config, feature_dimensions, selected_features):
    """Hash all experiment choices that make a CV result reusable."""
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": config,
        "feature_dimensions": feature_dimensions,
        "selected_features": selected_features,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def checkpoint_path(save_dir, fold_number):
    return os.path.join(save_dir, f"cv_fold_{fold_number}.json")


def load_compatible_fold_checkpoint(save_dir, fold_number, fingerprint, config, feature_dimensions):
    """Return a completed checkpoint or reject an incompatible one.

    A checkpoint is only consulted by callers that received ``--resume``.
    Existing files with a mismatched fingerprint are errors, never candidates
    for partial reuse.
    """
    path = checkpoint_path(save_dir, fold_number)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fold_number": fold_number,
        "experiment": config["experiment"],
        "model": config["model"],
        "config_fingerprint": fingerprint,
        "feature_dimensions": feature_dimensions,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Refusing to reuse incompatible CV checkpoint: {path} ({key} differs)."
            )
    if not isinstance(payload.get("validation_metrics"), dict):
        raise ValueError(f"Refusing incomplete CV checkpoint: {path}.")
    return payload


def save_fold_checkpoint(save_dir, fold_number, validation_metrics, balancing_audit,
                         balancing_seconds, training_seconds, config,
                         feature_dimensions, fingerprint):
    """Write a completed fold checkpoint atomically after validation succeeds."""
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fold_number": int(fold_number),
        "validation_metrics": validation_metrics,
        "balancing_audit": balancing_audit,
        "balancing_seconds": float(balancing_seconds),
        "training_seconds": float(training_seconds),
        "experiment": config["experiment"],
        "model": config["model"],
        "feature_dimensions": feature_dimensions,
        "split": {"method": "stratified_80_20", "random_state": config["random_state"]},
        "random_state": config["random_state"],
        "config_fingerprint": fingerprint,
    }
    os.makedirs(save_dir, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".cv_fold_{fold_number}.", suffix=".json", dir=save_dir, text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary_path, checkpoint_path(save_dir, fold_number))
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    return payload
