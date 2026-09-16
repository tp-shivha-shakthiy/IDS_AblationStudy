"""Canonical corrected-baseline configuration loader."""

from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("configs/corrected_baseline.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the one authoritative corrected-baseline configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
