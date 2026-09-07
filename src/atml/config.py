"""Config loading. The YAML file at the repo root is the only place settings live."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@functools.lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache config.yaml. Pass a path to override (used in tests)."""
    cfg_path = Path(path) if path else _DEFAULT_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def seed(cfg: dict[str, Any] | None = None) -> int:
    return int((cfg or load_config())["seed"])
