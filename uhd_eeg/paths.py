"""Load repository path configuration.

Copy ``configs/paths.yaml.example`` to ``configs/paths.yaml`` and edit
the paths for your machine. No absolute paths are stored in tracked files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yaml"
EXAMPLE_PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yaml.example"


def load_paths(paths_file: Path | None = None) -> dict[str, Any]:
    """Load path configuration from YAML.

    Parameters
    ----------
    paths_file:
        Optional explicit path. Defaults to ``configs/paths.yaml``, falling
        back to ``configs/paths.yaml.example`` if the local file is absent.
    """
    candidate = paths_file
    if candidate is None:
        if DEFAULT_PATHS_FILE.is_file():
            candidate = DEFAULT_PATHS_FILE
        elif EXAMPLE_PATHS_FILE.is_file():
            candidate = EXAMPLE_PATHS_FILE
        else:
            raise FileNotFoundError(
                "No path config found. Copy configs/paths.yaml.example to "
                "configs/paths.yaml and set bids_root / output_root."
            )
    with open(candidate, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Path config must be a mapping: {candidate}")
    return cfg


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve a configured path; relative paths are relative to project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    base = root if root is not None else PROJECT_ROOT
    return (base / path).resolve()


def get_bids_root(paths_file: Path | None = None) -> Path:
    cfg = load_paths(paths_file)
    if "bids_root" not in cfg:
        raise KeyError("paths config missing required key 'bids_root'")
    return resolve_path(cfg["bids_root"])


def get_output_root(paths_file: Path | None = None) -> Path:
    cfg = load_paths(paths_file)
    if "output_root" not in cfg:
        raise KeyError("paths config missing required key 'output_root'")
    return resolve_path(cfg["output_root"])
