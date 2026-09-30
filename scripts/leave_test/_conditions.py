"""BIDS run keys for offline calibration sessions (Supplementary Tables S1–S2)."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.figures._bids_runs import BidsRun, OFFLINE_RUNS

_BIDS_RUN_KEY = re.compile(
    r"^(?P<sub>sub-\d+)_task-(?P<task>[^_]+)_acq-(?P<acq>[^_]+)_run-(?P<run>\d+)$"
)


def run_key_from_bids_run(run: BidsRun) -> str:
    return run.key


def behavior_from_run_key(run_key: str) -> str:
    match = _BIDS_RUN_KEY.match(run_key)
    if match is None:
        raise ValueError(f"Not a BIDS run key: {run_key}")
    task = match.group("task")
    if task == "minimallyovert":
        return "minimally_overt"
    return task


def subject_from_run_key(run_key: str) -> str:
    match = _BIDS_RUN_KEY.match(run_key)
    if match is None:
        raise ValueError(f"Not a BIDS run key: {run_key}")
    return match.group("sub")


def default_calibration_run_keys() -> list[str]:
    """Offline calibration runs used in cross-modal controls (ds007591 only)."""
    return [run.key for run in OFFLINE_RUNS]


def discover_calibration_run_keys(bids_root: Path) -> list[str]:
    """Infer calibration run keys from ``*_events.tsv`` under a BIDS root."""
    keys: list[str] = []
    for events_path in sorted(bids_root.glob("sub-*/ses-*/eeg/*_events.tsv")):
        stem = events_path.name.replace("_events.tsv", "")
        if "_acq-calibration_" not in stem:
            continue
        run_key = stem.replace("_eeg", "")
        if _BIDS_RUN_KEY.match(run_key):
            keys.append(run_key)
    return sorted(set(keys))
