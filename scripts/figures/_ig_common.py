"""Shared helpers for IG figure scripts (BIDS ``sub-N`` IDs only)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

CM_TO_INCH = 1 / 2.54
DEFAULT_COLORS = ["green", "magenta", "orange", "violet", "yellow"]
DEFAULT_TASKS = ["overt", "minimally_overt", "covert"]
DEFAULT_SUBJECTS = [f"sub-{i}" for i in range(1, 10)]
LABEL_COLORS = {i: name for i, name in enumerate(DEFAULT_COLORS)}

NAME_RE = re.compile(
    r"^(?P<condition>.+?)_(?P<model>EEGNet_wo_adapt_filt|EMG_EEGNet|EEGNet)_"
    r"(?P<run_date>\d{4}-\d{2}-\d{2})_(?P<run_time>\d{2}-\d{2}-\d{2})_"
    r"(?P<task>.+)_cv(?P<cv>\d+)_(?P<kind>igs|trial_predictions)\.(?P<ext>pt|csv)$"
)

_SUBJECT_RE = re.compile(r"^subject(\d+)$", re.IGNORECASE)


def normalize_subject_id(raw: str) -> str:
    """Map legacy ``subjectN`` tokens to BIDS ``sub-N``."""
    text = str(raw).strip()
    if text.startswith("sub-"):
        return text
    match = _SUBJECT_RE.match(text)
    if match:
        return f"sub-{match.group(1)}"
    return text


def normalize_condition(raw: str) -> str:
    """Normalize ``subject6-overt`` → ``sub-6-overt`` (first hyphen segment only)."""
    text = str(raw).strip()
    if "-" not in text:
        return normalize_subject_id(text)
    head, tail = text.split("-", 1)
    return f"{normalize_subject_id(head)}-{tail}"


def parse_ig_name(path: Path) -> dict[str, str] | None:
    match = NAME_RE.match(path.name)
    if match is None:
        return None
    meta = match.groupdict()
    meta["condition"] = normalize_condition(meta["condition"])
    meta["subject"] = normalize_subject_id(meta["condition"].split("-")[0])
    return meta


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_ig_tensor(path: Path) -> np.ndarray:
    """Load ``*_igs.pt`` as ``(trial, channel, time)``."""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "torch is required to load integrated-gradient tensors"
        ) from exc

    igs = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(igs, "detach"):
        igs = igs.detach().cpu().numpy()
    igs = np.asarray(igs)
    if igs.ndim == 4 and igs.shape[1] == 1:
        igs = igs[:, 0]
    if igs.ndim != 3:
        raise ValueError(f"Expected IG shape (trial, channel, time), got {igs.shape}")
    return igs


def safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
