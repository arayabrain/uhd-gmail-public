"""Extract per-trial arrays from OpenNeuro ds007591 BIDS recordings.

Supports:
- sub-1..sub-3: EDF continuous recordings with TRIGGER onsets
- sub-4..sub-9: EEGLAB ``.set`` continuous recordings with analysis-window
  starts in the ``sample`` column of ``*_events.tsv``

Output layout (BIDS subject IDs only; no ``subjectN`` directories):

    {output_root}/
      sub-1/
        ses-20230511/
          task-minimallyovert_acq-calibration_run-01/
            trials/000.npy   # shape (139, 2880)
            labels.csv
            run.json
      manifest.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import mne
import numpy as np
import pandas as pd

from uhd_eeg.bids.constants import (
    EPOCH_SAMPLES,
    N_CH_TOTAL,
    SFREQ,
    SUBJECT_IDS,
    TASK_LABELS,
    UNIT_COEFF,
)

_BIDS_STEM_RE = re.compile(
    r"^(?P<sub>sub-\d+)_"
    r"(?P<ses>ses-\d+)_"
    r"task-(?P<task>[^_]+)_"
    r"acq-(?P<acq>[^_]+)_"
    r"run-(?P<run>\d+)_eeg$"
)


@dataclass(frozen=True)
class ExtractedRun:
    """One BIDS run converted to per-trial arrays."""

    subject_id: str
    session_id: str
    task: str
    acquisition: str
    run: str
    source_format: str  # "edf" | "eeglab"
    epochs: np.ndarray  # (n_trials, n_ch, n_samples)
    labels: np.ndarray  # (n_trials,) int

    @property
    def n_trials(self) -> int:
        return int(self.epochs.shape[0])

    @property
    def run_key(self) -> str:
        return f"task-{self.task}_acq-{self.acquisition}_run-{self.run}"


def parse_bids_eeg_stem(stem: str) -> dict[str, str]:
    """Parse a BIDS ``*_eeg`` filename stem into entities."""
    match = _BIDS_STEM_RE.match(stem)
    if match is None:
        raise ValueError(f"Unrecognized BIDS EEG stem: {stem}")
    return match.groupdict()


def iter_bids_runs(
    bids_root: Path,
    subject_ids: Sequence[str] | None = None,
) -> Iterator[Path]:
    """Yield EEG data files (``.edf`` or ``.set``) under ``bids_root``."""
    subjects = list(subject_ids) if subject_ids is not None else list(SUBJECT_IDS)
    for subject_id in subjects:
        sub_dir = bids_root / subject_id
        if not sub_dir.is_dir():
            continue
        for ses_dir in sorted(p for p in sub_dir.iterdir() if p.is_dir() and p.name.startswith("ses-")):
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.is_dir():
                continue
            files = sorted(eeg_dir.glob("*_eeg.edf")) + sorted(eeg_dir.glob("*_eeg.set"))
            for path in files:
                yield path


def load_continuous_raw_adc(eeg_path: Path) -> tuple[np.ndarray, str]:
    """Load continuous EEG and return ``(n_ch, n_samples)`` in raw ADC units."""
    suffix = eeg_path.suffix.lower()
    if suffix == ".edf":
        raw = mne.io.read_raw_edf(str(eeg_path), preload=True, verbose=False)
        data = raw.get_data()
        # EDF stores Volts; undo unit conversion used at publication.
        data = data.copy()
        data[: N_CH_TOTAL - 1] /= UNIT_COEFF
        return data, "edf"
    if suffix == ".set":
        raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
        data = raw.get_data().copy()
        # EEGLAB µV → MNE Volts; restore ADC for signal chans and 0/1 for stim.
        data[: N_CH_TOTAL - 1] /= UNIT_COEFF
        data[N_CH_TOTAL - 1] *= 1e6
        return data, "eeglab"
    raise ValueError(f"Unsupported EEG format: {eeg_path}")


def _trigger_onsets(data: np.ndarray) -> np.ndarray:
    trigger = data[N_CH_TOTAL - 1]
    return np.where(np.diff(trigger) > 0.5)[0] + 1


def extract_epochs_from_trigger(
    data: np.ndarray,
    n_events: int,
    *,
    keep: str = "last",
) -> List[np.ndarray]:
    """Extract fixed-length epochs from TRIGGER rising edges.

    Parameters
    ----------
    keep:
        When more triggers than events exist, keep ``"last"`` (EDF /
        continuous H5 style) or ``"first"`` (EEGLAB fallback).
    """
    onsets = _trigger_onsets(data)
    if len(onsets) == 0:
        return []
    if len(onsets) > n_events:
        onsets = onsets[-n_events:] if keep == "last" else onsets[:n_events]
    elif len(onsets) < n_events:
        n_events = len(onsets)

    epochs: List[np.ndarray] = []
    for i in range(n_events):
        onset_sample = int(onsets[i])
        start = (onset_sample // 8 - 359) * 8
        end = start + EPOCH_SAMPLES
        if start < 0 or end > data.shape[1]:
            continue
        epochs.append(data[:, start:end].copy())
    return epochs


def extract_epochs_from_sample_column(
    data: np.ndarray,
    events_df: pd.DataFrame,
) -> List[np.ndarray]:
    """Extract epochs using the OpenNeuro ``sample`` column (sub-4..sub-9)."""
    if "sample" not in events_df.columns:
        raise KeyError("events.tsv is missing required 'sample' column")
    epochs: List[np.ndarray] = []
    for sample in events_df["sample"].astype(int).tolist():
        start = int(sample)
        end = start + EPOCH_SAMPLES
        if start < 0 or end > data.shape[1]:
            continue
        epochs.append(data[:, start:end].copy())
    return epochs


def extract_epochs_concatenated(
    data: np.ndarray,
    n_trials: int,
) -> List[np.ndarray]:
    """Fallback when continuous data is trial-concatenated with no triggers."""
    epochs: List[np.ndarray] = []
    for i in range(n_trials):
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > data.shape[1]:
            break
        epochs.append(data[:, start:end].copy())
    return epochs


def extract_run(eeg_path: Path, events_path: Path | None = None) -> ExtractedRun:
    """Extract one BIDS run into trial arrays and labels."""
    eeg_path = Path(eeg_path)
    if events_path is None:
        events_path = eeg_path.with_name(
            eeg_path.name.replace("_eeg.edf", "_events.tsv").replace("_eeg.set", "_events.tsv")
        )
    events_path = Path(events_path)
    if not events_path.is_file():
        raise FileNotFoundError(f"events.tsv not found for {eeg_path}")

    entities = parse_bids_eeg_stem(eeg_path.stem)
    events_df = pd.read_csv(events_path, sep="\t")
    if "value" not in events_df.columns:
        raise KeyError(f"{events_path} missing required 'value' column")

    data, source_format = load_continuous_raw_adc(eeg_path)
    n_events = len(events_df)

    if "sample" in events_df.columns:
        epochs = extract_epochs_from_sample_column(data, events_df)
    else:
        onsets = _trigger_onsets(data)
        if len(onsets) == 0:
            epochs = extract_epochs_concatenated(data, n_events)
        else:
            keep = "first" if source_format == "eeglab" else "last"
            epochs = extract_epochs_from_trigger(data, n_events, keep=keep)

    if not epochs:
        raise RuntimeError(f"No epochs extracted from {eeg_path}")

    labels = events_df["value"].to_numpy(dtype=int)[: len(epochs)]
    stacked = np.stack(epochs, axis=0)
    if stacked.shape[1:] != (N_CH_TOTAL, EPOCH_SAMPLES):
        raise RuntimeError(
            f"Unexpected epoch shape {stacked.shape[1:]} from {eeg_path}; "
            f"expected ({N_CH_TOTAL}, {EPOCH_SAMPLES})"
        )

    return ExtractedRun(
        subject_id=entities["sub"],
        session_id=entities["ses"],
        task=entities["task"],
        acquisition=entities["acq"],
        run=entities["run"],
        source_format=source_format,
        epochs=stacked,
        labels=labels,
    )


def write_extracted_run(run: ExtractedRun, output_root: Path) -> Path:
    """Write one extracted run under ``output_root`` using BIDS IDs only."""
    run_dir = (
        Path(output_root)
        / run.subject_id
        / run.session_id
        / run.run_key
    )
    trials_dir = run_dir / "trials"
    if trials_dir.exists():
        for old in trials_dir.glob("*.npy"):
            old.unlink()
    trials_dir.mkdir(parents=True, exist_ok=True)

    for i, epoch in enumerate(run.epochs):
        np.save(trials_dir / f"{i:03d}.npy", epoch)

    labels_path = run_dir / "labels.csv"
    np.savetxt(labels_path, run.labels, delimiter=",", fmt="%d")

    meta = {
        "subject_id": run.subject_id,
        "session_id": run.session_id,
        "task": run.task,
        "task_label": TASK_LABELS.get(run.task, run.task),
        "acquisition": run.acquisition,
        "run": run.run,
        "source_format": run.source_format,
        "n_trials": run.n_trials,
        "n_channels": int(run.epochs.shape[1]),
        "n_samples": int(run.epochs.shape[2]),
        "sfreq": SFREQ,
    }
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return run_dir


def extract_subject(
    bids_root: Path,
    subject_id: str,
    output_root: Path | None = None,
) -> List[ExtractedRun]:
    """Extract all runs for one subject; optionally write outputs."""
    runs: List[ExtractedRun] = []
    for eeg_path in iter_bids_runs(bids_root, subject_ids=[subject_id]):
        run = extract_run(eeg_path)
        runs.append(run)
        if output_root is not None:
            write_extracted_run(run, output_root)
    return runs


def extract_dataset(
    bids_root: Path,
    output_root: Path,
    subject_ids: Sequence[str] | None = None,
) -> dict:
    """Extract all configured subjects and write a dataset manifest."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = list(subject_ids) if subject_ids is not None else list(SUBJECT_IDS)
    manifest_runs = []
    for subject_id in subjects:
        sub_dir = Path(bids_root) / subject_id
        if not sub_dir.is_dir():
            continue
        for run in extract_subject(bids_root, subject_id, output_root=output_root):
            manifest_runs.append(
                {
                    "subject_id": run.subject_id,
                    "session_id": run.session_id,
                    "task": run.task,
                    "acquisition": run.acquisition,
                    "run": run.run,
                    "n_trials": run.n_trials,
                    "source_format": run.source_format,
                    "relative_dir": str(
                        Path(run.subject_id) / run.session_id / run.run_key
                    ),
                }
            )

    manifest = {
        "bids_root": str(Path(bids_root)),
        "output_root": str(output_root),
        "n_runs": len(manifest_runs),
        "runs": manifest_runs,
        "notes": [
            "Subject identifiers are BIDS IDs (sub-1 ... sub-9).",
        ],
    }
    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest
