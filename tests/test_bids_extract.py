"""CPU tests for BIDS → per-trial extraction (synthetic data only)."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import scipy.io

from uhd_eeg.bids.constants import EPOCH_SAMPLES, N_CH_TOTAL, SFREQ, UNIT_COEFF
from uhd_eeg.bids.extract import (
    extract_dataset,
    extract_epochs_from_trigger,
    extract_run,
    parse_bids_eeg_stem,
    write_extracted_run,
)
from uhd_eeg.paths import load_paths, resolve_path


WORD_LABELS = {0: "green", 1: "magenta", 2: "orange", 3: "violet", 4: "yellow"}


def _channel_names_and_types():
    ch_names = [f"EEG{i + 1:03d}" for i in range(128)]
    ch_names += [
        "DISPLAY+",
        "DISPLAY-",
        "MIC+",
        "MIC-",
        "EOG+",
        "EOG-",
        "EMG_UPPER_OOris+",
        "EMG_UPPER_OOris-",
        "EMG_LOWER_OOris+",
        "EMG_LOWER_OOris-",
        "TRIGGER",
    ]
    ch_types = (
        ["eeg"] * 128
        + ["misc", "misc"]
        + ["misc", "misc"]
        + ["eog", "eog"]
        + ["emg", "emg"]
        + ["emg", "emg"]
        + ["stim"]
    )
    return ch_names, ch_types


def _onset_sample_for_epoch_start(epoch_start: int) -> int:
    """Invert ``start = (onset // 8 - 359) * 8`` for synthetic triggers."""
    return (epoch_start // 8 + 359) * 8


def _make_continuous(n_trials: int, labels: list[int], *, gap: int = 104) -> tuple[np.ndarray, list[int]]:
    """Build continuous ADC data with isolated trial epochs and triggers.

    ``gap`` defaults to a multiple of 8 so epoch starts stay packet-aligned
    with the trigger→window formula used for EDF recordings.
    """
    if gap % 8 != 0:
        raise ValueError("gap must be a multiple of 8 for packet alignment")
    starts = []
    cursor = 0
    for _ in range(n_trials):
        starts.append(cursor)
        cursor += EPOCH_SAMPLES + gap
    n_samples = cursor
    data = np.zeros((N_CH_TOTAL, n_samples), dtype=np.float64)
    for i, start in enumerate(starts):
        # Encode trial index on channel 0 for identity checks.
        data[0, start : start + EPOCH_SAMPLES] = float(i + 1)
        data[1, start : start + EPOCH_SAMPLES] = float(labels[i])
        onset = _onset_sample_for_epoch_start(start)
        if onset < n_samples:
            data[-1, onset] = 1.0
    return data, starts


def _write_edf_bids_run(
    bids_root: Path,
    *,
    subject_id: str = "sub-1",
    session_id: str = "ses-20250101",
    task: str = "overt",
    acquisition: str = "calibration",
    run: str = "01",
    n_trials: int = 3,
    labels: list[int] | None = None,
) -> Path:
    labels = labels or [0, 2, 4][:n_trials]
    assert len(labels) == n_trials

    data_adc, starts = _make_continuous(n_trials, labels)
    data_volts = data_adc.copy()
    data_volts[: N_CH_TOTAL - 1] *= UNIT_COEFF

    ch_names, ch_types = _channel_names_and_types()
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(data_volts, info, verbose=False)

    eeg_dir = bids_root / subject_id / session_id / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject_id}_{session_id}_task-{task}_acq-{acquisition}_run-{run}"
    edf_path = eeg_dir / f"{prefix}_eeg.edf"
    mne.export.export_raw(str(edf_path), raw, fmt="edf", overwrite=True, verbose=False)

    rows = []
    for i, (start, label) in enumerate(zip(starts, labels)):
        rows.append(
            {
                "onset": start / SFREQ,
                "duration": EPOCH_SAMPLES / SFREQ,
                "trial_type": WORD_LABELS[label],
                "value": label,
                "session_type": acquisition,
                "task_condition": task if task != "minimallyovert" else "minimally overt",
            }
        )
    pd.DataFrame(rows).to_csv(eeg_dir / f"{prefix}_events.tsv", sep="\t", index=False)
    return edf_path


def _write_eeglab_bids_run(
    bids_root: Path,
    *,
    subject_id: str = "sub-7",
    session_id: str = "ses-20260520",
    task: str = "overt",
    acquisition: str = "calibration",
    run: str = "01",
    n_trials: int = 3,
    labels: list[int] | None = None,
) -> Path:
    labels = labels or [1, 3, 0][:n_trials]
    assert len(labels) == n_trials

    data_adc, starts = _make_continuous(n_trials, labels)
    # EEGLAB convention in this dataset: store ADC counts as µV-scale values.
    data_uv = data_adc.copy()
    data_uv[N_CH_TOTAL - 1] = data_adc[N_CH_TOTAL - 1]  # keep stim 0/1

    ch_names, _ = _channel_names_and_types()
    n_ch, n_samp = data_uv.shape
    chanlocs = np.zeros(n_ch, dtype=[("labels", "U64")])
    for i, name in enumerate(ch_names):
        chanlocs[i] = (name,)

    eeg_dir = bids_root / subject_id / session_id / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject_id}_{session_id}_task-{task}_acq-{acquisition}_run-{run}"
    set_path = eeg_dir / f"{prefix}_eeg.set"
    mat_dict = {
        "data": data_uv,
        "setname": "",
        "nbchan": np.float64(n_ch),
        "pnts": np.float64(n_samp),
        "trials": np.float64(1),
        "srate": np.float64(SFREQ),
        "xmin": np.float64(0),
        "xmax": np.float64((n_samp - 1) / SFREQ),
        "ref": "",
        "chanlocs": chanlocs,
        "icawinv": np.array([]),
        "icasphere": np.array([]),
        "icaweights": np.array([]),
    }
    scipy.io.savemat(str(set_path), mat_dict, do_compression=False)

    rows = []
    for i, (start, label) in enumerate(zip(starts, labels)):
        rows.append(
            {
                "onset": start / SFREQ,
                "duration": EPOCH_SAMPLES / SFREQ,
                "trial_type": WORD_LABELS[label],
                "value": label,
                "session_type": acquisition,
                "task_condition": task,
                "sample": start,
                "trial_index": i,
            }
        )
    pd.DataFrame(rows).to_csv(eeg_dir / f"{prefix}_events.tsv", sep="\t", index=False)
    return set_path


def test_parse_bids_eeg_stem():
    entities = parse_bids_eeg_stem(
        "sub-3_ses-20230524_task-minimallyovert_acq-calibration_run-01_eeg"
    )
    assert entities["sub"] == "sub-3"
    assert entities["ses"] == "ses-20230524"
    assert entities["task"] == "minimallyovert"
    assert entities["acq"] == "calibration"
    assert entities["run"] == "01"


def test_trigger_epoch_identity():
    labels = [0, 4]
    data, starts = _make_continuous(2, labels)
    epochs = extract_epochs_from_trigger(data, n_events=2, keep="last")
    assert len(epochs) == 2
    assert epochs[0].shape == (N_CH_TOTAL, EPOCH_SAMPLES)
    assert epochs[0][0, 0] == pytest.approx(1.0)
    assert epochs[1][0, 0] == pytest.approx(2.0)
    assert starts[0] == 0


def test_extract_synthetic_edf_bids(tmp_path: Path):
    bids_root = tmp_path / "bids"
    labels = [0, 2, 4]
    edf_path = _write_edf_bids_run(bids_root, n_trials=3, labels=labels)

    run = extract_run(edf_path)
    assert run.subject_id == "sub-1"
    assert run.task == "overt"
    assert run.acquisition == "calibration"
    assert run.source_format == "edf"
    assert run.n_trials == 3
    assert run.epochs.shape == (3, N_CH_TOTAL, EPOCH_SAMPLES)
    assert run.labels.tolist() == labels
    # Channel 0 holds 1-based trial index in the synthetic generator.
    assert run.epochs[0, 0, 0] == pytest.approx(1.0, abs=1e-3)
    assert run.epochs[2, 0, 0] == pytest.approx(3.0, abs=1e-3)

    out_root = tmp_path / "derived"
    run_dir = write_extracted_run(run, out_root)
    assert (run_dir / "labels.csv").is_file()
    assert (run_dir / "run.json").is_file()
    assert len(list((run_dir / "trials").glob("*.npy"))) == 3
    # Public outputs must use BIDS IDs, never subjectN directory names.
    assert "subject" not in str(run_dir).replace(str(tmp_path), "")
    assert run_dir.parts[-3] == "sub-1"


def test_extract_synthetic_eeglab_bids_with_sample_column(tmp_path: Path):
    bids_root = tmp_path / "bids"
    labels = [1, 3, 0]
    set_path = _write_eeglab_bids_run(bids_root, n_trials=3, labels=labels)

    run = extract_run(set_path)
    assert run.subject_id == "sub-7"
    assert run.source_format == "eeglab"
    assert run.n_trials == 3
    assert run.epochs.shape == (3, N_CH_TOTAL, EPOCH_SAMPLES)
    assert run.labels.tolist() == labels
    assert run.epochs[1, 0, 0] == pytest.approx(2.0, abs=1e-6)


def test_extract_dataset_manifest(tmp_path: Path):
    bids_root = tmp_path / "bids"
    _write_edf_bids_run(bids_root, subject_id="sub-1", n_trials=2, labels=[0, 1])
    _write_eeglab_bids_run(bids_root, subject_id="sub-7", n_trials=2, labels=[2, 3])
    out_root = tmp_path / "derived"

    manifest = extract_dataset(bids_root, out_root, subject_ids=["sub-1", "sub-7"])
    assert manifest["n_runs"] == 2
    assert (out_root / "manifest.json").is_file()
    subject_ids = {r["subject_id"] for r in manifest["runs"]}
    assert subject_ids == {"sub-1", "sub-7"}
    for run in manifest["runs"]:
        assert run["subject_id"].startswith("sub-")
        assert not run["subject_id"].startswith("subject")


def test_paths_example_loads(tmp_path: Path):
    example = Path("configs/paths.yaml.example")
    cfg = load_paths(example)
    assert "bids_root" in cfg
    assert "output_root" in cfg
    resolved = resolve_path(cfg["bids_root"], root=tmp_path)
    assert resolved.is_absolute()
    assert str(resolved).endswith(str(Path("data") / "ds007591"))
