#!/usr/bin/env python3
"""Extract per-trial npy files from BIDS EDF dataset.

This script reads the BIDS-formatted EEG dataset in data/ and creates
the per-trial npy files, word list CSVs, and metadata.json that the
existing preprocessing and training pipelines expect.

Usage:
    uv run python bids/extract_from_bids.py
"""

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# ============================================================
# Constants
# ============================================================
SFREQ = 256
N_CH_TOTAL = 139
N_CH_EEG = 128
EPOCH_SAMPLES = 2880  # samples per trial (6.25 sec * 256 Hz + margin)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIDS_ROOT = PROJECT_ROOT / "data"

# BIDS subject ID → old subject directory name
SUBJECT_MAP = {
    "sub-1": "subject1",
    "sub-2": "subject2",
    "sub-3": "subject3",
}

# BIDS task label → task string in metadata
TASK_MAP = {
    "overt": "overt",
    "minimallyovert": "minimally overt",
    "covert": "covert",
}

# Original calibration ordering: (sub_id, date) → [task_bids, ...] in calibrated index order.
# This preserves the original experiment ordering from the NAS metadata.
CALIBRATION_ORDER = {
    ("sub-1", "20230511"): ["minimallyovert"],
    ("sub-1", "20230529"): ["overt", "covert"],
    ("sub-2", "20230512"): ["minimallyovert", "overt"],
    ("sub-2", "20230516"): ["covert"],
    ("sub-3", "20230523"): ["overt", "minimallyovert"],
    ("sub-3", "20230524"): ["covert", "minimallyovert"],
}


def extract_epochs_from_trigger(data, events_df):
    """Extract epochs using trigger channel for H5-based (continuous) recordings.

    Args:
        data: (n_ch, n_samples) array in raw ADC units
        events_df: DataFrame from events.tsv

    Returns:
        List of (n_ch, EPOCH_SAMPLES) arrays
    """
    trigger = data[N_CH_TOTAL - 1]
    onsets = np.where(np.diff(trigger) > 0.5)[0] + 1
    n_events = len(events_df)

    if len(onsets) > n_events:
        onsets = onsets[-n_events:]
    elif len(onsets) < n_events:
        print(f"  WARNING: fewer triggers ({len(onsets)}) than events ({n_events})")
        n_events = len(onsets)

    epochs = []
    for i in range(n_events):
        onset_sample = onsets[i]
        start = (onset_sample // 8 - 359) * 8
        end = start + EPOCH_SAMPLES
        if start < 0 or end > data.shape[1]:
            print(f"  WARNING: epoch {i} out of bounds (start={start}, end={end}), skipping")
            continue
        epoch = data[:, start:end].copy()
        epochs.append(epoch)
    return epochs


def extract_epochs_concatenated(data, events_df):
    """Extract epochs from concatenated npy-fallback recordings.

    For npy-based sessions, trials are concatenated at fixed intervals.

    Args:
        data: (n_ch, n_samples) array in raw ADC units
        events_df: DataFrame from events.tsv

    Returns:
        List of (n_ch, EPOCH_SAMPLES) arrays
    """
    n_trials = len(events_df)
    epochs = []
    for i in range(n_trials):
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > data.shape[1]:
            print(f"  WARNING: epoch {i} exceeds data length, skipping")
            break
        epoch = data[:, start:end].copy()
        epochs.append(epoch)
    return epochs


def is_npy_fallback(data):
    """Check if recording has no real triggers (npy fallback)."""
    trigger = data[N_CH_TOTAL - 1]
    onsets = np.where(np.diff(trigger) > 0.5)[0] + 1
    return len(onsets) == 0


def load_edf_as_raw_adc(edf_path):
    """Load EDF and convert back to raw ADC units.

    EDF stores data in Volts. We undo the unit_coeff conversion
    to get back to the raw ADC values that the pipeline expects.

    Returns:
        (n_ch, n_samples) array in raw ADC units
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    data = raw.get_data()  # (n_ch, n_samples) in Volts
    # Convert non-trigger channels back to raw ADC units
    data[:N_CH_TOTAL - 1] /= 1.0e-6
    return data


def process_subject(sub_id):
    """Process all sessions for a subject."""
    old_name = SUBJECT_MAP[sub_id]
    sub_dir = BIDS_ROOT / sub_id
    if not sub_dir.exists():
        print(f"Subject directory {sub_dir} not found, skipping")
        return

    sessions = sorted([d.name for d in sub_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")])

    for ses in sessions:
        date = ses.replace("ses-", "")
        eeg_dir = sub_dir / ses / "eeg"
        if not eeg_dir.exists():
            continue

        # Group runs by (task, acq) to determine calibrated/online mapping
        metadata = {}
        calibration_runs = {}  # task → list of (run_label, edf_path, events_path)
        online_runs = {}

        edf_files = sorted(eeg_dir.glob("*_eeg.edf"))
        for edf_path in edf_files:
            stem = edf_path.stem.replace("_eeg", "")
            parts = {}
            for token in stem.split("_"):
                if "-" in token:
                    key, val = token.split("-", 1)
                    parts[key] = val

            task_bids = parts.get("task", "")
            acq = parts.get("acq", "")
            run = parts.get("run", "01")
            events_path = edf_path.parent / edf_path.name.replace("_eeg.edf", "_events.tsv")

            if not events_path.exists():
                continue

            entry = (run, edf_path, events_path, task_bids)
            if acq == "calibration":
                calibration_runs.setdefault(task_bids, []).append(entry)
            elif acq == "online":
                online_runs.setdefault(task_bids, []).append(entry)

        # Order calibration runs using original experiment ordering
        calib_order = CALIBRATION_ORDER.get((sub_id, date))
        if calib_order:
            all_calib = []
            for task_bids in calib_order:
                for entry in sorted(calibration_runs.get(task_bids, [])):
                    all_calib.append(entry)
        else:
            # Fallback: sort by task name
            all_calib = []
            for task_bids in sorted(calibration_runs.keys()):
                for entry in sorted(calibration_runs[task_bids]):
                    all_calib.append(entry)

        calib_idx = 0
        calib_task_to_idx = {}
        for run_label, edf_path, events_path, task_bids in all_calib:
            calib_idx += 1
            calib_key = f"backup_calibrated_{calib_idx}"
            task_name = TASK_MAP.get(task_bids, task_bids)

            metadata[calib_key] = {"task": task_name}
            calib_task_to_idx[task_bids] = calib_idx

            print(f"  Extracting {sub_id}/{ses} {calib_key} ({task_name})...")
            _extract_run(edf_path, events_path, old_name, date, calib_key)

        # Order online runs using same task ordering as calibration
        if calib_order:
            all_online = []
            for task_bids in calib_order:
                for entry in sorted(online_runs.get(task_bids, [])):
                    all_online.append(entry)
        else:
            all_online = []
            for task_bids in sorted(online_runs.keys()):
                for entry in sorted(online_runs[task_bids]):
                    all_online.append(entry)

        online_idx = 0
        for run_label, edf_path, events_path, task_bids in all_online:
            online_idx += 1
            online_key = f"backup_online_{online_idx}"
            task_name = TASK_MAP.get(task_bids, task_bids)
            calib_ref_idx = calib_task_to_idx.get(task_bids, 1)

            metadata[online_key] = {
                "task": task_name,
                "calibrated with": f"backup_calibrated_{calib_ref_idx}",
            }

            print(f"  Extracting {sub_id}/{ses} {online_key} ({task_name})...")
            _extract_run(edf_path, events_path, old_name, date, online_key)

        # Write metadata.json
        if metadata:
            meta_dir = BIDS_ROOT / old_name / date
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"  Wrote {meta_path}")


def _extract_run(edf_path, events_path, old_name, date, run_key):
    """Extract epochs from a single BIDS run and save as npy files."""
    data = load_edf_as_raw_adc(edf_path)
    events_df = pd.read_csv(str(events_path), sep="\t")

    # Determine extraction method
    if is_npy_fallback(data):
        epochs = extract_epochs_concatenated(data, events_df)
    else:
        epochs = extract_epochs_from_trigger(data, events_df)

    if len(epochs) == 0:
        print(f"    WARNING: no epochs extracted")
        return

    # Save per-trial npy files (clean directory first to remove stale files)
    npy_dir = BIDS_ROOT / old_name / date / "eeg_margin_before_preproc" / f"{date}_{run_key}"
    if npy_dir.exists():
        for old_file in npy_dir.glob("*.npy"):
            old_file.unlink()
    npy_dir.mkdir(parents=True, exist_ok=True)
    for i, epoch in enumerate(epochs):
        npy_path = npy_dir / f"{i}.npy"
        np.save(str(npy_path), epoch)

    # Save word list CSV
    labels = events_df["value"].values[:len(epochs)]
    csv_path = BIDS_ROOT / old_name / date / f"word_list_{run_key}.csv"
    np.savetxt(str(csv_path), labels, delimiter=",", fmt="%d")

    print(f"    Saved {len(epochs)} epochs to {npy_dir}")
    print(f"    Saved word list to {csv_path}")


def main():
    print("Extracting per-trial data from BIDS dataset...")
    print(f"BIDS root: {BIDS_ROOT}")
    print()

    for sub_id in sorted(SUBJECT_MAP.keys()):
        print(f"Processing {sub_id} ({SUBJECT_MAP[sub_id]})...")
        process_subject(sub_id)
        print()

    print("Done! The extracted data is ready for the preprocessing and training pipelines.")
    print()
    print("Next steps:")
    print("  1. Save preprocessed EEG/EMG:")
    print("     uv run python plot_figures/make_preproc_files.py")
    print("  2. Train decoders:")
    print("     uv run python uhd_eeg/trainers/trainer.py -m hydra/launcher=joblib parallel_sets=subject1-1")


if __name__ == "__main__":
    main()
