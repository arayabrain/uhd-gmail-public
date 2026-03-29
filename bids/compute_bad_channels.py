#!/usr/bin/env python3
"""Detect bad EEG channels for the gmail BIDS dataset using PyPREP.

For each run (H5 or npy-based), loads the continuous EEG data (128 channels),
runs PyPREP NoisyChannels detection (nan/flat, deviation, HF noise, correlation),
and updates the BIDS channels.tsv with bad channel status.

Also saves per-run bad_channels.json reports alongside the BIDS data.

Usage:
    uv run python bids/compute_bad_channels.py                    # default: BIDS_ROOT from .env
    uv run python bids/compute_bad_channels.py --bids-root data   # for data/ (EDF)
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import mne
import numpy as np
import pandas as pd
from natsort import natsorted
from pyprep.find_noisy_channels import NoisyChannels

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
SFREQ = 256
N_CH_EEG = 128
N_CH_TOTAL = 139
UNIT_COEFF = 1.0e-6
PREAMP_GAIN = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from uhd_eeg.env import get_bids_root, get_raw_root, load_participant_mapping as _load_mapping

RAW_ROOT = get_raw_root()


# ============================================================
# Data classes
# ============================================================
@dataclass
class BadChannelResult:
    file_name: str
    bad_by_nan: List[int] = field(default_factory=list)
    bad_by_flat: List[int] = field(default_factory=list)
    bad_by_deviation: List[int] = field(default_factory=list)
    bad_by_hf_noise: List[int] = field(default_factory=list)
    bad_by_correlation: List[int] = field(default_factory=list)
    bad_all: List[int] = field(default_factory=list)
    n_channels: int = N_CH_EEG
    sfreq: int = SFREQ


@dataclass
class RunBadChannels:
    bids_id: str
    session: str
    run_prefix: str
    result: BadChannelResult


# ============================================================
# Participant mapping
# ============================================================
def load_participant_mapping() -> Dict[str, str]:
    return _load_mapping()


# ============================================================
# EEG loading
# ============================================================
def load_h5_eeg_only(h5_paths: List[Path]) -> np.ndarray:
    """Load continuous EEG data (128 channels) from H5 files, in Volts."""
    segments = []
    for path in h5_paths:
        with h5py.File(str(path), "r") as f:
            data = f["EEG"]["EEG"][:].reshape(-1, N_CH_TOTAL)  # (n_samples, n_ch)
            segments.append(data[:, :N_CH_EEG])  # EEG channels only
    continuous = np.concatenate(segments, axis=0)
    # Convert to Volts with preamp gain correction
    continuous = continuous * UNIT_COEFF / PREAMP_GAIN
    return continuous.T  # (128, n_samples)


def load_npy_eeg_only(npy_dir: Path) -> np.ndarray:
    """Load concatenated trial npy files (128 EEG channels), in Volts."""
    npy_files = natsorted(npy_dir.glob("*.npy"))
    trials = []
    for npy_file in npy_files:
        trial = np.load(str(npy_file))[:N_CH_EEG]  # (128, 2880)
        trials.append(trial)
    continuous = np.concatenate(trials, axis=1)
    continuous = continuous * UNIT_COEFF / PREAMP_GAIN
    return continuous


# ============================================================
# Bad channel detection
# ============================================================
def detect_bad_channels(data: np.ndarray, sfreq: int = SFREQ) -> BadChannelResult:
    """Run PyPREP bad channel detection on EEG data.

    Args:
        data: EEG data (n_channels, n_times) in Volts.
        sfreq: Sampling frequency.

    Returns:
        BadChannelResult with per-criterion bad channel indices (0-indexed).
    """
    n_channels = data.shape[0]
    ch_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    nc = NoisyChannels(raw, do_detrend=True, random_state=42)
    nc.find_bad_by_nan_flat()
    nc.find_bad_by_deviation()
    nc.find_bad_by_hfnoise()
    nc.find_bad_by_correlation()

    def _names_to_indices(names: List[str]) -> List[int]:
        return sorted([ch_names.index(n) for n in names if n in ch_names])

    bad_nan = _names_to_indices(nc.bad_by_nan)
    bad_flat = _names_to_indices(nc.bad_by_flat)
    bad_dev = _names_to_indices(nc.bad_by_deviation)
    bad_hf = _names_to_indices(nc.bad_by_hf_noise)
    bad_corr = _names_to_indices(nc.bad_by_correlation)
    bad_all = sorted(set(bad_nan + bad_flat + bad_dev + bad_hf + bad_corr))

    return BadChannelResult(
        file_name="",
        bad_by_nan=bad_nan,
        bad_by_flat=bad_flat,
        bad_by_deviation=bad_dev,
        bad_by_hf_noise=bad_hf,
        bad_by_correlation=bad_corr,
        bad_all=bad_all,
        n_channels=n_channels,
        sfreq=sfreq,
    )


# ============================================================
# BIDS integration
# ============================================================
def update_channels_tsv(channels_tsv_path: Path, bad_indices: List[int]) -> None:
    """Update channels.tsv status column with bad channel info."""
    df = pd.read_csv(channels_tsv_path, sep="\t")
    statuses = []
    for i in range(len(df)):
        if df.iloc[i]["type"] == "EEG" and i in bad_indices:
            statuses.append("bad")
        else:
            statuses.append(df.iloc[i].get("status", "good"))
    df["status"] = statuses
    df.to_csv(channels_tsv_path, sep="\t", index=False)


def save_bad_channels_json(result: BadChannelResult, output_path: Path) -> None:
    """Save bad channel detection result as JSON."""
    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)


# ============================================================
# Run discovery
# ============================================================
TASK_BIDS = {
    "overt": "overt",
    "minimally overt": "minimallyovert",
    "covert": "covert",
}


def _resolve_h5(name: str, date: str, calib_key: str) -> Tuple[List[Path], bool]:
    base_dir = RAW_ROOT / name / date
    h5_path = base_dir / f"EEG_{calib_key}.h5"
    if h5_path.exists():
        return [h5_path], False
    split_h5s = sorted(base_dir.glob(f"EEG_{calib_key}_*.h5"))
    if split_h5s:
        return split_h5s, False
    return [], True


def discover_runs(bids_root: Path) -> List[dict]:
    """Discover all BIDS runs and their data sources."""
    from collections import defaultdict
    from omegaconf import OmegaConf

    name_to_bids = load_participant_mapping()
    config_path = PROJECT_ROOT / "configs" / "trainer" / "config_color_within_offline_split.yaml"
    config = OmegaConf.load(str(config_path))

    # Parse config
    entries = []
    for key in config:
        if not isinstance(key, str) or "-" not in key:
            continue
        parts = key.split("-")
        if not parts[0].startswith("subject"):
            continue
        entry = config[key]
        if not OmegaConf.is_dict(entry) or "npy_dir" not in entry:
            continue
        npy_dir = str(entry["npy_dir"])
        path_parts = npy_dir.split("/")
        entries.append((parts[0], path_parts[2], path_parts[-1].split("_", 1)[1]))

    # Match codes to names
    code_dates = defaultdict(set)
    for code, date, _ in entries:
        code_dates[code].add(date)

    code_to_name = {}
    for code, dates in code_dates.items():
        for name in name_to_bids:
            name_dir = RAW_ROOT / name
            if not name_dir.exists():
                continue
            available = {d.name for d in name_dir.iterdir() if d.is_dir() and d.name.isdigit()}
            if dates.issubset(available):
                code_to_name[code] = name
                break

    runs = []
    for code, date, calib_name in entries:
        name = code_to_name[code]
        bids_id = name_to_bids[name]

        # Load metadata for task
        meta_path = RAW_ROOT / name / date / "metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        if calib_name in metadata:
            task = metadata[calib_name]["task"]
        else:
            split_keys = sorted(k for k in metadata if k.startswith(f"{calib_name}_") and k != "subject")
            task = metadata[split_keys[0]]["task"] if split_keys else None
        if task is None:
            continue

        task_bids = TASK_BIDS[task]
        h5_paths, use_npy = _resolve_h5(name, date, calib_name)
        npy_dir = RAW_ROOT / name / date / "eeg_margin_before_preproc" / f"{date}_{calib_name}"

        # Calibration run
        runs.append({
            "bids_id": bids_id, "name": name, "date": date,
            "task_bids": task_bids, "acq": "calibration",
            "h5_paths": h5_paths, "use_npy": use_npy, "npy_dir": npy_dir,
            "calib_name": calib_name,
        })

        # Online runs
        for exp_name, exp_meta in metadata.items():
            if exp_name == "subject" or not isinstance(exp_meta, dict):
                continue
            calib_with = exp_meta.get("calibrated with", "")
            if calib_name not in calib_with or exp_name.startswith("backup_calibrated"):
                continue
            online_task = exp_meta["task"]
            online_h5 = RAW_ROOT / name / date / f"EEG_{exp_name}.h5"
            online_npy = RAW_ROOT / name / date / "eeg_margin_before_preproc" / f"{date}_{exp_name}"
            online_use_npy = not online_h5.exists()
            runs.append({
                "bids_id": bids_id, "name": name, "date": date,
                "task_bids": TASK_BIDS[online_task], "acq": "online",
                "h5_paths": [online_h5] if not online_use_npy else [],
                "use_npy": online_use_npy, "npy_dir": online_npy,
                "calib_name": exp_name,
            })

    return runs


def find_bids_files(bids_root: Path, bids_id: str, date: str, task_bids: str, acq: str):
    """Find channels.tsv files matching the run."""
    ses_dir = bids_root / bids_id / f"ses-{date}" / "eeg"
    channels_files = sorted(ses_dir.glob(f"*_task-{task_bids}_acq-{acq}_*_channels.tsv"))
    return channels_files


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Detect bad EEG channels for gmail BIDS")
    parser.add_argument("--bids-root", type=Path, default=get_bids_root())
    args = parser.parse_args()
    bids_root = args.bids_root

    logger.info("BIDS root: %s", bids_root)
    logger.info("Discovering runs...")
    runs = discover_runs(bids_root)
    logger.info("Found %d runs", len(runs))

    # Assign run indices (same logic as convert_to_bids.py)
    from collections import defaultdict
    counter = defaultdict(int)
    for run in runs:
        key = (run["bids_id"], run["date"], run["task_bids"], run["acq"])
        counter[key] += 1
        run["run_idx"] = counter[key]

    all_results = []
    for run in runs:
        bids_id = run["bids_id"]
        date = run["date"]
        task_bids = run["task_bids"]
        acq = run["acq"]
        run_idx = run["run_idx"]
        prefix = f"{bids_id}/ses-{date} task-{task_bids} acq-{acq} run-{run_idx:02d}"

        logger.info("Processing %s...", prefix)

        try:
            if run["use_npy"] and run["npy_dir"].exists():
                eeg_data = load_npy_eeg_only(run["npy_dir"])
                source = "npy"
            elif run["h5_paths"]:
                eeg_data = load_h5_eeg_only(run["h5_paths"])
                source = "h5"
            else:
                logger.warning("  No data source for %s, skipping", prefix)
                continue
        except Exception as e:
            logger.error("  Failed to load data for %s: %s", prefix, e)
            continue

        logger.info("  Data shape: %s, source: %s", eeg_data.shape, source)

        result = detect_bad_channels(eeg_data)
        result.file_name = f"{run['calib_name']} ({source})"

        logger.info(
            "  Bad channels (%d): %s",
            len(result.bad_all),
            result.bad_all if result.bad_all else "none",
        )
        if result.bad_by_flat:
            logger.info("    flat: %s", result.bad_by_flat)
        if result.bad_by_deviation:
            logger.info("    deviation: %s", result.bad_by_deviation)
        if result.bad_by_hf_noise:
            logger.info("    HF noise: %s", result.bad_by_hf_noise)
        if result.bad_by_correlation:
            logger.info("    correlation: %s", result.bad_by_correlation)

        # Update BIDS channels.tsv
        bids_prefix = (
            f"{bids_id}_ses-{date}_task-{task_bids}_acq-{acq}_run-{run_idx:02d}"
        )
        channels_path = bids_root / bids_id / f"ses-{date}" / "eeg" / f"{bids_prefix}_channels.tsv"
        if channels_path.exists():
            update_channels_tsv(channels_path, result.bad_all)
            logger.info("  Updated %s", channels_path.name)
        else:
            logger.warning("  channels.tsv not found: %s", channels_path)

        # Save JSON report
        json_path = bids_root / bids_id / f"ses-{date}" / "eeg" / f"{bids_prefix}_bad_channels.json"
        save_bad_channels_json(result, json_path)

        all_results.append(result)

    # Summary
    all_bad = set()
    for r in all_results:
        all_bad.update(r.bad_all)
    logger.info(
        "\nDone! Processed %d runs. Unique bad channels across all runs: %d — %s",
        len(all_results),
        len(all_bad),
        sorted(all_bad) if all_bad else "none",
    )


if __name__ == "__main__":
    main()
