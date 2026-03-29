#!/usr/bin/env python3
"""Convert raw EEG data to BIDS format for OpenNeuro publication.

Usage:
    uv run python convert_to_bids.py                    # BrainVision (default, fp32)
    uv run python convert_to_bids.py --format edf       # EDF (16-bit)
    uv run python convert_to_bids.py --output /path/to  # Custom output directory
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import mne
import numpy as np
import pandas as pd
from natsort import natsorted
from omegaconf import OmegaConf

# ============================================================
# Constants
# ============================================================
SFREQ = 256
N_CH_TOTAL = 139
N_CH_EEG = 128
TRIAL_DURATION_SEC = 6.25  # 5 repetitions × 1.25 sec
UNIT_COEFF = 1.0e-6  # Raw ADC → Volts

WORD_LABELS = {0: "green", 1: "magenta", 2: "orange", 3: "violet", 4: "yellow"}
TASK_BIDS = {
    "overt": "overt",
    "minimally overt": "minimallyovert",
    "covert": "covert",
}

from uhd_eeg.env import get_raw_root, get_bids_root, load_participant_mapping as _load_participant_mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = get_raw_root()
DEFAULT_BIDS_ROOT = get_bids_root()
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "trainer"
    / "config_color_within_offline_split.yaml"
)
COORDS_PATH = (
    PROJECT_ROOT / "plot_figures" / "coordinates_colorless.npy"
)


# ============================================================
# Channel definitions
# ============================================================
def get_channel_names_and_types() -> Tuple[List[str], List[str]]:
    ch_names = [f"EEG{i + 1:03d}" for i in range(128)]
    ch_names += [
        "DISPLAY+", "DISPLAY-",
        "MIC+", "MIC-",
        "EOG+", "EOG-",
        "EMG_UPPER_OOris+", "EMG_UPPER_OOris-",
        "EMG_LOWER_OOris+", "EMG_LOWER_OOris-",
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


# ============================================================
# Data loading
# ============================================================
def load_participant_mapping() -> Dict[str, str]:
    """Load name→BIDS-ID mapping from path in .env"""
    return _load_participant_mapping()


def _export_raw_eeglab_fp64(raw: mne.io.RawArray, fpath: str) -> None:
    """Export MNE Raw to EEGLAB .set format preserving float64 precision.

    MNE's built-in EEGLAB export uses float32 via eeglabio.
    This function writes the .set file directly with scipy.io.savemat
    to preserve full float64 precision.
    """
    import scipy.io

    data_v = raw.get_data()  # (n_ch, n_samp) in Volts
    data_uv = data_v * 1e6  # EEGLAB convention: microvolts
    # Keep stim channel in original units (not microvolts)
    ch_types = raw.get_channel_types()
    for i, ct in enumerate(ch_types):
        if ct == "stim":
            data_uv[i] = data_v[i]  # trigger is 0/1, no scaling

    ch_names = raw.ch_names
    n_ch, n_samp = data_uv.shape

    chanlocs = np.zeros(n_ch, dtype=[("labels", "U64")])
    for i, name in enumerate(ch_names):
        chanlocs[i] = (name,)

    mat_dict = {
        "data": data_uv,
        "setname": "",
        "nbchan": np.float64(n_ch),
        "pnts": np.float64(n_samp),
        "trials": np.float64(1),
        "srate": np.float64(raw.info["sfreq"]),
        "xmin": np.float64(0),
        "xmax": np.float64((n_samp - 1) / raw.info["sfreq"]),
        "ref": "",
        "chanlocs": chanlocs,
        "icawinv": np.array([]),
        "icasphere": np.array([]),
        "icaweights": np.array([]),
    }
    scipy.io.savemat(fpath, mat_dict, do_compression=False)


def load_h5_continuous(h5_paths: List[Path]) -> np.ndarray:
    """Load and concatenate H5 files into continuous data.
    Returns: (n_channels, n_samples) array in volts.
    """
    segments = []
    for path in h5_paths:
        with h5py.File(str(path), "r") as f:
            data = f["EEG"]["EEG"][:].reshape(-1, N_CH_TOTAL)  # (n_samples, n_ch)
            segments.append(data)
    continuous = np.concatenate(segments, axis=0)  # (n_samples, n_ch)
    # Apply unit conversion (raw → Volts) for all channels except trigger
    continuous[:, :N_CH_TOTAL - 1] *= UNIT_COEFF
    return continuous.T  # (n_ch, n_samples)


def load_npy_concatenated(npy_dir: Path) -> np.ndarray:
    """Load and concatenate trial npy files as fallback.
    Returns: (n_channels, n_samples) array in volts.
    """
    npy_files = natsorted(npy_dir.glob("*.npy"))
    trials = []
    for npy_file in npy_files:
        trial = np.load(str(npy_file))  # (139, 2880)
        trials.append(trial)
    continuous = np.concatenate(trials, axis=1)  # (139, n_trials * 2880)
    # Apply unit conversion for all channels except trigger
    continuous[:N_CH_TOTAL - 1] *= UNIT_COEFF
    return continuous


def extract_events_from_trigger(
    data: np.ndarray, word_list: np.ndarray
) -> np.ndarray:
    """Extract events from trigger channel (ch 138).
    data: (n_ch, n_samples)
    Returns: MNE events array (n_events, 3)
    """
    trigger = data[N_CH_TOTAL - 1]  # channel 138
    onsets = np.where(np.diff(trigger) > 0.5)[0] + 1
    n_events = min(len(onsets), len(word_list))
    if len(onsets) > len(word_list):
        # Extra triggers at start are likely warmup; use last N
        onsets = onsets[-n_events:]
    word_list = word_list[:n_events]
    events = np.column_stack(
        [onsets[:n_events], np.zeros(n_events, dtype=int), word_list.astype(int)]
    )
    return events


def create_events_from_npy(
    npy_dir: Path, word_list: np.ndarray, samples_per_trial: int = 2880
) -> np.ndarray:
    """Create synthetic events for concatenated npy data."""
    n_trials = len(word_list)
    onsets = np.arange(n_trials) * samples_per_trial
    events = np.column_stack(
        [onsets, np.zeros(n_trials, dtype=int), word_list.astype(int)]
    )
    return events


# ============================================================
# Session discovery
# ============================================================
def parse_config_sessions() -> Dict[str, List[Tuple[str, int]]]:
    """Parse config to get (subject_code → [(date, sub_idx), ...])."""
    config = OmegaConf.load(str(CONFIG_PATH))
    sessions: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for key in config:
        if not isinstance(key, str) or "-" not in key:
            continue
        parts = key.split("-")
        if not parts[0].startswith("subject"):
            continue
        subject_code = parts[0]
        entry = config[key]
        if not OmegaConf.is_dict(entry) or "npy_dir" not in entry:
            continue
        npy_dir = str(entry["npy_dir"])
        path_parts = npy_dir.split("/")
        date = path_parts[2]
        calibrated_name = path_parts[-1]
        sub_idx = int(calibrated_name.rsplit("_", 1)[1])
        sessions[subject_code].append((date, sub_idx))
    return dict(sessions)


def match_codes_to_names(
    sessions: Dict[str, List], name_to_bids: Dict[str, str]
) -> Dict[str, str]:
    """Match subject codes to names by checking NAS date directories."""
    code_to_name = {}
    for code, date_list in sessions.items():
        dates_needed = {d for d, _ in date_list}
        for name in name_to_bids:
            name_dir = RAW_ROOT / name
            if not name_dir.exists():
                continue
            available = {
                d.name for d in name_dir.iterdir() if d.is_dir() and d.name.isdigit()
            }
            if dates_needed.issubset(available):
                code_to_name[code] = name
                break
    return code_to_name


def build_run_list(
    sessions: Dict[str, List[Tuple[str, int]]],
    code_to_name: Dict[str, str],
    name_to_bids: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Build complete list of runs including online runs."""
    runs: List[Dict[str, Any]] = []

    for code in sorted(sessions.keys()):
        name = code_to_name[code]
        bids_id = name_to_bids[name]

        # Group by date to avoid loading metadata multiple times
        by_date: Dict[str, List[int]] = defaultdict(list)
        for date, sub_idx in sessions[code]:
            by_date[date].append(sub_idx)

        for date, sub_idxs in sorted(by_date.items()):
            meta_path = RAW_ROOT / name / date / "metadata.json"
            with open(meta_path) as f:
                metadata = json.load(f)

            for sub_idx in sorted(sub_idxs):
                calib_key = f"backup_calibrated_{sub_idx}"

                # Determine task from metadata
                if calib_key in metadata:
                    calib_task = metadata[calib_key]["task"]
                else:
                    # Split keys (e.g., backup_calibrated_2_1, backup_calibrated_2_2)
                    split_keys = sorted(
                        k
                        for k in metadata
                        if k.startswith(f"{calib_key}_") and k != "subject"
                    )
                    if split_keys:
                        calib_task = metadata[split_keys[0]]["task"]
                    else:
                        print(f"  WARNING: {calib_key} not found in {meta_path}")
                        continue

                task_bids = TASK_BIDS[calib_task]

                # Resolve H5 files for calibrated run
                h5_paths, use_npy = _resolve_h5(name, date, calib_key)
                csv_path = RAW_ROOT / name / date / f"word_list_{calib_key}.csv"
                npy_dir = (
                    RAW_ROOT
                    / name
                    / date
                    / "eeg_margin_before_preproc"
                    / f"{date}_{calib_key}"
                )

                runs.append(
                    {
                        "bids_id": bids_id,
                        "date": date,
                        "task": calib_task,
                        "task_bids": task_bids,
                        "session_type": "calibration",
                        "run_name": calib_key,
                        "h5_paths": h5_paths,
                        "csv_path": csv_path,
                        "npy_dir": npy_dir if use_npy else None,
                    }
                )

                # Find online runs calibrated with this session
                for exp_name, exp_meta in metadata.items():
                    if exp_name == "subject" or not isinstance(exp_meta, dict):
                        continue
                    calib_with = exp_meta.get("calibrated with", "")
                    if calib_key not in calib_with:
                        continue
                    if exp_name.startswith("backup_calibrated"):
                        continue

                    online_task = exp_meta["task"]
                    online_h5 = RAW_ROOT / name / date / f"EEG_{exp_name}.h5"
                    online_csv = RAW_ROOT / name / date / f"word_list_{exp_name}.csv"
                    online_npy = (
                        RAW_ROOT
                        / name
                        / date
                        / "eeg_margin_before_preproc"
                        / f"{date}_{exp_name}"
                    )

                    online_use_npy = not online_h5.exists()

                    runs.append(
                        {
                            "bids_id": bids_id,
                            "date": date,
                            "task": online_task,
                            "task_bids": TASK_BIDS[online_task],
                            "session_type": "online",
                            "run_name": exp_name,
                            "h5_paths": [online_h5] if not online_use_npy else [],
                            "csv_path": online_csv,
                            "npy_dir": online_npy if online_use_npy else None,
                        }
                    )

    return runs


def _resolve_h5(
    name: str, date: str, calib_key: str
) -> Tuple[List[Path], bool]:
    """Resolve H5 file paths. Returns (h5_paths, use_npy_fallback)."""
    base_dir = RAW_ROOT / name / date

    # Standard naming
    h5_path = base_dir / f"EEG_{calib_key}.h5"
    if h5_path.exists():
        return [h5_path], False

    # Split files (e.g., backup_calibrated_2_1.h5, backup_calibrated_2_2.h5)
    split_h5s = sorted(base_dir.glob(f"EEG_{calib_key}_*.h5"))
    if split_h5s:
        return split_h5s, False

    # Fallback to npy
    return [], True


# ============================================================
# BIDS file creation
# ============================================================
def create_mne_raw(data: np.ndarray) -> mne.io.RawArray:
    """Create MNE RawArray from data (n_ch, n_samples)."""
    ch_names, ch_types = get_channel_names_and_types()
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def write_events_tsv(
    events: np.ndarray,
    session_type: str,
    task: str,
    output_path: Path,
) -> None:
    """Write BIDS events.tsv file."""
    rows = []
    for i in range(len(events)):
        onset_sample = events[i, 0]
        label_idx = events[i, 2]
        rows.append(
            {
                "onset": onset_sample / SFREQ,
                "duration": TRIAL_DURATION_SEC,
                "trial_type": WORD_LABELS.get(label_idx, f"unknown_{label_idx}"),
                "value": int(label_idx),
                "session_type": session_type,
                "task_condition": task,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)


def write_channels_tsv(output_path: Path) -> None:
    """Write BIDS channels.tsv."""
    ch_names, ch_types = get_channel_names_and_types()
    bids_types = []
    units = []
    for ch_type in ch_types:
        if ch_type == "eeg":
            bids_types.append("EEG")
            units.append("V")
        elif ch_type == "eog":
            bids_types.append("EOG")
            units.append("V")
        elif ch_type == "emg":
            bids_types.append("EMG")
            units.append("V")
        elif ch_type == "stim":
            bids_types.append("TRIG")
            units.append("n/a")
        else:
            bids_types.append("MISC")
            units.append("V")

    df = pd.DataFrame(
        {
            "name": ch_names,
            "type": bids_types,
            "units": units,
            "sampling_frequency": [SFREQ] * len(ch_names),
            "status": ["good"] * len(ch_names),
        }
    )
    df.to_csv(output_path, sep="\t", index=False)


def write_sidecar_json(
    task: str,
    task_bids: str,
    session_type: str,
    n_trials: int,
    output_path: Path,
) -> None:
    """Write BIDS EEG sidecar JSON."""
    sidecar = {
        "TaskName": task_bids,
        "TaskDescription": (
            f"Speech decoding task ({task}). "
            f"Participants produced one of 5 color words (green, magenta, orange, violet, yellow) "
            f"in {task} speech condition. Each trial consists of 5 repetitions of the same word "
            f"(1.25 sec per repetition, 6.25 sec total). "
            f"Session type: {session_type}."
        ),
        "InstitutionName": "Araya Inc.",
        "Manufacturer": "g.tec medical engineering GmbH",
        "ManufacturersModelName": "Pangolin",
        "SamplingFrequency": SFREQ,
        "EEGChannelCount": N_CH_EEG,
        "EOGChannelCount": 2,
        "EMGChannelCount": 4,
        "MiscChannelCount": 4,
        "TriggerChannelCount": 1,
        "EEGReference": "n/a (raw, pre-reference)",
        "EEGGround": "left mastoid",
        "PowerLineFrequency": 50,
        "SoftwareFilters": "n/a",
        "RecordingType": "continuous" if session_type != "npy_fallback" else "epoched",
    }
    with open(output_path, "w") as f:
        json.dump(sidecar, f, indent=4)


def write_electrodes_tsv(output_path: Path) -> None:
    """Write BIDS electrodes.tsv from 3D head coordinates in XML."""
    import xml.etree.ElementTree as ET

    xml_path = Path(__file__).resolve().parent / "electrodes_uhd.xml"
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    electrodes = root.findall("Electrode")

    ch_names = [f"EEG{i + 1:03d}" for i in range(128)]
    rows = []
    for i, e in enumerate(electrodes[:128]):
        head = e.find(".//Positions/Subject/Head").text
        x, y, z = [float(v) for v in head.split(",")]
        rows.append({"name": ch_names[i], "x": round(x, 4), "y": round(y, 4), "z": round(z, 4)})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)


def write_coordsystem_json(output_path: Path) -> None:
    """Write BIDS coordsystem.json."""
    cs = {
        "EEGCoordinateSystem": "Other",
        "EEGCoordinateSystemDescription": (
            "Subject head coordinate system from g.tec electrode digitization "
            "(electrodes_uhd.xml). 3D coordinates in the subject's head space."
        ),
        "EEGCoordinateUnits": "mm",
    }
    with open(output_path, "w") as f:
        json.dump(cs, f, indent=4)


def write_dataset_description(bids_root: Path = DEFAULT_BIDS_ROOT) -> None:
    """Write top-level dataset_description.json."""
    desc = {
        "Name": "Delineating neural contributions to EEG-based speech decoding",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "License": "CC0",
        "Authors": [
            "Motoshige Sato",
            "Yasuo Kabe",
            "Sensho Nobe",
            "Akito Yoshida",
            "Masakazu Inoue",
            "Mayumi Shimizu",
            "Kenichi Tomeoka",
            "Shuntaro Sasai",
        ],
        "Acknowledgements": (
            "We thank Ryota Kanai, for helpful discussions, and Rousslan Fernand "
            "Julien Dossa for data collection. Special thanks to Anna Maria Hadjiev "
            "for her meticulous proofreading, significantly enhancing our manuscript's "
            "quality."
        ),
        "Funding": ["JST, Moonshot R&D Grant Number JPMJMS2012"],
        "ReferencesAndLinks": ["https://doi.org/10.1101/2024.05.09.591996"],
        "HowToAcknowledge": (
            "Please cite the following paper "
            "(doi: https://doi.org/10.1101/2024.05.09.591996) "
            "and this dataset using its OpenNeuro DOI."
        ),
        "EthicsApprovals": [
            "Shiba Palace Clinic Ethics Review Committee",
            "Declaration of Helsinki",
        ],
    }
    output_path = bids_root / "dataset_description.json"
    with open(output_path, "w") as f:
        json.dump(desc, f, indent=4)


def write_participants_tsv(name_to_bids: Dict[str, str], bids_root: Path = DEFAULT_BIDS_ROOT) -> None:
    """Write participants.tsv and participants.json."""
    rows = []
    for name, bids_id in sorted(name_to_bids.items(), key=lambda x: x[1]):
        rows.append({"participant_id": bids_id, "age": "n/a", "sex": "n/a"})
    df = pd.DataFrame(rows)
    df.to_csv(bids_root / "participants.tsv", sep="\t", index=False)

    participants_json = {
        "participant_id": {"Description": "Unique participant identifier"},
        "age": {"Description": "Age of participant", "Units": "years"},
        "sex": {"Description": "Sex of participant", "Levels": {"M": "male", "F": "female"}},
    }
    with open(bids_root / "participants.json", "w") as f:
        json.dump(participants_json, f, indent=4)


def write_readme(bids_root: Path = DEFAULT_BIDS_ROOT) -> None:
    """Write dataset README."""
    readme = (
        "# Delineating neural contributions to EEG-based speech decoding\n\n"
        "## Overview\n"
        "128-channel EEG recordings during speech production tasks.\n"
        "Participants produced one of 5 color words (green, magenta, orange, violet, yellow)\n"
        "under three speech conditions: overt, minimally overt, and covert.\n\n"
        "Each trial consists of 5 repetitions of the same word (1.25 sec per repetition).\n\n"
        "## Channel layout (139 channels total)\n"
        "- Channels 1-128: EEG\n"
        "- Channels 129-130: DISPLAY (bipolar pair, misc)\n"
        "- Channels 131-132: MIC (bipolar pair, misc)\n"
        "- Channels 133-134: EOG (bipolar pair)\n"
        "- Channels 135-136: EMG upper orbicularis oris (bipolar pair)\n"
        "- Channels 137-138: EMG lower orbicularis oris (bipolar pair)\n"
        "- Channel 139: TRIGGER (marks trial onsets)\n\n"
        "## Session types\n"
        "- calibration: Offline data collection for decoder training\n"
        "- online: Real-time decoding with trained decoder\n\n"
        "## Preprocessing note\n"
        "The EEG channels were recorded with a 10x preamp gain.\n"
        "Raw values have been converted to Volts (×1e-6).\n\n"
        "## Code\n"
        "Code for data loading, preprocessing, and decoding models is available at:\n"
        "https://github.com/arayabrain/uhd-gmail-public\n"
    )
    with open(bids_root / "README", "w") as f:
        f.write(readme)


def write_events_json(bids_root: Path = DEFAULT_BIDS_ROOT) -> None:
    """Write top-level events.json describing custom columns in events.tsv."""
    events_desc = {
        "onset": {
            "Description": "Onset time of the trial in seconds from the start of the recording",
            "Units": "s",
        },
        "duration": {
            "Description": "Duration of the trial (5 repetitions x 1.25 sec each)",
            "Units": "s",
        },
        "trial_type": {
            "Description": "Color word spoken in the trial",
            "Levels": {
                "green": "The word 'green'",
                "magenta": "The word 'magenta'",
                "orange": "The word 'orange'",
                "violet": "The word 'violet'",
                "yellow": "The word 'yellow'",
            },
        },
        "value": {
            "Description": "Numeric label index corresponding to the color word (0-4)",
            "Levels": {"0": "green", "1": "magenta", "2": "orange", "3": "violet", "4": "yellow"},
        },
        "session_type": {
            "Description": "Whether data was collected during offline calibration or online decoding",
            "Levels": {
                "calibration": "Offline data collection for decoder training",
                "online": "Real-time decoding session with trained decoder",
            },
        },
        "task_condition": {
            "Description": "Speech production condition",
            "Levels": {
                "overt": "Full voice speech production",
                "minimally overt": "Minimally overt (whispered) speech production",
                "covert": "Silent/imagined speech production",
            },
        },
    }
    with open(bids_root / "events.json", "w") as f:
        json.dump(events_desc, f, indent=4)


# ============================================================
# Main processing
# ============================================================
def process_run(
    run_info: Dict[str, Any],
    bids_root: Path = DEFAULT_BIDS_ROOT,
    fmt: str = "brainvision",
) -> Optional[str]:
    """Process a single run and create BIDS files.
    Returns: warning message if any, None otherwise.
    """
    bids_id = run_info["bids_id"]
    date = run_info["date"]
    task_bids = run_info["task_bids"]
    session_type = run_info["session_type"]
    csv_path = run_info["csv_path"]
    h5_paths = run_info["h5_paths"]
    npy_dir = run_info["npy_dir"]
    run_name = run_info["run_name"]
    task = run_info["task"]

    # Load word list
    if not csv_path.exists():
        return f"Word list not found: {csv_path}"
    word_list = np.loadtxt(str(csv_path), delimiter=",", dtype=int)

    # Load data and extract events
    warning = None
    if npy_dir is not None and npy_dir.exists():
        # NPY fallback
        print(f"    Using npy fallback for {run_name}")
        data = load_npy_concatenated(npy_dir)
        events = create_events_from_npy(npy_dir, word_list)
        recording_type = "epoched_concatenated"
    elif h5_paths:
        print(f"    Loading H5 for {run_name}")
        data = load_h5_continuous(h5_paths)
        events = extract_events_from_trigger(data, word_list)
        recording_type = "continuous"
        if len(events) != len(word_list):
            warning = (
                f"{bids_id}/ses-{date}/{run_name}: "
                f"trigger count ({len(events)}) != word count ({len(word_list)})"
            )
    else:
        return f"No data source found for {bids_id}/ses-{date}/{run_name}"

    # Determine run number within (session, task, acq) group
    acq = session_type  # "calibration" or "online"

    return _write_bids_files(
        bids_id=bids_id,
        date=date,
        task_bids=task_bids,
        task=task,
        acq=acq,
        run_idx=run_info["run_idx"],
        data=data,
        events=events,
        session_type=session_type,
        recording_type=recording_type,
        n_trials=len(word_list),
        warning=warning,
        bids_root=bids_root,
        fmt=fmt,
    )


def _write_bids_files(
    bids_id: str,
    date: str,
    task_bids: str,
    task: str,
    acq: str,
    run_idx: int,
    data: np.ndarray,
    events: np.ndarray,
    session_type: str,
    recording_type: str,
    n_trials: int,
    warning: Optional[str],
    bids_root: Path = DEFAULT_BIDS_ROOT,
    fmt: str = "brainvision",
) -> Optional[str]:
    """Write all BIDS files for a single run."""
    ses = f"ses-{date}"
    run_label = f"run-{run_idx:02d}"
    prefix = f"{bids_id}_{ses}_task-{task_bids}_acq-{acq}_{run_label}"

    eeg_dir = bids_root / bids_id / ses / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)

    # Create MNE Raw and export
    raw = create_mne_raw(data)
    if fmt == "brainvision":
        out_path = eeg_dir / f"{prefix}_eeg.vhdr"
        raw.export(str(out_path), fmt=fmt, overwrite=True, verbose=False)
    elif fmt == "eeglab":
        out_path = eeg_dir / f"{prefix}_eeg.set"
        _export_raw_eeglab_fp64(raw, str(out_path))
    else:
        out_path = eeg_dir / f"{prefix}_eeg.edf"
        raw.export(str(out_path), fmt=fmt, overwrite=True, verbose=False)
    print(f"    Wrote {out_path.relative_to(bids_root)}")

    # Write sidecar files
    write_events_tsv(events, session_type, task, eeg_dir / f"{prefix}_events.tsv")
    write_channels_tsv(eeg_dir / f"{prefix}_channels.tsv")
    write_sidecar_json(task, task_bids, session_type, n_trials, eeg_dir / f"{prefix}_eeg.json")

    elec_path = eeg_dir / f"{bids_id}_{ses}_electrodes.tsv"
    if not elec_path.exists():
        write_electrodes_tsv(elec_path)
    coord_path = eeg_dir / f"{bids_id}_{ses}_coordsystem.json"
    if not coord_path.exists():
        write_coordsystem_json(coord_path)

    return warning


def assign_run_indices(runs: List[Dict[str, Any]]) -> None:
    """Assign run indices within each (bids_id, date, task_bids, acq) group."""
    counter: Dict[Tuple[str, ...], int] = defaultdict(int)
    for run in runs:
        key = (run["bids_id"], run["date"], run["task_bids"], run["session_type"])
        counter[key] += 1
        run["run_idx"] = counter[key]


def main():
    parser = argparse.ArgumentParser(description="Convert raw EEG data to BIDS format")
    parser.add_argument(
        "--format", choices=["brainvision", "edf", "eeglab"], default="brainvision",
        help="Output format: brainvision (fp32, default), edf (16-bit), or eeglab (fp64)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_BIDS_ROOT,
        help=f"Output BIDS directory (default: {DEFAULT_BIDS_ROOT})",
    )
    args = parser.parse_args()
    bids_root = args.output
    fmt = args.format

    print(f"Output: {bids_root}")
    print(f"Format: {fmt}")

    print("Loading participant mapping...")
    name_to_bids = load_participant_mapping()

    print("Parsing config for analysis sessions...")
    sessions = parse_config_sessions()

    print("Matching subject codes to participant names...")
    code_to_name = match_codes_to_names(sessions, name_to_bids)

    for code, name in sorted(code_to_name.items()):
        bids_id = name_to_bids[name]
        print(f"  {code} → {bids_id}")

    print("Building run list...")
    runs = build_run_list(sessions, code_to_name, name_to_bids)
    assign_run_indices(runs)

    print(f"Found {len(runs)} runs to process.\n")

    bids_root.mkdir(parents=True, exist_ok=True)

    warnings = []
    for run in runs:
        bids_id = run["bids_id"]
        date = run["date"]
        task_bids = run["task_bids"]
        acq = run["session_type"]
        run_idx = run["run_idx"]
        print(
            f"  Processing {bids_id}/ses-{date} "
            f"task-{task_bids} acq-{acq} run-{run_idx:02d}..."
        )
        result = process_run(run, bids_root=bids_root, fmt=fmt)
        if result:
            warnings.append(result)

    print("\nWriting top-level BIDS files...")
    write_dataset_description(bids_root)
    write_participants_tsv(name_to_bids, bids_root)
    write_readme(bids_root)
    write_events_json(bids_root)

    bidsignore_path = bids_root / ".bidsignore"
    with open(bidsignore_path, "w") as f:
        f.write(".gitkeep\n")

    print("\nDone!")

    if warnings:
        print("\n=== Warnings ===")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
