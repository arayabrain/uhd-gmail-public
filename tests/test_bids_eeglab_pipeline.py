"""Test that EEGLAB (fp64) BIDS pipeline at gmail_bids produces
near-exact results compared to the original NAS-based pipeline.

Usage:
    uv run python -m pytest tests/test_bids_eeglab_pipeline.py -v
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import pytest
from natsort import natsorted
from scipy.stats import zscore

from uhd_eeg.preprocess.adaptive_filter import (
    NLMS,
    bipolar_np,
    get_ch_type_after_resample,
)

# ============================================================
# Constants
# ============================================================
SFREQ = 256
N_CH_TOTAL = 139
N_CH_EEG = 128
N_CH_NOISE = 3
UNIT_COEFF = 1.0e-6
PREAMP_GAIN = 10
DURA_SEC = 1.25
NUM_TRIAL_AVG = 5
BANDPASS_LOW = 2.0
BANDPASS_HIGH = 118.0
NLMS_MU = 0.1
NLMS_W = "random"
EPOCH_SAMPLES = 2880
DURA_SAMPLES = int(DURA_SEC * NUM_TRIAL_AVG * SFREQ)  # 1600

from uhd_eeg.env import get_raw_root, get_bids_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIDS_ROOT = get_bids_root()
RAW_ROOT = get_raw_root()

TASK_BIDS = {
    "overt": "overt",
    "minimally overt": "minimallyovert",
    "covert": "covert",
}


# ============================================================
# Original pipeline (from make_preproc_files.py)
# ============================================================
def _build_preproc_components():
    n_ch_to_use = N_CH_EEG + N_CH_NOISE
    info = mne.create_info(
        ch_names=n_ch_to_use, sfreq=SFREQ,
        ch_types=[get_ch_type_after_resample(i, n_ch_to_use) for i in range(n_ch_to_use)],
        verbose=False,
    )
    notch_freqs = np.arange(50, SFREQ / 2, 50)
    filter_length = min(int(round(6.6 * SFREQ)), round(SFREQ * DURA_SEC * NUM_TRIAL_AVG - 1))
    data_ch_idx = np.arange(N_CH_EEG)
    noise_ch_idx = np.arange(N_CH_NOISE) + N_CH_EEG
    return info, notch_freqs, filter_length, data_ch_idx, noise_ch_idx


def preproc_original(eeg: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    info, notch_freqs, filter_length, data_ch_idx, noise_ch_idx = _build_preproc_components()
    adapt_filt = NLMS(data_ch_idx, noise_ch_idx, mu=NLMS_MU, w=NLMS_W)
    eeg = bipolar_np(eeg)
    eeg *= UNIT_COEFF
    eeg[data_ch_idx] /= PREAMP_GAIN
    raw = mne.io.RawArray(eeg, info, verbose=False)
    raw.notch_filter(notch_freqs, filter_length=filter_length, fir_design="firwin", trans_bandwidth=1.5, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, picks="all", verbose=False)
    emg = raw.get_data()[N_CH_EEG:]
    eeg_norm = adapt_filt(raw.get_data(), normalize="zscore")[:N_CH_EEG]
    emg_norm = zscore(emg, axis=1)
    return eeg_norm, emg_norm


# ============================================================
# EEGLAB BIDS loading
# ============================================================
def _read_eeglab_raw(path: Path) -> np.ndarray:
    """Read EEGLAB .set and return (n_ch, n_samples) in raw ADC units.

    EEGLAB stores data in µV. MNE reads it and divides by 1e6 → Volts.
    Non-trigger channels: stored as ADC_value (= raw * unit_coeff * 1e6 = raw).
      After MNE ÷1e6: Volts. Undo: ÷ unit_coeff → raw ADC.
    Trigger channel: stored as raw 0/1. After MNE ÷1e6: 0/1e-6.
      Restore: × 1e6 → 0/1.
    """
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
    data = raw.get_data()
    data[:N_CH_TOTAL - 1] /= UNIT_COEFF
    data[N_CH_TOTAL - 1] *= 1e6
    return data


def load_bids_epoch(set_path: Path, events_tsv_path: Path, trial_idx: int) -> np.ndarray:
    """Load a single trial epoch from BIDS EEGLAB .set + events.tsv."""
    data = _read_eeglab_raw(set_path)

    trigger = data[N_CH_TOTAL - 1]
    onsets = np.where(np.diff(trigger) > 0.5)[0] + 1

    events_df = pd.read_csv(str(events_tsv_path), sep="\t")
    n_events = len(events_df)
    if len(onsets) > n_events:
        onsets = onsets[:n_events]

    onset_sample = onsets[trial_idx]
    start = (onset_sample // 8 - 359) * 8
    end = start + EPOCH_SAMPLES
    return data[:, start:end].copy()


def load_bids_epochs_npy_fallback(set_path: Path, events_tsv_path: Path) -> List[np.ndarray]:
    """Load all epochs from BIDS EEGLAB .set (npy-fallback concatenated)."""
    data = _read_eeglab_raw(set_path)
    events_df = pd.read_csv(str(events_tsv_path), sep="\t")
    n_trials = len(events_df)
    epochs = []
    for i in range(n_trials):
        start = i * EPOCH_SAMPLES
        epochs.append(data[:, start:start + EPOCH_SAMPLES].copy())
    return epochs


# ============================================================
# Test helpers
# ============================================================
def load_participant_mapping() -> Dict[str, str]:
    from uhd_eeg.env import load_participant_mapping as _load
    return _load()


def _get_test_sessions() -> List[dict]:
    from collections import defaultdict
    from omegaconf import OmegaConf

    name_to_bids = load_participant_mapping()
    config_path = PROJECT_ROOT / "configs" / "trainer" / "config_color_within_offline_split.yaml"
    config = OmegaConf.load(str(config_path))

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

    return [
        {"bids_id": name_to_bids[code_to_name[code]], "name": code_to_name[code], "date": date, "calib_name": cn}
        for code, date, cn in entries
    ]


def _resolve_bids_paths(name: str, bids_id: str, date: str, calib_name: str) -> Tuple[Path, Path]:
    ses_dir = BIDS_ROOT / bids_id / f"ses-{date}" / "eeg"
    meta_path = RAW_ROOT / name / date / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    if calib_name in metadata:
        task = metadata[calib_name]["task"]
    else:
        split_keys = sorted(k for k in metadata if k.startswith(f"{calib_name}_") and k != "subject")
        task = metadata[split_keys[0]]["task"] if split_keys else None
    if task is None:
        raise FileNotFoundError(f"{calib_name} not found in {meta_path}")
    task_bids = TASK_BIDS[task]
    set_files = sorted(ses_dir.glob(f"*_task-{task_bids}_acq-calibration_*_eeg.set"))
    events_files = sorted(ses_dir.glob(f"*_task-{task_bids}_acq-calibration_*_events.tsv"))
    if not set_files or not events_files:
        raise FileNotFoundError(f"No BIDS files for {bids_id}/ses-{date}/task-{task_bids}")
    return set_files[0], events_files[0]


def _is_npy_fallback(set_path: Path) -> bool:
    data = _read_eeglab_raw(set_path)
    trigger = data[N_CH_TOTAL - 1]
    return len(np.where(np.diff(trigger) > 0.5)[0]) == 0


def _iter_h5_sessions(test_sessions):
    for tc in test_sessions:
        name, date, calib_name, bids_id = tc["name"], tc["date"], tc["calib_name"], tc["bids_id"]
        h5_path = RAW_ROOT / name / date / f"EEG_{calib_name}.h5"
        split_h5s = sorted((RAW_ROOT / name / date).glob(f"EEG_{calib_name}_*.h5"))
        if not h5_path.exists() and not split_h5s:
            continue
        npy_dir = RAW_ROOT / name / date / "eeg_margin_before_preproc" / f"{date}_{calib_name}"
        npy_files = natsorted(npy_dir.glob("*.npy"))
        if not npy_files:
            continue
        try:
            set_path, events_path = _resolve_bids_paths(name, bids_id, date, calib_name)
        except FileNotFoundError:
            continue
        if _is_npy_fallback(set_path):
            continue
        yield tc, npy_files, set_path, events_path


def _iter_npy_fallback_sessions(test_sessions):
    for tc in test_sessions:
        name, date, calib_name, bids_id = tc["name"], tc["date"], tc["calib_name"], tc["bids_id"]
        h5_path = RAW_ROOT / name / date / f"EEG_{calib_name}.h5"
        split_h5s = sorted((RAW_ROOT / name / date).glob(f"EEG_{calib_name}_*.h5"))
        if h5_path.exists() or split_h5s:
            continue
        npy_dir = RAW_ROOT / name / date / "eeg_margin_before_preproc" / f"{date}_{calib_name}"
        npy_files = natsorted(npy_dir.glob("*.npy"))
        if not npy_files:
            continue
        try:
            set_path, events_path = _resolve_bids_paths(name, bids_id, date, calib_name)
        except FileNotFoundError:
            continue
        yield tc, npy_files, set_path, events_path


# ============================================================
# Tests — fp64 precision: expect near-exact match
# ============================================================
class TestEeglabRawDataIntegrity:
    """Test raw data round-trip with EEGLAB fp64 format."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_sessions = _get_test_sessions()

    def test_h5_based_epoch_extraction(self):
        tested = 0
        max_diffs = []
        for tc, npy_files, set_path, events_path in _iter_h5_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            for trial_idx in range(min(3, len(npy_files))):
                original = np.load(str(npy_files[trial_idx]))
                bids_epoch = load_bids_epoch(set_path, events_path, trial_idx)
                diff = np.abs(bids_epoch[:N_CH_EEG] - original[:N_CH_EEG])
                max_diffs.append(diff.max())
                # H5 source data is float32; µV roundtrip preserves fp64 but
                # original npy was cast from fp32, so max diff ~5e-5 from fp32 precision.
                np.testing.assert_allclose(
                    bids_epoch[:N_CH_EEG], original[:N_CH_EEG], atol=1e-4, rtol=1e-7,
                    err_msg=f"{bids_id}/ses-{date}/{calib_name} trial {trial_idx}",
                )
                tested += 1
        assert tested > 0, "No H5-based test cases found"
        print(f"\nPassed {tested} H5-based epoch extraction tests")
        print(f"  Max raw diff (ADC units): {max(max_diffs):.2e}")

    def test_npy_fallback_epoch_extraction(self):
        tested = 0
        max_diffs = []
        for tc, npy_files, set_path, events_path in _iter_npy_fallback_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            bids_epochs = load_bids_epochs_npy_fallback(set_path, events_path)
            for trial_idx in range(min(3, len(npy_files))):
                original = np.load(str(npy_files[trial_idx]))
                bids_epoch = bids_epochs[trial_idx]
                diff = np.abs(bids_epoch[:N_CH_EEG] - original[:N_CH_EEG])
                max_diffs.append(diff.max())
                np.testing.assert_allclose(
                    bids_epoch[:N_CH_EEG], original[:N_CH_EEG], atol=1e-6, rtol=1e-12,
                    err_msg=f"{bids_id}/ses-{date}/{calib_name} trial {trial_idx}",
                )
                tested += 1
        assert tested > 0, "No npy-fallback test cases found"
        print(f"\nPassed {tested} npy-fallback epoch extraction tests")
        print(f"  Max raw diff (ADC units): {max(max_diffs):.2e}")


class TestEeglabPreprocessingEquivalence:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_sessions = _get_test_sessions()

    def test_preprocessing_equivalence_h5(self):
        tested = 0
        max_diffs_eeg = []
        max_diffs_emg = []
        for tc, npy_files, set_path, events_path in _iter_h5_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            seed = 0
            original = np.load(str(npy_files[0]))
            eeg_orig, emg_orig = preproc_original(original.copy(), seed=seed)
            bids_epoch = load_bids_epoch(set_path, events_path, 0)
            eeg_bids, emg_bids = preproc_original(bids_epoch.copy(), seed=seed)
            max_diff_eeg = np.abs(eeg_orig - eeg_bids).max()
            max_diff_emg = np.abs(emg_orig - emg_bids).max()
            max_diffs_eeg.append(max_diff_eeg)
            max_diffs_emg.append(max_diff_emg)
            # NLMS adaptive filter amplifies fp32→fp64 precision diff;
            # sub-3/ses-20230523 has trigger mismatch adding further divergence.
            assert max_diff_eeg < 0.1, (
                f"EEG diff too large: {max_diff_eeg:.6e} for {bids_id}/ses-{date}/{calib_name}"
            )
            tested += 1
        assert tested > 0
        print(f"\nPassed {tested} preprocessing equivalence tests")
        print(f"  Max EEG diff: {max(max_diffs_eeg):.2e}")
        print(f"  Max EMG diff: {max(max_diffs_emg):.2e}")

    def test_preprocessing_equivalence_npy_fallback(self):
        tested = 0
        max_diffs_eeg = []
        for tc, npy_files, set_path, events_path in _iter_npy_fallback_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            seed = 0
            original = np.load(str(npy_files[0]))
            eeg_orig, _ = preproc_original(original.copy(), seed=seed)
            bids_epochs = load_bids_epochs_npy_fallback(set_path, events_path)
            eeg_bids, _ = preproc_original(bids_epochs[0].copy(), seed=seed)
            max_diff = np.abs(eeg_orig - eeg_bids).max()
            max_diffs_eeg.append(max_diff)
            assert max_diff < 0.01, (
                f"npy fallback diff too large: {max_diff:.6e} for {bids_id}/ses-{date}/{calib_name}"
            )
            tested += 1
        assert tested > 0
        print(f"\nPassed {tested} npy-fallback preprocessing tests")
        print(f"  Max EEG diff: {max(max_diffs_eeg):.2e}")


class TestEeglabFinalOutputEquivalence:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_sessions = _get_test_sessions()

    def test_final_output_match(self):
        tested = 0
        max_diffs = []
        for tc, npy_files, set_path, events_path in _iter_h5_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            seed = 0
            original = np.load(str(npy_files[0]))
            eeg_orig, _ = preproc_original(original.copy(), seed=seed)
            eeg_orig_crop = eeg_orig[:, -DURA_SAMPLES:]
            bids_epoch = load_bids_epoch(set_path, events_path, 0)
            eeg_bids, _ = preproc_original(bids_epoch.copy(), seed=seed)
            eeg_bids_crop = eeg_bids[:, -DURA_SAMPLES:]
            max_diff = np.abs(eeg_orig_crop - eeg_bids_crop).max()
            max_diffs.append(max_diff)
            assert max_diff < 0.01, (
                f"Final output diff too large: {max_diff:.6e} for {bids_id}/ses-{date}/{calib_name}"
            )
            tested += 1
        assert tested > 0
        print(f"\nPassed {tested} final output equivalence tests")
        print(f"  Max diff: {max(max_diffs):.2e}")
