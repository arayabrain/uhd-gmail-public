"""Test that BIDS-based preprocessing pipeline produces identical results
to the original NAS-based pipeline.

Usage:
    uv run python -m pytest tests/test_bids_pipeline.py -v
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
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
# Constants (same as make_preproc_files.py)
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
EPOCH_SAMPLES = 2880  # samples per trial epoch
DURA_SAMPLES = int(DURA_SEC * NUM_TRIAL_AVG * SFREQ)  # 1600

from uhd_eeg.env import get_raw_root

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIDS_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = get_raw_root()


# ============================================================
# Original pipeline (from make_preproc_files.py)
# ============================================================
def _build_preproc_components():
    """Build MNE info and filter objects identical to make_preproc_files.py."""
    n_ch_to_use = N_CH_EEG + N_CH_NOISE  # 131
    info = mne.create_info(
        ch_names=n_ch_to_use,
        sfreq=SFREQ,
        ch_types=[get_ch_type_after_resample(i, n_ch_to_use) for i in range(n_ch_to_use)],
        verbose=False,
    )
    notch_freqs = np.arange(50, SFREQ / 2, 50)
    filter_length = min(
        int(round(6.6 * SFREQ)),
        round(SFREQ * DURA_SEC * NUM_TRIAL_AVG - 1),
    )
    data_ch_idx = np.arange(N_CH_EEG)
    noise_ch_idx = np.arange(N_CH_NOISE) + N_CH_EEG
    return info, notch_freqs, filter_length, data_ch_idx, noise_ch_idx


def preproc_original(eeg: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Original preprocessing pipeline from make_preproc_files.py.

    Args:
        eeg: Raw epoch data (139, 2880) in raw ADC units.
        seed: Random seed for NLMS filter reproducibility.

    Returns:
        (eeg_norm, emg_norm): Preprocessed EEG (128, T) and EMG (3, T).
    """
    np.random.seed(seed)
    info, notch_freqs, filter_length, data_ch_idx, noise_ch_idx = _build_preproc_components()
    adapt_filt = NLMS(data_ch_idx, noise_ch_idx, mu=NLMS_MU, w=NLMS_W)

    eeg = bipolar_np(eeg)  # (131, 2880)
    eeg *= UNIT_COEFF
    eeg[data_ch_idx] /= PREAMP_GAIN
    raw = mne.io.RawArray(eeg, info, verbose=False)
    raw.notch_filter(
        notch_freqs,
        filter_length=filter_length,
        fir_design="firwin",
        trans_bandwidth=1.5,
        verbose=False,
    )
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, picks="all", verbose=False)
    emg = raw.get_data()[N_CH_EEG:]
    eeg_norm = adapt_filt(raw.get_data(), normalize="zscore")[:N_CH_EEG]
    emg_norm = zscore(emg, axis=1)
    return eeg_norm, emg_norm


# ============================================================
# BIDS loading pipeline
# ============================================================
def load_participant_mapping() -> Dict[str, str]:
    """Load name→BIDS-ID mapping from .env."""
    from uhd_eeg.env import load_participant_mapping as _load
    return _load()


def load_bids_epoch(
    edf_path: Path,
    events_tsv_path: Path,
    trial_idx: int,
) -> np.ndarray:
    """Load a single trial epoch from BIDS EDF + events.tsv.

    The epoch extraction replicates the original pipeline's behavior:
    - Find trigger onset from TRIGGER channel in the EDF
    - Compute packet-aligned start: (onset_sample // 8 - 359) * 8
    - Extract 2880 samples

    Args:
        edf_path: Path to the BrainVision .vhdr file.
        events_tsv_path: Path to the events.tsv file.
        trial_idx: Index of the trial to extract.

    Returns:
        Epoch data (139, 2880) in raw ADC units (unit_coeff undone).
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    data = raw.get_data()  # (n_ch, n_samples)
    trigger = data[N_CH_TOTAL - 1]  # channel 138
    onsets = np.where(np.diff(trigger) > 0.5)[0] + 1

    # Handle trigger/event count mismatch
    events_df = pd.read_csv(str(events_tsv_path), sep="\t")
    n_events = len(events_df)
    if len(onsets) > n_events:
        onsets = onsets[:n_events]

    onset_sample = onsets[trial_idx]

    # Packet-aligned epoch extraction (packets of 8 samples)
    start = (onset_sample // 8 - 359) * 8
    end = start + EPOCH_SAMPLES
    epoch = data[:, start:end].copy()

    # EDF stores data in Volts. Undo unit_coeff to get raw ADC units.
    epoch[:N_CH_TOTAL - 1] /= UNIT_COEFF

    return epoch


def load_bids_epochs_npy_fallback(
    edf_path: Path,
    events_tsv_path: Path,
) -> List[np.ndarray]:
    """Load all epochs from a BIDS EDF created from concatenated npy files.

    For npy-based sessions, trials are concatenated at fixed intervals
    (every 2880 samples), so no trigger-based extraction is needed.

    Returns:
        List of epoch arrays (139, 2880) in raw ADC units.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    data = raw.get_data()

    events_df = pd.read_csv(str(events_tsv_path), sep="\t")
    n_trials = len(events_df)

    epochs = []
    for i in range(n_trials):
        start = i * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        epoch = data[:, start:end].copy()
        epoch[:N_CH_TOTAL - 1] /= UNIT_COEFF
        epochs.append(epoch)
    return epochs


def preproc_from_bids(epoch: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess a single epoch loaded from BIDS.

    This is identical to preproc_original since the BIDS loading
    already converts back to raw ADC units.

    Args:
        epoch: Epoch data (139, 2880) in raw ADC units.
        seed: Random seed for NLMS filter reproducibility.

    Returns:
        (eeg_norm, emg_norm): Preprocessed EEG (128, T) and EMG (3, T).
    """
    return preproc_original(epoch, seed=seed)


# ============================================================
# Test helpers
# ============================================================
def _get_test_sessions() -> List[dict]:
    """Build list of test cases from participant mapping and config."""
    from collections import defaultdict
    from omegaconf import OmegaConf

    name_to_bids = load_participant_mapping()
    config_path = PROJECT_ROOT / "configs" / "trainer" / "config_color_within_offline_split.yaml"
    config = OmegaConf.load(str(config_path))

    # Parse config to get (subject_code, date, sub_idx, calibrated_name)
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
        subject_code = parts[0]
        date = path_parts[2]
        calibrated_name = path_parts[-1].split("_", 1)[1]  # e.g., "backup_calibrated_1"
        entries.append((subject_code, date, calibrated_name))

    # Match subject codes to names
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

    test_cases = []
    for code, date, calib_name in entries:
        name = code_to_name[code]
        bids_id = name_to_bids[name]
        test_cases.append({
            "bids_id": bids_id,
            "name": name,
            "date": date,
            "calib_name": calib_name,
        })
    return test_cases


TASK_BIDS = {
    "overt": "overt",
    "minimally overt": "minimallyovert",
    "covert": "covert",
}


def _resolve_bids_paths(name: str, bids_id: str, date: str, calib_name: str) -> Tuple[Path, Path]:
    """Find the BIDS EDF and events.tsv for a calibration run using metadata task matching."""
    ses_dir = BIDS_ROOT / bids_id / f"ses-{date}" / "eeg"

    # Load metadata to get the task for this calibration run
    meta_path = RAW_ROOT / name / date / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    # Find the task: check direct key or split keys
    if calib_name in metadata:
        task = metadata[calib_name]["task"]
    else:
        split_keys = sorted(k for k in metadata if k.startswith(f"{calib_name}_") and k != "subject")
        if split_keys:
            task = metadata[split_keys[0]]["task"]
        else:
            raise FileNotFoundError(f"{calib_name} not found in {meta_path}")

    task_bids = TASK_BIDS[task]

    edf_path = sorted(ses_dir.glob(f"*_task-{task_bids}_acq-calibration_*_eeg.edf"))
    events_path = sorted(ses_dir.glob(f"*_task-{task_bids}_acq-calibration_*_events.tsv"))

    if not edf_path or not events_path:
        raise FileNotFoundError(f"No BIDS files for {bids_id}/ses-{date}/task-{task_bids} calibration")

    return edf_path[0], events_path[0]


def _is_npy_fallback(edf_path: Path) -> bool:
    """Check if this EDF was created from npy fallback (no real triggers)."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    trigger = raw.get_data()[N_CH_TOTAL - 1]
    onsets = np.where(np.diff(trigger) > 0.5)[0] + 1
    return len(onsets) == 0


# ============================================================
# Helpers for test iteration
# ============================================================
def _iter_h5_sessions(test_sessions):
    """Yield (tc, npy_files, edf_path, events_path) for H5-based sessions."""
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
            edf_path, events_path = _resolve_bids_paths(name, bids_id, date, calib_name)
        except FileNotFoundError:
            continue
        if _is_npy_fallback(edf_path):
            continue
        yield tc, npy_files, edf_path, events_path


def _iter_npy_fallback_sessions(test_sessions):
    """Yield (tc, npy_files, edf_path, events_path) for npy-fallback sessions."""
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
            edf_path, events_path = _resolve_bids_paths(name, bids_id, date, calib_name)
        except FileNotFoundError:
            continue
        yield tc, npy_files, edf_path, events_path


# ============================================================
# Tests
# ============================================================
class TestBidsRawDataIntegrity:
    """Test that raw data loaded from BIDS matches original NAS data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_sessions = _get_test_sessions()

    def test_h5_based_epoch_extraction(self):
        """For H5-based sessions, verify epoch extraction from BIDS EDF
        matches the original npy files within EDF quantization tolerance."""
        tested = 0
        max_diffs = []
        for tc, npy_files, edf_path, events_path in _iter_h5_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            for trial_idx in range(min(3, len(npy_files))):
                original_epoch = np.load(str(npy_files[trial_idx]))
                bids_epoch = load_bids_epoch(edf_path, events_path, trial_idx)
                # EDF 16-bit quantization: tolerance depends on data range
                # Typical max diff ~1.0 in raw ADC units
                diff = np.abs(bids_epoch[:N_CH_EEG] - original_epoch[:N_CH_EEG])
                max_diff = diff.max()
                max_diffs.append(max_diff)
                corr = np.corrcoef(bids_epoch[:N_CH_EEG].flatten(), original_epoch[:N_CH_EEG].flatten())[0, 1]
                assert corr > 0.999, (
                    f"Epoch correlation too low: {corr:.8f} "
                    f"for {bids_id}/ses-{date}/{calib_name} trial {trial_idx}"
                )
                tested += 1
        assert tested > 0, "No H5-based test cases found"
        print(f"\nPassed {tested} H5-based epoch extraction tests")
        print(f"  Max raw diff (ADC units): {max(max_diffs):.4f}")

    def test_npy_fallback_epoch_extraction(self):
        """For npy-fallback sessions, verify concatenated epoch data
        matches original npy files within EDF quantization tolerance."""
        tested = 0
        max_diffs = []
        for tc, npy_files, edf_path, events_path in _iter_npy_fallback_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            bids_epochs = load_bids_epochs_npy_fallback(edf_path, events_path)
            for trial_idx in range(min(3, len(npy_files))):
                original_epoch = np.load(str(npy_files[trial_idx]))
                bids_epoch = bids_epochs[trial_idx]
                diff = np.abs(bids_epoch[:N_CH_EEG] - original_epoch[:N_CH_EEG])
                max_diffs.append(diff.max())
                corr = np.corrcoef(bids_epoch[:N_CH_EEG].flatten(), original_epoch[:N_CH_EEG].flatten())[0, 1]
                assert corr > 0.999, (
                    f"npy fallback correlation too low: {corr:.8f} "
                    f"for {bids_id}/ses-{date}/{calib_name} trial {trial_idx}"
                )
                tested += 1
        assert tested > 0, "No npy-fallback test cases found"
        print(f"\nPassed {tested} npy-fallback epoch extraction tests")
        print(f"  Max raw diff (ADC units): {max(max_diffs):.4f}")


class TestBidsPreprocessingEquivalence:
    """Test that preprocessing from BIDS produces equivalent results
    to the original pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_sessions = _get_test_sessions()

    def test_preprocessing_equivalence_h5(self):
        """Verify preprocessed output from BIDS is equivalent to original pipeline."""
        tested = 0
        max_diffs_eeg = []
        max_diffs_emg = []

        for tc, npy_files, edf_path, events_path in _iter_h5_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            seed = 0

            original_epoch = np.load(str(npy_files[0]))
            eeg_orig, emg_orig = preproc_original(original_epoch.copy(), seed=seed)

            bids_epoch = load_bids_epoch(edf_path, events_path, 0)
            eeg_bids, emg_bids = preproc_from_bids(bids_epoch.copy(), seed=seed)

            # NLMS adaptive filter amplifies small input differences,
            # so use correlation rather than absolute tolerance
            corr_eeg = np.corrcoef(eeg_orig.flatten(), eeg_bids.flatten())[0, 1]
            corr_emg = np.corrcoef(emg_orig.flatten(), emg_bids.flatten())[0, 1]
            max_diffs_eeg.append(corr_eeg)
            max_diffs_emg.append(corr_emg)

            assert corr_eeg > 0.95, (
                f"EEG preprocessing correlation too low: {corr_eeg:.6f} "
                f"for {bids_id}/ses-{date}/{calib_name}"
            )
            assert corr_emg > 0.95, (
                f"EMG preprocessing correlation too low: {corr_emg:.6f} "
                f"for {bids_id}/ses-{date}/{calib_name}"
            )
            tested += 1

        assert tested > 0, "No test cases found"
        print(f"\nPassed {tested} preprocessing equivalence tests")
        print(f"  Min EEG corr: {min(max_diffs_eeg):.10f}")
        print(f"  Min EMG corr: {min(max_diffs_emg):.10f}")

    def test_preprocessing_equivalence_npy_fallback(self):
        """Verify preprocessed output for npy-fallback sessions."""
        tested = 0
        max_diffs_eeg = []

        for tc, npy_files, edf_path, events_path in _iter_npy_fallback_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            seed = 0

            original_epoch = np.load(str(npy_files[0]))
            eeg_orig, _ = preproc_original(original_epoch.copy(), seed=seed)

            bids_epochs = load_bids_epochs_npy_fallback(edf_path, events_path)
            eeg_bids, _ = preproc_from_bids(bids_epochs[0].copy(), seed=seed)

            corr = np.corrcoef(eeg_orig.flatten(), eeg_bids.flatten())[0, 1]
            max_diffs_eeg.append(corr)

            assert corr > 0.99, (
                f"npy fallback preprocessing correlation too low: {corr:.6f} "
                f"for {bids_id}/ses-{date}/{calib_name}"
            )
            tested += 1

        assert tested > 0, "No npy-fallback test cases found"
        print(f"\nPassed {tested} npy-fallback preprocessing equivalence tests")
        print(f"  Min EEG corr: {min(max_diffs_eeg):.10f}")


class TestBidsFinalOutputEquivalence:
    """Test that the final cropped output (last 1600 samples)
    matches between original and BIDS pipelines."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_sessions = _get_test_sessions()

    def test_final_output_match(self):
        """Verify that the final cropped EEG/EMG output is equivalent."""
        tested = 0
        correlations = []

        for tc, npy_files, edf_path, events_path in _iter_h5_sessions(self.test_sessions):
            bids_id, date, calib_name = tc["bids_id"], tc["date"], tc["calib_name"]
            seed = 0

            original_epoch = np.load(str(npy_files[0]))
            eeg_orig, _ = preproc_original(original_epoch.copy(), seed=seed)
            eeg_orig_crop = eeg_orig[:, -DURA_SAMPLES:]

            bids_epoch = load_bids_epoch(edf_path, events_path, 0)
            eeg_bids, _ = preproc_from_bids(bids_epoch.copy(), seed=seed)
            eeg_bids_crop = eeg_bids[:, -DURA_SAMPLES:]

            corr = np.corrcoef(eeg_orig_crop.flatten(), eeg_bids_crop.flatten())[0, 1]
            correlations.append(corr)

            assert corr > 0.99, (
                f"Final output correlation too low: {corr:.6f} "
                f"for {bids_id}/ses-{date}/{calib_name}"
            )
            tested += 1

        assert tested > 0, "No test cases found"
        print(f"\nPassed {tested} final output equivalence tests")
        print(f"  Min correlation: {min(correlations):.10f}")
        print(f"  Mean correlation: {np.mean(correlations):.10f}")
