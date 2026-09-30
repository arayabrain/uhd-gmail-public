"""Helpers for adaptive-filter waveform supplemental figures.

Computes before/after NLMS EEG and z-scored EMG from a raw trial array
(shape ``(139, n_samples)`` in ADC units), matching the public preprocess
pipeline used by ``scripts/figures/make_preproc_files.py``.

Default OpenNeuro-derived layout for Fig. S1 (sub-7, overt, offline)::

    {output_root}/sub-7/ses-20260520/task-overt_acq-calibration_run-01/trials/049.npy

where ``output_root`` comes from ``configs/paths.yaml`` (see
``uhd_eeg.paths.get_output_root``). Offline maps to ``acq-calibration``;
online maps to ``acq-online``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.stats import zscore

from scripts.figures._lib.default_plt import dark_blue, green, orange
from uhd_eeg.bids.constants import EPOCH_SAMPLES, N_CH_TOTAL, SFREQ
from uhd_eeg.paths import get_output_root
from uhd_eeg.preprocess.adaptive_filter import (
    NLMS,
    bipolar_np,
    get_ch_type_after_resample,
)

EMG_NAMES = ("EOG", "EMG upper", "EMG lower")
EMG_COLORS = (dark_blue, orange, green)

N_CH_EEG = 128
N_CH_NOISE = 3
UNIT_COEFF = 1.0e-6
PREAMP_GAIN = 10.0
BANDPASS_LOW = 2.0
BANDPASS_HIGH = 118.0
NUM_TRIAL_AVG = 5
DURA_SEC = 1.25
NLMS_MU = 0.1
NLMS_W = "random"

CONDITION_TO_TASK = {
    "overt": "overt",
    "minimally_overt": "minimallyovert",
    "minimallyovert": "minimallyovert",
    "covert": "covert",
}
RECORDING_TO_ACQ = {
    "offline": "calibration",
    "online": "online",
}

# Default session dates for the Fig. S1 subject (OpenNeuro BIDS ses-*).
DEFAULT_SESSION_DATES = {
    ("sub-7", "overt"): "20260520",
    ("sub-7", "minimally_overt"): "20260527",
    ("sub-7", "minimallyovert"): "20260527",
    ("sub-7", "covert"): "20260527",
}


@dataclass(frozen=True)
class TrialBundle:
    """Pre/post adaptive-filter arrays for one trial (speech window only)."""

    trial_index: int
    source_path: Path | None
    eeg_before: np.ndarray  # (128, n_speech_samples)
    eeg_after: np.ndarray  # (128, n_speech_samples)
    emg: np.ndarray  # (3, n_speech_samples)


def speech_samples(fs: int = SFREQ) -> int:
    return int(round(DURA_SEC * fs)) * NUM_TRIAL_AVG


def build_mne_info(fs: int = SFREQ) -> mne.Info:
    n_ch_to_use = N_CH_EEG + N_CH_NOISE
    return mne.create_info(
        ch_names=n_ch_to_use,
        sfreq=fs,
        ch_types=[get_ch_type_after_resample(i, n_ch_to_use) for i in range(n_ch_to_use)],
        verbose=False,
    )


def base_filtered_raw(trial: np.ndarray, *, fs: int = SFREQ) -> mne.io.RawArray:
    """Bipolarize, scale, notch, average-ref, and bandpass a raw trial."""
    if trial.ndim != 2 or trial.shape[0] < N_CH_TOTAL:
        raise ValueError(
            f"Expected trial with shape (>= {N_CH_TOTAL}, n_samples), got {trial.shape}"
        )
    info = build_mne_info(fs)
    data_ch_idx = np.arange(N_CH_EEG)
    notch_freqs = np.arange(50, fs / 2, 50)
    filter_length = min(
        int(round(6.6 * fs)),
        round(fs * DURA_SEC * NUM_TRIAL_AVG - 1),
    )

    eeg = bipolar_np(trial)  # (128 EEG + 3 noise, n_samples)
    eeg = eeg.astype(np.float64, copy=True)
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
    return raw


def compute_trial_bundle(
    trial: np.ndarray,
    *,
    trial_index: int = 0,
    source_path: Path | None = None,
    fs: int = SFREQ,
) -> TrialBundle:
    """Return before/after adaptive-filter EEG and z-scored EMG for one trial.

    ``eeg_before`` is channel-wise z-scored EEG after all shared preprocessing
    steps but without NLMS. ``eeg_after`` is the NLMS output (internal z-score
    normalization). Both retain only the trailing five-repeat speech window.
    """
    np.random.seed(0)
    dura = speech_samples(fs)
    raw = base_filtered_raw(trial, fs=fs)
    base_data = raw.get_data()

    data_ch_idx = np.arange(N_CH_EEG)
    noise_ch_idx = np.arange(N_CH_NOISE) + N_CH_EEG
    adapt_filt = NLMS(data_ch_idx, noise_ch_idx, mu=NLMS_MU, w=NLMS_W)

    eeg_before = zscore(base_data[:N_CH_EEG], axis=1)[:, -dura:]
    eeg_after = adapt_filt(base_data.copy(), normalize="zscore")[:N_CH_EEG, -dura:]
    emg = zscore(base_data[N_CH_EEG:], axis=1)[:, -dura:]
    return TrialBundle(
        trial_index=trial_index,
        source_path=source_path,
        eeg_before=eeg_before,
        eeg_after=eeg_after,
        emg=emg,
    )


def slice_window(
    bundle: TrialBundle,
    fs: int,
    start_sec: float,
    duration_sec: float | None,
) -> tuple[np.ndarray, slice]:
    """Return relative time axis and sample slice into the speech window."""
    n_samples = bundle.eeg_before.shape[-1]
    start = max(0, int(round(start_sec * fs)))
    if duration_sec is None:
        stop = n_samples
    else:
        stop = min(n_samples, start + int(round(duration_sec * fs)))
    if start >= stop:
        raise ValueError(f"empty plot window: start={start_sec}, duration={duration_sec}")
    time = np.arange(start, stop) / fs
    return time - time[0], slice(start, stop)


def default_session_date(subject: str, condition: str) -> str:
    key = (subject, condition)
    if key not in DEFAULT_SESSION_DATES:
        raise KeyError(
            f"No default session date for subject={subject!r}, condition={condition!r}. "
            "Pass --trial-npy or --session explicitly."
        )
    return DEFAULT_SESSION_DATES[key]


def default_trial_path(
    subject: str = "sub-7",
    condition: str = "overt",
    recording_type: str = "offline",
    trial_index: int = 49,
    *,
    session: str | None = None,
    run: str = "01",
    output_root: Path | None = None,
) -> Path:
    """Resolve the default derived-trial npy under the OpenNeuro extract layout."""
    if subject.startswith("subject"):
        raise ValueError(
            f"Use public BIDS IDs (e.g. 'sub-7'), not legacy names ({subject!r})."
        )
    task = CONDITION_TO_TASK.get(condition)
    if task is None:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {sorted(CONDITION_TO_TASK)}")
    acq = RECORDING_TO_ACQ.get(recording_type)
    if acq is None:
        raise ValueError(
            f"Unknown recording_type {recording_type!r}; expected one of {sorted(RECORDING_TO_ACQ)}"
        )
    ses = session or default_session_date(subject, condition)
    if not ses.startswith("ses-"):
        ses = f"ses-{ses}"
    root = Path(output_root) if output_root is not None else get_output_root()
    run_key = f"task-{task}_acq-{acq}_run-{run}"
    return root / subject / ses / run_key / "trials" / f"{trial_index:03d}.npy"


def load_trial_npy(path: Path) -> np.ndarray:
    """Load a raw trial array and validate channel count."""
    trial = np.load(path)
    if trial.ndim != 2 or trial.shape[0] < N_CH_TOTAL:
        raise ValueError(
            f"{path}: expected shape (>= {N_CH_TOTAL}, n_samples), got {trial.shape}"
        )
    if trial.shape[1] < speech_samples():
        raise ValueError(
            f"{path}: expected at least {speech_samples()} samples "
            f"(got {trial.shape[1]}); typical extract length is {EPOCH_SAMPLES}."
        )
    return trial


def load_or_compute_bundle(
    *,
    trial_npy: Path | None = None,
    eeg_before_npy: Path | None = None,
    eeg_after_npy: Path | None = None,
    emg_npy: Path | None = None,
    trial_index: int = 49,
    subject: str = "sub-7",
    condition: str = "overt",
    recording_type: str = "offline",
    session: str | None = None,
    run: str = "01",
    output_root: Path | None = None,
    fs: int = SFREQ,
) -> TrialBundle:
    """Load precomputed arrays or compute them from a raw trial npy."""
    if eeg_before_npy is not None or eeg_after_npy is not None or emg_npy is not None:
        missing = [
            name
            for name, path in (
                ("--eeg-before-npy", eeg_before_npy),
                ("--eeg-after-npy", eeg_after_npy),
                ("--emg-npy", emg_npy),
            )
            if path is None
        ]
        if missing:
            raise ValueError(
                "When supplying precomputed arrays, provide all of "
                "--eeg-before-npy, --eeg-after-npy, and --emg-npy "
                f"(missing: {', '.join(missing)})."
            )
        return TrialBundle(
            trial_index=trial_index,
            source_path=None,
            eeg_before=np.load(eeg_before_npy),
            eeg_after=np.load(eeg_after_npy),
            emg=np.load(emg_npy),
        )

    path = trial_npy
    if path is None:
        path = default_trial_path(
            subject=subject,
            condition=condition,
            recording_type=recording_type,
            trial_index=trial_index,
            session=session,
            run=run,
            output_root=output_root,
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trial array not found: {path}\n"
            "Extract OpenNeuro trials first, e.g.\n"
            "  uv run python bids/extract_from_bids.py --subjects sub-7\n"
            "or pass --trial-npy pointing at a (139, n_samples) ADC array."
        )
    trial = load_trial_npy(path)
    return compute_trial_bundle(
        trial,
        trial_index=trial_index,
        source_path=path,
        fs=fs,
    )
