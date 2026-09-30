"""Manuscript SNR for denoised EEG (linear ratio; Sahani–Linden equivalent).

For each trial, demean the five 1.25 s speech repetitions over time and average
them to obtain ``y_k``. For each word with ``K_w >= 4`` trials,

- ``σ_w²`` is the across-trial unbiased variance, averaged over electrodes and time
- ``P_w = ⟨ȳ_w²⟩ − σ_w² / K_w``
- ``SNR = ⟨P_w⟩_w / ⟨σ_w²⟩_w``

Public code exposes this as ``snr``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CHANNELS = 128
N_REPETITIONS = 5
SAMPLES_PER_REPETITION = 320
N_SPEECH_SAMPLES = N_REPETITIONS * SAMPLES_PER_REPETITION
MIN_TRIALS_PER_WORD = 4
WORDS = tuple(range(5))


@dataclass(frozen=True)
class SNREstimate:
    """Bias-corrected across-trial SNR components."""

    signal_power: float
    noise_variance: float
    snr: float


def trial_waveform(eeg: np.ndarray) -> np.ndarray:
    """Return the demeaned-and-averaged speech waveform ``y_k`` (128, 320).

    Parameters
    ----------
    eeg:
        Array with at least 128 channels and 1600 trailing speech samples.
        Extra leading samples (e.g. the full 2880-sample extracted epoch) are
        ignored; only the final five repetitions are used.
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2 or eeg.shape[0] < N_CHANNELS or eeg.shape[1] < N_SPEECH_SAMPLES:
        raise ValueError(
            f"Expected at least ({N_CHANNELS}, {N_SPEECH_SAMPLES}), got {eeg.shape}"
        )
    speech = eeg[:N_CHANNELS, -N_SPEECH_SAMPLES:]
    repetitions = speech.reshape(N_CHANNELS, N_REPETITIONS, SAMPLES_PER_REPETITION)
    repetitions = repetitions - repetitions.mean(axis=2, keepdims=True)
    return repetitions.mean(axis=1)


def estimate_word_snr(waveforms: np.ndarray) -> SNREstimate:
    """Estimate SNR from ``K`` trial waveforms of one word (shape ``K×128×320``)."""
    waveforms = np.asarray(waveforms, dtype=np.float64)
    if waveforms.ndim != 3 or waveforms.shape[1:] != (
        N_CHANNELS,
        SAMPLES_PER_REPETITION,
    ):
        raise ValueError(
            "Expected waveforms with shape "
            f"(K, {N_CHANNELS}, {SAMPLES_PER_REPETITION}), got {waveforms.shape}"
        )
    n_trials = waveforms.shape[0]
    if n_trials < 2:
        raise ValueError(f"Across-trial estimator requires K >= 2, got {n_trials}")

    mean_waveform = waveforms.mean(axis=0)
    # Unbiased variance over trials, then average over time per channel.
    sigma2_channel = waveforms.var(axis=0, ddof=1).mean(axis=1)
    signal_channel = (mean_waveform**2).mean(axis=1) - sigma2_channel / n_trials
    signal_power = float(signal_channel.mean())
    noise_variance = float(sigma2_channel.mean())
    if not np.isfinite(signal_power) or not np.isfinite(noise_variance):
        raise ValueError("Non-finite SNR components")
    if noise_variance <= 0:
        raise ValueError(f"Non-positive noise variance: {noise_variance}")
    return SNREstimate(signal_power, noise_variance, signal_power / noise_variance)


def compute_snr(
    waveforms: np.ndarray,
    labels: np.ndarray,
    *,
    min_trials_per_word: int = MIN_TRIALS_PER_WORD,
) -> tuple[SNREstimate, list[int]]:
    """Compute manuscript SNR, excluding words with fewer than ``min_trials_per_word``.

    Returns
    -------
    estimate:
        Subject/session-level SNR after averaging ``P_w`` and ``σ_w²`` over words.
    included_words:
        Word indices that passed the ``K_w`` threshold.
    """
    waveforms = np.asarray(waveforms, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if len(waveforms) != len(labels):
        raise ValueError("waveforms and labels must have the same length")

    estimates: list[SNREstimate] = []
    included: list[int] = []
    for word in WORDS:
        selected = waveforms[labels == word]
        if len(selected) < min_trials_per_word:
            continue
        estimates.append(estimate_word_snr(selected))
        included.append(int(word))
    if not estimates:
        raise ValueError(
            f"No word has at least {min_trials_per_word} trials; cannot compute SNR"
        )
    signal_power = float(np.mean([e.signal_power for e in estimates]))
    noise_variance = float(np.mean([e.noise_variance for e in estimates]))
    return SNREstimate(signal_power, noise_variance, signal_power / noise_variance), included
