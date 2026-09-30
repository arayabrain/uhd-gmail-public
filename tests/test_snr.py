"""CPU unit tests for manuscript SNR estimation."""

from __future__ import annotations

import numpy as np
import pytest

from uhd_eeg.analysis.snr import (
    MIN_TRIALS_PER_WORD,
    N_CHANNELS,
    N_REPETITIONS,
    SAMPLES_PER_REPETITION,
    compute_snr,
    estimate_word_snr,
    trial_waveform,
)


def test_trial_waveform_demean_and_average():
    eeg = np.zeros((N_CHANNELS, N_REPETITIONS * SAMPLES_PER_REPETITION))
    for rep in range(N_REPETITIONS):
        start = rep * SAMPLES_PER_REPETITION
        eeg[:, start : start + SAMPLES_PER_REPETITION] = rep + 1.0
    waveform = trial_waveform(eeg)
    assert waveform.shape == (N_CHANNELS, SAMPLES_PER_REPETITION)
    # Constant-per-repetition signals demean to zero.
    assert np.allclose(waveform, 0.0)


def test_snr_known_signal_close_to_truth():
    rng = np.random.default_rng(0)
    k = 80
    signal = np.sin(
        2.0 * np.pi * np.arange(SAMPLES_PER_REPETITION) / SAMPLES_PER_REPETITION
    )
    signal_power = float((signal**2).mean())
    noise_var = 0.25
    noise = rng.normal(scale=np.sqrt(noise_var), size=(k, N_CHANNELS, SAMPLES_PER_REPETITION))
    noise = noise - noise.mean(axis=2, keepdims=True)
    waveforms = signal[None, None, :] + noise
    estimate = estimate_word_snr(waveforms)
    expected = signal_power / noise_var
    assert abs(estimate.snr - expected) / expected < 0.10
    assert estimate.signal_power > 0
    assert estimate.noise_variance > 0


def test_snr_pure_noise_near_zero_or_negative():
    rng = np.random.default_rng(1)
    k = 30
    noise = rng.normal(size=(k, N_CHANNELS, SAMPLES_PER_REPETITION))
    noise = noise - noise.mean(axis=2, keepdims=True)
    estimate = estimate_word_snr(noise)
    assert estimate.snr < 0.05


def test_compute_snr_excludes_words_with_few_trials():
    rng = np.random.default_rng(2)
    # Words 0..3 have 5 trials; word 4 has only 2 (< MIN_TRIALS_PER_WORD).
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4])
    n = len(labels)
    signal = np.sin(
        2.0 * np.pi * np.arange(SAMPLES_PER_REPETITION) / SAMPLES_PER_REPETITION
    )
    waveforms = (
        signal[None, None, :]
        + 0.1 * rng.normal(size=(n, N_CHANNELS, SAMPLES_PER_REPETITION))
    )
    estimate, included = compute_snr(waveforms, labels)
    assert included == [0, 1, 2, 3]
    assert 4 not in included
    assert estimate.snr > 0.0


def test_compute_snr_requires_sufficient_words():
    waveforms = np.zeros((3, N_CHANNELS, SAMPLES_PER_REPETITION))
    labels = np.array([0, 0, 0])  # only 3 trials of one word
    with pytest.raises(ValueError, match="No word has at least"):
        compute_snr(waveforms, labels, min_trials_per_word=MIN_TRIALS_PER_WORD)
