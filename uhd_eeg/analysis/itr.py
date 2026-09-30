"""Wolpaw information transfer rate for N-class selection."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from uhd_eeg.analysis.manuscript_table2 import TABLE2_EEGNET

N_CLASSES = 5
CHANCE = 1.0 / N_CLASSES


def wolpaw_bits(probability: float, n_classes: int = N_CLASSES) -> float:
    """Bits per selection ``B(P)`` without chance clamping.

    ``B(P) = log2(N) + P log2(P) + (1-P) log2((1-P)/(N-1))``.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability outside [0, 1]: {probability}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")

    correct_term = 0.0 if probability == 0.0 else probability * math.log2(probability)
    error_probability = 1.0 - probability
    error_term = (
        0.0
        if error_probability == 0.0
        else error_probability * math.log2(error_probability / (n_classes - 1))
    )
    return math.log2(n_classes) + correct_term + error_term


def bits_per_selection(
    probability: float,
    n_classes: int = N_CLASSES,
    *,
    clamp_at_or_below_chance: bool = True,
) -> float:
    """Bits per selection with optional zeroing at or below chance."""
    chance = 1.0 / n_classes
    raw = wolpaw_bits(probability, n_classes=n_classes)
    if clamp_at_or_below_chance and probability <= chance:
        return 0.0
    return raw


def itr_bits_per_min(
    probability: float,
    duration_sec: float,
    n_classes: int = N_CLASSES,
    *,
    clamp_at_or_below_chance: bool = True,
) -> float:
    """Information transfer rate in bits/min: ``60 * B(P) / T``."""
    if duration_sec <= 0 or not math.isfinite(duration_sec):
        raise ValueError(f"duration_sec must be positive and finite, got {duration_sec}")
    bits = bits_per_selection(
        probability,
        n_classes=n_classes,
        clamp_at_or_below_chance=clamp_at_or_below_chance,
    )
    return bits * 60.0 / duration_sec


def manuscript_itr_summary(
    accuracies: Mapping[str, Sequence[float]] = TABLE2_EEGNET,
    *,
    durations_sec: Sequence[float] = (12.4, 6.25),
) -> dict[float, dict[str, float]]:
    """Mean per-condition ITR for each duration (chance-clamped)."""
    out: dict[float, dict[str, float]] = {}
    for duration in durations_sec:
        out[float(duration)] = {
            condition: float(
                sum(itr_bits_per_min(p, float(duration)) for p in values) / len(values)
            )
            for condition, values in accuracies.items()
        }
    return out
