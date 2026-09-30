"""CPU unit tests for Wolpaw ITR helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from uhd_eeg.analysis.itr import (
    CHANCE,
    TABLE2_EEGNET,
    bits_per_selection,
    itr_bits_per_min,
    wolpaw_bits,
)


def test_wolpaw_perfect_accuracy():
    assert wolpaw_bits(1.0) == pytest.approx(math.log2(5))


def test_wolpaw_chance_is_zero():
    assert wolpaw_bits(0.2) == pytest.approx(0.0, abs=1e-12)
    assert bits_per_selection(0.2) == 0.0
    assert bits_per_selection(0.19) == 0.0
    assert bits_per_selection(CHANCE) == 0.0


def test_wolpaw_below_chance_is_clamped():
    # Direct Wolpaw expression at P=0 equals 0 for N=5; values just below chance
    # can be positive, so the manuscript clamps using P <= 1/N.
    assert bits_per_selection(0.15) == 0.0
    assert bits_per_selection(0.15, clamp_at_or_below_chance=False) > 0.0


def test_known_intermediate_value():
    bits = wolpaw_bits(0.492)
    assert round(bits, 4) == 0.3061
    itr = itr_bits_per_min(0.492, duration_sec=6.25, clamp_at_or_below_chance=False)
    assert round(itr, 3) == 2.939


def test_itr_scales_with_duration():
    bits = bits_per_selection(1.0)
    assert itr_bits_per_min(1.0, 12.4) == pytest.approx(bits * 60.0 / 12.4)


def test_manuscript_itr_from_table2_eegnet():
    """Reproduce the ITR numbers reported in the main text from Table 2.

    Manuscript values (after rounding as printed):
    - overt / minimally overt / covert at T = 12.4 s → 1.8 / 0.9 / 0.15 bits/min
    - overt at T = 6.25 s → 3.6 bits/min

    Per-subject ITRs are averaged; chance-or-below accuracies contribute 0.
    Rounding matches the printed text: one decimal for 1.8 / 0.9 / 3.6 and
    two decimals for the smaller covert rate 0.15.
    """
    mean_12_4 = {
        condition: float(np.mean([itr_bits_per_min(p, 12.4) for p in accuracies]))
        for condition, accuracies in TABLE2_EEGNET.items()
    }
    assert round(mean_12_4["overt"], 1) == 1.8
    assert round(mean_12_4["minimally_overt"], 1) == 0.9
    assert round(mean_12_4["covert"], 2) == 0.15

    mean_overt_6_25 = float(
        np.mean([itr_bits_per_min(p, 6.25) for p in TABLE2_EEGNET["overt"]])
    )
    assert round(mean_overt_6_25, 1) == 3.6
