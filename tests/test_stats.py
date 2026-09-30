"""CPU unit tests for Bonferroni markers and significance symbols."""

from __future__ import annotations

import numpy as np

from uhd_eeg.analysis.stats import bonferroni, manuscript_star, significance_marker


def test_bonferroni_scales_and_clips():
    adjusted = bonferroni([0.01, 0.04, 0.5], n_comparisons=3)
    assert np.allclose(adjusted, [0.03, 0.12, 1.0])


def test_manuscript_star_single_asterisk():
    assert manuscript_star(0.0117) == "*"
    assert manuscript_star(0.049) == "*"
    assert manuscript_star(0.05) == ""
    assert manuscript_star(0.293) == ""


def test_significance_markers():
    # Figs. 1b / 2a / 2b legend: * for P_adj < 0.05 only.
    assert significance_marker(1e-4, 1e-4) == "*"
    assert significance_marker(0.005, 0.005) == "*"
    assert significance_marker(0.02, 0.02) == "*"
    assert significance_marker(0.02, 0.06) == ""
    assert significance_marker(0.2, 0.6) == ""
