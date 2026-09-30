"""Analysis helpers used by manuscript figures and tables."""

from uhd_eeg.analysis.electrode_subsets_kmedoids import (
    DENSITIES,
    EXPECTED_SUBSETS,
    solve_density_subsets,
    solve_kmedoids,
)
from uhd_eeg.analysis.itr import bits_per_selection, itr_bits_per_min, wolpaw_bits
from uhd_eeg.analysis.snr import compute_snr, estimate_word_snr, trial_waveform
from uhd_eeg.analysis.stats import bonferroni, manuscript_star, significance_marker

__all__ = [
    "DENSITIES",
    "EXPECTED_SUBSETS",
    "bits_per_selection",
    "bonferroni",
    "compute_snr",
    "estimate_word_snr",
    "itr_bits_per_min",
    "manuscript_star",
    "significance_marker",
    "solve_density_subsets",
    "solve_kmedoids",
    "trial_waveform",
    "wolpaw_bits",
]
