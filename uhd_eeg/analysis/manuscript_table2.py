"""Manuscript Table 2 EEGNet balanced accuracies (online, per subject)."""

from __future__ import annotations

# Subjects 1..9; conditions overt / minimally overt / covert.
TABLE2_EEGNET: dict[str, list[float]] = {
    "overt": [0.405, 0.711, 0.516, 0.640, 0.377, 0.517, 0.437, 0.622, 0.202],
    "minimally_overt": [
        0.508,
        0.490,
        0.212,
        0.400,
        0.294,
        0.519,
        0.500,
        0.441,
        0.246,
    ],
    "covert": [0.192, 0.175, 0.240, 0.340, 0.272, 0.265, 0.180, 0.405, 0.188],
}
