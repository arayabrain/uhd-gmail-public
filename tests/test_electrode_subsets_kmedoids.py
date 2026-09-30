"""CPU tests for exact k-medoids electrode subsets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from uhd_eeg.analysis.electrode_subsets_kmedoids import (
    DENSITIES,
    EXPECTED_SUBSETS,
    load_electrode_positions_tsv,
    load_electrode_positions_xml,
    solve_density_subsets,
    solve_kmedoids,
)

XML_PATH = Path("bids/electrodes_uhd.xml")
RUN_SLOW = os.environ.get("RUN_SLOW_KMEDOIDS", "").strip() in {"1", "true", "yes"}


@pytest.fixture(scope="module")
def xml_positions() -> np.ndarray:
    if not XML_PATH.is_file():
        pytest.skip("electrode XML not present")
    return load_electrode_positions_xml(XML_PATH)


def test_kmedoids_reproduces_expected_subsets():
    """Manuscript densitites 4/8/16/32 are recorded in EXPECTED_SUBSETS (Fig. S2).

    Fast structural check. Full MILP re-solves are gated behind
    ``RUN_SLOW_KMEDOIDS=1`` (see ``test_kmedoids_solves_milp_from_xml_coordinates``).
    """
    assert tuple(EXPECTED_SUBSETS) == DENSITIES
    for density in DENSITIES:
        channels = EXPECTED_SUBSETS[density]
        assert len(channels) == density
        assert len(set(channels)) == density
        assert all(0 <= c < 128 for c in channels)
        assert channels == sorted(channels)


@pytest.mark.skipif(not RUN_SLOW, reason="Set RUN_SLOW_KMEDOIDS=1 to re-solve MILP (slow).")
def test_kmedoids_solves_milp_from_xml_coordinates(xml_positions: np.ndarray):
    """Solve the MILP from 3-D XML coordinates (not a hardcoded lookup)."""
    distances = squareform(pdist(xml_positions))
    for density in DENSITIES:
        solution = solve_kmedoids(xml_positions, density)
        assert solution.channels == EXPECTED_SUBSETS[density]
        objective = float(distances[:, solution.channels].min(axis=1).sum())
        assert solution.objective == pytest.approx(objective, abs=1e-9)


@pytest.mark.skipif(not RUN_SLOW, reason="Set RUN_SLOW_KMEDOIDS=1 to re-solve MILP (slow).")
def test_kmedoids_openneuro_rounded_tsv_matches_expected(
    xml_positions: np.ndarray, tmp_path: Path
):
    """OpenNeuro electrodes.tsv rounds Head coords to ~1e-4 mm."""
    rounded = np.round(xml_positions, decimals=4)
    assert np.max(np.abs(rounded - xml_positions)) <= 5.1e-5

    tsv_path = tmp_path / "electrodes.tsv"
    pd.DataFrame(
        {
            "name": [f"EEG{i + 1:03d}" for i in range(128)],
            "x": rounded[:, 0],
            "y": rounded[:, 1],
            "z": rounded[:, 2],
        }
    ).to_csv(tsv_path, sep="\t", index=False)

    positions = load_electrode_positions_tsv(tsv_path)
    subsets = solve_density_subsets(positions, densities=DENSITIES)
    for density in DENSITIES:
        assert subsets[density] == EXPECTED_SUBSETS[density]


def test_tie_break_prefers_lower_channel_index():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    solution = solve_kmedoids(positions, k=2)
    assert solution.channels == [0, 2]
    assert solution.index_sum == 2


def test_invalid_k_raises():
    positions = np.eye(3)
    with pytest.raises(ValueError):
        solve_kmedoids(positions, k=0)
