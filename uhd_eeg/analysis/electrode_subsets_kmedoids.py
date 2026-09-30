"""Exact k-medoids electrode subsets via integer linear programming.

Selects ``k`` of 128 electrodes minimizing the sum of 3-D Euclidean distances
from every electrode to its nearest selected electrode. Among optima, the
subset with the smallest sum of zero-based channel indices is chosen.

Authoritative coordinates ship with the package as ``bids/electrodes_uhd.xml``
(no personal data). OpenNeuro ``*_electrodes.tsv`` Head columns are the same
positions rounded to ~1e-4 mm; both sources reproduce ``EXPECTED_SUBSETS``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack
from scipy.spatial.distance import pdist, squareform

DENSITIES = (4, 8, 16, 32)
SECOND_STAGE_TOLERANCE = 1e-6

# Subsets reported in the manuscript (Supplementary Fig. S2).
EXPECTED_SUBSETS: dict[int, list[int]] = {
    4: [37, 61, 78, 115],
    8: [8, 24, 40, 59, 71, 87, 108, 123],
    16: [7, 17, 27, 34, 43, 50, 58, 63, 68, 81, 91, 96, 104, 111, 114, 123],
    32: [
        0,
        7,
        10,
        12,
        18,
        19,
        24,
        31,
        34,
        39,
        41,
        47,
        48,
        53,
        55,
        63,
        64,
        71,
        73,
        81,
        83,
        86,
        88,
        95,
        96,
        101,
        103,
        113,
        115,
        118,
        120,
        127,
    ],
}


@dataclass(frozen=True)
class KMedoidsSolution:
    channels: list[int]
    objective: float
    index_sum: int


def load_electrode_positions_xml(xml_path: Path) -> np.ndarray:
    """Load the first 128 Subject/Head coordinates from ``electrodes_uhd.xml``."""
    electrodes = ET.parse(xml_path).getroot().findall("Electrode")
    if len(electrodes) < 128:
        raise ValueError(f"Expected >= 128 electrodes in {xml_path}; found {len(electrodes)}")
    positions = []
    for channel, electrode in enumerate(electrodes[:128]):
        head = electrode.findtext("./Positions/Subject/Head")
        if head is None:
            raise ValueError(f"Electrode {channel} has no Subject/Head position")
        positions.append([float(value) for value in head.split(",")])
    array = np.asarray(positions, dtype=float)
    if array.shape != (128, 3) or not np.isfinite(array).all():
        raise ValueError("Electrode positions must be a finite (128, 3) array")
    return array


def load_electrode_positions_tsv(tsv_path: Path) -> np.ndarray:
    """Load BIDS ``*_electrodes.tsv`` with columns ``x,y,z`` for 128 EEG channels."""
    import pandas as pd

    frame = pd.read_csv(tsv_path, sep="\t")
    required = {"x", "y", "z"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{tsv_path} must contain columns {sorted(required)}")
    # Prefer EEG-named rows when present; otherwise take the first 128.
    if "name" in frame.columns:
        eeg = frame[frame["name"].astype(str).str.startswith("EEG")]
        if len(eeg) >= 128:
            frame = eeg.iloc[:128]
        else:
            frame = frame.iloc[:128]
    else:
        frame = frame.iloc[:128]
    array = frame[["x", "y", "z"]].to_numpy(dtype=float)
    if array.shape != (128, 3) or not np.isfinite(array).all():
        raise ValueError(f"Could not read finite (128, 3) positions from {tsv_path}")
    return array


def _p_median_constraints(n: int, k: int) -> tuple[LinearConstraint, np.ndarray, Bounds]:
    pairs = n * n
    x_indices = n + np.arange(pairs)
    j_indices = np.tile(np.arange(n), n)
    assignment_rows = np.repeat(np.arange(n), n)
    linking_rows = n + np.arange(pairs)
    cardinality_row = n + pairs
    row = np.concatenate(
        [assignment_rows, linking_rows, linking_rows, np.full(n, cardinality_row)]
    )
    col = np.concatenate([x_indices, x_indices, j_indices, np.arange(n)])
    data = np.concatenate(
        [np.ones(pairs), np.ones(pairs), -np.ones(pairs), np.ones(n)]
    )
    matrix = coo_matrix((data, (row, col)), shape=(cardinality_row + 1, n + pairs)).tocsr()
    lower = np.full(cardinality_row + 1, -np.inf)
    upper = np.full(cardinality_row + 1, np.inf)
    lower[:n] = upper[:n] = 1.0
    upper[n : n + pairs] = 0.0
    lower[cardinality_row] = upper[cardinality_row] = float(k)
    integrality = np.concatenate([np.ones(n, dtype=int), np.zeros(pairs, dtype=int)])
    bounds = Bounds(np.zeros(n + pairs), np.ones(n + pairs))
    return LinearConstraint(matrix, lower, upper), integrality, bounds


def solve_kmedoids(positions: np.ndarray, k: int) -> KMedoidsSolution:
    """Solve the exact k-medoids / p-median problem for ``k`` electrodes."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (n, 3), got {positions.shape}")
    if k < 1 or k > len(positions):
        raise ValueError(f"k must be in 1..{len(positions)}, got {k}")

    distances = squareform(pdist(positions))
    n = len(distances)
    constraints, integrality, bounds = _p_median_constraints(n, k)
    primary_cost = np.concatenate([np.zeros(n), distances.ravel()])
    first = milp(
        primary_cost,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"mip_rel_gap": 0.0},
    )
    if first.status != 0 or first.fun is None:
        raise RuntimeError(f"{k}-medoids first stage failed: {first.message}")
    optimal_objective = float(first.fun)

    objective_row = coo_matrix(primary_cost.reshape(1, -1)).tocsr()
    second_matrix = vstack([constraints.A, objective_row], format="csr")
    second_constraints = LinearConstraint(
        second_matrix,
        np.append(constraints.lb, -np.inf),
        np.append(constraints.ub, optimal_objective + SECOND_STAGE_TOLERANCE),
    )
    secondary_cost = np.concatenate([np.arange(n, dtype=float), np.zeros(n * n)])
    second = milp(
        secondary_cost,
        integrality=integrality,
        bounds=bounds,
        constraints=second_constraints,
        options={"mip_rel_gap": 0.0},
    )
    if second.status != 0 or second.fun is None or second.x is None:
        raise RuntimeError(f"{k}-medoids second stage failed: {second.message}")

    channels = np.flatnonzero(second.x[:n] > 0.5).astype(int).tolist()
    if len(channels) != k or len(set(channels)) != k:
        raise RuntimeError(f"Expected {k} distinct channels, got {channels}")
    true_objective = float(distances[:, channels].min(axis=1).sum())
    index_sum = int(sum(channels))
    return KMedoidsSolution(channels, true_objective, index_sum)


def solve_density_subsets(
    positions: np.ndarray,
    densities: Sequence[int] = DENSITIES,
) -> dict[int, list[int]]:
    """Return ``{k: channel_list}`` for each requested density."""
    return {int(k): solve_kmedoids(positions, int(k)).channels for k in densities}


def write_subsets_json(subsets: Mapping[int, Sequence[int]], path: Path) -> None:
    payload = {str(k): list(map(int, subsets[k])) for k in sorted(subsets)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
