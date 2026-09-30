"""Shared selection of the first N valid online trials (manuscript rule).

Post-hoc online analyses and Supplementary Tables S3/S4 use the first
``max_trials`` labels with a word class in ``0 .. n_class-1``, in recording
order (word-list CSV row order, or ``*_events.tsv`` onset order). Rows without
a valid label are skipped. Set ``max_trials <= 0`` to keep every valid label.

The pseudo-online evaluator applies this to ``word_list*.csv``; BIDS trial
counting applies it to ``*_events.tsv``. Online evaluation / manuscript tables
use ``acq-online`` ``run-01`` only (sub-1 minimally overt ``run-02`` is unused).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_N_CLASS = 5
DEFAULT_ONLINE_MAX_TRIALS = 50
MANUSCRIPT_ONLINE_RUN = "01"

VALUE_TO_WORD = {
    0: "green",
    1: "magenta",
    2: "orange",
    3: "violet",
    4: "yellow",
}
WORD_TO_VALUE = {name: idx for idx, name in VALUE_TO_WORD.items()}


def normalize_run_id(run: str | int) -> str:
    """Normalize BIDS run entity to zero-padded two digits (``01``)."""
    text = str(run).strip()
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def is_manuscript_online_run(run: str | int) -> bool:
    """True if this online run is used in the manuscript / evaluation manifest."""
    return normalize_run_id(run) == MANUSCRIPT_ONLINE_RUN


def parse_word_label(value: object, n_class: int = DEFAULT_N_CLASS) -> int | None:
    """Parse a single label; valid means integer in ``[0, n_class)``."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        label = int(float(text)) if not isinstance(value, bool) else int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= label < n_class:
        return label
    return None


def select_first_valid_labels(
    values: Sequence[object],
    *,
    n_class: int = DEFAULT_N_CLASS,
    max_trials: int = DEFAULT_ONLINE_MAX_TRIALS,
) -> tuple[list[int], list[int]]:
    """Keep the first ``max_trials`` valid labels in sequence order.

    Returns
    -------
    original_indices, labels
        Indices into ``values`` and the corresponding integer labels.
    """
    original_indices: list[int] = []
    labels: list[int] = []
    for original_idx, value in enumerate(values):
        label = parse_word_label(value, n_class)
        if label is None:
            continue
        original_indices.append(original_idx)
        labels.append(label)
        if max_trials > 0 and len(labels) >= max_trials:
            break
    return original_indices, labels


def read_valid_online_labels(
    csv_path: Path,
    n_class: int = DEFAULT_N_CLASS,
    max_trials: int = DEFAULT_ONLINE_MAX_TRIALS,
) -> tuple[list[int], list[int], int]:
    """Read a one-column word-list CSV; return first valid labels (evaluator API)."""
    import csv

    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    values = [row[0] if row else "" for row in rows]
    original_indices, labels = select_first_valid_labels(
        values, n_class=n_class, max_trials=max_trials
    )
    return original_indices, labels, len(rows)


def event_row_to_label(
    row: pd.Series, n_class: int = DEFAULT_N_CLASS
) -> int | None:
    """Map one BIDS events.tsv row to a word label index, or None if invalid."""
    if "value" in row.index and pd.notna(row["value"]):
        return parse_word_label(row["value"], n_class)
    if "trial_type" in row.index and pd.notna(row["trial_type"]):
        word = str(row["trial_type"]).strip().lower().replace(" ", "_")
        # Accept "minimally overt" style elsewhere; trial_type is a color word.
        if word in WORD_TO_VALUE:
            label = WORD_TO_VALUE[word]
            if 0 <= label < n_class:
                return label
    return None


def select_first_valid_events(
    events_df: pd.DataFrame,
    *,
    n_class: int = DEFAULT_N_CLASS,
    max_trials: int = DEFAULT_ONLINE_MAX_TRIALS,
) -> pd.DataFrame:
    """Return the first ``max_trials`` valid events in onset (recording) order."""
    df = events_df.copy()
    if "onset" in df.columns:
        df = df.sort_values("onset", kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    keep_positions: list[int] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        if event_row_to_label(row, n_class) is None:
            continue
        keep_positions.append(pos)
        if max_trials > 0 and len(keep_positions) >= max_trials:
            break
    return df.iloc[keep_positions].reset_index(drop=True)
