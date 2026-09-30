"""Count trials from BIDS ``*_events.tsv`` files.

Offline (``acq-calibration``) counts every valid word event.
Online (``acq-online``) matches the post-hoc evaluator: only ``run-01``, and
the first 50 valid word labels (0–4) in onset order, skipping rows without a
valid label (see ``uhd_eeg.analysis.online_trial_selection``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from uhd_eeg.analysis.manuscript_trial_counts import WORD_COLUMNS
from uhd_eeg.analysis.online_trial_selection import (
    DEFAULT_N_CLASS,
    DEFAULT_ONLINE_MAX_TRIALS,
    VALUE_TO_WORD,
    event_row_to_label,
    is_manuscript_online_run,
    select_first_valid_events,
)
from uhd_eeg.bids.extract import parse_bids_eeg_stem

_EVENTS_STEM = re.compile(
    r"^(?P<sub>sub-\d+)_(?P<ses>ses-\d+)_task-(?P<task>[^_]+)_acq-(?P<acq>[^_]+)_run-(?P<run>\d+)$"
)


def task_to_condition(task: str) -> str:
    if task == "minimallyovert":
        return "minimally_overt"
    return task


def acq_to_recording_type(acq: str) -> str:
    if acq == "calibration":
        return "offline"
    if acq == "online":
        return "online"
    return acq


def word_label_from_event(row: pd.Series) -> str | None:
    label = event_row_to_label(row, DEFAULT_N_CLASS)
    if label is None:
        return None
    return VALUE_TO_WORD.get(label)


def count_events_dataframe(events_df: pd.DataFrame) -> dict[str, int]:
    counts = {word: 0 for word in WORD_COLUMNS}
    for _, row in events_df.iterrows():
        word = word_label_from_event(row)
        if word is None or word not in counts:
            continue
        counts[word] += 1
    counts["n_total"] = int(sum(counts[w] for w in WORD_COLUMNS))
    return counts


def count_events_file(
    events_path: Path,
    *,
    online_max_trials: int = DEFAULT_ONLINE_MAX_TRIALS,
    n_class: int = DEFAULT_N_CLASS,
) -> dict[str, object] | None:
    """Count one events file. Returns None when an online run is unused."""
    stem = events_path.name.replace("_events.tsv", "")
    if stem.endswith("_eeg"):
        stem = stem[: -len("_eeg")]
    match = _EVENTS_STEM.match(stem)
    if match is None:
        entities = parse_bids_eeg_stem(stem + "_eeg")
        subject = entities["sub"]
        task = entities["task"]
        acq = entities["acq"]
        session = entities["ses"]
        run = entities["run"]
    else:
        subject = match.group("sub")
        session = match.group("ses")
        task = match.group("task")
        acq = match.group("acq")
        run = match.group("run")

    if acq == "online" and not is_manuscript_online_run(run):
        return None

    events_df = pd.read_csv(events_path, sep="\t")
    if acq == "online":
        events_df = select_first_valid_events(
            events_df, n_class=n_class, max_trials=online_max_trials
        )

    counts = count_events_dataframe(events_df)
    return {
        "subject": subject,
        "session": session,
        "task": task,
        "acq": acq,
        "run": run,
        "recording_type": acq_to_recording_type(acq),
        "condition": task_to_condition(task),
        **counts,
    }


def count_bids_root(
    bids_root: Path,
    *,
    online_max_trials: int = DEFAULT_ONLINE_MAX_TRIALS,
    n_class: int = DEFAULT_N_CLASS,
) -> pd.DataFrame:
    rows = []
    for events_path in sorted(bids_root.glob("sub-*/ses-*/eeg/*_events.tsv")):
        row = count_events_file(
            events_path,
            online_max_trials=online_max_trials,
            n_class=n_class,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "subject",
                "session",
                "task",
                "acq",
                "run",
                "recording_type",
                "condition",
                "n_total",
                *WORD_COLUMNS,
            ]
        )
    return pd.DataFrame(rows)


def aggregate_subject_condition(per_run: pd.DataFrame) -> pd.DataFrame:
    """Sum runs to subject × recording_type × condition (Table S3 grain)."""
    if per_run.empty:
        return per_run
    return (
        per_run.groupby(["subject", "recording_type", "condition"], as_index=False)[
            ["n_total", *WORD_COLUMNS]
        ]
        .sum()
        .sort_values(["subject", "recording_type", "condition"])
    )


def aggregate_totals(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return per_run
    grouped = (
        per_run.groupby(["recording_type", "condition"], as_index=False)[
            ["n_total", *WORD_COLUMNS]
        ]
        .sum()
        .sort_values(["recording_type", "condition"])
    )
    return grouped
