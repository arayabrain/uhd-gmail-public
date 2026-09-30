"""CPU tests for BIDS events.tsv trial counting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from uhd_eeg.analysis.bids_trial_counts import (
    aggregate_totals,
    count_bids_root,
    count_events_dataframe,
)
from uhd_eeg.analysis.manuscript_trial_counts import SUPPLEMENTARY_TABLE_S3, SUPPLEMENTARY_TABLE_S4
from uhd_eeg.analysis.online_trial_selection import (
    read_valid_online_labels,
    select_first_valid_events,
    select_first_valid_labels,
)


def test_count_events_dataframe_synthetic():
    events = pd.DataFrame(
        {
            "onset": [0.0, 1.0, 2.0, 3.0, 4.0],
            "duration": [1.0] * 5,
            "trial_type": ["green", "magenta", "orange", "violet", "yellow"],
            "value": [0, 1, 2, 3, 4],
        }
    )
    counts = count_events_dataframe(events)
    assert counts["n_total"] == 5
    assert counts["green"] == 1
    assert counts["yellow"] == 1


def test_select_first_valid_labels_skips_invalid_and_caps():
    values = ["", "x", "0", "1", "9", "2", "3", "4", "0"]
    idxs, labels = select_first_valid_labels(values, n_class=5, max_trials=5)
    assert labels == [0, 1, 2, 3, 4]
    assert idxs == [2, 3, 5, 6, 7]


def test_select_first_valid_indices_match_sub6_overt_wordlist_pattern():
    """Original eval for sub-6 overt online used CSV rows 0, 1, 4–51.

    Rows 2 and 3 had no valid label; 54 source rows total → first 50 valid.
    """
    # Valid word labels stay in 0..4; pattern matches the blank-row index gaps.
    values: list[object] = [0, 1, "n/a", "", *[i % 5 for i in range(50)]]
    assert len(values) == 54
    idxs, labels = select_first_valid_labels(values, n_class=5, max_trials=50)
    assert idxs == [0, 1, *range(4, 52)]
    assert len(idxs) == 50
    assert labels == [0, 1, *[i % 5 for i in range(48)]]


def test_select_first_valid_events_skips_invalid_rows_in_place():
    """Events table with invalid rows kept in place (same index pattern as above)."""
    word = ["green", "magenta", "orange", "violet", "yellow"]
    rows = []
    for i in range(54):
        if i in (2, 3):
            rows.append({"onset": float(i), "value": "n/a", "trial_type": "n/a"})
        elif i < 2:
            rows.append({"onset": float(i), "value": i, "trial_type": word[i]})
        else:
            lab = (i - 4) % 5
            rows.append({"onset": float(i), "value": lab, "trial_type": word[lab]})
    events = pd.DataFrame(rows)
    assert len(events) == 54
    selected = select_first_valid_events(events, max_trials=50)
    assert len(selected) == 50
    # Onset equals original row index for this synthetic table.
    assert list(selected["onset"].astype(int)) == [0, 1, *range(4, 52)]
    idxs, _ = select_first_valid_labels(list(events["value"]), max_trials=50)
    assert idxs == [0, 1, *range(4, 52)]


_SUB6_OVERT_ONLINE_EVENTS = Path(
    "data/ds007591/sub-6/ses-20260518/eeg/"
    "sub-6_ses-20260518_task-overt_acq-online_run-01_events.tsv"
)


@pytest.mark.skipif(
    not _SUB6_OVERT_ONLINE_EVENTS.is_file(),
    reason="OpenNeuro ds007591 events not present under data/ds007591",
)
def test_real_sub6_overt_online_first50_trial_indices():
    """ds007591 ≥1.0.3: first 50 valid trials are trial_index 0, 1, 4–51."""
    # Keep literal "n/a" strings (pandas would otherwise coerce them to NaN).
    df = pd.read_csv(_SUB6_OVERT_ONLINE_EVENTS, sep="\t", keep_default_na=False)
    assert len(df) == 53
    assert "trial_index" in df.columns
    for ti in (2, 3, 52):
        row = df.loc[df["trial_index"].astype(int) == ti].iloc[0]
        assert str(row["value"]).strip().lower() in {"n/a", "na", ""}
        assert str(row["trial_type"]).strip().lower() in {"n/a", "na", ""}

    selected = select_first_valid_events(df, max_trials=50)
    assert list(selected["trial_index"].astype(int)) == [0, 1, *range(4, 52)]
    idxs, _ = select_first_valid_labels(list(df["value"]), max_trials=50)
    assert idxs == [0, 1, *range(4, 52)]



def test_select_first_valid_events_onset_order_and_cap():
    events = pd.DataFrame(
        {
            "onset": [10.0, 1.0, 2.0, 3.0] + [4.0 + i for i in range(50)],
            "value": [4, 0, 1, 9] + ([2] * 50),
        }
    )
    selected = select_first_valid_events(events, max_trials=50)
    assert len(selected) == 50
    assert list(selected["onset"].head(3)) == [1.0, 2.0, 4.0]
    assert int(selected.iloc[0]["value"]) == 0


def test_read_valid_online_labels_matches_selector(tmp_path: Path):
    csv_path = tmp_path / "word_list.csv"
    csv_path.write_text("0\n1\nbad\n2\n3\n4\n0\n", encoding="utf-8")
    idxs, labels, n_rows = read_valid_online_labels(csv_path, n_class=5, max_trials=5)
    assert n_rows == 7
    assert labels == [0, 1, 2, 3, 4]
    assert idxs == [0, 1, 3, 4, 5]


def test_count_bids_root_online_uses_first_50_and_skips_run02(tmp_path: Path):
    eeg_dir = tmp_path / "sub-1" / "ses-20250101" / "eeg"
    eeg_dir.mkdir(parents=True)
    run01 = pd.DataFrame(
        {
            "onset": list(range(55)),
            "duration": [6.25] * 55,
            "trial_type": ["green"] * 55,
            "value": [0] * 55,
        }
    )
    run01.to_csv(
        eeg_dir / "sub-1_ses-20250101_task-overt_acq-online_run-01_events.tsv",
        sep="\t",
        index=False,
    )
    run01.to_csv(
        eeg_dir / "sub-1_ses-20250101_task-overt_acq-online_run-02_events.tsv",
        sep="\t",
        index=False,
    )
    df = count_bids_root(tmp_path)
    assert len(df) == 1
    assert str(df.iloc[0]["run"]).lstrip("0") in {"1", ""} or int(df.iloc[0]["run"]) == 1
    assert int(df.iloc[0]["n_total"]) == 50
    assert int(df.iloc[0]["green"]) == 50


def test_count_bids_root_synthetic(tmp_path: Path):
    eeg_dir = tmp_path / "sub-1" / "ses-20250101" / "eeg"
    eeg_dir.mkdir(parents=True)
    prefix = "sub-1_ses-20250101_task-overt_acq-calibration_run-01"
    events = pd.DataFrame(
        {
            "onset": [0.0, 1.0],
            "duration": [6.25, 6.25],
            "trial_type": ["green", "green"],
            "value": [0, 0],
        }
    )
    events.to_csv(eeg_dir / f"{prefix}_events.tsv", sep="\t", index=False)
    df = count_bids_root(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["subject"] == "sub-1"
    assert row["recording_type"] == "offline"
    assert row["condition"] == "overt"
    assert row["n_total"] == 2
    assert row["green"] == 2


def test_manuscript_expected_tables_self_consistent():
    """Embedded S3 rows sum to S4 totals for each type/condition."""
    s3 = pd.DataFrame(
        [
            {
                "subject": r.subject,
                "recording_type": r.recording_type,
                "condition": r.condition,
                "n_total": r.n_total,
                "green": r.green,
                "magenta": r.magenta,
                "orange": r.orange,
                "violet": r.violet,
                "yellow": r.yellow,
            }
            for r in SUPPLEMENTARY_TABLE_S3
        ]
    )
    totals = aggregate_totals(s3.rename(columns={"subject": "subject"}))
    for exp in SUPPLEMENTARY_TABLE_S4:
        hit = totals[
            (totals["recording_type"] == exp.recording_type)
            & (totals["condition"] == exp.condition)
        ]
        assert len(hit) == 1
        row = hit.iloc[0]
        assert int(row["n_total"]) == exp.n_total
        assert int(row["green"]) == exp.green
        assert int(row["yellow"]) == exp.yellow


def test_supplementary_s3_row_count():
    assert len(SUPPLEMENTARY_TABLE_S3) == 54  # 9 subjects × 3 conditions × 2 types
