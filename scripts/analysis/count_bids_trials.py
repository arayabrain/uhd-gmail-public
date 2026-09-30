#!/usr/bin/env python3
"""Count trials per subject / recording type / condition from BIDS events.

Writes per-run counts and manuscript totals (Supplementary Tables S3–S4) to
``outputs/bids_trial_counts/`` by default. Use ``--verify-manuscript`` to compare
against embedded counts from ``supplementary.tex``.

Online rows follow the post-hoc evaluator rule: ``acq-online`` ``run-01`` only,
first 50 valid word labels (0–4) in onset order (skip rows without a valid label).

Example::

    uv run python scripts/analysis/count_bids_trials.py --verify-manuscript
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from uhd_eeg.analysis.bids_trial_counts import (
    aggregate_subject_condition,
    aggregate_totals,
    count_bids_root,
)
from uhd_eeg.analysis.manuscript_trial_counts import (
    SUPPLEMENTARY_TABLE_S3,
    SUPPLEMENTARY_TABLE_S4,
    WORD_COLUMNS,
)
from uhd_eeg.paths import get_bids_root

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "bids_trial_counts"


def _expected_s3_frame() -> pd.DataFrame:
    rows = []
    for row in SUPPLEMENTARY_TABLE_S3:
        rows.append(
            {
                "subject": row.subject,
                "recording_type": row.recording_type,
                "condition": row.condition,
                "n_total": row.n_total,
                **row.word_counts(),
            }
        )
    return pd.DataFrame(rows)


def _expected_s4_frame() -> pd.DataFrame:
    rows = []
    for row in SUPPLEMENTARY_TABLE_S4:
        rows.append(
            {
                "recording_type": row.recording_type,
                "condition": row.condition,
                "n_total": row.n_total,
                "green": row.green,
                "magenta": row.magenta,
                "orange": row.orange,
                "violet": row.violet,
                "yellow": row.yellow,
            }
        )
    return pd.DataFrame(rows)


def verify_against_manuscript(per_run: pd.DataFrame, totals: pd.DataFrame) -> tuple[bool, list[str]]:
    issues: list[str] = []
    expected_s3 = _expected_s3_frame()
    observed_s3 = aggregate_subject_condition(per_run)
    merged = observed_s3.merge(
        expected_s3,
        on=["subject", "recording_type", "condition"],
        how="outer",
        suffixes=("_obs", "_exp"),
        indicator=True,
    )
    missing = merged[merged["_merge"] != "both"]
    for _, row in missing.iterrows():
        issues.append(
            f"S3 key missing or extra: subject={row.get('subject')} "
            f"type={row.get('recording_type')} condition={row.get('condition')} "
            f"merge={row['_merge']}"
        )
    both = merged[merged["_merge"] == "both"]
    for col in ("n_total", *WORD_COLUMNS):
        obs = f"{col}_obs"
        exp = f"{col}_exp"
        if obs not in both.columns:
            continue
        diff = both[obs].astype(int) - both[exp].astype(int)
        bad = both.loc[diff != 0]
        for _, row in bad.iterrows():
            issues.append(
                f"S3 mismatch {row['subject']} {row['recording_type']} {row['condition']} "
                f"{col}: observed={int(row[obs])} expected={int(row[exp])}"
            )

    expected_s4 = _expected_s4_frame()
    tmerge = totals.merge(
        expected_s4,
        on=["recording_type", "condition"],
        how="outer",
        suffixes=("_obs", "_exp"),
        indicator=True,
    )
    for _, row in tmerge[tmerge["_merge"] != "both"].iterrows():
        issues.append(
            f"S4 key missing or extra: type={row.get('recording_type')} "
            f"condition={row.get('condition')} merge={row['_merge']}"
        )
    tboth = tmerge[tmerge["_merge"] == "both"]
    for col in ("n_total", *WORD_COLUMNS):
        obs = f"{col}_obs"
        exp = f"{col}_exp"
        diff = tboth[obs].astype(int) - tboth[exp].astype(int)
        for _, row in tboth.loc[diff != 0].iterrows():
            issues.append(
                f"S4 mismatch {row['recording_type']} {row['condition']} {col}: "
                f"observed={int(row[obs])} expected={int(row[exp])}"
            )
    return len(issues) == 0, issues


def print_verification_summary(ok: bool, issues: list[str]) -> None:
    if ok:
        print("Manuscript verification: PASS (Supplementary Tables S3 and S4 match BIDS counts)")
        return
    print("Manuscript verification: FAIL")
    for line in issues[:50]:
        print(f"  - {line}")
    if len(issues) > 50:
        print(f"  ... and {len(issues) - 50} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--paths-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--verify-manuscript",
        action="store_true",
        help="Compare counts to embedded supplementary.tex Table S3/S4 values",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected_s3 = _expected_s3_frame()
    expected_s4 = _expected_s4_frame()
    expected_s3.to_csv(args.out_dir / "expected_table_s3.csv", index=False)
    expected_s4.to_csv(args.out_dir / "expected_table_s4.csv", index=False)

    bids_root = args.bids_root
    if bids_root is None:
        try:
            bids_root = get_bids_root(args.paths_file)
        except (FileNotFoundError, KeyError):
            bids_root = None

    if bids_root is None or not Path(bids_root).is_dir():
        print(
            "BIDS root not available; wrote expected_table_s3.csv / expected_table_s4.csv only. "
            "Copy configs/paths.yaml.example to configs/paths.yaml and download ds007591 to count trials."
        )
        if args.verify_manuscript:
            print("Skipping live verify (no BIDS data). Embedded expected counts are in uhd_eeg/analysis/manuscript_trial_counts.py")
        return

    per_run = count_bids_root(Path(bids_root))
    totals = aggregate_totals(per_run)
    per_run.to_csv(args.out_dir / "trial_counts_by_run.csv", index=False)
    totals.to_csv(args.out_dir / "trial_counts_totals.csv", index=False)
    print(f"Wrote {args.out_dir / 'trial_counts_by_run.csv'} ({len(per_run)} runs)")
    print(f"Wrote {args.out_dir / 'trial_counts_totals.csv'}")

    if args.verify_manuscript:
        ok, issues = verify_against_manuscript(per_run, totals)
        print_verification_summary(ok, issues)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
