#!/usr/bin/env python3
"""Build the per-subject tables behind Supplementary Figs. S3/S4.

Writes::

    data/intersubject/subject_measures.csv
        columns: subject, condition, recording_type, snr, accuracy
    data/intersubject/subject_condition_summary.csv
        columns: subject, condition, recording_type, emg_rms,
                 eeg_emg_mutual_information, accuracy

Inputs are intermediate analysis CSVs (one row per subject × condition ×
recording type) produced after SNR / EMG RMS / EEG–EMG MI / decoding evaluation.
Pass paths with the flags below, or place defaults under ``outputs/intersubject/``.

Example
-------
::

    uv run python scripts/analysis/make_intersubject_tables.py \\
      --snr-csv outputs/intersubject/snr_by_subject.csv \\
      --summary-csv outputs/intersubject/emg_mi_accuracy_by_subject.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "intersubject"


def _normalize_subject(series: pd.Series) -> pd.Series:
    def one(value: object) -> str:
        text = str(value).strip().lower().replace("_", "-")
        if text.startswith("sub-"):
            return f"sub-{int(text.split('-', 1)[1])}"
        if text.startswith("subject"):
            return f"sub-{int(text.replace('subject', ''))}"
        return f"sub-{int(text)}"

    return series.map(one)


def build_tables(
    snr_csv: Path,
    summary_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble public-column tables from analysis intermediates.

    ``snr_csv`` must provide subject, condition, recording_type, snr,
    accuracy. ``summary_csv`` must provide subject, condition, recording_type,
    emg_rms, eeg_emg_mutual_information,
    accuracy.
    """
    snr = pd.read_csv(snr_csv)
    summary = pd.read_csv(summary_csv)
    for frame in (snr, summary):
        frame["subject"] = _normalize_subject(frame["subject"])

    measures = snr[
        ["subject", "condition", "recording_type", "snr", "accuracy"]
    ].copy()
    condition_summary = summary[
        [
            "subject",
            "condition",
            "recording_type",
            "emg_rms",
            "eeg_emg_mutual_information",
            "accuracy",
        ]
    ].copy()
    return measures, condition_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snr-csv",
        type=Path,
        default=ROOT / "outputs/intersubject/snr_by_subject.csv",
        help="Per-subject SNR + accuracy table.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "outputs/intersubject/emg_mi_accuracy_by_subject.csv",
        help="Per-subject EMG RMS, EEG–EMG MI, and accuracy table.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.snr_csv.is_file() or not args.summary_csv.is_file():
        raise SystemExit(
            "Missing intermediate CSVs. Compute per-subject SNR / EMG RMS / "
            "EEG–EMG MI / accuracy first, then re-run this script.\n"
            f"  snr:     {args.snr_csv}\n"
            f"  summary: {args.summary_csv}"
        )
    measures, summary = build_tables(args.snr_csv, args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measures_path = args.output_dir / "subject_measures.csv"
    summary_path = args.output_dir / "subject_condition_summary.csv"
    measures.to_csv(measures_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {measures_path} ({len(measures)} rows)")
    print(f"Wrote {summary_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
