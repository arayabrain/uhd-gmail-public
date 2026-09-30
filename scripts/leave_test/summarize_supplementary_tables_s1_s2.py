#!/usr/bin/env python3
"""Build Supplementary Table S1/S2 summaries from cross-modal result CSVs.

Table S1 — EMG-trained decoder tested on MI-top-3 denoised EEG
(``cross_modal_control_summary_by_condition.csv``, experiment
``emg3_to_denoised_eeg_mi_top3``).

Table S2 — EEG (3 ch) decoders trained on MI-top-3 EEG; Test: EEG vs Test: EMG
(``eeg3_mi_artifact_control_summary_by_condition.csv`` plus optional
``{run_key}_eeg_test.csv`` fold exports for the EEG baseline column).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.leave_test._conditions import behavior_from_run_key, subject_from_run_key

BEHAVIOR_ORDER = ("overt", "minimally_overt", "covert")
BEHAVIOR_HEADER = {
    "overt": "overt",
    "minimally_overt": "min-overt",
    "covert": "covert",
}


def _subject_sort_key(subject: str) -> int:
    return int(subject.split("-", 1)[1])


def table_s1_from_cross_modal(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    mask = df["experiment"] == "emg3_to_denoised_eeg_mi_top3"
    subset = df.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"No emg3_to_denoised_eeg_mi_top3 rows in {summary_csv}")
    subset["subject"] = subset["subject"].astype(str)
    rows = []
    for subject in sorted(subset["subject"].unique(), key=_subject_sort_key):
        row = {"subject": subject}
        for behavior in BEHAVIOR_ORDER:
            hit = subset[(subset["subject"] == subject) & (subset["behavior"] == behavior)]
            row[behavior] = float(hit["balanced_acc_mean"].iloc[0]) if len(hit) else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    avg = {"subject": "avg."}
    for behavior in BEHAVIOR_ORDER:
        avg[behavior] = out[behavior].mean()
    return pd.concat([out, pd.DataFrame([avg])], ignore_index=True)


def _eeg_baseline_from_folds(folds_dir: Path, run_key: str) -> float | None:
    path = folds_dir / f"{run_key}_eeg_test.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    return float(df["balanced_acc"].mean())


def table_s2_from_eeg3_controls(
    summary_csv: Path,
    folds_dir: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if "run_key" in df.columns:
        df["condition"] = df["run_key"]
    elif "condition" not in df.columns:
        raise KeyError(f"{summary_csv} must contain run_key or condition")

    emg = (
        df.groupby(["condition", "subject", "behavior"], as_index=False)["balanced_acc_mean"]
        .mean()
        .rename(columns={"balanced_acc_mean": "test_emg"})
    )
    rows = []
    for condition in sorted(emg["condition"].unique(), key=lambda k: (_subject_sort_key(subject_from_run_key(k)), k)):
        subject = subject_from_run_key(condition)
        behavior = behavior_from_run_key(condition)
        test_emg = float(emg.loc[emg["condition"] == condition, "test_emg"].iloc[0])
        test_eeg = None
        if folds_dir is not None:
            test_eeg = _eeg_baseline_from_folds(folds_dir, condition)
        rows.append(
            {
                "subject": subject,
                "behavior": behavior,
                "test_eeg": test_eeg,
                "test_emg": test_emg,
            }
        )
    long = pd.DataFrame(rows)
    wide_rows = []
    for subject in sorted(long["subject"].unique(), key=_subject_sort_key):
        row: dict[str, object] = {"subject": subject}
        for behavior in BEHAVIOR_ORDER:
            hit = long[(long["subject"] == subject) & (long["behavior"] == behavior)]
            if len(hit):
                row[f"{behavior}_eeg"] = hit["test_eeg"].iloc[0]
                row[f"{behavior}_emg"] = hit["test_emg"].iloc[0]
        wide_rows.append(row)
    out = pd.DataFrame(wide_rows)
    avg: dict[str, object] = {"subject": "avg."}
    for behavior in BEHAVIOR_ORDER:
        avg[f"{behavior}_eeg"] = out[f"{behavior}_eeg"].mean(skipna=True)
        avg[f"{behavior}_emg"] = out[f"{behavior}_emg"].mean()
    return pd.concat([out, pd.DataFrame([avg])], ignore_index=True)


def print_table_s1(table: pd.DataFrame) -> None:
    print("Supplementary Table S1 (EMG train → EEG MI-top-3 test)")
    header = ["subject"] + [BEHAVIOR_HEADER[b] for b in BEHAVIOR_ORDER]
    print("\t".join(header))
    for _, row in table.iterrows():
        values = [row["subject"]]
        for behavior in BEHAVIOR_ORDER:
            val = row[behavior]
            values.append("" if pd.isna(val) else f"{float(val):.3f}")
        print("\t".join(str(v) for v in values))


def print_table_s2(table: pd.DataFrame) -> None:
    print("Supplementary Table S2 (EEG 3ch train → EEG vs EMG test)")
    print(
        "subject\t"
        + "\t".join(f"{BEHAVIOR_HEADER[b]} EEG" for b in BEHAVIOR_ORDER)
        + "\t"
        + "\t".join(f"{BEHAVIOR_HEADER[b]} EMG" for b in BEHAVIOR_ORDER)
    )
    for _, row in table.iterrows():
        parts = [row["subject"]]
        for behavior in BEHAVIOR_ORDER:
            val = row.get(f"{behavior}_eeg")
            parts.append("" if val is None or pd.isna(val) else f"{float(val):.3f}")
        for behavior in BEHAVIOR_ORDER:
            parts.append(f"{float(row[f'{behavior}_emg']):.3f}")
        print("\t".join(str(p) for p in parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-modal-dir",
        type=Path,
        default=Path("outputs/cross_modal_controls"),
        help="Directory with cross_modal_control_summary_by_condition.csv",
    )
    parser.add_argument(
        "--eeg3-dir",
        type=Path,
        default=Path("outputs/eeg3_mi_artifact_controls"),
        help="Directory with eeg3_mi_artifact_control_summary_by_condition.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/supplementary_tables"),
        help="Where to write table_s1.csv and table_s2.csv",
    )
    args = parser.parse_args()

    s1_path = args.cross_modal_dir / "cross_modal_control_summary_by_condition.csv"
    s2_path = args.eeg3_dir / "eeg3_mi_artifact_control_summary_by_condition.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if s1_path.is_file():
        s1 = table_s1_from_cross_modal(s1_path)
        s1.to_csv(args.out_dir / "table_s1.csv", index=False)
        print_table_s1(s1)
        print(f"\nWrote {args.out_dir / 'table_s1.csv'}")
    else:
        print(f"Skip Table S1: missing {s1_path}")

    if s2_path.is_file():
        s2 = table_s2_from_eeg3_controls(s2_path, folds_dir=args.eeg3_dir / "folds")
        s2.to_csv(args.out_dir / "table_s2.csv", index=False)
        print_table_s2(s2)
        print(f"\nWrote {args.out_dir / 'table_s2.csv'}")
    else:
        print(f"Skip Table S2: missing {s2_path}")


if __name__ == "__main__":
    main()
