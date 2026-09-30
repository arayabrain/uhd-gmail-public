#!/usr/bin/env python3
"""Supplementary Fig. S2b (online): EEGNet channel decimation from aggregate online CSVs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from scripts.figures._channel_decimation_helpers import (
    BEHAVIOR_ORDER,
    CHANNEL_ORDER,
    aggregate_seed_subject_values,
    make_32_vs_128_stats,
    make_kendall_stats,
    make_per_seed_group_summary,
    make_repeated_measure_stats,
    make_seed_variability,
    make_wide_table,
    normalize_subject_id,
    parse_seed_path_specs,
    plot_decimation_from_summary_csvs,
    plot_overall,
    plot_subject_panels,
    subject_sort_key,
    summarize_folds,
    write_latex_table,
    write_markdown_table,
)

DEFAULT_OUTPUT = Path("outputs/figures/s2b_channel_decimation_online")


def infer_n_channels(row: pd.Series, path: Path) -> int:
    value = row.get("n_channels", "")
    if pd.notna(value) and str(value).strip():
        return int(float(value))
    model_name = str(row.get("model_name", ""))
    match = re.search(r"(\d+)ch", model_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer n_channels for {path}")


def collect_results(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].copy()
        rows.append(
            {
                "subject": normalize_subject_id(str(row["subject"])),
                "behavior": str(row["behavior"]),
                "n_channels": infer_n_channels(row, path),
                "model_name": str(row["model_name"]),
                "n_valid_used": int(row.get("n_valid_used", len(df))),
                "trial_rows": len(df),
                "acc_test": float(row["acc"]),
                "balanced_acc_test": float(row["balanced_acc"]),
                "source_file": path.name,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No online aggregate CSVs found in {input_dir}")
    result = pd.DataFrame(rows)
    result = result[result["n_channels"].isin(CHANNEL_ORDER)].copy()
    result["behavior"] = pd.Categorical(result["behavior"], categories=BEHAVIOR_ORDER, ordered=True)
    result = result.sort_values(["subject", "behavior", "n_channels", "source_file"])
    keys = ["subject", "behavior", "n_channels"]
    duplicates = result.duplicated(keys, keep=False)
    if duplicates.any():
        example = result.loc[duplicates, keys + ["source_file"]].head().to_dict("records")
        raise ValueError(f"Duplicate online result cells in {input_dir}: {example}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-summary-csv", type=Path, default=None)
    parser.add_argument("--subject-summary-csv", type=Path, default=None)
    parser.add_argument("--friedman-stats-csv", type=Path, default=None)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/online/channel_decimation_EEGNet"),
    )
    parser.add_argument("--seed-input-dir", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--seed-128-input-dir", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--expected-trials", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.group_summary_csv is not None:
        plot_decimation_from_summary_csvs(
            args.group_summary_csv,
            args.subject_summary_csv,
            figures_dir,
            subject_metric_col="balanced_acc_test_mean",
            figure_stem="channel_decimation_online_balanced_acc",
            stats_csv=args.friedman_stats_csv,
        )
        print(f"Plotted online channel decimation from {args.group_summary_csv}")
        return

    tables_dir = args.output_dir / "tables"
    stats_dir = args.output_dir / "stats"
    for directory in (tables_dir, stats_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seeded_mode = bool(args.seed_input_dir)
    if seeded_mode:
        full_overrides = parse_seed_path_specs(args.seed_128_input_dir)
        override_seeds = {seed for seed, _ in full_overrides}
        if len(override_seeds) != len(full_overrides):
            raise ValueError("Each seed may have at most one --seed-128-input-dir")
        frames = []
        for seed, input_dir in parse_seed_path_specs(args.seed_input_dir):
            frame = collect_results(input_dir)
            if seed in override_seeds:
                frame = frame[frame["n_channels"].ne(128)].copy()
            frame.insert(0, "seed", seed)
            frame["source_dir"] = str(input_dir)
            frames.append(frame)
        for seed, input_dir in full_overrides:
            frame = collect_results(input_dir)
            frame = frame[frame["n_channels"].eq(128)].copy()
            frame.insert(0, "seed", seed)
            frame["source_dir"] = str(input_dir)
            frames.append(frame)
        folds = pd.concat(frames, ignore_index=True)
        keys = ["seed", "subject", "behavior", "n_channels"]
        duplicates = folds.duplicated(keys, keep=False)
        if duplicates.any():
            example = folds.loc[duplicates, keys].head().to_dict("records")
            raise ValueError(f"Duplicate seeded online cells: {example}")
        if args.expected_trials is not None:
            if not folds["n_valid_used"].eq(args.expected_trials).all() or not folds[
                "trial_rows"
            ].eq(args.expected_trials).all():
                raise ValueError(
                    f"Every seeded online result must contain {args.expected_trials} used trials"
                )
        seed_subject = folds.rename(
            columns={
                "acc_test": "acc_test_mean",
                "balanced_acc_test": "balanced_acc_test_mean",
            }
        )
        subject_summary, group_summary = aggregate_seed_subject_values(
            seed_subject,
            ["acc_test", "balanced_acc_test"],
        )
    else:
        folds = collect_results(args.input_dir)
        subject_summary, group_summary = summarize_folds(folds, ["acc_test", "balanced_acc_test"])

    folds.to_csv(tables_dir / "channel_decimation_online_trials.csv", index=False)
    subject_summary.to_csv(tables_dir / "channel_decimation_online_subject_summary.csv", index=False)
    if seeded_mode:
        seed_subject.to_csv(
            tables_dir / "channel_decimation_online_per_seed_subject_summary.csv",
            index=False,
        )
        make_per_seed_group_summary(seed_subject, "balanced_acc_test_mean").to_csv(
            tables_dir / "channel_decimation_online_per_seed_group_summary.csv",
            index=False,
        )
        per_subject_variability, variability_summary = make_seed_variability(
            seed_subject, "balanced_acc_test_mean"
        )
        per_subject_variability.to_csv(
            tables_dir / "channel_decimation_online_per_subject_seed_variability.csv",
            index=False,
        )
        variability_summary.to_csv(
            tables_dir / "channel_decimation_online_seed_variability_summary.csv",
            index=False,
        )

    wide = make_wide_table(subject_summary, "balanced_acc_test_mean")
    wide.to_csv(tables_dir / "channel_decimation_online_balanced_acc_wide.csv", index=False)
    write_markdown_table(wide, tables_dir / "channel_decimation_online_balanced_acc_table.md")
    write_latex_table(
        wide,
        tables_dir / "channel_decimation_online_balanced_acc_table.tex",
        "Balanced online accuracy for EEGNet channel decimation.",
        "tab:channel_decimation_online",
    )

    global_stats_for_plot = None
    for metric_base in ["acc", "balanced_acc"]:
        metric_col = f"{metric_base}_test_mean"
        kendall = make_kendall_stats(subject_summary, metric_col)
        global_stats, pairwise_stats = make_repeated_measure_stats(subject_summary, metric_col)
        if metric_base == "balanced_acc":
            global_stats_for_plot = global_stats
        kendall.to_csv(
            stats_dir / f"channel_decimation_online_{metric_base}_kendall.csv",
            index=False,
        )
        global_stats.to_csv(
            stats_dir / f"channel_decimation_online_{metric_base}_friedman.csv",
            index=False,
        )
        pairwise_stats.to_csv(
            stats_dir / f"channel_decimation_online_{metric_base}_wilcoxon_pairwise.csv",
            index=False,
        )
        make_32_vs_128_stats(subject_summary, metric_col).to_csv(
            stats_dir / f"channel_decimation_online_{metric_base}_32_vs_128.csv",
            index=False,
        )

    plot_overall(
        group_summary,
        subject_summary,
        "balanced_acc",
        "balanced_acc_test_mean",
        figures_dir / "channel_decimation_online_balanced_acc",
        global_stats_for_plot,
    )
    plot_subject_panels(
        subject_summary,
        "balanced_acc_test_mean",
        figures_dir / "channel_decimation_online_balanced_acc_by_subject",
    )

    run_info = [
        f"input_dir: {args.input_dir}",
        f"output_dir: {args.output_dir}",
        f"seeded_mode: {seeded_mode}",
        f"rows: {len(folds)}",
        f"subjects: {', '.join(sorted(folds['subject'].unique(), key=subject_sort_key))}",
        f"behaviors: {', '.join(BEHAVIOR_ORDER)}",
        f"channels: {', '.join(map(str, CHANNEL_ORDER))}",
    ]
    if seeded_mode:
        run_info.append(f"seeds: {', '.join(map(str, sorted(folds['seed'].unique())))}")
    (args.output_dir / "run_info.txt").write_text("\n".join(run_info) + "\n")
    print(f"Saved online channel decimation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
