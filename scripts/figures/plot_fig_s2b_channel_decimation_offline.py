#!/usr/bin/env python3
"""Supplementary Fig. S2b (offline): EEGNet channel decimation from rotating CV histories."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.figures._channel_decimation_helpers import (
    BEHAVIOR_ORDER,
    CHANNEL_ORDER,
    METRIC_COLUMNS,
    deduplicate,
    keep_single_eval,
    make_32_vs_128_stats,
    make_kendall_stats,
    make_per_seed_group_summary,
    make_repeated_measure_stats,
    make_seed_variability,
    make_wide_table,
    plot_decimation_from_summary_csvs,
    plot_overall,
    plot_subject_panels,
    read_full_channel_history,
    read_histories,
    read_seeded_histories,
    subject_sort_key,
    summarize_folds,
    summarize_seeded_folds,
    write_latex_table,
    write_markdown_table,
)

DEFAULT_OUTPUT = Path("outputs/figures/s2b_channel_decimation_offline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-summary-csv",
        type=Path,
        default=None,
        help="Plot directly from a group summary CSV (behavior, n_channels, balanced_acc_mean).",
    )
    parser.add_argument(
        "--subject-summary-csv",
        type=Path,
        default=None,
        help="Optional subject-level CSV for scatter overlays and per-subject panels.",
    )
    parser.add_argument(
        "--friedman-stats-csv",
        type=Path,
        default=None,
        help="Optional Friedman stats CSV (scope, p_value) for significance stars.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/offline/channel_decimation_EEGNet"),
    )
    parser.add_argument(
        "--seed-input-dir",
        action="append",
        default=[],
        metavar="SEED=PATH",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--full-test-history",
        type=Path,
        default=Path("outputs/offline/baseline/history_color_rotating_test_fold_test.csv"),
    )
    parser.add_argument(
        "--full-cv-history",
        type=Path,
        default=Path("outputs/offline/baseline/history_color_rotating_test_fold_cv.csv"),
    )
    parser.add_argument("--full-model-name", default="EEGNet")
    parser.add_argument("--test-eval-type", default="single")
    parser.add_argument("--deduplicate-folds", action="store_true")
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
            figure_stem="channel_decimation_offline_balanced_acc",
            stats_csv=args.friedman_stats_csv,
        )
        print(f"Plotted offline channel decimation from {args.group_summary_csv}")
        return

    tables_dir = args.output_dir / "tables"
    stats_dir = args.output_dir / "stats"
    for directory in (tables_dir, stats_dir):
        directory.mkdir(parents=True, exist_ok=True)

    seeded_mode = bool(args.seed_input_dir)
    if seeded_mode:
        test_folds = read_seeded_histories(args.seed_input_dir, "test")
        cv_folds = read_seeded_histories(args.seed_input_dir, "cv")
    else:
        import pandas as pd

        test_folds = pd.concat(
            [
                read_histories(args.input_dir, "test"),
                read_full_channel_history(args.full_test_history, "test", args.full_model_name),
            ],
            ignore_index=True,
        )
        cv_folds = pd.concat(
            [
                read_histories(args.input_dir, "cv"),
                read_full_channel_history(args.full_cv_history, "cv", args.full_model_name),
            ],
            ignore_index=True,
        )
    test_folds = keep_single_eval(test_folds, args.test_eval_type)
    if args.deduplicate_folds:
        test_folds = deduplicate(test_folds, "test")
        cv_folds = deduplicate(cv_folds, "cv")

    if seeded_mode:
        test_seed_subject, test_subject, test_group = summarize_seeded_folds(
            test_folds, METRIC_COLUMNS["test"]
        )
        cv_seed_subject, cv_subject, cv_group = summarize_seeded_folds(
            cv_folds, METRIC_COLUMNS["cv"]
        )
    else:
        test_subject, test_group = summarize_folds(test_folds, METRIC_COLUMNS["test"])
        cv_subject, cv_group = summarize_folds(cv_folds, METRIC_COLUMNS["cv"])

    test_folds.to_csv(tables_dir / "channel_decimation_test_folds.csv", index=False)
    cv_folds.to_csv(tables_dir / "channel_decimation_validation_folds.csv", index=False)
    test_subject.to_csv(tables_dir / "channel_decimation_test_subject_summary.csv", index=False)
    cv_subject.to_csv(tables_dir / "channel_decimation_validation_subject_summary.csv", index=False)

    if seeded_mode:
        test_seed_subject.to_csv(
            tables_dir / "channel_decimation_test_per_seed_subject_summary.csv", index=False
        )
        cv_seed_subject.to_csv(
            tables_dir / "channel_decimation_validation_per_seed_subject_summary.csv", index=False
        )
        for split, seed_subject, metric_col in [
            ("test", test_seed_subject, "balanced_acc_test_mean"),
            ("validation", cv_seed_subject, "balanced_acc_val_mean"),
        ]:
            make_per_seed_group_summary(seed_subject, metric_col).to_csv(
                tables_dir / f"channel_decimation_{split}_per_seed_group_summary.csv",
                index=False,
            )
            per_subject_variability, variability_summary = make_seed_variability(
                seed_subject, metric_col
            )
            per_subject_variability.to_csv(
                tables_dir / f"channel_decimation_{split}_per_subject_seed_variability.csv",
                index=False,
            )
            variability_summary.to_csv(
                tables_dir / f"channel_decimation_{split}_seed_variability_summary.csv",
                index=False,
            )

    wide_test = make_wide_table(test_subject, "balanced_acc_test_mean")
    wide_cv = make_wide_table(cv_subject, "balanced_acc_val_mean")
    wide_test.to_csv(tables_dir / "channel_decimation_test_balanced_acc_wide.csv", index=False)
    wide_cv.to_csv(tables_dir / "channel_decimation_validation_balanced_acc_wide.csv", index=False)
    write_markdown_table(wide_test, tables_dir / "channel_decimation_test_balanced_acc_table.md")
    write_markdown_table(wide_cv, tables_dir / "channel_decimation_validation_balanced_acc_table.md")
    write_latex_table(
        wide_test,
        tables_dir / "channel_decimation_test_balanced_acc_table.tex",
        "Balanced test accuracy for offline EEGNet channel decimation.",
        "tab:channel_decimation_offline_test",
    )
    write_latex_table(
        wide_cv,
        tables_dir / "channel_decimation_validation_balanced_acc_table.tex",
        "Balanced validation accuracy for offline EEGNet channel decimation.",
        "tab:channel_decimation_offline_validation",
    )

    global_stats_by_split_metric = {}
    for split, subject_summary, metric_suffix in [
        ("test", test_subject, "test"),
        ("validation", cv_subject, "val"),
    ]:
        for metric_base in ["acc", "balanced_acc"]:
            metric_col = f"{metric_base}_{metric_suffix}_mean"
            kendall = make_kendall_stats(subject_summary, metric_col)
            global_stats, pairwise_stats = make_repeated_measure_stats(subject_summary, metric_col)
            global_stats_by_split_metric[(split, metric_base)] = global_stats
            kendall.to_csv(stats_dir / f"channel_decimation_{split}_{metric_base}_kendall.csv", index=False)
            global_stats.to_csv(
                stats_dir / f"channel_decimation_{split}_{metric_base}_friedman.csv",
                index=False,
            )
            pairwise_stats.to_csv(
                stats_dir / f"channel_decimation_{split}_{metric_base}_wilcoxon_pairwise.csv",
                index=False,
            )
            make_32_vs_128_stats(subject_summary, metric_col).to_csv(
                stats_dir / f"channel_decimation_{split}_{metric_base}_32_vs_128.csv",
                index=False,
            )

    plot_overall(
        test_group,
        test_subject,
        "balanced_acc",
        "balanced_acc_test_mean",
        figures_dir / "channel_decimation_offline_test_balanced_acc",
        global_stats_by_split_metric[("test", "balanced_acc")],
    )
    plot_overall(
        cv_group,
        cv_subject,
        "balanced_acc",
        "balanced_acc_val_mean",
        figures_dir / "channel_decimation_offline_validation_balanced_acc",
        global_stats_by_split_metric[("validation", "balanced_acc")],
    )
    plot_subject_panels(
        test_subject,
        "balanced_acc_test_mean",
        figures_dir / "channel_decimation_offline_test_balanced_acc_by_subject",
    )
    plot_subject_panels(
        cv_subject,
        "balanced_acc_val_mean",
        figures_dir / "channel_decimation_offline_validation_balanced_acc_by_subject",
    )

    run_info = [
        f"input_dir: {args.input_dir}",
        f"full_test_history: {args.full_test_history}",
        f"full_cv_history: {args.full_cv_history}",
        f"full_model_name: {args.full_model_name}",
        f"output_dir: {args.output_dir}",
        f"test_eval_type: {args.test_eval_type}",
        f"deduplicate_folds: {args.deduplicate_folds}",
        f"seeded_mode: {seeded_mode}",
        f"subjects: {', '.join(sorted(test_folds['subject'].unique(), key=subject_sort_key))}",
        f"behaviors: {', '.join(BEHAVIOR_ORDER)}",
        f"channels: {', '.join(map(str, CHANNEL_ORDER))}",
    ]
    (args.output_dir / "run_info.txt").write_text("\n".join(run_info) + "\n")
    print(f"Saved offline channel decimation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
