#!/usr/bin/env python3
"""Supplementary Fig. S7: EEGNet training-jitter ablation (post-hoc online evaluation).

Manuscript panels use pseudo-online (post-hoc online) scores. Train jitter variants
with ``scripts/offline/run_jitter_ablation_offline.py``, evaluate with
``scripts/pseudo_online_test/run_jitter_ablation_pseudo_online.py``, then plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from scripts.figures._jitter_ablation_helpers import (
    BEHAVIOR_ORDER,
    JITTER_COLOR_BY_CONDITION,
    JITTER_ORDER,
    JITTER_PALETTE,
    aggregate_subject_jitter_rows,
    behavior_label,
    compute_statistical_tests,
    draw_significance_bracket,
    filter_jitter_conditions,
    format_p,
    infer_jitter_condition,
    jitter_label,
    load_test_history,
    make_markdown_condition,
    make_markdown_condition_behavior,
    normalize_subject_id,
    ordered_subject_offsets,
    plot_condition,
    plot_condition_behavior,
    slugify,
    summarize_condition,
    summarize_condition_behavior,
    write_stats_files,
    write_summary_files,
)
from scripts.figures._plt_style import despine

DEFAULT_OUTPUT = Path("outputs/figures/s7_jitter_ablation")
DEFAULT_ONLINE_JITTER_DIRS = {
    "fixed_jitter_m100ms": Path("outputs/online/jitter_ablation/fixed_jitter_m100ms"),
    "fixed_jitter_m50ms": Path("outputs/online/jitter_ablation/fixed_jitter_m50ms"),
    "fixed_jitter_p50ms": Path("outputs/online/jitter_ablation/fixed_jitter_p50ms"),
    "fixed_jitter_p100ms": Path("outputs/online/jitter_ablation/fixed_jitter_p100ms"),
    "no_jitter": Path("outputs/online/jitter_ablation/no_jitter"),
}


def read_online_aggregate_csv(path: Path, jitter_condition: str) -> dict | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Skipping {path}: {exc}")
        return None
    if df.empty:
        print(f"Skipping {path}: empty CSV")
        return None

    row = df.iloc[0]
    model_name = str(row.get("model_name", row.get("training_model_name", "")))
    subject = row.get("subject", row.get("sbj"))
    behavior = row.get("behavior")
    ensemble_method = row.get("ensemble_method", "")
    balanced_acc = row.get("balanced_acc", row.get("balanced_acc_test"))
    if pd.isna(subject) or pd.isna(behavior) or pd.isna(model_name) or pd.isna(balanced_acc):
        print(f"Skipping {path}: missing subject/behavior/model/balanced_acc")
        return None

    return {
        "sbj": normalize_subject_id(str(subject)),
        "behavior": str(behavior),
        "model_name": model_name,
        "ensemble_method": str(ensemble_method),
        "jitter_condition": jitter_condition,
        "balanced_acc_test": float(balanced_acc),
        "source_file": str(path),
    }


def load_online_condition_dir(
    path: Path,
    jitter_condition: str,
    model_name: str,
    ensemble_method: str,
    emg_input_mode: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input directory not found: {path}")

    rows = []
    for csv_path in sorted(path.glob("*.csv")):
        row = read_online_aggregate_csv(csv_path, jitter_condition)
        if row is not None:
            rows.append(row)

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError(f"No online jitter rows loaded from {path}")

    data = data[data["model_name"] == model_name].copy()
    data = data[data["ensemble_method"] == ensemble_method].copy()

    if emg_input_mode:
        keep_rows = []
        for _, loaded_row in data.iterrows():
            source = pd.read_csv(loaded_row["source_file"], nrows=1).iloc[0]
            keep_rows.append(str(source.get("emg_input_mode", "")) == emg_input_mode)
        data = data[keep_rows].copy()

    return data


def parse_condition_dir_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Condition directories must be jitter_condition=path, got: {value}")
    condition, path = value.split("=", 1)
    return condition.strip(), Path(path.strip())


def load_offline_histories(input_files: list[Path], model_name: str, ensemble_method: str) -> pd.DataFrame:
    frames = []
    for path in input_files:
        df = load_test_history(path)
        df["jitter_condition"] = infer_jitter_condition(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data[data["model_name"] == model_name].copy()
    data = data[data["ensemble_method"] == ensemble_method].copy()
    if data.empty:
        raise ValueError(
            f"No rows after filtering model_name={model_name!r} ensemble_method={ensemble_method!r}"
        )
    return filter_jitter_conditions(data)


def load_online_data(args: argparse.Namespace) -> pd.DataFrame:
    condition_dirs = DEFAULT_ONLINE_JITTER_DIRS.copy()
    for item in args.condition_dir:
        condition, path = parse_condition_dir_arg(item)
        condition_dirs[condition] = path

    frames = [
        load_online_condition_dir(
            args.baseline_dir,
            "random_jitter_±100ms",
            args.model_name,
            args.ensemble_method,
            args.emg_input_mode,
        )
    ]
    for condition in JITTER_ORDER:
        if condition == "random_jitter_±100ms" or condition not in condition_dirs:
            continue
        frames.append(
            load_online_condition_dir(
                condition_dirs[condition],
                condition,
                args.model_name,
                args.ensemble_method,
                args.emg_input_mode,
            )
        )
    return filter_jitter_conditions(pd.concat(frames, ignore_index=True))


def plot_online_condition(
    summary: pd.DataFrame,
    data_subj: pd.DataFrame,
    out_path: Path,
    stats_tables: dict[str, pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    jitter_levels = [
        condition for condition in JITTER_ORDER if condition in set(summary["jitter_condition"].astype(str))
    ]
    x = np.arange(len(jitter_levels))
    offset_map = ordered_subject_offsets(data_subj["sbj"])
    behavior_offsets = {
        behavior: offset
        for behavior, offset in zip(BEHAVIOR_ORDER, np.linspace(-0.06, 0.06, len(BEHAVIOR_ORDER)))
    }
    plot_data = data_subj.copy()
    plot_data["jitter_condition"] = plot_data["jitter_condition"].astype(str)
    for i, condition in enumerate(jitter_levels):
        condition_df = plot_data[plot_data["jitter_condition"] == condition]
        xs = [
            i
            + offset_map.get(str(row.sbj), 0.0)
            + behavior_offsets.get(str(row.behavior), 0.0)
            for row in condition_df.itertuples()
        ]
        ax.scatter(
            xs,
            condition_df["balanced_acc_test"],
            s=24,
            marker="o",
            color=JITTER_PALETTE[i % len(JITTER_PALETTE)],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.82,
            zorder=3,
        )
        ax.hlines(
            condition_df["balanced_acc_test"].mean(),
            i - 0.28,
            i + 0.28,
            color="black",
            linewidth=3.2,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([jitter_label(condition) for condition in jitter_levels], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Training jitter condition")
    ax.set_ylabel("Online balanced accuracy")
    ax.set_title("EEGNet online balanced accuracy by training jitter", pad=110)
    ax.axhline(0.2, color="0.45", linestyle="--", linewidth=0.8, zorder=1)
    ax.grid(False)
    despine(ax)

    pairwise = stats_tables["overall_pairwise"]
    if not pairwise.empty:
        bracket_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        y = 1.08
        for _, row in pairwise.iterrows():
            star = row.get("significance_bonferroni", "")
            comparison = str(row["comparison"])
            if star == "n.s." or comparison not in jitter_levels:
                continue
            draw_significance_bracket(
                ax,
                jitter_levels.index("random_jitter_±100ms"),
                jitter_levels.index(comparison),
                y,
                star,
                transform=bracket_transform,
            )
            y += 0.15

    fig.subplots_adjust(top=0.58, bottom=0.24, left=0.12, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_online_condition_behavior(
    data_subj: pd.DataFrame,
    out_path: Path,
    stats_tables: dict[str, pd.DataFrame],
) -> None:
    jitter_levels = [
        condition
        for condition in JITTER_ORDER
        if condition in set(data_subj["jitter_condition"].astype(str))
    ]
    plot_data = data_subj.copy()
    plot_data["jitter_condition"] = plot_data["jitter_condition"].astype(str)

    for behavior in BEHAVIOR_ORDER:
        behavior_data = plot_data[plot_data["behavior"] == behavior]
        if behavior_data.empty:
            continue
        behavior_out_path = out_path.with_name(f"{out_path.stem}_{slugify(behavior)}{out_path.suffix}")
        _plot_single_online_behavior(behavior, jitter_levels, behavior_data, behavior_out_path, stats_tables)


def _plot_single_online_behavior(
    behavior: str,
    jitter_levels: list[str],
    behavior_data: pd.DataFrame,
    out_path: Path,
    stats_tables: dict[str, pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(jitter_levels))
    subject_offsets = ordered_subject_offsets(behavior_data["sbj"])

    for i, condition in enumerate(jitter_levels):
        condition_df = behavior_data[behavior_data["jitter_condition"] == condition]
        if condition_df.empty:
            continue
        xs = [i + subject_offsets.get(str(subject), 0.0) for subject in condition_df["sbj"]]
        ax.scatter(
            xs,
            condition_df["balanced_acc_test"],
            s=24,
            marker="o",
            color=JITTER_COLOR_BY_CONDITION[condition],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.82,
            zorder=3,
        )
        ax.hlines(
            condition_df["balanced_acc_test"].mean(),
            i - 0.28,
            i + 0.28,
            color="black",
            linewidth=3.2,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([jitter_label(condition) for condition in jitter_levels], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Training jitter condition")
    ax.set_ylabel("Online balanced accuracy")
    ax.axhline(0.2, color="0.45", linestyle="--", linewidth=0.8, zorder=1)
    ax.grid(False)
    despine(ax)

    omnibus = stats_tables["behavior_omnibus"]
    omnibus_row = omnibus[omnibus["behavior"] == behavior]
    omnibus_text = ""
    if not omnibus_row.empty:
        p_value = omnibus_row.iloc[0]["p_value"]
        omnibus_text = f"\nFriedman p={format_p(p_value)}"
    ax.set_title(
        f"EEGNet online balanced accuracy by training jitter ({behavior_label(behavior)})"
        f"{omnibus_text}",
        pad=48,
    )

    pairwise = stats_tables["behavior_pairwise"]
    if not pairwise.empty:
        bracket_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        y = 1.04
        behavior_rows = pairwise[pairwise["behavior"] == behavior]
        for _, row in behavior_rows.iterrows():
            star = row.get("significance_bonferroni", "")
            comparison = str(row["comparison"])
            if star == "n.s." or comparison not in jitter_levels:
                continue
            draw_significance_bracket(
                ax,
                jitter_levels.index("random_jitter_±100ms"),
                jitter_levels.index(comparison),
                y,
                star,
                transform=bracket_transform,
            )
            y += 0.11

    fig.subplots_adjust(top=0.76, bottom=0.24, left=0.12, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_offline(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.input_files:
        raise ValueError("Offline mode requires --input-files with one or more test-history CSV paths.")
    input_files = [Path(p) for p in args.input_files]
    for path in input_files:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    data = load_offline_histories(input_files, args.model_name, args.ensemble_method)
    data_subj = aggregate_subject_jitter_rows(data)
    stats_tables = compute_statistical_tests(data_subj)
    write_stats_files(stats_tables, output_dir)

    summary_condition = summarize_condition(data_subj)
    condition_dir = output_dir / "by_jitter_condition"
    condition_plot = condition_dir / "offline_balanced_acc_by_jitter_condition.png"
    write_summary_files(
        summary_condition,
        make_markdown_condition(summary_condition),
        condition_dir,
        prefix="offline_jitter_condition_summary",
        plot_path=condition_plot,
    )
    plot_condition(summary_condition, data_subj, condition_plot, stats_tables)

    summary_behavior = summarize_condition_behavior(data_subj)
    behavior_dir = output_dir / "by_jitter_condition_behavior"
    behavior_plot = behavior_dir / "offline_balanced_acc_by_jitter_condition_behavior.png"
    write_summary_files(
        summary_behavior,
        make_markdown_condition_behavior(summary_behavior),
        behavior_dir,
        prefix="offline_jitter_condition_behavior_summary",
        plot_path=behavior_plot,
    )
    plot_condition_behavior(summary_behavior, data_subj, behavior_plot, stats_tables)


def run_online(args: argparse.Namespace, output_dir: Path) -> None:
    data = load_online_data(args)
    if data.empty:
        raise ValueError("No online jitter data loaded.")
    data_subj = aggregate_subject_jitter_rows(data)
    stats_tables = compute_statistical_tests(data_subj)
    write_stats_files(stats_tables, output_dir)

    summary_condition = summarize_condition(data_subj)
    condition_dir = output_dir / "by_jitter_condition"
    condition_plot = condition_dir / "online_balanced_acc_by_jitter_condition.png"
    write_summary_files(
        summary_condition,
        make_markdown_condition(summary_condition),
        condition_dir,
        prefix="online_jitter_condition_summary",
        plot_path=condition_plot,
    )
    plot_online_condition(summary_condition, data_subj, condition_plot, stats_tables)

    summary_behavior = summarize_condition_behavior(data_subj)
    behavior_dir = output_dir / "by_jitter_condition_behavior"
    behavior_plot = behavior_dir / "online_balanced_acc_by_jitter_condition_behavior.png"
    write_summary_files(
        summary_behavior,
        make_markdown_condition_behavior(summary_behavior),
        behavior_dir,
        prefix="online_jitter_condition_behavior_summary",
        plot_path=behavior_plot,
    )
    plot_online_condition_behavior(data_subj, behavior_plot, stats_tables)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-type",
        choices=("offline", "online"),
        default="offline",
        help="Offline rotating test histories vs online aggregate CSV directories.",
    )
    parser.add_argument(
        "--input-files",
        type=Path,
        nargs="*",
        default=[],
        help="Offline mode: test-history CSV files (jitter inferred from filename).",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/online/jitter_ablation/baseline"),
        help="Online mode: directory for random ±100 ms baseline aggregate CSVs.",
    )
    parser.add_argument(
        "--condition-dir",
        action="append",
        default=[],
        help="Online mode: override jitter_condition=path (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for summaries, statistics, and figures.",
    )
    parser.add_argument("--model-name", default="EEGNet")
    parser.add_argument("--ensemble-method", default="single")
    parser.add_argument(
        "--ensemble-method-online",
        dest="ensemble_method_online",
        default="zscore_mean",
        help="Default ensemble filter when --recording-type online.",
    )
    parser.add_argument("--emg-input-mode", default="real")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.recording_type
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.recording_type == "online":
        args.ensemble_method = args.ensemble_method_online
        run_online(args, output_dir)
    else:
        run_offline(args, output_dir)
    print(f"Jitter ablation outputs written to {output_dir}")


if __name__ == "__main__":
    main()
