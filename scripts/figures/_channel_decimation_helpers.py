"""Shared helpers for Supplementary Fig. S2b channel-decimation plots."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kendalltau, wilcoxon

from scripts.figures._plt_style import apply_rc, despine
from scripts.figures.condition_colors import condition_colors


def normalize_subject_id(subject: str) -> str:
    text = str(subject).strip()
    if text.startswith("sub-"):
        return text
    match = re.search(r"(\d+)", text)
    if match:
        return f"sub-{match.group(1)}"
    return text

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

apply_rc(PAPER_STYLE)

CHANNEL_ORDER = [4, 8, 16, 32, 128]
RANK_DECIMALS = 12
BEHAVIOR_ORDER = ["overt", "minimally_overt", "covert"]
BEHAVIOR_LABELS = {
    "overt": "overt",
    "minimally_overt": "minimally overt",
    "covert": "covert",
}
METRIC_COLUMNS = {
    "test": ["acc_test", "balanced_acc_test"],
    "cv": ["acc_val", "balanced_acc_val"],
}


def subject_sort_key(subject: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(subject))
    return (int(match.group(1)), str(subject)) if match else (10**9, str(subject))


def subject_label(subject: str) -> str:
    text = normalize_subject_id(subject)
    if text.startswith("sub-"):
        return text.replace("sub-", "sub. ")
    return text


def behavior_label(behavior: str) -> str:
    return BEHAVIOR_LABELS.get(str(behavior), str(behavior))


def infer_n_channels(path: Path) -> int:
    match = re.search(r"with_mask_(\d+)ch", path.name)
    if not match:
        raise ValueError(f"Could not infer channel count from filename: {path}")
    return int(match.group(1))


def infer_group(path: Path) -> str:
    match = re.search(r"_(g\d+)_(?:cv|test)\.csv$", path.name)
    return match.group(1) if match else ""


def read_histories(input_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted(
        input_dir.glob(f"history_color_rotating_test_fold_EEGNet_with_mask_*ch_*_{split}.csv")
    )
    if not paths:
        raise FileNotFoundError(f"No {split} CSVs found in {input_dir}")

    dfs = []
    required = {"sbj", "behavior", "model_name", "CV", *METRIC_COLUMNS[split]}
    if split == "test":
        required.update({"eval_type", "test_fold"})

    for path in paths:
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        df = df.copy()
        df["n_channels"] = infer_n_channels(path)
        df["channel_label"] = df["n_channels"].astype(str) + "ch"
        df["source_file"] = path.name
        df["group"] = infer_group(path)
        df["split"] = split
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out.rename(columns={"sbj": "subject"})
    out["subject"] = out["subject"].astype(str).map(normalize_subject_id)
    out["behavior"] = out["behavior"].astype(str)
    return out


def read_full_channel_history(path: Path, split: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sbj", "behavior", "model_name", "CV", *METRIC_COLUMNS[split]}
    if split == "test":
        required.update({"eval_type", "test_fold"})
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = df[df["model_name"].eq(model_name)].copy()
    if df.empty:
        raise ValueError(f"{path} has no rows for model_name={model_name!r}")

    df["n_channels"] = 128
    df["channel_label"] = "128ch"
    df["source_file"] = path.name
    df["group"] = "full"
    df["split"] = split
    df = df.rename(columns={"sbj": "subject"})
    df["subject"] = df["subject"].astype(str).map(normalize_subject_id)
    df["behavior"] = df["behavior"].astype(str)
    return df


def keep_single_eval(df: pd.DataFrame, test_eval_type: str) -> pd.DataFrame:
    if "eval_type" not in df.columns or test_eval_type == "all":
        return df.copy()
    return df[df["eval_type"].eq(test_eval_type)].copy()


def deduplicate(df: pd.DataFrame, split: str) -> pd.DataFrame:
    fold_col = "test_fold" if split == "test" and "test_fold" in df.columns else "CV"
    keys = ["subject", "behavior", "n_channels", fold_col]
    if "seed" in df.columns:
        keys.insert(0, "seed")
    return df.drop_duplicates(subset=keys, keep="last").copy()


def parse_seed_path_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Seed input must have the form SEED=PATH: {spec!r}")
        seed, raw_path = spec.split("=", 1)
        if not seed.strip() or not raw_path.strip():
            raise ValueError(f"Seed input must have the form SEED=PATH: {spec!r}")
        parsed.append((seed.strip(), Path(raw_path)))
    return parsed


def read_seeded_histories(specs: list[str], split: str) -> pd.DataFrame:
    frames = []
    for seed, input_dir in parse_seed_path_specs(specs):
        frame = read_histories(input_dir, split)
        frame.insert(0, "seed", seed)
        frame["source_dir"] = str(input_dir)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one --seed-input-dir is required in seeded mode")
    result = pd.concat(frames, ignore_index=True)
    fold_col = "test_fold" if split == "test" else "CV"
    keys = ["seed", "subject", "behavior", "n_channels", fold_col]
    duplicates = result.duplicated(keys, keep=False)
    if duplicates.any():
        example = result.loc[duplicates, keys].head().to_dict("records")
        raise ValueError(f"Duplicate seeded fold rows: {example}")
    return result


def summarize_seeded_folds(
    df: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-seed, seed-averaged subject, and across-subject summaries."""

    keys = ["seed", "subject", "behavior", "n_channels"]
    seed_subject = df.groupby(keys, observed=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    seed_subject.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in seed_subject.columns
    ]
    for metric in metrics:
        bad = seed_subject[seed_subject[f"{metric}_count"].ne(10)]
        if not bad.empty:
            raise ValueError(
                f"Expected 10 folds for every seed/subject/condition/density; "
                f"{len(bad)} {metric} cells differ"
            )

    subject_summary, group_summary = aggregate_seed_subject_values(seed_subject, metrics)
    return seed_subject, subject_summary, group_summary


def aggregate_seed_subject_values(
    seed_subject: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_keys = ["subject", "behavior", "n_channels"]
    mean_columns = [f"{metric}_mean" for metric in metrics]
    grouped = seed_subject.groupby(subject_keys, observed=False)
    subject_summary = grouped[mean_columns].mean().reset_index()
    seed_counts = grouped["seed"].nunique().rename("n_seeds").reset_index()
    subject_summary = subject_summary.merge(seed_counts, on=subject_keys, validate="one_to_one")
    expected_seeds = seed_subject["seed"].nunique()
    if not subject_summary["n_seeds"].eq(expected_seeds).all():
        raise ValueError("At least one subject/condition/density cell is missing a seed")
    for column in mean_columns:
        seed_std = grouped[column].std().rename(f"{column}_seed_std").reset_index()
        subject_summary = subject_summary.merge(seed_std, on=subject_keys, validate="one_to_one")
    return subject_summary, summarize_subject_values(subject_summary, metrics)


def summarize_subject_values(subject_summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    value_columns = {metric: f"{metric}_mean" for metric in metrics}
    grouped = subject_summary.groupby(["behavior", "n_channels"], observed=False)
    result = grouped.size().rename("n_subjects").reset_index()
    for metric, column in value_columns.items():
        metric_name = "balanced_acc" if metric.startswith("balanced_acc") else "acc"
        values = grouped[column].agg(["mean", "std"]).reset_index()
        values = values.rename(
            columns={"mean": f"{metric_name}_mean", "std": f"{metric_name}_std"}
        )
        result = result.merge(values, on=["behavior", "n_channels"], validate="one_to_one")
        result[f"{metric_name}_sem"] = result[f"{metric_name}_std"] / np.sqrt(
            result["n_subjects"]
        )
    return result


def make_per_seed_group_summary(seed_subject: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    long = (
        seed_subject.groupby(["seed", "behavior", "n_channels"], observed=False)[metric_col]
        .agg(n_subjects="count", across_subject_mean="mean")
        .reset_index()
    )
    wide = long.pivot_table(
        index=["behavior", "n_channels"],
        columns="seed",
        values="across_subject_mean",
        observed=False,
    )
    wide = wide.rename(columns={column: f"seed_{column}_mean" for column in wide.columns})
    wide["across_seed_sd"] = wide.std(axis=1, ddof=1)
    return wide.reset_index()


def make_seed_variability(
    seed_subject: pd.DataFrame,
    metric_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["subject", "behavior", "n_channels"]
    per_subject = (
        seed_subject.groupby(keys, observed=False)[metric_col]
        .agg(n_seeds="count", seed_mean="mean", seed_sd="std")
        .reset_index()
    )
    summary = (
        per_subject.groupby(["behavior", "n_channels"], observed=False)["seed_sd"]
        .agg(n_subjects="count", median_seed_sd="median", max_seed_sd="max")
        .reset_index()
    )
    return per_subject, summary


def make_32_vs_128_stats(subject_summary: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows = []
    for behavior in BEHAVIOR_ORDER:
        table = make_paired_table(subject_summary, metric_col, behavior)
        values_32 = table[32].to_numpy(dtype=float)
        values_128 = table[128].to_numpy(dtype=float)
        difference = np.round(values_128 - values_32, decimals=RANK_DECIMALS)
        if np.allclose(difference, 0):
            statistic, p_raw = np.nan, 1.0
        else:
            result = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
            statistic, p_raw = result.statistic, result.pvalue
        rows.append(
            {
                "scope": behavior,
                "metric": metric_col,
                "channel_a": 32,
                "channel_b": 128,
                "n_units": len(table),
                "mean_32": values_32.mean(),
                "mean_128": values_128.mean(),
                "mean_diff_128_minus_32": difference.mean(),
                "statistic": statistic,
                "p_raw": p_raw,
            }
        )
    adjusted = bonferroni_adjust([row["p_raw"] for row in rows])
    for row, p_value in zip(rows, adjusted):
        row["p_bonferroni_across_conditions"] = p_value
        row["significant_bonferroni_0.05"] = bool(p_value < 0.05)
    return pd.DataFrame(rows)


def summarize_folds(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_summary = (
        df.groupby(["subject", "behavior", "n_channels"], observed=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    subject_summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in subject_summary.columns
    ]

    group_summary = (
        subject_summary.groupby(["behavior", "n_channels"], observed=False)
        .agg(
            n_subjects=("subject", "count"),
            acc_mean=("acc_test_mean" if "acc_test_mean" in subject_summary.columns else "acc_val_mean", "mean"),
            acc_std=("acc_test_mean" if "acc_test_mean" in subject_summary.columns else "acc_val_mean", "std"),
            balanced_acc_mean=(
                "balanced_acc_test_mean"
                if "balanced_acc_test_mean" in subject_summary.columns
                else "balanced_acc_val_mean",
                "mean",
            ),
            balanced_acc_std=(
                "balanced_acc_test_mean"
                if "balanced_acc_test_mean" in subject_summary.columns
                else "balanced_acc_val_mean",
                "std",
            ),
        )
        .reset_index()
    )
    group_summary["acc_sem"] = group_summary["acc_std"] / np.sqrt(group_summary["n_subjects"])
    group_summary["balanced_acc_sem"] = group_summary["balanced_acc_std"] / np.sqrt(
        group_summary["n_subjects"]
    )
    return subject_summary, group_summary


def bonferroni_adjust(p_values: list[float]) -> np.ndarray:
    return np.minimum(np.asarray(p_values, dtype=float) * len(p_values), 1.0)


def make_paired_table(
    subject_summary: pd.DataFrame,
    metric_col: str,
    behavior: str | None,
) -> pd.DataFrame:
    df = subject_summary.copy()
    if behavior is not None:
        df = df[df["behavior"].eq(behavior)]
        index_cols = ["subject"]
    else:
        index_cols = ["subject", "behavior"]
    table = df.pivot_table(
        index=index_cols,
        columns="n_channels",
        values=metric_col,
        observed=False,
    )
    table = table.reindex(columns=CHANNEL_ORDER)
    return table.dropna().round(RANK_DECIMALS)


def make_kendall_stats(subject_summary: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows = []
    for (subject, behavior), group in subject_summary.groupby(["subject", "behavior"], observed=False):
        ordered = group.set_index("n_channels").reindex(CHANNEL_ORDER)
        values = np.round(
            ordered[metric_col].to_numpy(dtype=float), decimals=RANK_DECIMALS
        )
        valid = ~np.isnan(values)
        tau, p_value = (np.nan, np.nan)
        if valid.sum() >= 2:
            tau, p_value = kendalltau(np.asarray(CHANNEL_ORDER)[valid], values[valid])
        rows.append(
            {
                "subject": subject,
                "behavior": behavior,
                "metric": metric_col,
                "n_channels": int(valid.sum()),
                "kendall_tau": tau,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def make_repeated_measure_stats(subject_summary: pd.DataFrame, metric_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows = []
    pairwise_rows = []
    scopes = [(behavior, behavior) for behavior in BEHAVIOR_ORDER]
    scopes.append(("all_behaviors", None))

    for scope_name, behavior in scopes:
        table = make_paired_table(subject_summary, metric_col, behavior)
        if table.empty:
            continue

        friedman = friedmanchisquare(*(table[channel].to_numpy() for channel in CHANNEL_ORDER))
        global_rows.append(
            {
                "scope": scope_name,
                "metric": metric_col,
                "test": "Friedman",
                "n_units": len(table),
                "n_channels": len(CHANNEL_ORDER),
                "statistic": friedman.statistic,
                "p_value": friedman.pvalue,
            }
        )

        raw_p_values = []
        row_start = len(pairwise_rows)
        for channel_a, channel_b in itertools.combinations(CHANNEL_ORDER, 2):
            values_a = table[channel_a].to_numpy(dtype=float)
            values_b = table[channel_b].to_numpy(dtype=float)
            diff = np.round(values_b - values_a, decimals=RANK_DECIMALS)
            if np.allclose(diff, 0):
                stat, p_value = np.nan, np.nan
            else:
                result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                stat, p_value = result.statistic, result.pvalue
            raw_p_values.append(p_value)
            pairwise_rows.append(
                {
                    "scope": scope_name,
                    "metric": metric_col,
                    "channel_a": channel_a,
                    "channel_b": channel_b,
                    "test": "Wilcoxon signed-rank",
                    "n_units": len(table),
                    "mean_a": values_a.mean(),
                    "mean_b": values_b.mean(),
                    "mean_diff_b_minus_a": diff.mean(),
                    "median_diff_b_minus_a": np.median(diff),
                    "statistic": stat,
                    "p_raw": p_value,
                }
            )

        adjusted = bonferroni_adjust([1.0 if pd.isna(p) else p for p in raw_p_values])
        for i, p_adjusted in enumerate(adjusted, start=row_start):
            pairwise_rows[i]["p_bonferroni"] = p_adjusted
            pairwise_rows[i]["significant_bonferroni_0.05"] = p_adjusted < 0.05

    return pd.DataFrame(global_rows), pd.DataFrame(pairwise_rows)


def make_wide_table(subject_summary: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    table = subject_summary.pivot_table(
        index="subject",
        columns=["behavior", "n_channels"],
        values=metric_col,
        observed=False,
    )
    ordered_columns = pd.MultiIndex.from_product([BEHAVIOR_ORDER, CHANNEL_ORDER])
    table = table.reindex(columns=ordered_columns)
    table.loc["mean"] = table.mean(axis=0)
    table.loc["std"] = table.iloc[:-1].std(axis=0)
    n_subjects = len([idx for idx in table.index if str(idx).startswith("sub-")])
    table.loc["sem"] = table.iloc[:-2].std(axis=0) / np.sqrt(max(n_subjects, 1))
    table.columns = [f"{behavior_label(behavior)} / {channel}ch" for behavior, channel in table.columns]
    return table.reset_index().rename(columns={"index": "subject"})


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(table.columns) - 1)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("--")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def escape_latex(text) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def write_latex_table(table: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    column_spec = "l" + "c" * (len(table.columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(escape_latex(col) for col in table.columns) + r" \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("--")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.3f}")
            else:
                cells.append(escape_latex(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            rf"\caption{{{escape_latex(caption)}}}",
            rf"\label{{{escape_latex(label)}}}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def p_to_star(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plot_overall(
    summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
    metric: str,
    subject_metric_col: str,
    out_base: Path,
    stats_df: pd.DataFrame | None = None,
) -> None:
    value_col = f"{metric}_mean"
    palette = condition_colors(BEHAVIOR_ORDER)
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    channel_positions = np.arange(len(CHANNEL_ORDER), dtype=float)
    group_gap = 1.65
    group_width = len(CHANNEL_ORDER)
    xticks = []
    xticklabels = []
    subjects = sorted(subject_summary["subject"].dropna().unique(), key=subject_sort_key)
    subject_offsets = {
        subject: offset
        for subject, offset in zip(subjects, np.linspace(-0.16, 0.16, max(len(subjects), 1)))
    }

    for group_idx, (color, behavior) in enumerate(zip(palette, BEHAVIOR_ORDER)):
        offset = group_idx * (group_width + group_gap)
        x = offset + channel_positions
        mean_data = (
            summary[summary["behavior"].eq(behavior)]
            .set_index("n_channels")
            .reindex(CHANNEL_ORDER)
        )
        subject_data = subject_summary[subject_summary["behavior"].eq(behavior)]
        for channel_idx, channel in enumerate(CHANNEL_ORDER):
            channel_subjects = subject_data[subject_data["n_channels"].eq(channel)]
            xs = [
                x[channel_idx] + subject_offsets.get(subject, 0.0)
                for subject in channel_subjects["subject"]
            ]
            ax.scatter(
                xs,
                channel_subjects[subject_metric_col],
                s=18,
                marker="o",
                color=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.78,
                zorder=3,
            )
            mean_value = mean_data.loc[channel, value_col]
            if pd.notna(mean_value):
                ax.hlines(
                    mean_value,
                    x[channel_idx] - 0.28,
                    x[channel_idx] + 0.28,
                    color="black",
                    linewidth=2.7,
                    zorder=4,
                )
        xticks.extend(x)
        xticklabels.extend([str(channel) for channel in CHANNEL_ORDER])
        center = offset + (group_width - 1) / 2
        ax.text(
            center,
            -0.19,
            behavior_label(behavior),
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
        )

        if stats_df is not None:
            row = stats_df[stats_df["scope"].eq(behavior)]
            if not row.empty:
                star = p_to_star(float(row.iloc[0]["p_value"]))
                if star:
                    y = 0.98
                    ax.plot([x[0], x[-1]], [y, y], color="black", linewidth=0.8)
                    ax.text(center, y + 0.015, star, ha="center", va="bottom", fontsize=10)

    ax.axhline(0.2, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("# of electrodes used for decoding", labelpad=34)
    ax.set_ylabel("balanced accuracy" if metric == "balanced_acc" else "accuracy")
    ax.set_ylim(0, 1.04)
    ax.set_xlim(-0.7, xticks[-1] + 0.7)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(False)
    ax.yaxis.grid(False)
    despine(ax)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.30)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_subject_panels(subject_summary: pd.DataFrame, metric_col: str, out_base: Path) -> None:
    subjects = sorted(subject_summary["subject"].unique(), key=subject_sort_key)
    palette = condition_colors(BEHAVIOR_ORDER)
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.8), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for ax, subject in zip(axes, subjects):
        subject_df = subject_summary[subject_summary["subject"].eq(subject)]
        for color, behavior in zip(palette, BEHAVIOR_ORDER):
            data = (
                subject_df[subject_df["behavior"].eq(behavior)]
                .set_index("n_channels")
                .reindex(CHANNEL_ORDER)
            )
            ax.plot(
                np.arange(len(CHANNEL_ORDER)),
                data[metric_col],
                marker="o",
                linewidth=1.0,
                label=behavior_label(behavior),
                color=color,
            )
        ax.axhline(0.2, color="black", linestyle="--", linewidth=0.6)
        ax.set_title(subject_label(subject))
        ax.set_xticks(np.arange(len(CHANNEL_ORDER)))
        ax.set_xticklabels([str(channel) for channel in CHANNEL_ORDER])
        ax.set_ylim(0, 1)
        ax.grid(False)
        despine(ax)

    for ax in axes[len(subjects) :]:
        ax.axis("off")

    for ax in axes[::3]:
        ax.set_ylabel("Balanced accuracy")
    for ax in axes[-3:]:
        ax.set_xlabel("Channels")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def load_group_summary_csv(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    required = {"behavior", "n_channels", "balanced_acc_mean"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return summary


def load_subject_summary_csv(path: Path) -> pd.DataFrame:
    subject_summary = pd.read_csv(path)
    if "subject" not in subject_summary.columns:
        raise ValueError(f"{path} must include a subject column")
    subject_summary = subject_summary.copy()
    subject_summary["subject"] = subject_summary["subject"].astype(str).map(normalize_subject_id)
    return subject_summary


def plot_decimation_from_summary_csvs(
    group_summary_csv: Path,
    subject_summary_csv: Path | None,
    figures_dir: Path,
    *,
    metric: str = "balanced_acc",
    subject_metric_col: str = "balanced_acc_test_mean",
    figure_stem: str = "channel_decimation_balanced_acc",
    stats_csv: Path | None = None,
) -> None:
    """Plot balanced accuracy vs channel count from pre-aggregated CSV tables."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    group_summary = load_group_summary_csv(group_summary_csv)
    subject_summary = (
        load_subject_summary_csv(subject_summary_csv)
        if subject_summary_csv is not None
        else pd.DataFrame()
    )
    stats_df = pd.read_csv(stats_csv) if stats_csv is not None else None
    plot_overall(
        group_summary,
        subject_summary,
        metric,
        subject_metric_col,
        figures_dir / figure_stem,
        stats_df,
    )
    if not subject_summary.empty and subject_metric_col in subject_summary.columns:
        plot_subject_panels(
            subject_summary,
            subject_metric_col,
            figures_dir / f"{figure_stem}_by_subject",
        )


