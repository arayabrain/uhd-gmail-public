"""Shared helpers for Supplementary Fig. S7 jitter-ablation plots."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from scipy import stats

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

JITTER_ORDER = [
    "random_jitter_±100ms",
    "fixed_jitter_m100ms",
    "fixed_jitter_m50ms",
    "fixed_jitter_p50ms",
    "fixed_jitter_p100ms",
    "no_jitter",
]

BEHAVIOR_ORDER = ["overt", "minimally_overt", "covert"]

# Shared speech-condition colors: overt is darkest, then min-overt, then covert.
PALETTE = condition_colors(BEHAVIOR_ORDER)
JITTER_PALETTE = [matplotlib.colormaps["viridis"](value) for value in np.linspace(0.18, 0.82, len(JITTER_ORDER))]
JITTER_COLOR_BY_CONDITION = dict(zip(JITTER_ORDER, JITTER_PALETTE))

# Behavior label mapping (kept small and consistent with other plotting modules)
BEHAVIOR_LABELS = {
    "overt": "overt",
    "minimally_overt": "min-overt",
    "covert": "covert",
}


def behavior_label(behavior: str) -> str:
    return BEHAVIOR_LABELS.get(behavior, behavior)


def jitter_label(condition: str) -> str:
    labels = {
        "random_jitter_±100ms": "random ±100ms",
        "fixed_jitter_m100ms": "-100ms",
        "fixed_jitter_m50ms": "-50ms",
        "fixed_jitter_p50ms": "+50ms",
        "fixed_jitter_p100ms": "+100ms",
        "no_jitter": "no jitter",
    }
    return labels.get(str(condition), str(condition))


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace(" ", "_")
    return "".join([c if c.isalnum() or c == "_" else "_" for c in text]).strip("_")


def load_test_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sbj", "behavior", "model_name", "balanced_acc_test"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    if "ensemble_method" not in df.columns:
        df["ensemble_method"] = "single"
    else:
        df["ensemble_method"] = df["ensemble_method"].fillna("single")

    df["sbj"] = df["sbj"].astype(str).map(normalize_subject_id)
    return df


def infer_jitter_condition(path: Path) -> str:
    stem = path.stem
    match = stem.replace("history_color_rotating_test_fold_", "")
    if match.endswith("_test"):
        match = match[: -len("_test")]
    # If the result is just "test" or empty, it indicates random jitter with ±100ms range
    if match == "test" or match == "":
        return "random_jitter_±100ms"
    return match


def normalize_jitter_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN jitter_condition with random_jitter_±100ms label."""
    df = df.copy()
    df["jitter_condition"] = df["jitter_condition"].fillna("random_jitter_±100ms")
    return df


def summarize_condition(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("jitter_condition", dropna=False)["balanced_acc_test"].agg(
        balanced_acc_test_mean="mean",
        balanced_acc_test_std="std",
        count="count",
    )
    summary = summary.reset_index()
    summary["jitter_condition"] = summary["jitter_condition"].fillna("random_jitter_±100ms")
    # standard error of the mean for error bars
    summary["balanced_acc_test_sem"] = summary["balanced_acc_test_std"] / np.sqrt(summary["count"].replace(0, np.nan))
    summary["jitter_condition"] = pd.Categorical(
        summary["jitter_condition"], categories=JITTER_ORDER, ordered=True
    )
    return summary.sort_values("jitter_condition")


def summarize_condition_behavior(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(["jitter_condition", "behavior"], dropna=False)["balanced_acc_test"].agg(
        balanced_acc_test_mean="mean",
        balanced_acc_test_std="std",
        count="count",
    )
    summary = summary.reset_index()
    summary["jitter_condition"] = summary["jitter_condition"].fillna("random_jitter_±100ms")
    summary["balanced_acc_test_sem"] = summary["balanced_acc_test_std"] / np.sqrt(summary["count"].replace(0, np.nan))
    summary["jitter_condition"] = pd.Categorical(
        summary["jitter_condition"], categories=JITTER_ORDER, ordered=True
    )
    summary["behavior"] = pd.Categorical(
        summary["behavior"], categories=BEHAVIOR_ORDER, ordered=True
    )
    return summary.sort_values(["jitter_condition", "behavior"])


def make_markdown_condition(summary: pd.DataFrame) -> str:
    lines = [
        "| jitter_condition | balanced_acc_test_mean | balanced_acc_test_std | count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['jitter_condition']} | {row['balanced_acc_test_mean']:.3f} | {row['balanced_acc_test_std']:.3f} | {int(row['count'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def make_markdown_condition_behavior(summary: pd.DataFrame) -> str:
    pivot = summary.pivot(index="jitter_condition", columns="behavior", values="balanced_acc_test_mean")
    pivot = pivot.reindex(JITTER_ORDER)
    lines = [
        "| jitter_condition | overt | minimally_overt | covert |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in pivot.index:
        row = pivot.loc[condition]
        lines.append(
            f"| {condition} | {row.get('overt', np.nan):.3f} | {row.get('minimally_overt', np.nan):.3f} | {row.get('covert', np.nan):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def bonferroni_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return adjusted.tolist()

    m = int(valid.sum())
    adjusted[valid] = np.minimum(p[valid] * m, 1.0)
    return adjusted.tolist()


def complete_wide(
    df: pd.DataFrame,
    unit_cols: list[str],
    condition_col: str = "jitter_condition",
    value_col: str = "balanced_acc_test",
) -> pd.DataFrame:
    wide = df.pivot_table(
        index=unit_cols,
        columns=condition_col,
        values=value_col,
        aggfunc="mean",
    )
    wide = wide.reindex(columns=JITTER_ORDER)
    return wide.dropna(axis=0, how="any")


def friedman_from_wide(wide: pd.DataFrame) -> dict:
    if len(wide) < 2 or wide.shape[1] < 3:
        return {"test": "Friedman", "statistic": np.nan, "p_value": np.nan, "n_units": len(wide)}
    stat, p = stats.friedmanchisquare(*[wide[col].to_numpy() for col in wide.columns])
    return {"test": "Friedman", "statistic": stat, "p_value": p, "n_units": len(wide)}


def pairwise_vs_reference_from_wide(
    wide: pd.DataFrame,
    reference: str = "random_jitter_±100ms",
) -> pd.DataFrame:
    rows = []
    if reference not in wide.columns:
        return pd.DataFrame(rows)
    for condition in wide.columns:
        if condition == reference:
            continue
        diff = wide[condition] - wide[reference]
        try:
            stat, p = stats.wilcoxon(wide[condition], wide[reference], zero_method="wilcox")
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append(
            {
                "reference": reference,
                "comparison": condition,
                "test": "Wilcoxon signed-rank",
                "statistic": stat,
                "p_value": p,
                "mean_difference_vs_reference": diff.mean(),
                "n_units": int(diff.notna().sum()),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bonferroni"] = bonferroni_adjust(result["p_value"].tolist())
        result["significance_bonferroni"] = result["p_bonferroni"].map(p_to_stars)
    return result


def pairwise_all_from_wide(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, condition_a in enumerate(wide.columns):
        for condition_b in wide.columns[i + 1 :]:
            diff = wide[condition_b] - wide[condition_a]
            try:
                stat, p = stats.wilcoxon(
                    wide[condition_b],
                    wide[condition_a],
                    zero_method="wilcox",
                )
            except ValueError:
                stat, p = np.nan, np.nan
            rows.append(
                {
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "test": "Wilcoxon signed-rank",
                    "statistic": stat,
                    "p_value": p,
                    "mean_difference_b_minus_a": diff.mean(),
                    "n_units": int(diff.notna().sum()),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bonferroni"] = bonferroni_adjust(result["p_value"].tolist())
        result["significance_bonferroni"] = result["p_bonferroni"].map(p_to_stars)
    return result


def compute_statistical_tests(data_subj: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = data_subj.copy()
    data["unit_behavior"] = data["sbj"].astype(str) + ":" + data["behavior"].astype(str)
    overall_wide = complete_wide(data, ["unit_behavior"])
    overall_omnibus = pd.DataFrame(
        [{**friedman_from_wide(overall_wide), "scope": "overall_subject_behavior"}]
    )
    overall_pairwise = pairwise_vs_reference_from_wide(overall_wide)
    if not overall_pairwise.empty:
        overall_pairwise.insert(0, "scope", "overall_subject_behavior")
    overall_all_pairwise = pairwise_all_from_wide(overall_wide)
    if not overall_all_pairwise.empty:
        overall_all_pairwise.insert(0, "scope", "overall_subject_behavior")

    behavior_rows = []
    behavior_pairwise = []
    behavior_all_pairwise = []
    for behavior in BEHAVIOR_ORDER:
        sub = data[data["behavior"] == behavior]
        wide = complete_wide(sub, ["sbj"])
        behavior_rows.append({**friedman_from_wide(wide), "behavior": behavior})
        pairwise = pairwise_vs_reference_from_wide(wide)
        if not pairwise.empty:
            pairwise.insert(0, "behavior", behavior)
            behavior_pairwise.append(pairwise)
        all_pairwise = pairwise_all_from_wide(wide)
        if not all_pairwise.empty:
            all_pairwise.insert(0, "behavior", behavior)
            behavior_all_pairwise.append(all_pairwise)

    return {
        "overall_omnibus": overall_omnibus,
        "overall_pairwise": overall_pairwise,
        "overall_all_pairwise": overall_all_pairwise,
        "behavior_omnibus": pd.DataFrame(behavior_rows),
        "behavior_pairwise": (
            pd.concat(behavior_pairwise, ignore_index=True)
            if behavior_pairwise
            else pd.DataFrame()
        ),
        "behavior_all_pairwise": (
            pd.concat(behavior_all_pairwise, ignore_index=True)
            if behavior_all_pairwise
            else pd.DataFrame()
        ),
    }


def stats_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No statistical results._\n"
    out = df.copy()
    for col in [
        "p_value",
        "p_bonferroni",
        "statistic",
        "mean_difference_vs_reference",
        "mean_difference_b_minus_a",
    ]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.6g}")
    headers = list(out.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines) + "\n"


def write_stats_files(stats_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for name, table in stats_tables.items():
        csv_path = stats_dir / f"{name}.csv"
        md_path = stats_dir / f"{name}.md"
        table.to_csv(csv_path, index=False)
        md_path.write_text(stats_markdown_table(table))
        print(f"Statistics saved to {csv_path}")


def ordered_subject_offsets(subjects: pd.Series, width: float = 0.26) -> dict[str, float]:
    subject_order = sorted(subjects.dropna().astype(str).unique())
    if len(subject_order) <= 1:
        offsets = [0.0] * len(subject_order)
    else:
        offsets = np.linspace(-width / 2, width / 2, len(subject_order))
    return dict(zip(subject_order, offsets))


def draw_significance_bracket(
    ax,
    x0: float,
    x1: float,
    y: float,
    label: str,
    color: str = "black",
    transform=None,
) -> None:
    if label in {"", "n.s.", "NA"}:
        return
    height = 0.025
    text_gap = 0.006
    if transform is None:
        transform = ax.transData
    ax.plot(
        [x0, x0, x1, x1],
        [y, y + height, y + height, y],
        color=color,
        linewidth=0.9,
        clip_on=False,
        transform=transform,
    )
    ax.text(
        (x0 + x1) / 2,
        y + height + text_gap,
        label,
        ha="center",
        va="bottom",
        color=color,
        fontsize=8,
        clip_on=False,
        transform=transform,
    )


def plot_condition(
    summary: pd.DataFrame,
    data_subj: pd.DataFrame,
    out_path: Path,
    stats_tables: dict[str, pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    jitter_levels = [condition for condition in JITTER_ORDER if condition in set(summary["jitter_condition"].astype(str))]
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
        mean_value = condition_df["balanced_acc_test"].mean()
        ax.hlines(mean_value, i - 0.28, i + 0.28, color="black", linewidth=3.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([jitter_label(condition) for condition in jitter_levels], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Jitter condition")
    ax.set_ylabel("Balanced test accuracy")
    ax.set_title("EEGNet balanced_acc_test by jitter condition", pad=110)
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
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def plot_condition_behavior(
    summary: pd.DataFrame,
    data_subj: pd.DataFrame,
    out_path: Path,
    stats_tables: dict[str, pd.DataFrame],
) -> None:
    jitter_levels = [condition for condition in JITTER_ORDER if condition in set(summary["jitter_condition"].astype(str))]
    output_dir = out_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data = data_subj.copy()
    plot_data["jitter_condition"] = plot_data["jitter_condition"].astype(str)

    for behavior in BEHAVIOR_ORDER:
        behavior_data = plot_data[plot_data["behavior"] == behavior]
        if behavior_data.empty:
            continue
        behavior_out_path = out_path.with_name(
            f"{out_path.stem}_{slugify(behavior)}{out_path.suffix}"
        )
        plot_single_behavior_condition(
            behavior,
            jitter_levels,
            behavior_data,
            behavior_out_path,
            stats_tables,
        )


def plot_single_behavior_condition(
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
    ax.set_xlabel("Jitter condition")
    ax.set_ylabel("Balanced test accuracy")
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
        f"EEGNet balanced_acc_test by jitter condition ({behavior_label(behavior)})"
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
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def write_summary_files(
    summary: pd.DataFrame,
    markdown: str,
    output_dir: Path,
    prefix: str,
    plot_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    md_path = output_dir / f"{prefix}.md"
    csv_path.write_text(summary.to_csv(index=False))
    md_path.write_text(markdown)
    print(f"Summary CSV saved to {csv_path}")
    print(f"Markdown table saved to {md_path}")

def aggregate_subject_jitter_rows(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["sbj", "jitter_condition", "behavior", "model_name", "ensemble_method"])[
            "balanced_acc_test"
        ]
        .mean()
        .reset_index()
    )


def filter_jitter_conditions(data: pd.DataFrame) -> pd.DataFrame:
    data = normalize_jitter_condition(data)
    invalid = data.loc[~data["jitter_condition"].isin(JITTER_ORDER), "jitter_condition"].unique()
    if len(invalid) > 0:
        print(f"Warning: dropping unsupported jitter conditions: {list(invalid)}")
        data = data[data["jitter_condition"].isin(JITTER_ORDER)].copy()
    return data


