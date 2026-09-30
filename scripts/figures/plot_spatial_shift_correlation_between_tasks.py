#!/usr/bin/env python3
"""Fig. 5 / S6 scaffolding: channel-level correlations of spatial shift maps.

Inputs are difference maps from ``plot_eegnet_wo_adapt_diff_spatial_contribution.py``.
Public default is ``--levels channel`` only (no mixed_effects).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from plot_spatial_correlation_between_tasks import (
        DEFAULT_MI_DIR,
        DEFAULT_MI_EMGS,
        compute_mi_channel_weights,
        write_channel_weights,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.figures.plot_spatial_correlation_between_tasks import (
        DEFAULT_MI_DIR,
        DEFAULT_MI_EMGS,
        compute_mi_channel_weights,
        write_channel_weights,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "IG_spatial_final_clip"
    / "wo_adapt_filt_spatial_contribution"
)
DEFAULT_NO_FINAL_CLIP_INPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "IG_spatial"
    / "wo_adapt_filt_spatial_contribution"
)
DEFAULT_SAVE_DIR = (
    REPO_ROOT
    / "outputs"
    / "IG_spatial_final_clip"
    / "spatial_shift_correlation"
)
DEFAULT_NO_FINAL_CLIP_SAVE_DIR = (
    REPO_ROOT
    / "outputs"
    / "IG_spatial"
    / "spatial_shift_correlation"
)
DEFAULT_COLORS = ["green", "magenta", "orange", "violet", "yellow"]
DEFAULT_TASKS = ["overt", "minimally_overt", "covert"]
TASK_LABELS = {"overt": "overt", "minimally_overt": "min overt", "covert": "covert"}
CM_TO_INCH = 1 / 2.54
SUBJECT_CONDITION_RE = re.compile(r"^(?P<subject>subject\d+)_(?P<task>.+)_avg\.csv$")
SUBJECT_LABEL_RE = re.compile(
    r"^(?P<subject>subject\d+)_(?P<task>.+)_label(?P<label>\d+)_(?P<color>.+)\.csv$"
)
CONDITION_RE = re.compile(r"^(?P<task>.+)_avg\.csv$")


@dataclass(frozen=True)
class MatrixResult:
    matrix: object
    sem: object
    p_values: object
    p_values_corrected: object
    significant: object
    n_units: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", default=[f"sub-{i}" for i in range(1, 10)])
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--colors", nargs="*", default=DEFAULT_COLORS)
    parser.add_argument("--plot-vmin", type=float, default=-1.0)
    parser.add_argument("--plot-vmax", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--corr-type",
        choices=("pearson", "weighted_pearson"),
        default="pearson",
        help="Correlation used for channel_level. weighted_pearson is implemented only for channel_level.",
    )
    parser.add_argument("--mi-dir", type=Path, default=DEFAULT_MI_DIR)
    parser.add_argument("--mi-eeg-type", default="raw")
    parser.add_argument("--mi-emg-type", default="raw")
    parser.add_argument("--mi-emgs", nargs="*", default=DEFAULT_MI_EMGS)
    parser.add_argument("--weight-eps", type=float, default=1e-12)
    parser.add_argument("--weighted-num-shuffle", type=int, default=9999)
    parser.add_argument("--weighted-seed", type=int, default=0)
    parser.add_argument(
        "--levels",
        nargs="*",
        choices=("subject", "subject_word_lme", "channel"),
        default=None,
        help="Analysis levels to write.",
    )
    parser.add_argument(
        "--no-final-clip",
        dest="no_final_clip",
        action="store_true",
        default=True,
        help="Read/write the no-final-clip spatial shift maps by default.",
    )
    parser.add_argument(
        "--final-clip",
        dest="no_final_clip",
        action="store_false",
        help="Read/write final-clipped spatial shift maps.",
    )
    parser.add_argument(
        "--title",
        default="spatial contribution v.s. spatial contribution",
        help="Title shown above the combined grid.",
    )
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = DEFAULT_NO_FINAL_CLIP_INPUT_DIR if args.no_final_clip else DEFAULT_INPUT_DIR
    if args.levels is None:
        # Public port: channel-level Pearson only (no mixed_effects).
        args.levels = ["channel"]
    if args.corr_type == "weighted_pearson" and args.levels != ["channel"]:
        raise ValueError("weighted_pearson is implemented only for --levels channel")
    if "subject_word_lme" in args.levels:
        raise ValueError(
            "subject_word_lme requires mixed_effects, which is not included in the "
            "public package. Use --levels channel (default)."
        )
    if args.save_dir is None:
        if args.corr_type == "weighted_pearson":
            args.save_dir = (
                DEFAULT_NO_FINAL_CLIP_SAVE_DIR.with_name("spatial_shift_weighted_correlation")
                if args.no_final_clip
                else DEFAULT_SAVE_DIR.with_name("spatial_shift_weighted_correlation")
            )
        else:
            args.save_dir = DEFAULT_NO_FINAL_CLIP_SAVE_DIR if args.no_final_clip else DEFAULT_SAVE_DIR
    return args


def read_shift_table(path: Path) -> tuple[list[int], object]:
    import numpy as np

    channels = []
    values = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            channels.append(int(row["channel"]))
            values.append(float(row["spatial_difference"]))
    order = np.argsort(np.asarray(channels, dtype=int))
    sorted_channels = [channels[int(idx)] for idx in order]
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    return sorted_channels, sorted_values


def load_subject_condition_vectors(input_dir: Path, subjects: set[str], tasks: set[str]) -> dict[tuple[str, str], object]:
    vectors = {}
    for path in sorted((input_dir / "subject_condition" / "tables").glob("*.csv")):
        match = SUBJECT_CONDITION_RE.match(path.name)
        if match is None:
            continue
        subject = match.group("subject")
        task = match.group("task")
        if subject not in subjects or task not in tasks:
            continue
        vectors[(subject, task)] = read_shift_table(path)
    return vectors


def load_subject_label_vectors(
    input_dir: Path,
    subjects: set[str],
    tasks: set[str],
) -> dict[tuple[str, str, int], object]:
    vectors = {}
    for path in sorted((input_dir / "subject_condition_label" / "tables").glob("*.csv")):
        match = SUBJECT_LABEL_RE.match(path.name)
        if match is None:
            continue
        subject = match.group("subject")
        task = match.group("task")
        label = int(match.group("label"))
        if subject not in subjects or task not in tasks:
            continue
        vectors[(subject, task, label)] = read_shift_table(path)
    return vectors


def correlation(left, right) -> float:
    import numpy as np

    if np.allclose(left.std(), 0) or np.allclose(right.std(), 0):
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(left * right) / denom)


def fisher_z(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    return math.atanh(max(min(value, 0.999999), -0.999999))


def inv_fisher_z(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    return math.tanh(value)


def two_sided_ttest(values) -> float:
    import numpy as np
    from scipy.stats import ttest_1samp

    values = np.asarray(values, dtype=np.float64)
    result = ttest_1samp(values, 0.0, nan_policy="omit")
    if not math.isfinite(float(result.statistic)):
        return float("nan")
    return float(result.pvalue)


def corrected_p_values(p_values, alpha: float):
    import numpy as np

    corrected = np.full_like(p_values, np.nan, dtype=np.float64)
    significant = np.zeros_like(p_values, dtype=bool)
    pair_indices = [(i, j) for i, j in itertools.combinations(range(p_values.shape[0]), 2)]
    n_tests = len(pair_indices)
    for i, j in pair_indices:
        if np.isnan(p_values[i, j]):
            continue
        value = min(float(p_values[i, j]) * n_tests, 1.0)
        corrected[i, j] = corrected[j, i] = value
        significant[i, j] = significant[j, i] = value < alpha
    return corrected, significant


def summarize_pairs(z_values_by_pair: dict[tuple[int, int], list[float]], n_tasks: int, alpha: float) -> MatrixResult:
    import numpy as np

    matrix = np.eye(n_tasks, dtype=np.float64)
    sem = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
    p_values = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
    n_units = np.zeros((n_tasks, n_tasks), dtype=int)

    for i, j in itertools.combinations(range(n_tasks), 2):
        values = np.asarray([value for value in z_values_by_pair[(i, j)] if not np.isnan(value)])
        n_units[i, j] = n_units[j, i] = len(values)
        if len(values) == 0:
            matrix[i, j] = matrix[j, i] = np.nan
            continue
        mean_z = float(values.mean())
        matrix[i, j] = matrix[j, i] = inv_fisher_z(mean_z)
        if len(values) > 1:
            sem_z = float(values.std(ddof=1) / math.sqrt(len(values)))
            sem[i, j] = sem[j, i] = sem_z
            p_values[i, j] = p_values[j, i] = two_sided_ttest(values)

    corrected, significant = corrected_p_values(p_values, alpha)
    return MatrixResult(matrix, sem, p_values, corrected, significant, n_units)


def compute_subject_result(
    vectors: dict[tuple, object],
    subjects: list[str],
    tasks: list[str],
    alpha: float,
    label: int | None = None,
):
    z_values_by_pair = defaultdict(list)
    rows = []
    for subject in subjects:
        for i, j in itertools.combinations(range(len(tasks)), 2):
            left_key = (subject, tasks[i]) if label is None else (subject, tasks[i], label)
            right_key = (subject, tasks[j]) if label is None else (subject, tasks[j], label)
            if left_key not in vectors or right_key not in vectors:
                continue
            left_channels, left_values = vectors[left_key]
            right_channels, right_values = vectors[right_key]
            if left_channels != right_channels:
                raise ValueError(f"Channel mismatch for {left_key} and {right_key}")
            corr = correlation(left_values, right_values)
            z_value = fisher_z(corr)
            z_values_by_pair[(i, j)].append(z_value)
            rows.append(
                {
                    "subject": subject,
                    "task_1": tasks[i],
                    "task_2": tasks[j],
                    "label": "" if label is None else label,
                    "correlation": corr,
                    "fisher_z": z_value,
                    "n_channels": len(left_channels),
                }
            )
    return summarize_pairs(z_values_by_pair, len(tasks), alpha), rows


def compute_subject_word_lme_result(vectors: dict[tuple, object], subjects: list[str], tasks: list[str], colors: list[str], alpha: float):
    raise NotImplementedError(
        "subject_word_lme is not available in the public package (no mixed_effects). "
        "Use --levels channel."
    )


def load_condition_vectors(input_dir: Path, tasks: set[str]) -> dict[str, object]:
    vectors = {}
    for path in sorted((input_dir / "condition" / "tables").glob("*.csv")):
        match = CONDITION_RE.match(path.name)
        if match is None:
            continue
        task = match.group("task")
        if task not in tasks:
            continue
        vectors[task] = read_shift_table(path)
    return vectors


def average_subject_label_vectors(
    vectors: dict[tuple[str, str, int], object],
    subjects: list[str],
    tasks: list[str],
    label: int,
) -> dict[str, object]:
    import numpy as np

    averaged = {}
    for task in tasks:
        channels = None
        values = []
        for subject in subjects:
            key = (subject, task, label)
            if key not in vectors:
                continue
            current_channels, current_values = vectors[key]
            if channels is None:
                channels = current_channels
            elif channels != current_channels:
                raise ValueError(f"Channel mismatch while averaging {task} label {label}")
            values.append(current_values)
        if channels is not None and values:
            averaged[task] = (channels, np.mean(np.asarray(values, dtype=np.float64), axis=0))
    return averaged


def compute_channel_result(vectors: dict[str, object], tasks: list[str], alpha: float, args) -> tuple[MatrixResult, list[dict]]:
    import numpy as np

    if args.corr_type == "weighted_pearson":
        try:
            from weighted_corr import WeightedCorr
        except ModuleNotFoundError:
            from weighted_corr import WeightedCorr

        weighted_corr = WeightedCorr(
            w=args.channel_weights,
            num_shuffle=args.weighted_num_shuffle,
            seed=args.weighted_seed,
        )
        pearsonr = None
    else:
        weighted_corr = None
        try:
            from scipy.stats import pearsonr
        except ModuleNotFoundError:
            pearsonr = None

    n_tasks = len(tasks)
    matrix = np.eye(n_tasks, dtype=np.float64)
    sem = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
    p_values = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
    n_units = np.zeros((n_tasks, n_tasks), dtype=int)
    rows = []

    for i, j in itertools.combinations(range(n_tasks), 2):
        task_1 = tasks[i]
        task_2 = tasks[j]
        if task_1 not in vectors or task_2 not in vectors:
            matrix[i, j] = matrix[j, i] = np.nan
            continue
        left_channels, left_values = vectors[task_1]
        right_channels, right_values = vectors[task_2]
        if left_channels != right_channels:
            raise ValueError(f"Channel mismatch for {task_1} and {task_2}")
        valid = np.isfinite(left_values) & np.isfinite(right_values)
        if args.corr_type == "weighted_pearson":
            valid &= np.isfinite(args.channel_weights) & (args.channel_weights > 0)
        n_channels = int(valid.sum())
        n_units[i, j] = n_units[j, i] = n_channels
        if n_channels < 2:
            matrix[i, j] = matrix[j, i] = np.nan
            continue
        if weighted_corr is not None:
            weights = args.channel_weights[valid]
            weights = weights / weights.sum()
            corr, p_value = weighted_corr(left_values[valid], right_values[valid], weights)
            corr = float(corr)
            p_value = float(p_value)
        elif pearsonr is not None:
            corr, p_value = pearsonr(left_values[valid], right_values[valid])
            corr = float(corr)
            p_value = float(p_value)
        else:
            corr = correlation(left_values[valid], right_values[valid])
            p_value = float("nan")
        matrix[i, j] = matrix[j, i] = corr
        p_values[i, j] = p_values[j, i] = p_value
        rows.append(
            {
                "task_1": task_1,
                "task_2": task_2,
                "correlation": corr,
                "p_value": p_value,
                "corr_type": args.corr_type,
                "n_channels": n_channels,
            }
        )

    corrected, significant = corrected_p_values(p_values, alpha)
    return MatrixResult(matrix, sem, p_values, corrected, significant, n_units), rows


def significance_marker(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plot_matrix(
    ax,
    result: MatrixResult,
    tasks: list[str],
    vmin: float,
    vmax: float,
    show_y_labels: bool,
):
    import numpy as np
    from matplotlib import pyplot as plt

    values = result.matrix.copy()
    np.fill_diagonal(values, np.nan)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    labels = [TASK_LABELS.get(task, task) for task in tasks]
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels if show_y_labels else [""] * len(tasks), fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    for i, j in zip(*result.significant.nonzero()):
        if i == j:
            continue
        marker = significance_marker(result.p_values_corrected[i, j])
        if marker:
            ax.text(j, i, marker, ha="center", va="center", fontsize=8, color="black")
    return im


def plot_single(path: Path, result: MatrixResult, tasks: list[str], title: str, vmin: float, vmax: float) -> None:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(2.4, 2.15))
    im = plot_matrix(ax, result, tasks, vmin, vmax, show_y_labels=True)
    if title:
        ax.set_title(title, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def plot_combined(
    path: Path,
    results_by_key: dict[str, MatrixResult],
    tasks: list[str],
    colors: list[str],
    vmin: float,
    vmax: float,
    title: str,
) -> None:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(colors) + 1,
        figsize=((len(colors) + 1) * 3.8 * CM_TO_INCH, 6.4 * CM_TO_INCH),
        squeeze=False,
    )
    last_im = None
    for col_idx, _ in enumerate(colors):
        ax = axes[0, col_idx]
        result = results_by_key.get(str(col_idx))
        if result is None:
            ax.axis("off")
            continue
        last_im = plot_matrix(ax, result, tasks, vmin, vmax, show_y_labels=col_idx == 0)

    ax = axes[0, -1]
    result = results_by_key.get("avg")
    if result is None:
        ax.axis("off")
    else:
        last_im = plot_matrix(ax, result, tasks, vmin, vmax, show_y_labels=False)

    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.22, top=0.76, wspace=0.26)
    fig.text(0.11, 0.80, title, ha="left", va="bottom", fontsize=8)
    cbar_ax = fig.add_axes([0.925, 0.22, 0.012, 0.54])
    if last_im is None:
        cmap = plt.get_cmap("jet").copy()
        cmap.set_bad("white")
        last_im = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap)
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("correlation", fontsize=7, labelpad=4)
    cbar.ax.tick_params(labelsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def output_paths(save_dir: Path, level: str, group: str, name: str) -> tuple[Path, Path]:
    fig_path = save_dir / level / group / "figures" / f"{name}.png"
    table_path = save_dir / level / group / "tables" / f"{name}.csv"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    return fig_path, table_path


def write_matrix_table(path: Path, result: MatrixResult, tasks: list[str], unit_name: str) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", *tasks])
        writer.writeheader()
        for task, values in zip(tasks, result.matrix):
            writer.writerow({"task": task, **{other: float(value) for other, value in zip(tasks, values)}})

    long_path = path.with_name(path.stem + "_long.csv")
    with long_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_1",
                "task_2",
                "mean_correlation",
                "sem_fisher_z",
                unit_name,
                "p_value",
                "p_value_corrected",
                "significant",
            ],
        )
        writer.writeheader()
        for i, task_1 in enumerate(tasks):
            for j, task_2 in enumerate(tasks):
                writer.writerow(
                    {
                        "task_1": task_1,
                        "task_2": task_2,
                        "mean_correlation": float(result.matrix[i, j]),
                        "sem_fisher_z": float(result.sem[i, j]) if not np.isnan(result.sem[i, j]) else "",
                        unit_name: int(result.n_units[i, j]),
                        "p_value": float(result.p_values[i, j]) if not np.isnan(result.p_values[i, j]) else "",
                        "p_value_corrected": (
                            float(result.p_values_corrected[i, j])
                            if not np.isnan(result.p_values_corrected[i, j])
                            else ""
                        ),
                        "significant": bool(result.significant[i, j]),
                    }
                )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(save_dir: Path, rows: list[dict[str, object]]) -> None:
    path = save_dir / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "label", "color", "figure", "table", "detail_table"])
        writer.writeheader()
        writer.writerows(rows)


def run_subject_level(args, condition_vectors, label_vectors) -> list[dict[str, object]]:
    results_by_key = {}
    summary_rows = []

    for label, color in enumerate(args.colors):
        result, rows = compute_subject_result(label_vectors, args.subjects, args.tasks, args.alpha, label)
        results_by_key[str(label)] = result
        fig_path, table_path = output_paths(args.save_dir, "subject_level", "condition_label", f"label{label}_{color}")
        detail_path = table_path.with_name(table_path.stem + "_subjects.csv")
        write_matrix_table(table_path, result, args.tasks, "n_subjects")
        write_rows(detail_path, rows)
        plot_single(fig_path, result, args.tasks, color, args.plot_vmin, args.plot_vmax)
        summary_rows.append(
            {
                "level": "subject_level",
                "label": label,
                "color": color,
                "figure": str(fig_path),
                "table": str(table_path),
                "detail_table": str(detail_path),
            }
        )

    result, rows = compute_subject_result(condition_vectors, args.subjects, args.tasks, args.alpha, None)
    results_by_key["avg"] = result
    fig_path, table_path = output_paths(args.save_dir, "subject_level", "condition", "avg")
    detail_path = table_path.with_name(table_path.stem + "_subjects.csv")
    write_matrix_table(table_path, result, args.tasks, "n_subjects")
    write_rows(detail_path, rows)
    plot_single(fig_path, result, args.tasks, "avg.", args.plot_vmin, args.plot_vmax)
    summary_rows.append(
        {
            "level": "subject_level",
            "label": "",
            "color": "avg",
            "figure": str(fig_path),
            "table": str(table_path),
            "detail_table": str(detail_path),
        }
    )

    plot_combined(
        args.save_dir / "subject_level" / "combined" / "figures" / "between_tasks_grid.png",
        results_by_key,
        args.tasks,
        args.colors,
        args.plot_vmin,
        args.plot_vmax,
        args.title,
    )
    return summary_rows


def run_subject_word_lme(args, label_vectors) -> list[dict[str, object]]:
    result, rows, lme_rows = compute_subject_word_lme_result(
        label_vectors,
        args.subjects,
        args.tasks,
        args.colors,
        args.alpha,
    )
    fig_path, table_path = output_paths(args.save_dir, "subject_word_lme", "condition_label", "all_labels")
    detail_path = table_path.with_name(table_path.stem + "_observations.csv")
    lme_path = table_path.with_name(table_path.stem + "_model.csv")
    write_matrix_table(table_path, result, args.tasks, "n_observations")
    write_rows(detail_path, rows)
    write_rows(lme_path, lme_rows)
    plot_single(fig_path, result, args.tasks, "LME", args.plot_vmin, args.plot_vmax)
    plot_combined(
        args.save_dir / "subject_word_lme" / "combined" / "figures" / "between_tasks_grid.png",
        {"avg": result},
        args.tasks,
        [],
        args.plot_vmin,
        args.plot_vmax,
        args.title,
    )
    return [
        {
            "level": "subject_word_lme",
            "label": "all",
            "color": "all",
            "figure": str(fig_path),
            "table": str(table_path),
            "detail_table": str(detail_path),
        }
    ]


def run_channel_level(args, condition_vectors, label_vectors) -> list[dict[str, object]]:
    results_by_key = {}
    summary_rows = []
    if args.corr_type == "weighted_pearson":
        write_channel_weights(
            args.save_dir / "channel_level" / "tables" / "channel_weights.csv",
            args.channel_weights,
            args.channel_mean_mi,
            args.channel_weight_metadata,
        )

    for label, color in enumerate(args.colors):
        vectors = average_subject_label_vectors(label_vectors, args.subjects, args.tasks, label)
        result, rows = compute_channel_result(vectors, args.tasks, args.alpha, args)
        results_by_key[str(label)] = result
        fig_path, table_path = output_paths(args.save_dir, "channel_level", "condition_label", f"label{label}_{color}")
        detail_path = table_path.with_name(table_path.stem + "_channels.csv")
        write_matrix_table(table_path, result, args.tasks, "n_channels")
        write_rows(detail_path, rows)
        plot_single(fig_path, result, args.tasks, color, args.plot_vmin, args.plot_vmax)
        summary_rows.append(
            {
                "level": "channel_level",
                "label": label,
                "color": color,
                "figure": str(fig_path),
                "table": str(table_path),
                "detail_table": str(detail_path),
            }
        )

    result, rows = compute_channel_result(condition_vectors, args.tasks, args.alpha, args)
    results_by_key["avg"] = result
    fig_path, table_path = output_paths(args.save_dir, "channel_level", "condition", "avg")
    detail_path = table_path.with_name(table_path.stem + "_channels.csv")
    write_matrix_table(table_path, result, args.tasks, "n_channels")
    write_rows(detail_path, rows)
    plot_single(fig_path, result, args.tasks, "avg.", args.plot_vmin, args.plot_vmax)
    summary_rows.append(
        {
            "level": "channel_level",
            "label": "",
            "color": "avg",
            "figure": str(fig_path),
            "table": str(table_path),
            "detail_table": str(detail_path),
        }
    )

    plot_combined(
        args.save_dir / "channel_level" / "combined" / "figures" / "between_tasks_grid.png",
        results_by_key,
        args.tasks,
        args.colors,
        args.plot_vmin,
        args.plot_vmax,
        args.title,
    )
    return summary_rows


def main() -> None:
    args = parse_args()
    subjects = set(args.subjects)
    tasks = set(args.tasks)

    subject_condition_vectors = load_subject_condition_vectors(args.input_dir, subjects, tasks)
    subject_label_vectors = load_subject_label_vectors(args.input_dir, subjects, tasks)
    condition_vectors = load_condition_vectors(args.input_dir, tasks)
    if not subject_condition_vectors and not subject_label_vectors and not condition_vectors:
        raise ValueError(f"No shift-map tables found in {args.input_dir}")

    if args.corr_type == "weighted_pearson":
        args.channel_weights, args.channel_mean_mi, args.channel_weight_metadata = compute_mi_channel_weights(
            args.mi_dir,
            subjects,
            args.mi_eeg_type,
            args.mi_emg_type,
            args.mi_emgs,
            n_channels=128,
            eps=args.weight_eps,
        )

    summary_rows = []
    if "subject" in args.levels:
        summary_rows.extend(run_subject_level(args, subject_condition_vectors, subject_label_vectors))
    if "subject_word_lme" in args.levels:
        summary_rows.extend(run_subject_word_lme(args, subject_label_vectors))
    if "channel" in args.levels:
        summary_rows.extend(run_channel_level(args, condition_vectors, subject_label_vectors))

    write_summary(args.save_dir, summary_rows)
    print(f"subject-condition vectors: {len(subject_condition_vectors)}")
    print(f"subject-condition-label vectors: {len(subject_label_vectors)}")
    print(f"condition vectors: {len(condition_vectors)}")
    print(f"matrices: {len(summary_rows)}")
    print(f"saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

