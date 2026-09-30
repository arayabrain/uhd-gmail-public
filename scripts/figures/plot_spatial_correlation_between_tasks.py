#!/usr/bin/env python3
"""Fig. 4 / S5 scaffolding: channel-level spatial-contribution Pearson correlations.

Public port keeps **channel-level only** (no subject LME / mixed_effects).
Use ``--model EEGNet`` (Fig. 4) or ``--model EEGNet_wo_adapt_filt`` (Fig. S5).

Weighted Pearson (Fig. S6-related channel weighting) is available via
``--corr-type weighted_pearson`` and requires MI weights (see ``--mi-dir``).

Example
-------
::

    uv run python scripts/figures/plot_spatial_correlation_between_tasks.py \\
      --ig-dir outputs/integrated_gradients --levels channel
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.figures._ig_common import (
        CM_TO_INCH,
        DEFAULT_COLORS,
        DEFAULT_SUBJECTS,
        DEFAULT_TASKS,
        load_ig_tensor,
        parse_ig_name,
        read_prediction_rows,
    )
    from scripts.figures.weighted_corr import WeightedCorr
except ImportError:  # pragma: no cover
    from _ig_common import (  # type: ignore[no-redef]
        CM_TO_INCH,
        DEFAULT_COLORS,
        DEFAULT_SUBJECTS,
        DEFAULT_TASKS,
        load_ig_tensor,
        parse_ig_name,
        read_prediction_rows,
    )
    from weighted_corr import WeightedCorr  # type: ignore[no-redef]

TASK_LABELS = {"overt": "overt", "minimally_overt": "min overt", "covert": "covert"}
DEFAULT_IG_DIR = REPO_ROOT / "outputs" / "integrated_gradients"
DEFAULT_SAVE_DIR = (
    REPO_ROOT / "outputs" / "IG_spatial" / "eegnet_spatial_contribution_correlation"
)
DEFAULT_MI_DIR = REPO_ROOT / "outputs" / "mutual_information"
DEFAULT_MI_EMGS = ["EOG", "EMG_upper", "EMG_lower"]
COMBINED_PANEL_WIDTH_CM = 3.8
COMBINED_HEIGHT_CM = 6.4
COMBINED_ADJUST = dict(left=0.10, right=0.96, bottom=0.22, top=0.76, wspace=0.26)
COMBINED_TITLE_POS = (0.11, 0.80)
COMBINED_CBAR_AX = [0.84, 0.86, 0.12, 0.035]


@dataclass(frozen=True)
class Record:
    condition: str
    subject: str
    task: str
    cv: int
    ig_path: Path
    predictions_path: Path


@dataclass
class Accumulator:
    total: object | None = None
    n_trials: int = 0


@dataclass(frozen=True)
class MatrixResult:
    matrix: object
    sem: object
    p_values: object
    p_values_corrected: object
    significant: object
    n_units: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ig-dir", type=Path, default=DEFAULT_IG_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--colors", nargs="*", default=DEFAULT_COLORS)
    parser.add_argument(
        "--model",
        default="EEGNet",
        choices=("EEGNet", "EEGNet_wo_adapt_filt"),
        help="EEGNet → Fig. 4 channel grids; EEGNet_wo_adapt_filt → Fig. S5.",
    )
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--plot-vmin", type=float, default=-1.0)
    parser.add_argument("--plot-vmax", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--corr-type",
        choices=("pearson", "weighted_pearson"),
        default="pearson",
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
        choices=("channel",),
        default=["channel"],
        help="Public port supports channel-level only (no mixed_effects).",
    )
    return parser.parse_args()


def discover_records(ig_dir: Path, model: str) -> list[Record]:
    grouped: dict[tuple[str, str, int], dict[str, Path]] = defaultdict(dict)
    metadata: dict[tuple[str, str, int], dict[str, str]] = {}
    for path in sorted(ig_dir.glob("*")):
        meta = parse_ig_name(path)
        if meta is None or meta["model"] != model:
            continue
        key = (meta["condition"], meta["task"], int(meta["cv"]))
        grouped[key][meta["kind"]] = path
        metadata[key] = meta

    records = []
    for key, paths in grouped.items():
        if "igs" not in paths or "trial_predictions" not in paths:
            continue
        meta = metadata[key]
        records.append(
            Record(
                condition=meta["condition"],
                subject=meta["subject"],
                task=meta["task"],
                cv=int(meta["cv"]),
                ig_path=paths["igs"],
                predictions_path=paths["trial_predictions"],
            )
        )
    return sorted(records, key=lambda record: (record.subject, record.task, record.cv))


@lru_cache(maxsize=None)
def _load_igs_cached(path_str: str):
    return load_ig_tensor(Path(path_str))


def normalize_trial(ig, clip: float):
    import numpy as np

    ig = np.abs(ig)
    return np.clip((ig - ig.mean()) / max(ig.std(), 1e-12), -clip, clip)


def spatial_vector(ig, clip: float):
    return normalize_trial(ig, clip).mean(axis=1)


def zscore(values):
    return (values - values.mean()) / max(values.std(), 1e-12)


def add_vector(acc: Accumulator, values) -> None:
    import numpy as np

    if acc.total is None:
        acc.total = np.zeros_like(values, dtype=np.float64)
    acc.total += values
    acc.n_trials += 1


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


def normalize_mi_name(value: str) -> str:
    return value.strip().replace(" ", "_")


def compute_mi_channel_weights(
    mi_dir: Path,
    subjects: set[str],
    eeg_type: str,
    emg_type: str,
    emgs: list[str],
    n_channels: int,
    eps: float,
):
    import numpy as np

    wanted_emgs = {normalize_mi_name(emg) for emg in emgs}
    channel_sum = np.zeros(n_channels, dtype=np.float64)
    channel_count = np.zeros(n_channels, dtype=np.float64)
    used_rows = 0
    used_subjects: set[str] = set()

    for subject in sorted(subjects):
        table_path = mi_dir / subject / "data" / "mis_table.csv"
        values_path = mi_dir / subject / "data" / "mis.npy"
        if not table_path.exists() or not values_path.exists():
            continue
        with table_path.open(newline="", encoding="utf-8") as handle:
            table = list(csv.DictReader(handle))
        values = np.load(values_path)
        if values.ndim != 3 or values.shape[-1] != n_channels:
            raise ValueError(
                f"Expected MI shape (series, trial, {n_channels}), got {values.shape}: {values_path}"
            )
        for row_idx, row in enumerate(table):
            if row.get("eeg_type") != eeg_type:
                continue
            if row.get("emg_type") != emg_type:
                continue
            if row.get("surrogate_type", "none") != "none":
                continue
            if normalize_mi_name(row.get("emg_name", "")) not in wanted_emgs:
                continue
            series = np.asarray(values[row_idx], dtype=np.float64)
            finite = np.isfinite(series)
            channel_sum += np.nansum(series, axis=0)
            channel_count += finite.sum(axis=0)
            used_rows += 1
            used_subjects.add(subject)

    valid = channel_count > 0
    if used_rows == 0 or not np.any(valid):
        raise ValueError(
            "No MI values matched the requested filters under "
            f"mi_dir={mi_dir}, eeg_type={eeg_type}, emg_type={emg_type}, emgs={sorted(wanted_emgs)}"
        )
    mean_mi = np.full(n_channels, np.nan, dtype=np.float64)
    mean_mi[valid] = channel_sum[valid] / channel_count[valid]
    safe_mi = np.where(np.isfinite(mean_mi), mean_mi, np.nanmedian(mean_mi[valid]))
    safe_mi = np.maximum(safe_mi, eps)
    weights = 1.0 / safe_mi
    weights /= weights.sum()
    return weights, mean_mi, {"used_mi_rows": used_rows, "used_mi_subjects": len(used_subjects)}


def write_channel_weights(
    path: Path,
    weights,
    mean_mi,
    metadata: dict | None = None,
) -> None:
    """Write per-channel MI weights used by weighted Pearson (Fig. S6 helper path)."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["channel", "weight", "mean_mi"],
        )
        writer.writeheader()
        for channel, (weight, mi) in enumerate(
            zip(np.asarray(weights).ravel(), np.asarray(mean_mi).ravel())
        ):
            writer.writerow(
                {
                    "channel": channel,
                    "weight": float(weight),
                    "mean_mi": float(mi) if np.isfinite(mi) else "",
                }
            )
        if metadata:
            meta_path = path.with_name(path.stem + "_metadata.txt")
            meta_path.write_text(
                "\n".join(f"{key}={value}" for key, value in sorted(metadata.items()))
                + "\n",
                encoding="utf-8",
            )


def corrected_p_values(p_values, alpha: float):
    import numpy as np

    corrected = np.full_like(p_values, np.nan, dtype=np.float64)
    significant = np.zeros_like(p_values, dtype=bool)
    pair_indices = list(itertools.combinations(range(p_values.shape[0]), 2))
    n_tests = len(pair_indices)
    for i, j in pair_indices:
        if np.isnan(p_values[i, j]):
            continue
        value = min(float(p_values[i, j]) * n_tests, 1.0)
        corrected[i, j] = corrected[j, i] = value
        significant[i, j] = significant[j, i] = value < alpha
    return corrected, significant


def compute_channel_level(records, subjects, tasks, label, args):
    import numpy as np

    if args.corr_type == "weighted_pearson":
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

    groups: dict[str, Accumulator] = defaultdict(Accumulator)
    for record in records:
        if record.subject not in subjects or record.task not in tasks:
            continue
        predictions = read_prediction_rows(record.predictions_path)
        igs = _load_igs_cached(str(record.ig_path))
        for row in predictions:
            if row["correct"].lower() != "true":
                continue
            if label is not None and int(row["label"]) != label:
                continue
            add_vector(groups[record.task], spatial_vector(igs[int(row["ig_index"])], args.clip))

    vectors = {}
    counts = {}
    for task, acc in groups.items():
        if acc.total is None or acc.n_trials == 0:
            continue
        vectors[task] = zscore(acc.total / acc.n_trials)
        counts[task] = acc.n_trials

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
        left = vectors[task_1]
        right = vectors[task_2]
        valid = np.isfinite(left) & np.isfinite(right)
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
            corr, p_value = weighted_corr(left[valid], right[valid], weights)
        elif pearsonr is not None:
            corr, p_value = pearsonr(left[valid], right[valid])
            corr = float(corr)
            p_value = float(p_value)
        else:
            corr = correlation(left[valid], right[valid])
            p_value = float("nan")
        matrix[i, j] = matrix[j, i] = float(corr)
        p_values[i, j] = p_values[j, i] = float(p_value)
        rows.append(
            {
                "task_1": task_1,
                "task_2": task_2,
                "correlation": float(corr),
                "p_value": float(p_value),
                "corr_type": args.corr_type,
                "n_channels": n_channels,
                "n_correct_trials_task_1": counts[task_1],
                "n_correct_trials_task_2": counts[task_2],
            }
        )

    corrected, significant = corrected_p_values(p_values, args.alpha)
    return MatrixResult(matrix, sem, p_values, corrected, significant, n_units), rows


def output_paths(save_dir: Path, level: str, group: str, name: str) -> tuple[Path, Path]:
    fig_path = save_dir / level / group / "figures" / f"{name}.png"
    table_path = save_dir / level / group / "tables" / f"{name}.csv"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    return fig_path, table_path


def write_matrix_table(path: Path, result: MatrixResult, tasks: list[str], unit_name: str) -> None:
    import numpy as np

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", *tasks])
        writer.writeheader()
        for task, values in zip(tasks, result.matrix):
            writer.writerow({"task": task, **{other: float(value) for other, value in zip(tasks, values)}})

    long_path = path.with_name(path.stem + "_long.csv")
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def significance_marker(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plot_matrix(ax, result, tasks, title, vmin, vmax, show_y_labels=True):
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
    if title:
        ax.set_title(title, fontsize=7)
    for i, j in zip(*result.significant.nonzero()):
        if i == j:
            continue
        marker = significance_marker(result.p_values_corrected[i, j])
        if marker:
            ax.text(j, i, marker, ha="center", va="center", fontsize=8, color="black")
    return im


def plot_single(path, result, tasks, title, vmin, vmax) -> None:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(2.4, 2.15))
    im = plot_matrix(ax, result, tasks, title, vmin, vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def plot_combined(path, results_by_key, tasks, colors, vmin, vmax) -> None:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(colors) + 1,
        figsize=((len(colors) + 1) * COMBINED_PANEL_WIDTH_CM * CM_TO_INCH, COMBINED_HEIGHT_CM * CM_TO_INCH),
        squeeze=False,
    )
    last_im = None
    for col_idx, color in enumerate(colors):
        ax = axes[0, col_idx]
        result = results_by_key.get(str(col_idx))
        if result is None:
            ax.axis("off")
            continue
        last_im = plot_matrix(ax, result, tasks, "", vmin, vmax, show_y_labels=col_idx == 0)

    ax = axes[0, -1]
    result = results_by_key.get("avg")
    if result is None:
        ax.axis("off")
    else:
        last_im = plot_matrix(ax, result, tasks, "", vmin, vmax, show_y_labels=False)

    fig.subplots_adjust(**COMBINED_ADJUST)
    fig.text(*COMBINED_TITLE_POS, "spatial contribution v.s. spatial contribution", ha="left", va="bottom", fontsize=8)
    cbar_ax = fig.add_axes(COMBINED_CBAR_AX)
    if last_im is None:
        cmap = plt.get_cmap("jet").copy()
        cmap.set_bad("white")
        last_im = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap)
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.set_label("correlation", fontsize=7, labelpad=1)
    cbar.ax.tick_params(labelsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def run_channel_level(args, records, subjects) -> list[dict[str, object]]:
    results_by_key = {}
    summary_rows = []
    for label, color in enumerate(args.colors):
        result, rows = compute_channel_level(records, subjects, args.tasks, label, args)
        results_by_key[str(label)] = result
        fig_path, table_path = output_paths(
            args.save_dir, "channel_level", "condition_label", f"label{label}_{color}"
        )
        write_matrix_table(table_path, result, args.tasks, "n_channels")
        detail_path = table_path.with_name(table_path.stem + "_pairs.csv")
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

    result, rows = compute_channel_level(records, subjects, args.tasks, None, args)
    results_by_key["avg"] = result
    fig_path, table_path = output_paths(args.save_dir, "channel_level", "condition", "avg")
    write_matrix_table(table_path, result, args.tasks, "n_channels")
    detail_path = table_path.with_name(table_path.stem + "_pairs.csv")
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
    )
    return summary_rows


def main() -> None:
    args = parse_args()
    records = discover_records(args.ig_dir, args.model)
    subjects = {str(s) for s in args.subjects}
    records = [record for record in records if record.subject in subjects and record.task in args.tasks]
    if not records:
        raise ValueError(f"No {args.model} IG/prediction pairs in {args.ig_dir}")

    if args.corr_type == "weighted_pearson":
        probe = _load_igs_cached(str(records[0].ig_path))
        weights, _, meta = compute_mi_channel_weights(
            args.mi_dir,
            subjects,
            args.mi_eeg_type,
            args.mi_emg_type,
            args.mi_emgs,
            probe.shape[1],
            args.weight_eps,
        )
        args.channel_weights = weights
        print(f"MI weights: {meta}")
    else:
        args.channel_weights = None

    summary_rows = []
    if "channel" in args.levels:
        summary_rows.extend(run_channel_level(args, records, subjects))

    summary_path = args.save_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["level", "label", "color", "figure", "table", "detail_table"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
