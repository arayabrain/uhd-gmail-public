#!/usr/bin/env python3
"""Fig. 4 scaffolding: EEGNet spatial IG contribution maps (correct trials).

Loads precomputed ``*_igs.pt`` + ``*_trial_predictions.csv`` from ``--ig-dir``.
Subject IDs are normalized to BIDS ``sub-N``.

Example
-------
::

    uv run python scripts/figures/plot_eegnet_spatial_contribution.py \\
      --ig-dir outputs/integrated_gradients \\
      --combined-only
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
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
        safe_name,
    )
except ImportError:  # pragma: no cover
    from _ig_common import (  # type: ignore[no-redef]
        CM_TO_INCH,
        DEFAULT_COLORS,
        DEFAULT_SUBJECTS,
        DEFAULT_TASKS,
        load_ig_tensor,
        parse_ig_name,
        read_prediction_rows,
        safe_name,
    )

MODEL = "EEGNet"
DEFAULT_IG_DIR = REPO_ROOT / "outputs" / "integrated_gradients"
DEFAULT_SAVE_DIR = REPO_ROOT / "outputs" / "IG_spatial" / "eegnet_spatial_contribution"


@dataclass(frozen=True)
class IGRecord:
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
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--plot-vmin", type=float, default=-1.0)
    parser.add_argument("--plot-vmax", type=float, default=1.0)
    parser.add_argument(
        "--no-final-clip",
        dest="no_final_clip",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--final-clip",
        dest="no_final_clip",
        action="store_false",
    )
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument(
        "--montage-image",
        type=Path,
        default=REPO_ROOT / "scripts" / "figures" / "assets" / "montage_colorless.png",
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=REPO_ROOT / "scripts" / "figures" / "assets" / "coordinates_colorless.npy",
    )
    return parser.parse_args()


def discover_records(ig_dir: Path) -> list[IGRecord]:
    grouped: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    metadata: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in sorted(ig_dir.glob("*")):
        meta = parse_ig_name(path)
        if meta is None or meta["model"] != MODEL:
            continue
        key = (meta["condition"], meta["task"], meta["cv"])
        grouped[key][meta["kind"]] = path
        metadata[key] = meta

    records = []
    for key, paths in grouped.items():
        if "igs" not in paths or "trial_predictions" not in paths:
            continue
        meta = metadata[key]
        records.append(
            IGRecord(
                condition=meta["condition"],
                subject=meta["subject"],
                task=meta["task"],
                cv=int(meta["cv"]),
                ig_path=paths["igs"],
                predictions_path=paths["trial_predictions"],
            )
        )
    return records


def correct_indices_by_label(path: Path) -> dict[int, list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for row in read_prediction_rows(path):
        if row["correct"].lower() != "true":
            continue
        by_label[int(row["label"])].append(int(row["ig_index"]))
    return by_label


def normalize_trials(igs, clip: float):
    import numpy as np

    igs = np.abs(igs)
    mean = igs.mean(axis=(1, 2), keepdims=True)
    std = igs.std(axis=(1, 2), keepdims=True)
    return np.clip((igs - mean) / np.maximum(std, 1e-12), -clip, clip)


def add_trials(acc: Accumulator, igs, indices: list[int], clip: float) -> None:
    import numpy as np

    if not indices:
        return
    selected = normalize_trials(igs[np.asarray(indices, dtype=int)], clip)
    spatial = selected.mean(axis=2)
    if acc.total is None:
        acc.total = np.zeros(spatial.shape[1], dtype=np.float64)
    acc.total += spatial.sum(axis=0)
    acc.n_trials += spatial.shape[0]


def finalize(acc: Accumulator, plot_vmin: float, plot_vmax: float, no_final_clip: bool):
    import numpy as np

    if acc.total is None or acc.n_trials == 0:
        return None
    values = acc.total / acc.n_trials
    values = (values - values.mean()) / max(values.std(), 1e-12)
    if no_final_clip:
        return values
    return np.clip(values, plot_vmin, plot_vmax)


def output_paths(save_dir: Path, level: str, name: str) -> tuple[Path, Path]:
    fig_path = save_dir / level / "figures" / f"{name}.png"
    table_path = save_dir / level / "tables" / f"{name}.csv"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    return fig_path, table_path


def write_table(path: Path, values, coordinates, n_trials: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["channel", "x", "y", "spatial_contribution", "n_trials"],
        )
        writer.writeheader()
        for channel, value in enumerate(values):
            writer.writerow(
                {
                    "channel": channel,
                    "x": float(coordinates[channel, 0]),
                    "y": float(coordinates[channel, 1]),
                    "spatial_contribution": float(value),
                    "n_trials": n_trials,
                }
            )


def plot_montage(path, values, coordinates, image, vmin, vmax, title) -> None:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(3.1 * CM_TO_INCH, 3.1 * CM_TO_INCH))
    ax.imshow(image)
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=5,
        c=values,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_title(title, fontsize=7)
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.01, ticks=[vmin, 0, vmax])
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def plot_condition_label_grid(
    path,
    by_condition_label,
    by_condition,
    tasks,
    colors,
    coordinates,
    image,
    vmin,
    vmax,
    no_final_clip,
) -> None:
    import matplotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt

    display_task = {
        "overt": "overt",
        "minimally_overt": "min overt",
        "covert": "covert",
    }
    n_rows = len(tasks)
    n_cols = len(colors) + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.3 * CM_TO_INCH, n_rows * 2.2 * CM_TO_INCH),
        squeeze=False,
    )
    last_scatter = None
    for row_idx, task in enumerate(tasks):
        for col_idx, color in enumerate(colors):
            ax = axes[row_idx, col_idx]
            ax.imshow(image)
            values = finalize(by_condition_label[(task, col_idx)], vmin, vmax, no_final_clip)
            if values is not None:
                last_scatter = ax.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    s=3,
                    c=values,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0,
                )
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(color, fontsize=7)
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    display_task.get(task, task),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=7,
                )

        ax = axes[row_idx, -1]
        ax.imshow(image)
        values = finalize(by_condition[task], vmin, vmax, no_final_clip)
        if values is not None:
            last_scatter = ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=3,
                c=values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
            )
        ax.axis("off")
        if row_idx == 0:
            ax.set_title("avg.", fontsize=7)

    if last_scatter is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        last_scatter = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.subplots_adjust(wspace=0.05, hspace=0.15, right=0.93)
    cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    cbar_ticks = np.arange(vmin, vmax + 0.25, 0.5)
    cbar = fig.colorbar(last_scatter, cax=cbar_ax, ticks=cbar_ticks)
    cbar.ax.tick_params(labelsize=5)
    fig.text(1.03, 0.5, "Z-score", rotation=90, va="center", ha="center", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import numpy as np
    from matplotlib.image import imread

    records = discover_records(args.ig_dir)
    subject_set = {str(s) for s in args.subjects}
    task_set = set(args.tasks)
    records = [
        record
        for record in records
        if record.subject in subject_set and record.task in task_set
    ]
    if not records:
        raise ValueError(f"No EEGNet IG/prediction pairs found in {args.ig_dir}")

    image = imread(args.montage_image)
    coordinates = np.load(args.coordinates)

    by_subject_condition_label: dict[tuple[str, str, int], Accumulator] = defaultdict(Accumulator)
    by_subject_condition: dict[tuple[str, str], Accumulator] = defaultdict(Accumulator)
    by_condition_label: dict[tuple[str, int], Accumulator] = defaultdict(Accumulator)
    by_condition: dict[str, Accumulator] = defaultdict(Accumulator)

    for record in records:
        igs = load_ig_tensor(record.ig_path)
        correct_by_label = correct_indices_by_label(record.predictions_path)
        for label, indices in correct_by_label.items():
            add_trials(
                by_subject_condition_label[(record.subject, record.task, label)],
                igs,
                indices,
                args.clip,
            )
            add_trials(
                by_subject_condition[(record.subject, record.task)],
                igs,
                indices,
                args.clip,
            )
            add_trials(by_condition_label[(record.task, label)], igs, indices, args.clip)
            add_trials(by_condition[record.task], igs, indices, args.clip)

    summary_rows = []

    def save_group(level: str, key, acc: Accumulator, label: int | None = None) -> None:
        values = finalize(acc, args.plot_vmin, args.plot_vmax, args.no_final_clip)
        if values is None:
            return
        if level == "subject_condition_label":
            subject, condition, label_value = key
            color = args.colors[label_value] if label_value < len(args.colors) else str(label_value)
            name = f"{safe_name(subject)}_{safe_name(condition)}_label{label_value}_{safe_name(color)}"
            title = f"{subject} {condition} {color}"
        elif level == "subject_condition":
            subject, condition = key
            color = "avg"
            name = f"{safe_name(subject)}_{safe_name(condition)}_avg"
            title = f"{subject} {condition} avg."
        else:
            condition = key
            subject = "all"
            color = "avg"
            name = f"{safe_name(condition)}_avg"
            title = f"{condition} avg."
        fig_path, table_path = output_paths(args.save_dir, level, name)
        plot_montage(fig_path, values, coordinates, image, args.plot_vmin, args.plot_vmax, title)
        write_table(table_path, values, coordinates, acc.n_trials)
        summary_rows.append(
            {
                "level": level,
                "subject": subject,
                "condition": condition,
                "label": "" if label is None else label,
                "color": color,
                "n_trials": acc.n_trials,
                "figure": str(fig_path),
                "table": str(table_path),
            }
        )

    if not args.combined_only:
        for key in sorted(by_subject_condition_label):
            _, _, label = key
            save_group("subject_condition_label", key, by_subject_condition_label[key], label=label)
        for key in sorted(by_subject_condition):
            save_group("subject_condition", key, by_subject_condition[key])
        for key in sorted(by_condition):
            save_group("condition", key, by_condition[key])

    plot_condition_label_grid(
        args.save_dir / "combined" / "figures" / "condition_by_label_grid.png",
        by_condition_label,
        by_condition,
        args.tasks,
        args.colors,
        coordinates,
        image,
        args.plot_vmin,
        args.plot_vmax,
        args.no_final_clip,
    )
    summary_path = args.save_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["level", "subject", "condition", "label", "color", "n_trials", "figure", "table"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
