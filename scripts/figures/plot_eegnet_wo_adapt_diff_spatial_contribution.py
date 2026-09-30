#!/usr/bin/env python3
"""Fig. 5 scaffolding: spatial IG contribution differences (EEGNet − wo_adapt).

Paired EEGNet-correct trials by default. Loads precomputed IGs from ``--ig-dir``.
Subject IDs use public ``sub-N`` form.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IG_DIR = REPO_ROOT / "outputs" / "integrated_gradients"
DEFAULT_SAVE_DIR = (
    REPO_ROOT
    / "outputs"
    / "IG_spatial_final_clip"
    / "wo_adapt_filt_spatial_contribution"
)
DEFAULT_NO_FINAL_CLIP_SAVE_DIR = (
    REPO_ROOT
    / "outputs"
    / "IG_spatial"
    / "wo_adapt_filt_spatial_contribution"
)
DEFAULT_COLORS = ["green", "magenta", "orange", "violet", "yellow"]
DEFAULT_TASKS = ["overt", "minimally_overt", "covert"]
MODEL_A = "EEGNet"
MODEL_B = "EEGNet_wo_adapt_filt"
DIFF_CMAP = "bwr"
CM_TO_INCH = 1 / 2.54
NAME_RE = re.compile(
    r"^(?P<condition>.+?)_(?P<model>EEGNet_wo_adapt_filt|EMG_EEGNet|EEGNet)_"
    r"(?P<run_date>\d{4}-\d{2}-\d{2})_(?P<run_time>\d{2}-\d{2}-\d{2})_"
    r"(?P<task>.+)_cv(?P<cv>\d+)_(?P<kind>igs|trial_predictions)\.(?P<ext>pt|csv)$"
)


@dataclass(frozen=True)
class ModelRecord:
    condition: str
    subject: str
    task: str
    cv: int
    model: str
    ig_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class PairedRecord:
    condition: str
    subject: str
    task: str
    cv: int
    eegnet: ModelRecord
    wo_adapt: ModelRecord


@dataclass
class DiffAccumulator:
    total_diff: object | None = None
    n_trials: int = 0
    total_a: object | None = None
    total_b: object | None = None
    n_a: int = 0
    n_b: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ig-dir", type=Path, default=DEFAULT_IG_DIR)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--colors", nargs="*", default=DEFAULT_COLORS)
    parser.add_argument(
        "--correct-mode",
        choices=["eegnet", "both", "each_model"],
        default="eegnet",
        help=(
            "eegnet: paired difference on EEGNet-correct trials; "
            "both: paired difference on trials correct in both models; "
            "each_model: subtract model-wise means from each model's own correct trials."
        ),
    )
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--plot-vmin", type=float, default=-1.0)
    parser.add_argument("--plot-vmax", type=float, default=1.0)
    parser.add_argument(
        "--no-final-clip",
        dest="no_final_clip",
        action="store_true",
        default=True,
        help="Do not clip the final z-scored spatial difference before writing tables/figures.",
    )
    parser.add_argument(
        "--final-clip",
        dest="no_final_clip",
        action="store_false",
        help="Clip the final z-scored spatial difference before writing tables/figures.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only write the combined condition-by-label grid figure.",
    )
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
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = DEFAULT_NO_FINAL_CLIP_SAVE_DIR if args.no_final_clip else DEFAULT_SAVE_DIR
    return args


def parse_name(path: Path) -> dict[str, str] | None:
    match = NAME_RE.match(path.name)
    return None if match is None else match.groupdict()


def discover_records(ig_dir: Path) -> list[PairedRecord]:
    grouped: dict[tuple[str, str, int, str], dict[str, Path]] = defaultdict(dict)
    metadata: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for path in sorted(ig_dir.glob("*")):
        meta = parse_name(path)
        if meta is None or meta["model"] not in {MODEL_A, MODEL_B}:
            continue
        key = (meta["condition"], meta["task"], int(meta["cv"]), meta["model"])
        grouped[key][meta["kind"]] = path
        metadata[key] = meta

    records_by_model: dict[tuple[str, str, int], dict[str, ModelRecord]] = defaultdict(dict)
    for key, paths in grouped.items():
        if "igs" not in paths or "trial_predictions" not in paths:
            continue
        meta = metadata[key]
        condition, task, cv, model = key
        records_by_model[(condition, task, cv)][model] = ModelRecord(
            condition=condition,
            subject=condition.split("-")[0],
            task=task,
            cv=cv,
            model=model,
            ig_path=paths["igs"],
            predictions_path=paths["trial_predictions"],
        )

    paired = []
    for key, by_model in records_by_model.items():
        if MODEL_A not in by_model or MODEL_B not in by_model:
            continue
        condition, task, cv = key
        paired.append(
            PairedRecord(
                condition=condition,
                subject=condition.split("-")[0],
                task=task,
                cv=cv,
                eegnet=by_model[MODEL_A],
                wo_adapt=by_model[MODEL_B],
            )
        )
    return sorted(paired, key=lambda record: (record.condition, record.task, record.cv))


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def trial_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    source_trial = row.get("source_trial") or row.get("dataset_index") or row["ig_index"]
    inner_trial = row.get("inner_trial") or ""
    trial_file = row.get("trial_file") or ""
    label = row["label"]
    return source_trial, inner_trial, trial_file, label


def prediction_map(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {trial_key(row): row for row in read_prediction_rows(path)}


def load_igs(path: Path):
    import numpy as np
    import torch

    igs = torch.load(path, map_location="cpu")
    if hasattr(igs, "detach"):
        igs = igs.detach().cpu().numpy()
    igs = np.asarray(igs)
    if igs.ndim == 4 and igs.shape[1] == 1:
        igs = igs[:, 0]
    if igs.ndim != 3:
        raise ValueError(f"Expected IG shape (trial, channel, time), got {igs.shape}")
    return igs


def normalize_trials(igs, clip: float):
    import numpy as np

    igs = np.abs(igs)
    mean = igs.mean(axis=(1, 2), keepdims=True)
    std = igs.std(axis=(1, 2), keepdims=True)
    return np.clip((igs - mean) / np.maximum(std, 1e-12), -clip, clip)


def spatial_trials(igs, indices: list[int], clip: float):
    import numpy as np

    if not indices:
        return None
    selected = normalize_trials(igs[np.asarray(indices, dtype=int)], clip)
    return selected.mean(axis=2)


def add_paired_diff(acc: DiffAccumulator, spatial_a, spatial_b) -> None:
    import numpy as np

    if spatial_a is None or spatial_b is None or spatial_a.shape != spatial_b.shape:
        return
    diff = spatial_a - spatial_b
    if acc.total_diff is None:
        acc.total_diff = np.zeros(diff.shape[1], dtype=np.float64)
    acc.total_diff += diff.sum(axis=0)
    acc.n_trials += diff.shape[0]
    acc.n_a += diff.shape[0]
    acc.n_b += diff.shape[0]


def add_model_trials(acc: DiffAccumulator, spatial, model: str) -> None:
    import numpy as np

    if spatial is None:
        return
    if model == MODEL_A:
        if acc.total_a is None:
            acc.total_a = np.zeros(spatial.shape[1], dtype=np.float64)
        acc.total_a += spatial.sum(axis=0)
        acc.n_a += spatial.shape[0]
    else:
        if acc.total_b is None:
            acc.total_b = np.zeros(spatial.shape[1], dtype=np.float64)
        acc.total_b += spatial.sum(axis=0)
        acc.n_b += spatial.shape[0]


def finalize(acc: DiffAccumulator, plot_vmin: float, plot_vmax: float, no_final_clip: bool = False):
    import numpy as np

    if acc.total_diff is not None and acc.n_trials > 0:
        values = acc.total_diff / acc.n_trials
    elif acc.total_a is not None and acc.total_b is not None and acc.n_a > 0 and acc.n_b > 0:
        values = (acc.total_a / acc.n_a) - (acc.total_b / acc.n_b)
    else:
        return None
    values = (values - values.mean()) / max(values.std(), 1e-12)
    if no_final_clip:
        return values
    return np.clip(values, plot_vmin, plot_vmax)


def add_to_groups(
    groups: tuple[
        dict[tuple[str, str, int], DiffAccumulator],
        dict[tuple[str, str], DiffAccumulator],
        dict[tuple[str, int], DiffAccumulator],
        dict[str, DiffAccumulator],
    ],
    subject: str,
    task: str,
    label: int,
) -> list[DiffAccumulator]:
    by_subject_condition_label, by_subject_condition, by_condition_label, by_condition = groups
    return [
        by_subject_condition_label[(subject, task, label)],
        by_subject_condition[(subject, task)],
        by_condition_label[(task, label)],
        by_condition[task],
    ]


def add_record_paired(
    record: PairedRecord,
    correct_mode: str,
    groups: tuple[
        dict[tuple[str, str, int], DiffAccumulator],
        dict[tuple[str, str], DiffAccumulator],
        dict[tuple[str, int], DiffAccumulator],
        dict[str, DiffAccumulator],
    ],
    clip: float,
) -> None:
    from collections import defaultdict as dd

    eeg_rows = prediction_map(record.eegnet.predictions_path)
    wo_rows = prediction_map(record.wo_adapt.predictions_path)
    common_keys = sorted(set(eeg_rows) & set(wo_rows))
    selected_by_label: dict[int, list[tuple[int, int]]] = dd(list)

    for key in common_keys:
        eeg_row = eeg_rows[key]
        wo_row = wo_rows[key]
        if eeg_row["correct"].lower() != "true":
            continue
        if correct_mode == "both" and wo_row["correct"].lower() != "true":
            continue
        label = int(eeg_row["label"])
        selected_by_label[label].append((int(eeg_row["ig_index"]), int(wo_row["ig_index"])))

    if not selected_by_label:
        return

    igs_a = load_igs(record.eegnet.ig_path)
    igs_b = load_igs(record.wo_adapt.ig_path)
    for label, index_pairs in selected_by_label.items():
        idx_a = [pair[0] for pair in index_pairs]
        idx_b = [pair[1] for pair in index_pairs]
        spatial_a = spatial_trials(igs_a, idx_a, clip)
        spatial_b = spatial_trials(igs_b, idx_b, clip)
        for acc in add_to_groups(groups, record.subject, record.task, label):
            add_paired_diff(acc, spatial_a, spatial_b)


def add_record_each_model(
    record: PairedRecord,
    groups: tuple[
        dict[tuple[str, str, int], DiffAccumulator],
        dict[tuple[str, str], DiffAccumulator],
        dict[tuple[str, int], DiffAccumulator],
        dict[str, DiffAccumulator],
    ],
    clip: float,
) -> None:
    for model_record in [record.eegnet, record.wo_adapt]:
        rows_by_label: dict[int, list[int]] = defaultdict(list)
        for row in read_prediction_rows(model_record.predictions_path):
            if row["correct"].lower() != "true":
                continue
            rows_by_label[int(row["label"])].append(int(row["ig_index"]))
        if not rows_by_label:
            continue
        igs = load_igs(model_record.ig_path)
        for label, indices in rows_by_label.items():
            spatial = spatial_trials(igs, indices, clip)
            for acc in add_to_groups(groups, record.subject, record.task, label):
                add_model_trials(acc, spatial, model_record.model)


def safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "-")


def output_paths(save_dir: Path, level: str, name: str) -> tuple[Path, Path]:
    fig_path = save_dir / level / "figures" / f"{name}.png"
    table_path = save_dir / level / "tables" / f"{name}.csv"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    return fig_path, table_path


def write_table(path: Path, values, coordinates, acc: DiffAccumulator) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "channel",
                "x",
                "y",
                "spatial_difference",
                "n_paired_trials",
                "n_eegnet_trials",
                "n_wo_adapt_trials",
            ],
        )
        writer.writeheader()
        for channel, value in enumerate(values):
            writer.writerow(
                {
                    "channel": channel,
                    "x": float(coordinates[channel, 0]),
                    "y": float(coordinates[channel, 1]),
                    "spatial_difference": float(value),
                    "n_paired_trials": acc.n_trials,
                    "n_eegnet_trials": acc.n_a,
                    "n_wo_adapt_trials": acc.n_b,
                }
            )


def plot_montage(path: Path, values, coordinates, image, vmin: float, vmax: float, title: str) -> None:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(3.1 * CM_TO_INCH, 3.1 * CM_TO_INCH))
    ax.imshow(image)
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=5,
        c=values,
        cmap=DIFF_CMAP,
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
    path: Path,
    by_condition_label: dict[tuple[str, int], DiffAccumulator],
    by_condition: dict[str, DiffAccumulator],
    tasks: list[str],
    colors: list[str],
    coordinates,
    image,
    vmin: float,
    vmax: float,
    no_final_clip: bool = False,
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
                    cmap=DIFF_CMAP,
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
                cmap=DIFF_CMAP,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
            )
        ax.axis("off")
        if row_idx == 0:
            ax.set_title("avg.", fontsize=7)

    if last_scatter is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        last_scatter = mpl.cm.ScalarMappable(norm=norm, cmap=DIFF_CMAP)
    fig.subplots_adjust(wspace=0.05, hspace=0.15, right=0.93)
    cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    cbar_ticks = np.arange(vmin, vmax + 0.25, 0.5)
    cbar = fig.colorbar(last_scatter, cax=cbar_ax, ticks=cbar_ticks)
    cbar.ax.tick_params(labelsize=5)
    fig.text(1.03, 0.5, "Z-score difference", rotation=90, va="center", ha="center", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def write_summary(save_dir: Path, rows: list[dict[str, object]]) -> None:
    path = save_dir / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "subject",
                "condition",
                "label",
                "color",
                "n_paired_trials",
                "n_eegnet_trials",
                "n_wo_adapt_trials",
                "figure",
                "table",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    import numpy as np
    from matplotlib.image import imread

    records = discover_records(args.ig_dir)
    if args.subjects is not None:
        subject_set = set(args.subjects)
        records = [record for record in records if record.subject in subject_set]
    task_set = set(args.tasks)
    records = [record for record in records if record.task in task_set]
    if not records:
        raise ValueError(f"No paired {MODEL_A}/{MODEL_B} IG records found in {args.ig_dir}")

    save_dir = args.save_dir
    image = imread(args.montage_image)
    coordinates = np.load(args.coordinates)

    by_subject_condition_label: dict[tuple[str, str, int], DiffAccumulator] = defaultdict(DiffAccumulator)
    by_subject_condition: dict[tuple[str, str], DiffAccumulator] = defaultdict(DiffAccumulator)
    by_condition_label: dict[tuple[str, int], DiffAccumulator] = defaultdict(DiffAccumulator)
    by_condition: dict[str, DiffAccumulator] = defaultdict(DiffAccumulator)
    groups = (
        by_subject_condition_label,
        by_subject_condition,
        by_condition_label,
        by_condition,
    )

    for record in records:
        if args.correct_mode == "each_model":
            add_record_each_model(record, groups, args.clip)
        else:
            add_record_paired(record, args.correct_mode, groups, args.clip)

    summary_rows = []

    def save_group(level: str, key, acc: DiffAccumulator, label: int | None = None) -> None:
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
        fig_path, table_path = output_paths(save_dir, level, name)
        plot_montage(fig_path, values, coordinates, image, args.plot_vmin, args.plot_vmax, title)
        write_table(table_path, values, coordinates, acc)
        summary_rows.append(
            {
                "level": level,
                "subject": subject,
                "condition": condition,
                "label": "" if label is None else label,
                "color": color,
                "n_paired_trials": acc.n_trials,
                "n_eegnet_trials": acc.n_a,
                "n_wo_adapt_trials": acc.n_b,
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
        save_dir / "combined" / "figures" / "condition_by_label_grid.png",
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

    if not args.combined_only:
        write_summary(save_dir, summary_rows)
    print(f"records: {len(records)}")
    print(f"correct_mode: {args.correct_mode}")
    print(f"figures/tables: {len(summary_rows)}")
    print(f"saved to: {save_dir}")


if __name__ == "__main__":
    main()

