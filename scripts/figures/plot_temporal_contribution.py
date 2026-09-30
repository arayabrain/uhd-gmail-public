#!/usr/bin/env python3
"""Fig. 3 scaffolding: temporal IG contribution tables from precomputed IGs.

Exports per-target temporal contribution CSVs (EEGNet, EEGNet_wo_adapt_filt,
EMG_EEGNet EOG/upper/lower) using EEGNet-correct trials by default and top-10
EEG channels. Loads ``*_igs.pt`` from ``--ig-dir`` (no checkpoint recompute).

Example
-------
::

    uv run python scripts/figures/plot_temporal_contribution.py \\
      --ig-dir outputs/integrated_gradients
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
        DEFAULT_COLORS,
        DEFAULT_SUBJECTS,
        DEFAULT_TASKS,
        load_ig_tensor,
        parse_ig_name,
        read_prediction_rows,
        safe_name,
    )

DEFAULT_IG_DIR = REPO_ROOT / "outputs" / "integrated_gradients"
DEFAULT_SAVE_DIR = REPO_ROOT / "outputs" / "IG_temporal_contribution" / "top10_ch"
REFERENCE_MODEL = "EEGNet"


@dataclass(frozen=True)
class ModelRecord:
    condition: str
    subject: str
    model: str
    task: str
    cv: int
    ig_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class TargetSpec:
    key: str
    model: str
    display_name: str
    channel_indices: tuple[int, ...] | None


@dataclass
class Accumulator:
    total: object | None = None
    total_map: object | None = None
    n_trials: int = 0


TARGETS = [
    TargetSpec("eegnet", "EEGNet", "EEGNet", None),
    TargetSpec("eegnet_wo_adapt_filt", "EEGNet_wo_adapt_filt", "EEGNet_wo_adapt_filt", None),
    TargetSpec("emg_eegnet_eog", "EMG_EEGNet", "EMG_EEGNet(EOG)", (0,)),
    TargetSpec("emg_eegnet_emg_upper", "EMG_EEGNet", "EMG_EEGNet(EMG upper)", (1,)),
    TargetSpec("emg_eegnet_emg_lower", "EMG_EEGNet", "EMG_EEGNet(EMG lower)", (2,)),
]


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
    parser.add_argument("--correct-mode", choices=["eegnet", "own"], default="eegnet")
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--output-clip", type=float, default=1.0)
    parser.add_argument("--eeg-channel-mode", choices=["mean", "top_k"], default="top_k")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-final-clip", dest="no_final_clip", action="store_true", default=True)
    parser.add_argument("--final-clip", dest="no_final_clip", action="store_false")
    return parser.parse_args()


def discover_records(ig_dir: Path) -> dict[tuple[str, str, int, str], ModelRecord]:
    grouped: dict[tuple[str, str, int, str], dict[str, Path]] = defaultdict(dict)
    metadata: dict[tuple[str, str, int, str], dict[str, str]] = {}
    target_models = {target.model for target in TARGETS}
    for path in sorted(ig_dir.glob("*")):
        meta = parse_ig_name(path)
        if meta is None or meta["model"] not in target_models:
            continue
        key = (meta["condition"], meta["task"], int(meta["cv"]), meta["model"])
        grouped[key][meta["kind"]] = path
        metadata[key] = meta

    records = {}
    for key, paths in grouped.items():
        if "igs" not in paths or "trial_predictions" not in paths:
            continue
        meta = metadata[key]
        records[key] = ModelRecord(
            condition=meta["condition"],
            subject=meta["subject"],
            model=meta["model"],
            task=meta["task"],
            cv=int(meta["cv"]),
            ig_path=paths["igs"],
            predictions_path=paths["trial_predictions"],
        )
    return records


def trial_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    source_trial = row.get("source_trial") or row.get("dataset_index") or row["ig_index"]
    inner_trial = row.get("inner_trial") or ""
    trial_file = row.get("trial_file") or ""
    return source_trial, inner_trial, trial_file, row["label"]


def prediction_map(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {trial_key(row): row for row in read_prediction_rows(path)}


def selected_indices_by_label(
    target_record: ModelRecord,
    reference_record: ModelRecord | None,
    correct_mode: str,
) -> dict[int, list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    if correct_mode == "own" or target_record.model == REFERENCE_MODEL:
        for row in read_prediction_rows(target_record.predictions_path):
            if row["correct"].lower() == "true":
                by_label[int(row["label"])].append(int(row["ig_index"]))
        return by_label
    if reference_record is None:
        return by_label
    reference_rows = prediction_map(reference_record.predictions_path)
    target_rows = prediction_map(target_record.predictions_path)
    for key in sorted(set(reference_rows) & set(target_rows)):
        reference_row = reference_rows[key]
        if reference_row["correct"].lower() != "true":
            continue
        target_row = target_rows[key]
        by_label[int(reference_row["label"])].append(int(target_row["ig_index"]))
    return by_label


def normalize_trials(igs, clip: float):
    import numpy as np

    igs = np.abs(igs)
    mean = igs.mean(axis=(1, 2), keepdims=True)
    std = igs.std(axis=(1, 2), keepdims=True)
    return np.clip((igs - mean) / np.maximum(std, 1e-12), -clip, clip)


def add_trials(
    acc: Accumulator,
    igs,
    indices: list[int],
    channels: tuple[int, ...] | None,
    clip: float,
    use_top_k: bool = False,
) -> None:
    import numpy as np

    if not indices:
        return
    selected = normalize_trials(igs[np.asarray(indices, dtype=int)], clip)
    if channels is not None:
        selected = selected[:, np.asarray(channels, dtype=int), :]
    if use_top_k:
        if acc.total_map is None:
            acc.total_map = np.zeros(selected.shape[1:], dtype=np.float64)
        acc.total_map += selected.sum(axis=0)
        acc.n_trials += selected.shape[0]
        return
    temporal = selected.mean(axis=1)
    if acc.total is None:
        acc.total = np.zeros(temporal.shape[1], dtype=np.float64)
    acc.total += temporal.sum(axis=0)
    acc.n_trials += temporal.shape[0]


def finalize(acc: Accumulator, output_clip: float, no_final_clip: bool, top_k: int | None):
    import numpy as np

    if acc.total is None or acc.n_trials == 0:
        if acc.total_map is None or acc.n_trials == 0:
            return None, None, None
        mean_map = acc.total_map / acc.n_trials
        spatial = mean_map.mean(axis=1)
        n_channels = mean_map.shape[0]
        k = min(int(top_k or n_channels), n_channels)
        top_indices = np.argsort(spatial)[::-1][:k]
        values = mean_map[top_indices].mean(axis=0)
        top_scores = spatial[top_indices]
    else:
        values = acc.total / acc.n_trials
        top_indices = None
        top_scores = None
    values = (values - values.mean()) / max(values.std(), 1e-12)
    if no_final_clip:
        return values, top_indices, top_scores
    return np.clip(values, -output_clip, output_clip), top_indices, top_scores


def table_path(save_dir: Path, target: TargetSpec, level: str, name: str) -> Path:
    path = save_dir / f"{target.key}_temporal_contribution" / level / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def top_channels_path(save_dir: Path, target: TargetSpec, level: str, name: str) -> Path:
    path = save_dir / f"{target.key}_temporal_contribution" / level / "top_channels" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_table(path: Path, values, n_trials: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_index", "temporal_contribution", "n_trials"])
        writer.writeheader()
        for time_index, value in enumerate(values):
            writer.writerow(
                {
                    "time_index": time_index,
                    "temporal_contribution": float(value),
                    "n_trials": n_trials,
                }
            )


def write_top_channels(path: Path, top_indices, top_scores) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "channel", "spatial_contribution"])
        writer.writeheader()
        if top_indices is None or top_scores is None:
            return
        for rank, (channel, score) in enumerate(zip(top_indices, top_scores), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "channel": int(channel),
                    "spatial_contribution": float(score),
                }
            )


def main() -> None:
    args = parse_args()
    records = discover_records(args.ig_dir)
    if not records:
        raise ValueError(f"No IG/prediction records found in {args.ig_dir}")

    summary_rows = []
    task_set = set(args.tasks)
    subject_set = set(args.subjects)

    for target in TARGETS:
        by_subject_condition_label: dict[tuple[str, str, int], Accumulator] = defaultdict(Accumulator)
        by_subject_condition: dict[tuple[str, str], Accumulator] = defaultdict(Accumulator)
        by_condition: dict[str, Accumulator] = defaultdict(Accumulator)

        target_records = [
            record
            for key, record in records.items()
            if record.model == target.model
            and record.task in task_set
            and record.subject in subject_set
        ]
        for record in sorted(target_records, key=lambda item: (item.condition, item.task, item.cv)):
            reference_record = records.get(
                (record.condition, record.task, record.cv, REFERENCE_MODEL)
            )
            selected_by_label = selected_indices_by_label(
                record, reference_record, args.correct_mode
            )
            if not selected_by_label:
                continue
            igs = load_ig_tensor(record.ig_path)
            if target.channel_indices is not None and max(target.channel_indices) >= igs.shape[1]:
                raise ValueError(
                    f"{target.display_name} needs channel {max(target.channel_indices)}, "
                    f"but {record.ig_path} has {igs.shape[1]} channels"
                )
            for label, indices in selected_by_label.items():
                use_top_k = args.eeg_channel_mode == "top_k" and target.channel_indices is None
                add_trials(
                    by_subject_condition_label[(record.subject, record.task, label)],
                    igs,
                    indices,
                    target.channel_indices,
                    args.clip,
                    use_top_k,
                )
                add_trials(
                    by_subject_condition[(record.subject, record.task)],
                    igs,
                    indices,
                    target.channel_indices,
                    args.clip,
                    use_top_k,
                )
                add_trials(
                    by_condition[record.task],
                    igs,
                    indices,
                    target.channel_indices,
                    args.clip,
                    use_top_k,
                )

        def save_group(level: str, key, acc: Accumulator, label: int | None = None) -> None:
            values, top_indices, top_scores = finalize(
                acc,
                args.output_clip,
                args.no_final_clip,
                args.top_k if args.eeg_channel_mode == "top_k" else None,
            )
            if values is None:
                return
            if level == "subject_condition_label":
                subject, condition, label_value = key
                color = args.colors[label_value] if label_value < len(args.colors) else str(label_value)
                name = f"{safe_name(subject)}_{safe_name(condition)}_label{label_value}_{safe_name(color)}"
            elif level == "subject_condition":
                subject, condition = key
                color = "avg"
                name = f"{safe_name(subject)}_{safe_name(condition)}_avg"
            else:
                condition = key
                subject = "all"
                color = "avg"
                name = f"{safe_name(condition)}_avg"
            path = table_path(args.save_dir, target, level, name)
            top_path = ""
            if args.eeg_channel_mode == "top_k" and target.channel_indices is None:
                top_path_obj = top_channels_path(args.save_dir, target, level, name)
                write_top_channels(top_path_obj, top_indices, top_scores)
                top_path = str(top_path_obj)
            write_table(path, values, acc.n_trials)
            summary_rows.append(
                {
                    "target": target.display_name,
                    "model": target.model,
                    "channels": (
                        f"top_{args.top_k}"
                        if args.eeg_channel_mode == "top_k" and target.channel_indices is None
                        else "all"
                        if target.channel_indices is None
                        else ",".join(map(str, target.channel_indices))
                    ),
                    "correct_mode": args.correct_mode,
                    "level": level,
                    "subject": subject,
                    "condition": condition,
                    "label": "" if label is None else label,
                    "color": color,
                    "n_trials": acc.n_trials,
                    "table": str(path),
                    "top_channels_table": top_path,
                }
            )

        for key in sorted(by_subject_condition_label):
            _, _, label = key
            save_group("subject_condition_label", key, by_subject_condition_label[key], label=label)
        for key in sorted(by_subject_condition):
            save_group("subject_condition", key, by_subject_condition[key])
        for key in sorted(by_condition):
            save_group("condition", key, by_condition[key])

    summary_path = args.save_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "model",
                "channels",
                "correct_mode",
                "level",
                "subject",
                "condition",
                "label",
                "color",
                "n_trials",
                "table",
                "top_channels_table",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"targets: {len(TARGETS)}")
    print(f"tables: {len(summary_rows)}")
    print(f"saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
