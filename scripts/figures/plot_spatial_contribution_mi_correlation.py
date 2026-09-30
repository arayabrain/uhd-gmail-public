#!/usr/bin/env python3
"""Fig. 4 scaffolding: channel-level spatial IG vs EEG–EMG MI Pearson correlation.

Loads precomputed IG tensors from ``--ig-dir`` and MI values from ``--mi-table``
or ``--mi-npy-dir``. Subject IDs are normalized to BIDS ``sub-N``.

Mixed-effects / subject-word LME paths are **not** ported (manuscript uses
channel-level Pearson + Bonferroni as described in the figure caption).

Example
-------
::

    uv run python scripts/figures/plot_spatial_contribution_mi_correlation.py \\
      --ig-dir outputs/integrated_gradients \\
      --mi-table outputs/mutual_information/subject_channel_mi_values.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import tempfile
import warnings
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
        normalize_subject_id,
        parse_ig_name,
        read_prediction_rows,
    )
except ImportError:  # pragma: no cover
    from _ig_common import (  # type: ignore[no-redef]
        CM_TO_INCH,
        DEFAULT_COLORS,
        DEFAULT_SUBJECTS,
        DEFAULT_TASKS,
        load_ig_tensor,
        normalize_subject_id,
        parse_ig_name,
        read_prediction_rows,
    )

DEFAULT_IG_DIR = REPO_ROOT / "outputs" / "integrated_gradients"
DEFAULT_SAVE_DIR = REPO_ROOT / "outputs" / "IG_spatial" / "spatial_contribution_mi_correlation"
DEFAULT_MI_TABLE = (
    REPO_ROOT / "outputs" / "mutual_information" / "subject_channel_mi_values.csv"
)
DEFAULT_EMGS = ["EOG", "EMG upper", "EMG lower"]
MODEL_A = "EEGNet"
MODEL_B = "EEGNet_wo_adapt_filt"
TASK_LABELS = {"overt": "overt", "minimally_overt": "min overt", "covert": "covert"}
COMBINED_PANEL_WIDTH_CM = 3.8
COMBINED_HEIGHT_CM = 6.4
COMBINED_ADJUST = dict(left=0.10, right=0.96, bottom=0.22, top=0.76, wspace=0.26)
COMBINED_TITLE_POS = (0.11, 0.80)
COMBINED_CBAR_AX = [0.84, 0.86, 0.12, 0.035]

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
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
    total: object | None = None
    n_trials: int = 0


@dataclass(frozen=True)
class CorrResult:
    values: object
    p_values: object
    p_values_corrected: object
    significant: object
    n_channels: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ig-dir", type=Path, default=DEFAULT_IG_DIR)
    parser.add_argument("--mi-table", type=Path, default=DEFAULT_MI_TABLE)
    parser.add_argument("--mi-npy-dir", type=Path, default=None)
    parser.add_argument("--mi-eeg-type", default="raw")
    parser.add_argument("--mi-k", type=int, default=3)
    parser.add_argument("--mi-seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--colors", nargs="*", default=DEFAULT_COLORS)
    parser.add_argument("--emgs", nargs="*", default=DEFAULT_EMGS)
    parser.add_argument("--correct-mode", choices=["eegnet", "both"], default="eegnet")
    parser.add_argument(
        "--contribution-mode",
        choices=["eegnet", "wo_adapt", "diff"],
        default="eegnet",
    )
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--plot-vmin", type=float, default=-1.0)
    parser.add_argument("--plot-vmax", type=float, default=1.0)
    parser.add_argument(
        "--levels",
        nargs="*",
        choices=("group",),
        default=["group"],
        help="Public port: channel/group-level Pearson only (no mixed_effects).",
    )
    return parser.parse_args()


def discover_records(ig_dir: Path) -> list[PairedRecord]:
    grouped: dict[tuple[str, str, int, str], dict[str, Path]] = defaultdict(dict)
    metadata: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for path in sorted(ig_dir.glob("*")):
        meta = parse_ig_name(path)
        if meta is None or meta["model"] not in {MODEL_A, MODEL_B}:
            continue
        key = (meta["condition"], meta["task"], int(meta["cv"]), meta["model"])
        grouped[key][meta["kind"]] = path
        metadata[key] = meta

    by_model: dict[tuple[str, str, int], dict[str, ModelRecord]] = defaultdict(dict)
    for key, paths in grouped.items():
        if "igs" not in paths or "trial_predictions" not in paths:
            continue
        condition, task, cv, model = key
        meta = metadata[key]
        by_model[(condition, task, cv)][model] = ModelRecord(
            condition=condition,
            subject=meta["subject"],
            task=task,
            cv=cv,
            model=model,
            ig_path=paths["igs"],
            predictions_path=paths["trial_predictions"],
        )

    paired = []
    for (condition, task, cv), records in by_model.items():
        if MODEL_A not in records or MODEL_B not in records:
            continue
        paired.append(
            PairedRecord(
                condition=condition,
                subject=records[MODEL_A].subject,
                task=task,
                cv=cv,
                eegnet=records[MODEL_A],
                wo_adapt=records[MODEL_B],
            )
        )
    return sorted(paired, key=lambda record: (record.subject, record.task, record.cv))


def trial_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("source_trial") or row.get("dataset_index") or row["ig_index"],
        row.get("inner_trial") or "",
        row.get("trial_file") or "",
        row["label"],
    )


def prediction_map(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {trial_key(row): row for row in read_prediction_rows(path)}


@lru_cache(maxsize=None)
def _load_igs(path_str: str):
    return load_ig_tensor(Path(path_str))


def normalize_trial(ig, clip: float):
    import numpy as np

    ig = np.abs(ig)
    return np.clip((ig - ig.mean()) / max(ig.std(), 1e-12), -clip, clip)


def spatial_vector(ig, clip: float):
    return normalize_trial(ig, clip).mean(axis=1)


def zscore(values):
    return (values - values.mean()) / max(values.std(), 1e-12)


def add_vector(acc: DiffAccumulator, values) -> None:
    import numpy as np

    if acc.total is None:
        acc.total = np.zeros_like(values, dtype=np.float64)
    acc.total += values
    acc.n_trials += 1


def selected_trial_rows(pair: PairedRecord, correct_mode: str) -> list[tuple[dict, dict]]:
    eeg_rows = prediction_map(pair.eegnet.predictions_path)
    wo_rows = prediction_map(pair.wo_adapt.predictions_path)
    out = []
    for key in sorted(set(eeg_rows) & set(wo_rows)):
        eeg_row = eeg_rows[key]
        wo_row = wo_rows[key]
        if eeg_row["correct"].lower() != "true":
            continue
        if correct_mode == "both" and wo_row["correct"].lower() != "true":
            continue
        out.append((eeg_row, wo_row))
    return out


def contribution_vector(pair: PairedRecord, eeg_row, wo_row, mode: str, clip: float):
    eeg_igs = _load_igs(str(pair.eegnet.ig_path))
    wo_igs = _load_igs(str(pair.wo_adapt.ig_path))
    eeg_vec = spatial_vector(eeg_igs[int(eeg_row["ig_index"])], clip)
    wo_vec = spatial_vector(wo_igs[int(wo_row["ig_index"])], clip)
    if mode == "eegnet":
        return eeg_vec
    if mode == "wo_adapt":
        return wo_vec
    if mode == "diff":
        return eeg_vec - wo_vec
    raise ValueError(mode)


def load_mi_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "subject" in row:
            row["subject"] = normalize_subject_id(row["subject"])
    return rows


def mi_channel_vector(rows: list[dict], subject: str | None, emg: str, n_channels: int):
    import numpy as np

    totals = np.zeros(n_channels, dtype=np.float64)
    counts = np.zeros(n_channels, dtype=np.float64)
    emg_key = emg.replace("_", " ")
    for row in rows:
        if subject is not None and row.get("subject") != subject:
            continue
        row_emg = row.get("emg") or row.get("emg_name") or ""
        if row_emg.replace("_", " ") != emg_key:
            continue
        channel = int(row.get("channel") or row.get("ch") or row["eeg_channel"])
        value = float(row.get("mi") or row.get("mutual_information") or row["value"])
        if 0 <= channel < n_channels and np.isfinite(value):
            totals[channel] += value
            counts[channel] += 1
    valid = counts > 0
    if not valid.any():
        return None
    values = np.full(n_channels, np.nan, dtype=np.float64)
    values[valid] = totals[valid] / counts[valid]
    return values


def pearson(left, right):
    import numpy as np

    try:
        from scipy.stats import pearsonr
    except ModuleNotFoundError:
        pearsonr = None
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    if pearsonr is not None:
        corr, p_value = pearsonr(left[mask], right[mask])
        return float(corr), float(p_value), int(mask.sum())
    x = left[mask] - left[mask].mean()
    y = right[mask] - right[mask].mean()
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    corr = float(np.sum(x * y) / denom) if denom else float("nan")
    return corr, float("nan"), int(mask.sum())


def corrected_p_values(p_values, alpha: float):
    import numpy as np

    flat = [(i, j) for i in range(p_values.shape[0]) for j in range(p_values.shape[1])]
    n_tests = len(flat)
    corrected = np.full_like(p_values, np.nan, dtype=np.float64)
    significant = np.zeros_like(p_values, dtype=bool)
    for i, j in flat:
        if np.isnan(p_values[i, j]):
            continue
        value = min(float(p_values[i, j]) * n_tests, 1.0)
        corrected[i, j] = value
        significant[i, j] = value < alpha
    return corrected, significant


def build_group_ig_vectors(pairs, subjects, tasks, label, args):
    groups: dict[str, DiffAccumulator] = defaultdict(DiffAccumulator)
    for pair in pairs:
        if pair.subject not in subjects or pair.task not in tasks:
            continue
        for eeg_row, wo_row in selected_trial_rows(pair, args.correct_mode):
            if label is not None and int(eeg_row["label"]) != label:
                continue
            add_vector(
                groups[pair.task],
                contribution_vector(
                    pair, eeg_row, wo_row, args.contribution_mode, args.clip
                ),
            )
    vectors = {}
    for task, acc in groups.items():
        if acc.total is None or acc.n_trials == 0:
            continue
        vectors[task] = zscore(acc.total / acc.n_trials)
    return vectors


def compute_group_correlations(ig_vectors, mi_rows, tasks, emgs, alpha):
    import numpy as np

    n_tasks = len(tasks)
    n_emgs = len(emgs)
    values = np.full((n_tasks, n_emgs), np.nan)
    p_values = np.full((n_tasks, n_emgs), np.nan)
    n_channels = np.zeros((n_tasks, n_emgs), dtype=int)
    probe = next(iter(ig_vectors.values()))
    for i, task in enumerate(tasks):
        if task not in ig_vectors:
            continue
        for j, emg in enumerate(emgs):
            mi = mi_channel_vector(mi_rows, None, emg, len(probe))
            if mi is None:
                continue
            corr, p_value, n_ch = pearson(ig_vectors[task], mi)
            values[i, j] = corr
            p_values[i, j] = p_value
            n_channels[i, j] = n_ch
    corrected, significant = corrected_p_values(p_values, alpha)
    return CorrResult(values, p_values, corrected, significant, n_channels)


def significance_marker(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plot_corr_grid(path, result, tasks, emgs, colors_title, vmin, vmax) -> None:
    import numpy as np
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cmap = plt.get_cmap("jet").copy()
    im = ax.imshow(result.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(emgs)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(emgs, rotation=90, fontsize=7)
    ax.set_yticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=7)
    ax.set_title(colors_title, fontsize=8)
    for i, j in itertools.product(range(len(tasks)), range(len(emgs))):
        if result.significant[i, j]:
            marker = significance_marker(result.p_values_corrected[i, j])
            if marker:
                ax.text(j, i, marker, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def write_corr_table(path: Path, result: CorrResult, tasks, emgs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "emg",
                "correlation",
                "p_value",
                "p_value_corrected",
                "significant",
                "n_channels",
            ],
        )
        writer.writeheader()
        for i, task in enumerate(tasks):
            for j, emg in enumerate(emgs):
                writer.writerow(
                    {
                        "task": task,
                        "emg": emg,
                        "correlation": float(result.values[i, j]),
                        "p_value": float(result.p_values[i, j]),
                        "p_value_corrected": float(result.p_values_corrected[i, j]),
                        "significant": bool(result.significant[i, j]),
                        "n_channels": int(result.n_channels[i, j]),
                    }
                )


def main() -> None:
    args = parse_args()
    pairs = discover_records(args.ig_dir)
    subjects = {normalize_subject_id(s) for s in args.subjects}
    pairs = [p for p in pairs if p.subject in subjects and p.task in args.tasks]
    if not pairs:
        raise ValueError(f"No paired EEGNet / wo_adapt IG records in {args.ig_dir}")
    if not args.mi_table.exists():
        raise FileNotFoundError(
            f"MI table not found: {args.mi_table}. Pass --mi-table or generate MI outputs first."
        )
    mi_rows = load_mi_table(args.mi_table)

    for label, color in enumerate(args.colors):
        ig_vectors = build_group_ig_vectors(pairs, subjects, args.tasks, label, args)
        if not ig_vectors:
            continue
        result = compute_group_correlations(
            ig_vectors, mi_rows, args.tasks, args.emgs, args.alpha
        )
        stem = f"label{label}_{color}"
        fig_path = args.save_dir / "group" / "figures" / f"{stem}.png"
        table_path = args.save_dir / "group" / "tables" / f"{stem}.csv"
        plot_corr_grid(fig_path, result, args.tasks, args.emgs, color, args.plot_vmin, args.plot_vmax)
        write_corr_table(table_path, result, args.tasks, args.emgs)
        print(fig_path)

    ig_vectors = build_group_ig_vectors(pairs, subjects, args.tasks, None, args)
    result = compute_group_correlations(
        ig_vectors, mi_rows, args.tasks, args.emgs, args.alpha
    )
    fig_path = args.save_dir / "group" / "figures" / "avg.png"
    table_path = args.save_dir / "group" / "tables" / "avg.csv"
    plot_corr_grid(fig_path, result, args.tasks, args.emgs, "avg.", args.plot_vmin, args.plot_vmax)
    write_corr_table(table_path, result, args.tasks, args.emgs)
    print(fig_path)
    print(f"saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
