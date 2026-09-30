#!/usr/bin/env python3
"""Fig. 3 scaffolding: trial-based temporal Jaccard / similarity matrices.

Builds per-trial temporal waveforms from precomputed IG outputs, computes
pairwise Jaccard (or related metrics), and tests significance. Subject IDs use
public ``sub-N`` form.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IG_DIR = REPO_ROOT / "outputs" / "integrated_gradients"
DEFAULT_SAVE_DIR = (
    REPO_ROOT
    / "outputs"
    / "IG_temporal_similarity"
    / "top10_ch"
    / "target_similarity_trial_based"
)
DEFAULT_COLORS = ["green", "magenta", "orange", "violet", "yellow"]
DEFAULT_TASKS = ["overt", "minimally_overt", "covert"]
METRIC_CHOICES = ("jaccard", "soft_jaccard", "pearson", "peak_latency")
SURROGATE_METHOD_CHOICES = (
    "time_permutation",
    "circular_shift",
    "block_permutation",
    "phase_randomization",
    "aaft",
    "iaaft",
    "target_label_permutation",
    "different_label_trial_pair_shuffle",
    "trial_pair_shuffle",
    "sign_flip",
)
TRIAL_LABEL_KEY = "__label__"
PAIR_SHUFFLE_METHODS = {"different_label_trial_pair_shuffle", "trial_pair_shuffle"}
REFERENCE_MODEL = "EEGNet"
TARGETS = [
    ("EEGNet", "EEGNet", None),
    ("EEGNet_wo_adapt_filt", "EEGNet_wo_adapt_filt", None),
    ("EMG_EEGNet(EOG)", "EMG_EEGNet", (0,)),
    ("EMG_EEGNet(EMG upper)", "EMG_EEGNet", (1,)),
    ("EMG_EEGNet(EMG lower)", "EMG_EEGNet", (2,)),
]
TARGET_ORDER = [target[0] for target in TARGETS]
TARGET_SHORT_NAMES = {
    "EEGNet": "denoised\nEEG",
    "EEGNet_wo_adapt_filt": "min pre-\nprocessed EEG",
    "EMG_EEGNet(EOG)": "EOG",
    "EMG_EEGNet(EMG upper)": "EMG upper",
    "EMG_EEGNet(EMG lower)": "EMG lower",
}
TARGET_LABEL_COLORS = {
    "EEGNet": "#2b8cbe",
    "EEGNet_wo_adapt_filt": "#d8b365",
    "EMG_EEGNet(EOG)": "#1a9850",
    "EMG_EEGNet(EMG upper)": "#b2182b",
    "EMG_EEGNet(EMG lower)": "#7b3294",
}
NAME_RE = re.compile(
    r"^(?P<condition>.+?)_(?P<model>EEGNet_wo_adapt_filt|EMG_EEGNet|EEGNet)_"
    r"(?P<run_date>\d{4}-\d{2}-\d{2})_(?P<run_time>\d{2}-\d{2}-\d{2})_"
    r"(?P<task>.+)_cv(?P<cv>\d+)_(?P<kind>igs|trial_predictions)\.(?P<ext>pt|csv)$"
)


@dataclass(frozen=True)
class Record:
    condition: str
    subject: str
    model: str
    task: str
    cv: int
    ig_path: Path
    predictions_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ig-dir", type=Path, default=DEFAULT_IG_DIR)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If omitted, the directory is derived from "
            "--num-surrogate: *_no_surrogate for 0, *_surrogate<N> otherwise."
        ),
    )
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--colors", nargs="*", default=DEFAULT_COLORS)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-surrogate", type=int, default=1000)
    parser.add_argument(
        "--surrogate-method",
        choices=SURROGATE_METHOD_CHOICES,
        default="time_permutation",
        help=(
            "Null model for surrogate tests. time_permutation shuffles time points; "
            "circular_shift preserves waveform shape and shifts timing; "
            "block_permutation shuffles local time blocks; phase_randomization "
            "preserves the amplitude spectrum; aaft and iaaft use amplitude-adjusted "
            "Fourier transform surrogates; target_label_permutation shuffles "
            "target labels within each trial; different_label_trial_pair_shuffle "
            "pairs each trial with a random different-label trial; trial_pair_shuffle "
            "pairs each trial with a random trial without label constraints; sign_flip "
            "flips waveform sign."
        ),
    )
    parser.add_argument(
        "--surrogate-block-size",
        type=int,
        default=16,
        help="Block length in samples for --surrogate-method block_permutation.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of worker processes for waveform-based surrogate iterations. "
            "Use 1 for serial execution, or 0/-1 to use all available CPUs."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["jaccard"],
        choices=(*METRIC_CHOICES, "all"),
        help=(
            "Similarity metrics for the five temporal contribution waveforms. "
            "Use 'pearson' for temporal contribution correlation; use 'all' to "
            "also compute Jaccard, soft Jaccard, and peak-latency difference."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help=(
            "Optional sampling rate in Hz for peak-latency differences. If omitted, "
            "peak latency is reported in sample indices."
        ),
    )
    parser.add_argument(
        "--stat-level",
        choices=("subject", "unique_trial"),
        default="subject",
        help=(
            "Unit used for matrix averaging and surrogate tests. "
            "subject averages trial matrices within subject first; unique_trial pools "
            "deduplicated source trials across subjects."
        ),
    )
    args = parser.parse_args()
    if "all" in args.metrics:
        args.metrics = list(METRIC_CHOICES)
    if args.save_dir is None:
        suffix = "no_surrogate" if args.num_surrogate == 0 else f"surrogate{args.num_surrogate}"
        args.save_dir = DEFAULT_SAVE_DIR.with_name(
            f"{DEFAULT_SAVE_DIR.name}_{args.surrogate_method}_{suffix}"
        )
    return args


def parse_name(path: Path) -> dict[str, str] | None:
    match = NAME_RE.match(path.name)
    return None if match is None else match.groupdict()


def discover_records(ig_dir: Path) -> dict[tuple[str, str, int, str], Record]:
    grouped: dict[tuple[str, str, int, str], dict[str, Path]] = defaultdict(dict)
    metadata: dict[tuple[str, str, int, str], dict[str, str]] = {}
    target_models = {target[1] for target in TARGETS}
    for path in sorted(ig_dir.glob("*")):
        meta = parse_name(path)
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
        records[key] = Record(
            condition=meta["condition"],
            subject=meta["condition"].split("-")[0],
            model=meta["model"],
            task=meta["task"],
            cv=int(meta["cv"]),
            ig_path=paths["igs"],
            predictions_path=paths["trial_predictions"],
        )
    return records


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def trial_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("source_trial") or row.get("dataset_index") or row["ig_index"],
        row.get("inner_trial") or "",
        row.get("trial_file") or "",
        row["label"],
    )


def unique_trial_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return trial_key(row)


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


def normalize_trial(ig, clip: float):
    import numpy as np

    ig = np.abs(ig)
    return np.clip((ig - ig.mean()) / max(ig.std(), 1e-12), -clip, clip)


def zscore_waveform(values):
    return (values - values.mean()) / max(values.std(), 1e-12)


def jaccard_index(left, right) -> float:
    import numpy as np

    left_set = set(np.where(left > left.mean())[0].tolist())
    right_set = set(np.where(right > right.mean())[0].tolist())
    union = len(left_set | right_set)
    return 0.0 if union == 0 else len(left_set & right_set) / union


def soft_jaccard_index(left, right) -> float:
    import numpy as np

    left = np.clip(left, 0, None)
    right = np.clip(right, 0, None)
    union = np.maximum(left, right).sum()
    return 0.0 if union == 0 else float(np.minimum(left, right).sum() / union)


def pearson_index(left, right) -> float:
    import numpy as np

    left = left - left.mean()
    right = right - right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return 0.0 if denom == 0 else float(np.dot(left, right) / denom)


def peak_latency_diff(left, right, sample_rate: float | None) -> float:
    import numpy as np

    value = abs(int(np.argmax(left)) - int(np.argmax(right)))
    return float(value / sample_rate) if sample_rate is not None else float(value)


def metric_index(left, right, metric: str, sample_rate: float | None) -> float:
    if metric == "jaccard":
        return jaccard_index(left, right)
    if metric == "soft_jaccard":
        return soft_jaccard_index(left, right)
    if metric == "pearson":
        return pearson_index(left, right)
    if metric == "peak_latency":
        return peak_latency_diff(left, right, sample_rate)
    raise ValueError(f"Unknown metric: {metric}")


def metric_diagonal(metric: str) -> float:
    return 0.0 if metric == "peak_latency" else 1.0


def metric_alternative(metric: str) -> str:
    return "less" if metric == "peak_latency" else "greater"


def metric_value_names(metric: str) -> tuple[str, str, str]:
    if metric == "soft_jaccard":
        return "mean_soft_jaccard", "sem_soft_jaccard", "Soft Jaccard index"
    if metric == "pearson":
        return "mean_pearson", "sem_pearson", "Pearson r"
    if metric == "peak_latency":
        return "mean_peak_latency_diff", "sem_peak_latency_diff", "Peak latency difference"
    return "mean_jaccard", "sem_jaccard", "Jaccard index"


def metric_output_dir(save_dir: Path, metric: str) -> Path:
    return save_dir if metric == "jaccard" else save_dir / safe_name(metric)


def metric_matrix(waveforms: dict[str, object], metric: str, sample_rate: float | None):
    import numpy as np

    matrix = np.full((len(TARGET_ORDER), len(TARGET_ORDER)), np.nan, dtype=np.float64)
    for i, left in enumerate(TARGET_ORDER):
        for j, right in enumerate(TARGET_ORDER):
            if left not in waveforms or right not in waveforms:
                continue
            matrix[i, j] = metric_diagonal(metric) if i == j else metric_index(
                waveforms[left], waveforms[right], metric, sample_rate
            )
    return matrix


def jaccard_matrix(waveforms: dict[str, object]):
    return metric_matrix(waveforms, "jaccard", None)


def build_group_waveforms(
    records: dict[tuple[str, str, int, str], Record],
    task: str,
    label: int | None,
    subjects: set[str] | None,
    clip: float,
    top_k: int,
):
    import numpy as np

    records_for_task = [
        record
        for key, record in records.items()
        if record.model == REFERENCE_MODEL
        and record.task == task
        and (subjects is None or record.subject in subjects)
    ]
    sum_maps: dict[tuple[str, str], object] = {}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    matched_rows = []

    for reference_record in records_for_task:
        model_records = {
            model: records.get((reference_record.condition, task, reference_record.cv, model))
            for _, model, _ in TARGETS
        }
        if any(record is None for record in model_records.values()):
            continue
        prediction_maps = {
            model: prediction_map(record.predictions_path)
            for model, record in model_records.items()
        }
        reference_rows = prediction_maps[REFERENCE_MODEL]
        common_keys = set(reference_rows)
        for model in model_records:
            common_keys &= set(prediction_maps[model])
        if not common_keys:
            continue
        igs_by_model = {model: load_igs(record.ig_path) for model, record in model_records.items()}
        for key in sorted(common_keys):
            reference_row = reference_rows[key]
            if reference_row["correct"].lower() != "true":
                continue
            if label is not None and int(reference_row["label"]) != label:
                continue
            normalized = {}
            for display_name, model, _ in TARGETS:
                row = prediction_maps[model][key]
                normalized[display_name] = normalize_trial(
                    igs_by_model[model][int(row["ig_index"])],
                    clip,
                )
            unique_key = unique_trial_key(reference_row)
            matched_rows.append((reference_record.subject, unique_key, normalized))
            for display_name, _, channels in TARGETS:
                if channels is not None:
                    continue
                acc_key = (reference_record.subject, display_name)
                if acc_key not in sum_maps:
                    sum_maps[acc_key] = np.zeros_like(normalized[display_name], dtype=np.float64)
                sum_maps[acc_key] += normalized[display_name]
                counts[acc_key] += 1

    top_indices: dict[tuple[str, str], object] = {}
    for acc_key, total in sum_maps.items():
        spatial = (total / counts[acc_key]).mean(axis=1)
        top_indices[acc_key] = np.argsort(spatial)[::-1][: min(top_k, len(spatial))]

    trial_waveforms: dict[str, dict[tuple[str, str, str, str], dict[str, list[object]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    n_ig_rows = 0
    for subject, unique_key, normalized in matched_rows:
        n_ig_rows += 1
        for display_name, _, channels in TARGETS:
            if channels is None:
                channels_to_use = top_indices[(subject, display_name)]
            else:
                channels_to_use = np.asarray(channels, dtype=int)
            waveform = normalized[display_name][channels_to_use].mean(axis=0)
            trial_waveforms[subject][unique_key][display_name].append(waveform)

    subject_trials: dict[str, list[dict[str, object]]] = defaultdict(list)
    for subject, by_trial in trial_waveforms.items():
        for unique_key, by_target in by_trial.items():
            if any(target not in by_target for target in TARGET_ORDER):
                continue
            trial = {
                target: zscore_waveform(np.mean(by_target[target], axis=0))
                for target in TARGET_ORDER
            }
            trial[TRIAL_LABEL_KEY] = unique_key[3]
            subject_trials[subject].append(trial)
    n_unique_trials = sum(len(trials) for trials in subject_trials.values())
    return subject_trials, n_ig_rows, n_unique_trials


def subject_matrices(subject_trials: dict[str, list[dict[str, object]]]):
    import numpy as np

    matrices = {}
    for subject, trials in subject_trials.items():
        if not trials:
            continue
            matrices[subject] = np.nanmean(np.asarray([jaccard_matrix(trial) for trial in trials]), axis=0)
    return matrices


def masks_by_subject(subject_trials):
    import numpy as np

    masks = {}
    for subject, trials in subject_trials.items():
        if not trials:
            continue
        masks[subject] = np.asarray(
            [
                [trial[target] > trial[target].mean() for target in TARGET_ORDER]
                for trial in trials
            ],
            dtype=bool,
        )
    return masks


def mask_jaccard_matrix(masks):
    import numpy as np

    matrix = np.full((len(TARGET_ORDER), len(TARGET_ORDER)), np.nan, dtype=np.float64)
    for i, j in itertools.product(range(len(TARGET_ORDER)), repeat=2):
        if i == j:
            matrix[i, j] = 1.0
            continue
        intersection = np.logical_and(masks[:, i, :], masks[:, j, :]).sum(axis=1)
        union = np.logical_or(masks[:, i, :], masks[:, j, :]).sum(axis=1)
        values = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float64),
            where=union != 0,
        )
        matrix[i, j] = values.mean()
    return matrix


def mask_jaccard_matrix_by_trial(masks):
    import numpy as np

    matrices = np.full(
        (masks.shape[0], len(TARGET_ORDER), len(TARGET_ORDER)),
        np.nan,
        dtype=np.float64,
    )
    for i, j in itertools.product(range(len(TARGET_ORDER)), repeat=2):
        if i == j:
            matrices[:, i, j] = 1.0
            continue
        intersection = np.logical_and(masks[:, i, :], masks[:, j, :]).sum(axis=1)
        union = np.logical_or(masks[:, i, :], masks[:, j, :]).sum(axis=1)
        matrices[:, i, j] = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float64),
            where=union != 0,
        )
    return matrices


def permute_masks(masks, rng):
    import numpy as np

    n_trials, n_targets, n_times = masks.shape
    permuted = np.empty_like(masks)
    for target_idx in range(n_targets):
        order = np.argsort(rng.random((n_trials, n_times)), axis=1)
        permuted[:, target_idx, :] = np.take_along_axis(
            masks[:, target_idx, :],
            order,
            axis=1,
        )
    return permuted


def mask_surrogate_matrix(masks, rng):
    import numpy as np

    permuted = permute_masks(masks, rng)
    matrix = np.full((len(TARGET_ORDER), len(TARGET_ORDER)), np.nan, dtype=np.float64)
    for i, j in itertools.product(range(len(TARGET_ORDER)), repeat=2):
        if i == j:
            matrix[i, j] = 1.0
            continue
        intersection = np.logical_and(permuted[:, i, :], masks[:, j, :]).sum(axis=1)
        union = np.logical_or(permuted[:, i, :], masks[:, j, :]).sum(axis=1)
        values = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float64),
            where=union != 0,
        )
        matrix[i, j] = values.mean()
    return matrix


def surrogate_subject_matrices(subject_trials, rng):
    import numpy as np

    matrices = {}
    for subject, trials in subject_trials.items():
        trial_matrices = []
        for trial in trials:
            matrix = np.full((len(TARGET_ORDER), len(TARGET_ORDER)), np.nan, dtype=np.float64)
            for i, left in enumerate(TARGET_ORDER):
                for j, right in enumerate(TARGET_ORDER):
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        matrix[i, j] = jaccard_index(rng.permutation(trial[left]), trial[right])
            trial_matrices.append(matrix)
        if trial_matrices:
            matrices[subject] = np.nanmean(np.asarray(trial_matrices), axis=0)
    return matrices


def trial_metric_matrices(subject_trials, metric: str, sample_rate: float | None):
    import numpy as np

    matrices = {}
    for subject, trials in subject_trials.items():
        values = [metric_matrix(trial, metric, sample_rate) for trial in trials]
        if values:
            matrices[subject] = np.asarray(values)
    return matrices


def subject_matrices_from_trials(subject_trials, metric: str, sample_rate: float | None):
    import numpy as np

    return {
        subject: np.nanmean(matrices, axis=0)
        for subject, matrices in trial_metric_matrices(subject_trials, metric, sample_rate).items()
    }


def pooled_trial_matrices(subject_trials, metric: str, sample_rate: float | None):
    import numpy as np

    matrices = list(trial_metric_matrices(subject_trials, metric, sample_rate).values())
    if not matrices:
        return None
    return np.concatenate(matrices, axis=0)


def different_label_trial_pair_matrices(
    trials,
    pool_trials,
    metric: str,
    sample_rate: float | None,
    rng,
    require_different_label: bool = True,
):
    import numpy as np

    if metric == "jaccard":
        return different_label_trial_pair_jaccard_matrices(trials, pool_trials, rng, require_different_label)
    if metric == "pearson":
        prepared = prepare_trial_pair_pearson(
            {"subject": trials},
            {"subject": pool_trials},
            require_different_label,
        )
        if "subject" not in prepared:
            return None
        return trial_pair_pearson_matrices_from_prepared(prepared["subject"], rng)

    matrices = []
    by_label = defaultdict(list)
    for pool_trial in pool_trials:
        by_label[str(pool_trial.get(TRIAL_LABEL_KEY, ""))].append(pool_trial)
    candidates_by_label = {
        label: [
            candidate
            for candidate_label, candidates in by_label.items()
            if (candidate_label != label or not require_different_label)
            for candidate in candidates
        ]
        for label in by_label
    }

    for trial in trials:
        label = str(trial.get(TRIAL_LABEL_KEY, ""))
        candidates = candidates_by_label.get(label, [])
        matrix = np.full((len(TARGET_ORDER), len(TARGET_ORDER)), np.nan, dtype=np.float64)
        for i, left in enumerate(TARGET_ORDER):
            for j, right in enumerate(TARGET_ORDER):
                if i == j:
                    matrix[i, j] = metric_diagonal(metric)
                    continue
                if not candidates or left not in trial:
                    continue
                paired_trial = candidates[int(rng.integers(0, len(candidates)))]
                matrix[i, j] = metric_index(trial[left], paired_trial[right], metric, sample_rate)
        matrices.append(matrix)

    return np.asarray(matrices, dtype=np.float64) if matrices else None


def different_label_trial_pair_jaccard_matrices(trials, pool_trials, rng, require_different_label: bool = True):
    import numpy as np

    if not trials or not pool_trials:
        return None

    trial_labels = np.asarray([str(trial.get(TRIAL_LABEL_KEY, "")) for trial in trials])
    pool_labels = np.asarray([str(trial.get(TRIAL_LABEL_KEY, "")) for trial in pool_trials])
    trial_masks = np.asarray(
        [[trial[target] > trial[target].mean() for target in TARGET_ORDER] for trial in trials],
        dtype=bool,
    )
    pool_masks = np.asarray(
        [[trial[target] > trial[target].mean() for target in TARGET_ORDER] for trial in pool_trials],
        dtype=bool,
    )

    candidates_by_label = {
        label: np.flatnonzero(pool_labels != label) if require_different_label else np.arange(len(pool_labels))
        for label in np.unique(trial_labels)
    }
    n_trials = len(trials)
    n_targets = len(TARGET_ORDER)
    matrices = np.full((n_trials, n_targets, n_targets), np.nan, dtype=np.float64)

    for i, j in itertools.product(range(n_targets), repeat=2):
        if i == j:
            matrices[:, i, j] = 1.0
            continue

        selected = np.empty(n_trials, dtype=int)
        valid = np.ones(n_trials, dtype=bool)
        for trial_idx, label in enumerate(trial_labels):
            candidates = candidates_by_label.get(label)
            if candidates is None or len(candidates) == 0:
                valid[trial_idx] = False
                selected[trial_idx] = 0
            else:
                selected[trial_idx] = candidates[int(rng.integers(0, len(candidates)))]
        if not np.any(valid):
            continue

        left = trial_masks[valid, i, :]
        right = pool_masks[selected[valid], j, :]
        intersection = np.logical_and(left, right).sum(axis=1)
        union = np.logical_or(left, right).sum(axis=1)
        values = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float64),
            where=union != 0,
        )
        matrices[valid, i, j] = values

    return matrices


def prepare_different_label_jaccard(subject_trials, pool_subject_trials, require_different_label: bool = True):
    import numpy as np

    prepared = {}
    for subject, trials in subject_trials.items():
        pool_trials = pool_subject_trials.get(subject, [])
        if not trials or not pool_trials:
            continue
        trial_labels = np.asarray([str(trial.get(TRIAL_LABEL_KEY, "")) for trial in trials])
        pool_labels = np.asarray([str(trial.get(TRIAL_LABEL_KEY, "")) for trial in pool_trials])
        trial_masks = np.asarray(
            [[trial[target] > trial[target].mean() for target in TARGET_ORDER] for trial in trials],
            dtype=bool,
        )
        pool_masks = np.asarray(
            [[trial[target] > trial[target].mean() for target in TARGET_ORDER] for trial in pool_trials],
            dtype=bool,
        )
        candidates_by_label = {
            label: np.flatnonzero(pool_labels != label) if require_different_label else np.arange(len(pool_labels))
            for label in np.unique(trial_labels)
        }
        prepared[subject] = (trial_masks, trial_labels, pool_masks, candidates_by_label)
    return prepared


def different_label_jaccard_matrices_from_prepared(prepared_subject, rng):
    import numpy as np

    trial_masks, trial_labels, pool_masks, candidates_by_label = prepared_subject
    n_trials = len(trial_masks)
    n_targets = len(TARGET_ORDER)
    matrices = np.full((n_trials, n_targets, n_targets), np.nan, dtype=np.float64)

    for i, j in itertools.product(range(n_targets), repeat=2):
        if i == j:
            matrices[:, i, j] = 1.0
            continue
        selected = np.empty(n_trials, dtype=int)
        valid = np.ones(n_trials, dtype=bool)
        for trial_idx, label in enumerate(trial_labels):
            candidates = candidates_by_label.get(label)
            if candidates is None or len(candidates) == 0:
                valid[trial_idx] = False
                selected[trial_idx] = 0
            else:
                selected[trial_idx] = candidates[int(rng.integers(0, len(candidates)))]
        if not np.any(valid):
            continue
        left = trial_masks[valid, i, :]
        right = pool_masks[selected[valid], j, :]
        intersection = np.logical_and(left, right).sum(axis=1)
        union = np.logical_or(left, right).sum(axis=1)
        matrices[valid, i, j] = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float64),
            where=union != 0,
        )
    return matrices


def different_label_subject_surrogate_jaccard_from_prepared(prepared, rng):
    import numpy as np

    matrices = [
        np.nanmean(different_label_jaccard_matrices_from_prepared(subject_prepared, rng), axis=0)
        for subject_prepared in prepared.values()
    ]
    if not matrices:
        return None
    return np.nanmean(np.asarray(matrices), axis=0)


def different_label_pooled_surrogate_jaccard_from_prepared(prepared, rng):
    import numpy as np

    matrices = [
        different_label_jaccard_matrices_from_prepared(subject_prepared, rng)
        for subject_prepared in prepared.values()
    ]
    if not matrices:
        return None
    return np.nanmean(np.concatenate(matrices, axis=0), axis=0)


def normalize_for_pearson(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean()
    norm = np.linalg.norm(values)
    if norm == 0:
        return np.zeros_like(values, dtype=np.float64)
    return values / norm


def prepare_trial_pair_pearson(subject_trials, pool_subject_trials, require_different_label: bool = True):
    import numpy as np

    prepared = {}
    for subject, trials in subject_trials.items():
        pool_trials = pool_subject_trials.get(subject, [])
        if not trials or not pool_trials:
            continue
        trial_labels = np.asarray([str(trial.get(TRIAL_LABEL_KEY, "")) for trial in trials])
        pool_labels = np.asarray([str(trial.get(TRIAL_LABEL_KEY, "")) for trial in pool_trials])
        trial_values = np.asarray(
            [[normalize_for_pearson(trial[target]) for target in TARGET_ORDER] for trial in trials],
            dtype=np.float64,
        )
        pool_values = np.asarray(
            [[normalize_for_pearson(trial[target]) for target in TARGET_ORDER] for trial in pool_trials],
            dtype=np.float64,
        )
        candidates_by_label = {
            label: np.flatnonzero(pool_labels != label) if require_different_label else np.arange(len(pool_labels))
            for label in np.unique(trial_labels)
        }
        prepared[subject] = (trial_values, trial_labels, pool_values, candidates_by_label)
    return prepared


def trial_pair_pearson_matrices_from_prepared(prepared_subject, rng):
    import numpy as np

    trial_values, trial_labels, pool_values, candidates_by_label = prepared_subject
    n_trials = len(trial_values)
    n_targets = len(TARGET_ORDER)
    matrices = np.full((n_trials, n_targets, n_targets), np.nan, dtype=np.float64)

    for i, j in itertools.product(range(n_targets), repeat=2):
        if i == j:
            matrices[:, i, j] = 1.0
            continue
        selected = np.empty(n_trials, dtype=int)
        valid = np.ones(n_trials, dtype=bool)
        for trial_idx, label in enumerate(trial_labels):
            candidates = candidates_by_label.get(label)
            if candidates is None or len(candidates) == 0:
                valid[trial_idx] = False
                selected[trial_idx] = 0
            else:
                selected[trial_idx] = candidates[int(rng.integers(0, len(candidates)))]
        if not np.any(valid):
            continue
        left = trial_values[valid, i, :]
        right = pool_values[selected[valid], j, :]
        matrices[valid, i, j] = np.sum(left * right, axis=1)
    return matrices


def trial_pair_subject_surrogate_pearson_from_prepared(prepared, rng):
    import numpy as np

    matrices = [
        np.nanmean(trial_pair_pearson_matrices_from_prepared(subject_prepared, rng), axis=0)
        for subject_prepared in prepared.values()
    ]
    if not matrices:
        return None
    return np.nanmean(np.asarray(matrices), axis=0)


def trial_pair_pooled_surrogate_pearson_from_prepared(prepared, rng):
    import numpy as np

    matrices = [
        trial_pair_pearson_matrices_from_prepared(subject_prepared, rng)
        for subject_prepared in prepared.values()
    ]
    if not matrices:
        return None
    return np.nanmean(np.concatenate(matrices, axis=0), axis=0)


def different_label_subject_surrogate_matrix(
    subject_trials,
    pool_subject_trials,
    metric: str,
    sample_rate: float | None,
    rng,
    require_different_label: bool = True,
):
    import numpy as np

    matrices = {}
    for subject, trials in subject_trials.items():
        pool_trials = pool_subject_trials.get(subject, [])
        values = different_label_trial_pair_matrices(
            trials,
            pool_trials,
            metric,
            sample_rate,
            rng,
            require_different_label,
        )
        if values is not None and len(values):
            matrices[subject] = np.nanmean(values, axis=0)
    if not matrices:
        return None
    return np.nanmean(np.asarray(list(matrices.values())), axis=0)


def different_label_pooled_surrogate_matrix(
    subject_trials,
    pool_subject_trials,
    metric: str,
    sample_rate: float | None,
    rng,
    require_different_label: bool = True,
):
    import numpy as np

    values = []
    for subject, trials in subject_trials.items():
        matrices = different_label_trial_pair_matrices(
            trials,
            pool_subject_trials.get(subject, []),
            metric,
            sample_rate,
            rng,
            require_different_label,
        )
        if matrices is not None and len(matrices):
            values.append(matrices)
    if not values:
        return None
    return np.nanmean(np.concatenate(values, axis=0), axis=0)


def time_permutation_waveform(waveform, rng):
    return rng.permutation(waveform)


def circular_shift_waveform(waveform, rng):
    import numpy as np

    if len(waveform) == 0:
        return waveform.copy()
    return np.roll(waveform, int(rng.integers(0, len(waveform))))


def block_permutation_waveform(waveform, rng, block_size: int):
    import numpy as np

    block_size = max(1, int(block_size))
    blocks = [waveform[start : start + block_size] for start in range(0, len(waveform), block_size)]
    if len(blocks) <= 1:
        return waveform.copy()
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[idx] for idx in order])


def phase_randomized_waveform(waveform, rng):
    import numpy as np

    spectrum = np.fft.rfft(waveform)
    if len(spectrum) <= 2:
        return waveform.copy()
    randomized = spectrum.copy()
    stop = -1 if len(waveform) % 2 == 0 else None
    phase_indices = slice(1, stop)
    phases = rng.uniform(0, 2 * np.pi, size=randomized[phase_indices].shape)
    randomized[phase_indices] = np.abs(randomized[phase_indices]) * np.exp(1j * phases)
    return np.fft.irfft(randomized, n=len(waveform))


def aaft_waveform(waveform, rng):
    import numpy as np

    waveform = np.asarray(waveform)
    n = len(waveform)
    if n <= 2:
        return waveform.copy()

    sorted_waveform = np.sort(waveform)
    ranks = np.argsort(np.argsort(waveform))

    noise = rng.normal(0, np.std(waveform, ddof=1), size=n)
    gaussianized = np.sort(noise)[ranks]

    randomized = phase_randomized_waveform(gaussianized, rng)
    randomized_ranks = np.argsort(np.argsort(randomized))
    return sorted_waveform[randomized_ranks]


def iaaft_waveform(waveform, rng, tol_pc: float = 5.0, maxiter: int = 10000):
    import numpy as np

    waveform = np.asarray(waveform)
    n = len(waveform)
    if n <= 2:
        return waveform.copy()

    indices = np.arange(n)
    target_amplitude = np.abs(np.fft.fft(waveform))
    sorted_waveform = np.sort(waveform)
    previous_rank = rng.permutation(indices)
    current_rank = np.argsort(waveform)
    surrogate = waveform[previous_rank].astype(np.complex128)
    percent_unequal = 100.0
    count = 0

    while percent_unequal > tol_pc and count < maxiter:
        previous_rank = current_rank
        phase = np.angle(np.fft.fft(surrogate))
        surrogate = np.fft.ifft(target_amplitude * np.exp(1j * phase))
        current_rank = np.argsort(surrogate, kind="quicksort")
        surrogate[current_rank] = sorted_waveform.copy()
        percent_unequal = ((current_rank != previous_rank).sum() * 100.0) / n
        count += 1

    return np.real(surrogate)


def surrogate_waveform(waveform, rng, method: str, block_size: int):
    if method == "time_permutation":
        return time_permutation_waveform(waveform, rng)
    if method == "circular_shift":
        return circular_shift_waveform(waveform, rng)
    if method == "block_permutation":
        return block_permutation_waveform(waveform, rng, block_size)
    if method == "phase_randomization":
        return phase_randomized_waveform(waveform, rng)
    if method == "aaft":
        return aaft_waveform(waveform, rng)
    if method == "iaaft":
        return iaaft_waveform(waveform, rng)
    if method == "sign_flip":
        return waveform * (1 if rng.random() < 0.5 else -1)
    raise ValueError(f"Unknown surrogate_method: {method}")


def surrogate_trial(trial, rng, method: str, block_size: int):
    targets = [target for target in TARGET_ORDER if target in trial]
    if method == "target_label_permutation":
        shuffled = list(rng.permutation(targets))
        return {target: trial[source] for target, source in zip(targets, shuffled)}
    return {
        target: surrogate_waveform(waveform, rng, method, block_size)
        for target, waveform in trial.items()
        if target in TARGET_ORDER
    }


def permuted_trial(trial, rng):
    return surrogate_trial(trial, rng, "time_permutation", 16)


def surrogate_subject_matrices_from_trials(
    subject_trials,
    metric: str,
    sample_rate: float | None,
    rng,
    surrogate_method: str = "time_permutation",
    block_size: int = 16,
):
    import numpy as np

    matrices = {}
    for subject, trials in subject_trials.items():
        values = [
            metric_matrix(surrogate_trial(trial, rng, surrogate_method, block_size), metric, sample_rate)
            for trial in trials
        ]
        if values:
            matrices[subject] = np.nanmean(np.asarray(values), axis=0)
    return matrices


def surrogate_pooled_trial_matrix(
    subject_trials,
    metric: str,
    sample_rate: float | None,
    rng,
    surrogate_method: str = "time_permutation",
    block_size: int = 16,
):
    import numpy as np

    values = []
    for trials in subject_trials.values():
        values.extend(
            metric_matrix(surrogate_trial(trial, rng, surrogate_method, block_size), metric, sample_rate)
            for trial in trials
        )
    if not values:
        return None
    return np.nanmean(np.asarray(values), axis=0)


def permuted_subject_trials(subject_trials, rng, surrogate_method: str = "time_permutation", block_size: int = 16):
    return {
        subject: [surrogate_trial(trial, rng, surrogate_method, block_size) for trial in trials]
        for subject, trials in subject_trials.items()
    }


def subject_surrogate_matrix_from_permuted(permuted_trials, metric: str, sample_rate: float | None):
    import numpy as np

    subj_mats = subject_matrices_from_trials(permuted_trials, metric, sample_rate)
    if not subj_mats:
        return None
    return np.nanmean(np.asarray(list(subj_mats.values())), axis=0)


def pooled_surrogate_matrix_from_permuted(permuted_trials, metric: str, sample_rate: float | None):
    import numpy as np

    trial_matrices = pooled_trial_matrices(permuted_trials, metric, sample_rate)
    if trial_matrices is None or len(trial_matrices) == 0:
        return None
    return np.nanmean(trial_matrices, axis=0)


def bonferroni_p_values(observed, surrogate_values, alpha: float, alternative: str = "greater"):
    import numpy as np

    p_values = np.full_like(observed, np.nan, dtype=np.float64)
    corrected = np.full_like(observed, np.nan, dtype=np.float64)
    significant = np.zeros_like(observed, dtype=bool)
    if len(surrogate_values) == 0:
        return p_values, corrected, significant
    pair_indices = [(i, j) for i, j in itertools.combinations(range(len(TARGET_ORDER)), 2)]
    for i, j in pair_indices:
        null_values = surrogate_values[:, i, j]
        if alternative == "greater":
            p_value = (np.sum(null_values >= observed[i, j]) + 1) / (len(null_values) + 1)
        elif alternative == "less":
            p_value = (np.sum(null_values <= observed[i, j]) + 1) / (len(null_values) + 1)
        elif alternative == "two-sided":
            p_greater = (np.sum(null_values >= observed[i, j]) + 1) / (len(null_values) + 1)
            p_less = (np.sum(null_values <= observed[i, j]) + 1) / (len(null_values) + 1)
            p_value = min(2.0 * min(p_greater, p_less), 1.0)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        p_values[i, j] = p_values[j, i] = p_value
    n_tests = len(pair_indices)
    for i, j in pair_indices:
        value = min(float(p_values[i, j]) * n_tests, 1.0)
        corrected[i, j] = corrected[j, i] = value
        significant[i, j] = significant[j, i] = value < alpha
    return p_values, corrected, significant


def progress_step(total: int) -> int:
    return max(1, total // 10)


def report_surrogate_progress(label: str | None, current: int, total: int, step: int) -> None:
    if label is None or total == 0:
        return
    if current == 1 or current == total or current % step == 0:
        print(f"  surrogate {current}/{total}: {label}", flush=True)


_WAVEFORM_SURROGATE_WORKER = {}


def resolve_n_jobs(n_jobs: int) -> int:
    import os

    if n_jobs <= 0:
        return max(1, os.cpu_count() or 1)
    return max(1, int(n_jobs))


def _init_waveform_surrogate_worker(
    subject_trials,
    stat_level: str,
    metric: str,
    sample_rate: float | None,
    surrogate_method: str,
    surrogate_block_size: int,
):
    _WAVEFORM_SURROGATE_WORKER.clear()
    _WAVEFORM_SURROGATE_WORKER.update(
        {
            "subject_trials": subject_trials,
            "stat_level": stat_level,
            "metric": metric,
            "sample_rate": sample_rate,
            "surrogate_method": surrogate_method,
            "surrogate_block_size": surrogate_block_size,
        }
    )


def _waveform_surrogate_worker(seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    subject_trials = _WAVEFORM_SURROGATE_WORKER["subject_trials"]
    metric = _WAVEFORM_SURROGATE_WORKER["metric"]
    sample_rate = _WAVEFORM_SURROGATE_WORKER["sample_rate"]
    surrogate_method = _WAVEFORM_SURROGATE_WORKER["surrogate_method"]
    surrogate_block_size = _WAVEFORM_SURROGATE_WORKER["surrogate_block_size"]

    if _WAVEFORM_SURROGATE_WORKER["stat_level"] == "subject":
        surrogate_mats = surrogate_subject_matrices_from_trials(
            subject_trials,
            metric,
            sample_rate,
            rng,
            surrogate_method,
            surrogate_block_size,
        )
        if not surrogate_mats:
            return None
        return np.nanmean(np.asarray(list(surrogate_mats.values())), axis=0)

    return surrogate_pooled_trial_matrix(
        subject_trials,
        metric,
        sample_rate,
        rng,
        surrogate_method,
        surrogate_block_size,
    )


def parallel_waveform_surrogate_values(
    subject_trials,
    stat_level: str,
    metric: str,
    sample_rate: float | None,
    surrogate_method: str,
    surrogate_block_size: int,
    num_surrogate: int,
    rng,
    n_jobs: int,
    observed,
    progress_label: str | None = None,
):
    import multiprocessing as mp
    import numpy as np

    n_jobs = min(resolve_n_jobs(n_jobs), max(1, num_surrogate))
    if n_jobs <= 1 or num_surrogate <= 1:
        return None

    seeds = [int(seed) for seed in rng.integers(0, np.iinfo(np.uint32).max, size=num_surrogate)]
    step = progress_step(num_surrogate)
    chunksize = max(1, num_surrogate // (n_jobs * 8))
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    values = []
    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_waveform_surrogate_worker,
        initargs=(
            subject_trials,
            stat_level,
            metric,
            sample_rate,
            surrogate_method,
            surrogate_block_size,
        ),
    ) as pool:
        for completed, matrix in enumerate(
            pool.imap_unordered(_waveform_surrogate_worker, seeds, chunksize=chunksize),
            start=1,
        ):
            values.append(matrix if matrix is not None else np.full_like(observed, np.nan))
            report_surrogate_progress(progress_label, completed, num_surrogate, step)
    return values


def compute_group(
    subject_trials,
    metric: str,
    sample_rate: float | None,
    surrogate_method: str,
    surrogate_block_size: int,
    num_surrogate: int,
    alpha: float,
    rng,
    surrogate_source_trials=None,
    n_jobs: int = 1,
    progress_label: str | None = None,
):
    import numpy as np

    if progress_label is not None:
        print(f"  observed: {progress_label}", flush=True)
    subj_mats = subject_matrices_from_trials(subject_trials, metric, sample_rate)
    if not subj_mats:
        return None
    subject_values = np.asarray(list(subj_mats.values()))
    observed = np.nanmean(subject_values, axis=0)
    sem = np.nanstd(subject_values, axis=0, ddof=1) / np.sqrt(max(len(subject_values), 1))
    surrogate_values = []
    step = progress_step(num_surrogate)
    prepared_jaccard = None
    prepared_pearson = None
    require_different_label = surrogate_method == "different_label_trial_pair_shuffle"
    if surrogate_method in PAIR_SHUFFLE_METHODS and metric == "jaccard":
        prepared_jaccard = prepare_different_label_jaccard(
            subject_trials,
            surrogate_source_trials or subject_trials,
            require_different_label,
        )
    if surrogate_method in PAIR_SHUFFLE_METHODS and metric == "pearson":
        prepared_pearson = prepare_trial_pair_pearson(
            subject_trials,
            surrogate_source_trials or subject_trials,
            require_different_label,
        )
    if surrogate_method not in PAIR_SHUFFLE_METHODS:
        parallel_values = parallel_waveform_surrogate_values(
            subject_trials,
            "subject",
            metric,
            sample_rate,
            surrogate_method,
            surrogate_block_size,
            num_surrogate,
            rng,
            n_jobs,
            observed,
            progress_label,
        )
        if parallel_values is not None:
            surrogate_values = parallel_values

    for surrogate_idx in range(0 if surrogate_values else num_surrogate):
        if prepared_jaccard is not None:
            matrix = different_label_subject_surrogate_jaccard_from_prepared(prepared_jaccard, rng)
            surrogate_values.append(matrix if matrix is not None else np.full_like(observed, np.nan))
        elif prepared_pearson is not None:
            matrix = trial_pair_subject_surrogate_pearson_from_prepared(prepared_pearson, rng)
            surrogate_values.append(matrix if matrix is not None else np.full_like(observed, np.nan))
        elif surrogate_method in PAIR_SHUFFLE_METHODS:
            matrix = different_label_subject_surrogate_matrix(
                subject_trials,
                surrogate_source_trials or subject_trials,
                metric,
                sample_rate,
                rng,
                require_different_label,
            )
            surrogate_values.append(matrix if matrix is not None else np.full_like(observed, np.nan))
        else:
            surrogate_mats = surrogate_subject_matrices_from_trials(
                subject_trials, metric, sample_rate, rng, surrogate_method, surrogate_block_size
            )
            surrogate_values.append(np.nanmean(np.asarray(list(surrogate_mats.values())), axis=0))
        report_surrogate_progress(progress_label, surrogate_idx + 1, num_surrogate, step)
    alternative = metric_alternative(metric)
    p_values, corrected, significant = bonferroni_p_values(
        observed,
        np.asarray(surrogate_values).reshape((-1, len(TARGET_ORDER), len(TARGET_ORDER))),
        alpha,
        alternative,
    )
    return {
        "observed": observed,
        "sem": sem,
        "p_values": p_values,
        "p_values_corrected": corrected,
        "significant": significant,
        "n_subjects": len(subject_values),
        "n_analysis_units": len(subject_values),
        "stat_level": "subject",
        "metric": metric,
        "surrogate_method": surrogate_method,
        "alternative": alternative,
    }


def compute_unique_trial_group(
    subject_trials,
    metric: str,
    sample_rate: float | None,
    surrogate_method: str,
    surrogate_block_size: int,
    num_surrogate: int,
    alpha: float,
    rng,
    surrogate_source_trials=None,
    n_jobs: int = 1,
    progress_label: str | None = None,
):
    import numpy as np

    if progress_label is not None:
        print(f"  observed: {progress_label}", flush=True)
    trial_matrices = pooled_trial_matrices(subject_trials, metric, sample_rate)
    if trial_matrices is None or len(trial_matrices) == 0:
        return None
    observed = np.nanmean(trial_matrices, axis=0)
    sem = np.nanstd(trial_matrices, axis=0, ddof=1) / np.sqrt(max(len(trial_matrices), 1))
    surrogate_values = []
    step = progress_step(num_surrogate)
    prepared_jaccard = None
    prepared_pearson = None
    require_different_label = surrogate_method == "different_label_trial_pair_shuffle"
    if surrogate_method in PAIR_SHUFFLE_METHODS and metric == "jaccard":
        prepared_jaccard = prepare_different_label_jaccard(
            subject_trials,
            surrogate_source_trials or subject_trials,
            require_different_label,
        )
    if surrogate_method in PAIR_SHUFFLE_METHODS and metric == "pearson":
        prepared_pearson = prepare_trial_pair_pearson(
            subject_trials,
            surrogate_source_trials or subject_trials,
            require_different_label,
        )
    if surrogate_method not in PAIR_SHUFFLE_METHODS:
        parallel_values = parallel_waveform_surrogate_values(
            subject_trials,
            "unique_trial",
            metric,
            sample_rate,
            surrogate_method,
            surrogate_block_size,
            num_surrogate,
            rng,
            n_jobs,
            observed,
            progress_label,
        )
        if parallel_values is not None:
            surrogate_values = parallel_values

    for surrogate_idx in range(0 if surrogate_values else num_surrogate):
        if prepared_jaccard is not None:
            matrix = different_label_pooled_surrogate_jaccard_from_prepared(prepared_jaccard, rng)
            surrogate_values.append(matrix if matrix is not None else np.full_like(observed, np.nan))
        elif prepared_pearson is not None:
            matrix = trial_pair_pooled_surrogate_pearson_from_prepared(prepared_pearson, rng)
            surrogate_values.append(matrix if matrix is not None else np.full_like(observed, np.nan))
        elif surrogate_method in PAIR_SHUFFLE_METHODS:
            matrix = different_label_pooled_surrogate_matrix(
                subject_trials,
                surrogate_source_trials or subject_trials,
                metric,
                sample_rate,
                rng,
                require_different_label,
            )
            surrogate_values.append(matrix if matrix is not None else np.full_like(observed, np.nan))
        else:
            surrogate_values.append(
                surrogate_pooled_trial_matrix(
                    subject_trials, metric, sample_rate, rng, surrogate_method, surrogate_block_size
                )
            )
        report_surrogate_progress(progress_label, surrogate_idx + 1, num_surrogate, step)
    alternative = metric_alternative(metric)
    p_values, corrected, significant = bonferroni_p_values(
        observed,
        np.asarray(surrogate_values).reshape((-1, len(TARGET_ORDER), len(TARGET_ORDER))),
        alpha,
        alternative,
    )
    return {
        "observed": observed,
        "sem": sem,
        "p_values": p_values,
        "p_values_corrected": corrected,
        "significant": significant,
        "n_subjects": len(subject_trials),
        "n_analysis_units": len(trial_matrices),
        "stat_level": "unique_trial",
        "metric": metric,
        "surrogate_method": surrogate_method,
        "alternative": alternative,
    }


def compute_result(
    subject_trials,
    stat_level: str,
    metric: str,
    sample_rate: float | None,
    surrogate_method: str,
    surrogate_block_size: int,
    num_surrogate: int,
    alpha: float,
    rng,
    surrogate_source_trials=None,
    n_jobs: int = 1,
    progress_label: str | None = None,
):
    if stat_level == "subject":
        return compute_group(
            subject_trials,
            metric,
            sample_rate,
            surrogate_method,
            surrogate_block_size,
            num_surrogate,
            alpha,
            rng,
            surrogate_source_trials,
            n_jobs,
            progress_label,
        )
    if stat_level == "unique_trial":
        return compute_unique_trial_group(
            subject_trials,
            metric,
            sample_rate,
            surrogate_method,
            surrogate_block_size,
            num_surrogate,
            alpha,
            rng,
            surrogate_source_trials,
            n_jobs,
            progress_label,
        )
    raise ValueError(f"Unknown stat_level: {stat_level}")


def compute_results_shared_surrogates(
    subject_trials,
    stat_level: str,
    metrics: list[str],
    sample_rate: float | None,
    surrogate_method: str,
    surrogate_block_size: int,
    num_surrogate: int,
    alpha: float,
    rng,
    surrogate_source_trials=None,
    progress_label: str | None = None,
):
    import numpy as np

    if stat_level not in {"subject", "unique_trial"}:
        raise ValueError(f"Unknown stat_level: {stat_level}")

    results = {}
    surrogate_values = {
        metric: np.empty((num_surrogate, len(TARGET_ORDER), len(TARGET_ORDER)), dtype=np.float64)
        for metric in metrics
    }

    for metric in metrics:
        if progress_label is not None:
            print(f"  observed: metric={metric}, {progress_label}", flush=True)
        if stat_level == "subject":
            subj_mats = subject_matrices_from_trials(subject_trials, metric, sample_rate)
            if not subj_mats:
                results[metric] = None
                continue
            values = np.asarray(list(subj_mats.values()))
            n_subjects = len(values)
            n_analysis_units = len(values)
        else:
            values = pooled_trial_matrices(subject_trials, metric, sample_rate)
            if values is None or len(values) == 0:
                results[metric] = None
                continue
            n_subjects = len(subject_trials)
            n_analysis_units = len(values)

        results[metric] = {
            "observed": np.nanmean(values, axis=0),
            "sem": np.nanstd(values, axis=0, ddof=1) / np.sqrt(max(len(values), 1)),
            "n_subjects": n_subjects,
            "n_analysis_units": n_analysis_units,
            "stat_level": stat_level,
            "metric": metric,
            "surrogate_method": surrogate_method,
            "alternative": metric_alternative(metric),
        }

    prepared_jaccard = None
    prepared_pearson = None
    require_different_label = surrogate_method == "different_label_trial_pair_shuffle"
    if surrogate_method in PAIR_SHUFFLE_METHODS and "jaccard" in metrics:
        prepared_jaccard = prepare_different_label_jaccard(
            subject_trials,
            surrogate_source_trials or subject_trials,
            require_different_label,
        )
    if surrogate_method in PAIR_SHUFFLE_METHODS and "pearson" in metrics:
        prepared_pearson = prepare_trial_pair_pearson(
            subject_trials,
            surrogate_source_trials or subject_trials,
            require_different_label,
        )
    step = progress_step(num_surrogate)
    for surrogate_idx in range(num_surrogate):
        if surrogate_method in PAIR_SHUFFLE_METHODS:
            permuted_trials = None
        else:
            permuted_trials = permuted_subject_trials(subject_trials, rng, surrogate_method, surrogate_block_size)
        for metric in metrics:
            if results.get(metric) is None:
                surrogate_values[metric][surrogate_idx] = np.nan
                continue
            if prepared_jaccard is not None and metric == "jaccard":
                if stat_level == "subject":
                    matrix = different_label_subject_surrogate_jaccard_from_prepared(prepared_jaccard, rng)
                else:
                    matrix = different_label_pooled_surrogate_jaccard_from_prepared(prepared_jaccard, rng)
            elif prepared_pearson is not None and metric == "pearson":
                if stat_level == "subject":
                    matrix = trial_pair_subject_surrogate_pearson_from_prepared(prepared_pearson, rng)
                else:
                    matrix = trial_pair_pooled_surrogate_pearson_from_prepared(prepared_pearson, rng)
            elif surrogate_method in PAIR_SHUFFLE_METHODS:
                if stat_level == "subject":
                    matrix = different_label_subject_surrogate_matrix(
                        subject_trials,
                        surrogate_source_trials or subject_trials,
                        metric,
                        sample_rate,
                        rng,
                        require_different_label,
                    )
                else:
                    matrix = different_label_pooled_surrogate_matrix(
                        subject_trials,
                        surrogate_source_trials or subject_trials,
                        metric,
                        sample_rate,
                        rng,
                        require_different_label,
                    )
            elif stat_level == "subject":
                matrix = subject_surrogate_matrix_from_permuted(permuted_trials, metric, sample_rate)
            else:
                matrix = pooled_surrogate_matrix_from_permuted(permuted_trials, metric, sample_rate)
            surrogate_values[metric][surrogate_idx] = matrix if matrix is not None else np.nan
        report_surrogate_progress(progress_label, surrogate_idx + 1, num_surrogate, step)

    for metric, result in list(results.items()):
        if result is None:
            continue
        p_values, corrected, significant = bonferroni_p_values(
            result["observed"],
            surrogate_values[metric],
            alpha,
            result["alternative"],
        )
        result["p_values"] = p_values
        result["p_values_corrected"] = corrected
        result["significant"] = significant
    return results


def safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "-")


def output_paths(save_dir: Path, level: str, name: str) -> tuple[Path, Path]:
    fig_path = save_dir / level / "figures" / f"{name}.png"
    table_path = save_dir / level / "tables" / f"{name}.csv"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    return fig_path, table_path


def figure_path_for_alternative(path: Path, alternative: str) -> Path:
    if alternative in {"greater", "less"} and not path.stem.endswith("_oneside"):
        return path.with_name(f"{path.stem}_oneside{path.suffix}")
    return path


def write_tables(path: Path, result, n_ig_rows: int, n_unique_trials: int) -> None:
    mean_name, sem_name, _ = metric_value_names(result.get("metric", "jaccard"))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", *TARGET_ORDER])
        writer.writeheader()
        for target, values in zip(TARGET_ORDER, result["observed"]):
            row = {"target": target}
            for other, value in zip(TARGET_ORDER, values):
                row[other] = float(value)
            writer.writerow(row)
    long_path = path.with_name(path.stem + "_long.csv")
    with long_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_1",
                "target_2",
                "metric",
                mean_name,
                sem_name,
                "surrogate_method",
                "n_ig_rows",
                "n_unique_trials",
                "n_subjects",
                "n_analysis_units",
                "stat_level",
                "alternative",
                "p_value",
                "p_value_corrected",
                "significant",
            ],
        )
        writer.writeheader()
        for i, target_1 in enumerate(TARGET_ORDER):
            for j, target_2 in enumerate(TARGET_ORDER):
                writer.writerow(
                    {
                        "target_1": target_1,
                        "target_2": target_2,
                        "metric": result.get("metric", "jaccard"),
                        mean_name: float(result["observed"][i, j]),
                        sem_name: float(result["sem"][i, j]) if result["sem"][i, j] == result["sem"][i, j] else "",
                        "surrogate_method": result.get("surrogate_method", "time_permutation"),
                        "n_ig_rows": n_ig_rows,
                        "n_unique_trials": n_unique_trials,
                        "n_subjects": result["n_subjects"],
                        "n_analysis_units": result["n_analysis_units"],
                        "stat_level": result["stat_level"],
                        "alternative": result.get("alternative", "greater"),
                        "p_value": float(result["p_values"][i, j]) if result["p_values"][i, j] == result["p_values"][i, j] else "",
                        "p_value_corrected": float(result["p_values_corrected"][i, j]) if result["p_values_corrected"][i, j] == result["p_values_corrected"][i, j] else "",
                        "significant": bool(result["significant"][i, j]),
                    }
                )


def significance_marker(p_value: float) -> str:
    import math

    if not math.isfinite(p_value) or p_value >= 0.05:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def plot_matrix(ax, matrix, metric: str = "jaccard", significant=None):
    import numpy as np
    from matplotlib import pyplot as plt

    values = matrix.copy()
    np.fill_diagonal(values, np.nan)
    if metric == "peak_latency":
        cmap_name = "viridis_r"
    elif metric == "pearson":
        cmap_name = "jet"
    else:
        cmap_name = None
    cmap = plt.get_cmap("jet" if cmap_name is None else cmap_name).copy()
    cmap.set_bad("white")
    if metric == "pearson":
        vmin, vmax = -1, 1
    elif metric == "peak_latency":
        finite = values[np.isfinite(values)]
        vmin, vmax = 0, float(finite.max()) if len(finite) else 1
    else:
        vmin, vmax = 0, 1
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(TARGET_ORDER)))
    ax.set_yticks(np.arange(len(TARGET_ORDER)))
    return im


def decorate_axis(ax, show_x: bool, show_y: bool) -> None:
    labels = [TARGET_SHORT_NAMES[target] for target in TARGET_ORDER]
    if show_x:
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        for tick, target in zip(ax.get_xticklabels(), TARGET_ORDER):
            tick.set_color(TARGET_LABEL_COLORS[target])
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_yticklabels(labels, fontsize=6)
        for tick, target in zip(ax.get_yticklabels(), TARGET_ORDER):
            tick.set_color(TARGET_LABEL_COLORS[target])
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)


def plot_single(path: Path, result, title: str) -> None:
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    metric = result.get("metric", "jaccard")
    im = plot_matrix(ax, result["observed"], metric, result["significant"])
    decorate_axis(ax, True, True)
    ax.set_title(title, fontsize=9)
    for i, j in zip(*result["significant"].nonzero()):
        if i != j:
            marker = significance_marker(result["p_values_corrected"][i, j])
            if marker:
                ax.text(j, i, marker, ha="center", va="center", color="k", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric_value_names(metric)[2])
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_combined(path: Path, results_by_key, tasks: list[str], colors: list[str]) -> None:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    display_task = {"overt": "overt", "minimally_overt": "min overt", "covert": "covert"}
    fig, axes = plt.subplots(len(tasks), len(colors) + 1, figsize=(8.7, 4.7), squeeze=False)
    last_im = None
    for row_idx, task in enumerate(tasks):
        for col_idx, color in enumerate(colors):
            ax = axes[row_idx, col_idx]
            result = results_by_key.get((task, str(col_idx)))
            if result is not None:
                last_im = plot_matrix(ax, result["observed"], result.get("metric", "jaccard"), result["significant"])
                for i, j in zip(*result["significant"].nonzero()):
                    if i != j:
                        marker = significance_marker(result["p_values_corrected"][i, j])
                        if marker:
                            ax.text(j, i, marker, ha="center", va="center", color="k", fontsize=6, fontweight="bold")
            decorate_axis(ax, row_idx == len(tasks) - 1, col_idx == 0)
            if col_idx == 0:
                ax.set_ylabel(display_task.get(task, task), fontsize=8)
            if row_idx == 0:
                ax.set_title(color, fontsize=8)
        ax = axes[row_idx, -1]
        result = results_by_key.get((task, "avg"))
        if result is not None:
            last_im = plot_matrix(ax, result["observed"], result.get("metric", "jaccard"), result["significant"])
            for i, j in zip(*result["significant"].nonzero()):
                if i != j:
                    marker = significance_marker(result["p_values_corrected"][i, j])
                    if marker:
                        ax.text(j, i, marker, ha="center", va="center", color="k", fontsize=6, fontweight="bold")
        decorate_axis(ax, row_idx == len(tasks) - 1, False)
        if row_idx == 0:
            ax.set_title("avg.", fontsize=8)
    fig.subplots_adjust(wspace=0.12, hspace=0.12, right=0.91, bottom=0.16)
    cbar_ax = fig.add_axes([0.93, 0.18, 0.018, 0.62])
    if last_im is None:
        cmap = plt.get_cmap("jet").copy()
        cmap.set_bad("white")
        last_im = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap)
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    metric = next((result.get("metric", "jaccard") for result in results_by_key.values() if result is not None), "jaccard")
    cbar.set_label(metric_value_names(metric)[2], fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def write_summary(save_dir: Path, rows: list[dict[str, object]]) -> None:
    path = save_dir / "summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "surrogate_method",
                "level",
                "condition",
                "label",
                "color",
                "n_ig_rows",
                "n_unique_trials",
                "n_subjects",
                "n_analysis_units",
                "stat_level",
                "alternative",
                "figure",
                "table",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    import numpy as np

    rng = np.random.default_rng(args.seed)
    records = discover_records(args.ig_dir)
    subject_set = set(args.subjects) if args.subjects is not None else None
    summary_rows = []

    print(f"planned metrics: {', '.join(args.metrics)}", flush=True)
    print(f"stat_level: {args.stat_level}", flush=True)
    print(f"num_surrogate: {args.num_surrogate}", flush=True)
    print(f"surrogate_method: {args.surrogate_method}", flush=True)
    print(f"n_jobs: {resolve_n_jobs(args.n_jobs)}", flush=True)
    if args.surrogate_method == "block_permutation":
        print(f"surrogate_block_size: {args.surrogate_block_size}", flush=True)
    print("building input waveforms...", flush=True)

    input_groups = []
    for task in args.tasks:
        all_label_subject_trials, all_label_n_ig_rows, all_label_n_unique_trials = build_group_waveforms(
            records,
            task,
            None,
            subject_set,
            args.clip,
            args.top_k,
        )
        for label, color in enumerate(args.colors):
            subject_trials, n_ig_rows, n_unique_trials = build_group_waveforms(
                records,
                task,
                label,
                subject_set,
                args.clip,
                args.top_k,
            )
            input_groups.append(
                (
                    (task, str(label)),
                    "condition_label",
                    task,
                    label,
                    color,
                    f"{safe_name(task)}_label{label}_{safe_name(color)}",
                    subject_trials,
                    all_label_subject_trials,
                    n_ig_rows,
                    n_unique_trials,
                )
            )
        input_groups.append(
            (
                (task, "avg"),
                "condition",
                task,
                "",
                "avg",
                f"{safe_name(task)}_avg",
                all_label_subject_trials,
                all_label_subject_trials,
                all_label_n_ig_rows,
                all_label_n_unique_trials,
            )
        )

    total_panels = len(input_groups)
    print(f"input panels: {total_panels}", flush=True)

    results_by_metric = {metric: {} for metric in args.metrics}
    for metric in args.metrics:
        print(f"[metric] planned {metric} -> {metric_output_dir(args.save_dir, metric)}", flush=True)

    if len(args.metrics) > 1:
        for panel_idx, (
            result_key,
            level,
            task,
            label,
            color,
            name,
            subject_trials,
            surrogate_source_trials,
            n_ig_rows,
            n_unique_trials,
        ) in enumerate(input_groups, start=1):
            progress_label = (
                f"metrics={','.join(args.metrics)}, panel={panel_idx}/{total_panels}, "
                f"condition={task}, color={color}, stat_level={args.stat_level}"
            )
            print(f"[panel] {progress_label}", flush=True)
            panel_results = compute_results_shared_surrogates(
                subject_trials,
                args.stat_level,
                args.metrics,
                args.sample_rate,
                args.surrogate_method,
                args.surrogate_block_size,
                args.num_surrogate,
                args.alpha,
                rng,
                surrogate_source_trials,
                progress_label,
            )
            for metric, result in panel_results.items():
                if result is None:
                    print(f"  skipped: no valid data for metric={metric}, {progress_label}", flush=True)
                    continue
                results_by_metric[metric][result_key] = result
                metric_save_dir = metric_output_dir(args.save_dir, metric)
                fig_path, table_path = output_paths(metric_save_dir, level, name)
                fig_path = figure_path_for_alternative(fig_path, result["alternative"])
                write_tables(table_path, result, n_ig_rows, n_unique_trials)
                plot_single(fig_path, result, f"{task} {color}")
                print(f"  saved panel: metric={metric} figure={fig_path} table={table_path}", flush=True)
                summary_rows.append(
                    {
                        "metric": metric,
                        "surrogate_method": result.get("surrogate_method", args.surrogate_method),
                        "level": level,
                        "condition": task,
                        "label": label,
                        "color": color,
                        "n_ig_rows": n_ig_rows,
                        "n_unique_trials": n_unique_trials,
                        "n_subjects": result["n_subjects"],
                        "n_analysis_units": result["n_analysis_units"],
                        "stat_level": result["stat_level"],
                        "alternative": result["alternative"],
                        "figure": str(fig_path),
                        "table": str(table_path),
                    }
                )
    else:
        metric = args.metrics[0]
        metric_save_dir = metric_output_dir(args.save_dir, metric)
        print(f"[metric] start {metric} -> {metric_save_dir}", flush=True)
        for panel_idx, (
            result_key,
            level,
            task,
            label,
            color,
            name,
            subject_trials,
            surrogate_source_trials,
            n_ig_rows,
            n_unique_trials,
        ) in enumerate(input_groups, start=1):
            progress_label = (
                f"metric={metric}, panel={panel_idx}/{total_panels}, "
                f"condition={task}, color={color}, stat_level={args.stat_level}"
            )
            print(f"[panel] {progress_label}", flush=True)
            result = compute_result(
                subject_trials,
                args.stat_level,
                metric,
                args.sample_rate,
                args.surrogate_method,
                args.surrogate_block_size,
                args.num_surrogate,
                args.alpha,
                rng,
                surrogate_source_trials,
                args.n_jobs,
                progress_label,
            )
            if result is None:
                print(f"  skipped: no valid data for {progress_label}", flush=True)
                continue
            results_by_metric[metric][result_key] = result
            fig_path, table_path = output_paths(metric_save_dir, level, name)
            fig_path = figure_path_for_alternative(fig_path, result["alternative"])
            write_tables(table_path, result, n_ig_rows, n_unique_trials)
            plot_single(fig_path, result, f"{task} {color}")
            print(f"  saved panel: figure={fig_path} table={table_path}", flush=True)
            summary_rows.append(
                {
                    "metric": metric,
                    "surrogate_method": result.get("surrogate_method", args.surrogate_method),
                    "level": level,
                    "condition": task,
                    "label": label,
                    "color": color,
                    "n_ig_rows": n_ig_rows,
                    "n_unique_trials": n_unique_trials,
                    "n_subjects": result["n_subjects"],
                    "n_analysis_units": result["n_analysis_units"],
                    "stat_level": result["stat_level"],
                    "alternative": result["alternative"],
                    "figure": str(fig_path),
                    "table": str(table_path),
                }
            )
        print(f"[metric] done {metric}", flush=True)

    for metric, results_by_key in results_by_metric.items():
        metric_save_dir = metric_output_dir(args.save_dir, metric)
        combined_path = metric_save_dir / "combined" / "figures" / "condition_by_label_grid.png"
        alternative = next(
            (result.get("alternative", "") for result in results_by_key.values() if result is not None),
            "",
        )
        plot_combined(
            figure_path_for_alternative(combined_path, alternative),
            results_by_key,
            args.tasks,
            args.colors,
        )
        print(f"[metric] done {metric}", flush=True)

    write_summary(args.save_dir, summary_rows)
    print(f"matrices: {len(summary_rows)}")
    print(f"saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

