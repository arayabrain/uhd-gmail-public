#!/usr/bin/env python3
"""Evaluate rotating-CV models on independent pseudo-online data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from scripts.leave_test._conditions import behavior_from_run_key, subject_from_run_key
from uhd_eeg.datasets.DatasetUHD import EEGDataset, EMGDataset
from uhd_eeg.trainers.trainer_within_offline_split import (
    build_model,
    combine_predictions,
    prepare_inputs,
)


SCRIPT_NAME = "pseudo_online_test"
DEFAULT_OUTPUT_DIR = Path("outputs") / SCRIPT_NAME
DEFAULT_TEST_HISTORIES = [
    Path("outputs/rotating/baseline/history_color_rotating_test_fold_test.csv"),
    Path("outputs/rotating/baseline/history_color_rotating_test_fold_test_cBraMod.csv"),
]
LEGACY_CONDITION_BEHAVIORS = {
    "sub-1_task-minimallyovert_acq-calibration": "minimally_overt",
    "sub-1_task-overt_acq-calibration": "overt",
    "sub-1_task-covert_acq-calibration": "covert",
    "sub-2_task-minimallyovert_acq-calibration": "minimally_overt",
    "sub-2_task-overt_acq-calibration": "overt",
    "sub-2_task-covert_acq-calibration": "covert",
    "sub-3_task-overt_acq-calibration": "overt",
    "sub-3_task-minimallyovert_acq-calibration": "minimally_overt",
    "sub-3_task-covert_acq-calibration": "covert",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pseudo-online evaluation for rotating-CV model directories. "
            "Online data are supplied by a manifest and are never used for model selection."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help=(
            "CSV with columns: condition,csv_dir,npy_dir,csv_header. "
            "Optional columns: online_label,model,run_dir."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["EEGNet", "RNN", "CovTanSVM", "cBraMod"],
        help="Models to evaluate when the manifest row does not specify model.",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=4,
        help="Number of rotating-CV models to ensemble.",
    )
    parser.add_argument(
        "--method",
        default="zscore_mean",
        help="Ensemble method passed to combine_predictions.",
    )
    parser.add_argument(
        "--rank-metric",
        default="balanced_acc_test",
        choices=["balanced_acc_test", "acc_test", "balanced_acc_val", "acc_val", "cv_order"],
        help="Metric used to select the top n rotating-CV models.",
    )
    parser.add_argument(
        "--test-history",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Rotating test history CSV(s). If omitted, the standard rotating test logs "
            "are searched. Used for balanced_acc_test/acc_test ranking."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-condition pseudo-online result files are written.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU id in loaded configs.")
    parser.add_argument(
        "--jitter",
        type=float,
        default=None,
        help="Override inference jitter in seconds. Omit to use the training run config.",
    )
    parser.add_argument(
        "--jitter-mode",
        choices=["random", "fixed"],
        default=None,
        help="Override inference jitter mode. Omit to use the training run config.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=50,
        help="Use the first N valid labels only. Set <=0 to use all valid labels.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing model/data combinations instead of failing.",
    )
    parser.add_argument(
        "--emg-input-modes",
        nargs="+",
        default=["real"],
        choices=["real", "pseudo_eeg_mean", "zero"],
        help=(
            "Input variants for EEG+EMG models: real keeps EMG channels, "
            "pseudo_eeg_mean replaces EMG with the mean of selected EEG channels, "
            "and zero sets EMG channels to zero."
        ),
    )
    parser.add_argument(
        "--pseudo-emg-eeg-channels",
        nargs="+",
        type=int,
        default=[64, 65, 67],
        help="0-based EEG channels averaged to make pseudo EMG for pseudo_eeg_mean.",
    )
    parser.add_argument(
        "--channel-decimation-channels",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Evaluate EEGNet channel-decimation runs for these channel counts. "
            "Run directories are resolved from --test-history; 128 uses EEGNet, "
            "smaller counts use EEGNet_with_mask."
        ),
    )
    parser.add_argument(
        "--disable-dataset-cache",
        action="store_true",
        help="Do not read or write the preprocessed dataset cache.",
    )
    return parser.parse_args()


def slug(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "_")
    )


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"condition", "csv_dir", "npy_dir", "csv_header"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def model_filename(config_name: str, model: str, n_trial_avg: int, cv: int) -> str:
    if model == "CovTanSVM":
        return f"CovTanSVM_{config_name}_N{n_trial_avg}_cv{cv}.dill"
    return f"model_weight_{config_name}_N{n_trial_avg}_cv{cv}.pth"


def load_config(run_dir: Path):
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing hydra config: {config_path}")
    args = OmegaConf.load(config_path)
    OmegaConf.set_struct(args, False)
    args.use_hydra_savedir = False
    if "gmail" not in args:
        if "leave_set" in args and "leave_sets" in args and args.leave_set in args.leave_sets:
            args.gmail = OmegaConf.create(
                {
                    "behavior": args.leave_sets[args.leave_set].behavior,
                    "csv_dir": "",
                    "npy_dir": "",
                    "csv_header": "",
                }
            )
        elif "parallel_sets" not in args or args.parallel_sets not in args:
            raise KeyError(
                f"Cannot reconstruct args.gmail from {config_path}; "
                "missing gmail and parallel_sets entry."
            )
        else:
            args.gmail = OmegaConf.create(OmegaConf.to_container(args[args.parallel_sets], resolve=True))
    if "behavior" not in args.gmail:
        _, behavior = condition_subject_behavior(str(args.parallel_sets))
        args.gmail.behavior = behavior
    return args


def discover_run_dir(
    condition: str,
    model: str,
    prefer_eeg_emg: bool = False,
    expected_decode_from: str | None = None,
) -> Path | None:
    if prefer_eeg_emg:
        expected_config_names = {
            f"config_color_rotating_test_fold_eeg_emg_{condition}_{model}",
        }
    else:
        expected_config_names = {
            f"config_color_rotating_test_fold_{condition}_{model}",
            f"config_color_rotating_test_fold_eeg_emg_{condition}_{model}",
        }
    matches: list[Path] = []
    for config_path in Path("outputs").glob("*/*/.hydra/config.yaml"):
        try:
            args = OmegaConf.load(config_path)
        except Exception:
            continue
        if (
            str(args.get("config_name", "")) in expected_config_names
            and str(args.get("parallel_sets", "")) == condition
            and str(args.get("model_name", "")) == model
        ):
            if expected_decode_from is not None and str(args.get("decode_from", "")).lower() != expected_decode_from:
                continue
            matches.append(config_path.parent.parent)
    return sorted(matches)[-1] if matches else None


def load_histories(paths: list[Path] | None) -> pd.DataFrame:
    if paths is None:
        paths = DEFAULT_TEST_HISTORIES
    frames = []
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path)
            frame["source_history"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def condition_subject_behavior(condition: str) -> tuple[str, str]:
    if "_run-" in condition:
        return subject_from_run_key(condition), behavior_from_run_key(condition)
    if condition in LEGACY_CONDITION_BEHAVIORS:
        subject = condition.split("_", 1)[0]
        return subject, LEGACY_CONDITION_BEHAVIORS[condition]
    if condition.startswith("sub-") and "_task-" in condition:
        subject = condition.split("_", 1)[0]
        task = condition.split("_task-", 1)[1].split("_", 1)[0]
        behavior = "minimally_overt" if task == "minimallyovert" else task
        return subject, behavior
    raise ValueError(f"Unrecognized BIDS condition key: {condition!r}")


def discover_run_dir_from_history(
    condition: str,
    model: str,
    history: pd.DataFrame,
    prefer_eeg_emg: bool = False,
    n_channels: int | None = None,
    expected_decode_from: str | None = None,
) -> Path | None:
    if history.empty or "config" not in history.columns:
        return None
    subject, behavior = condition_subject_behavior(condition)
    rows = history[
        (history["sbj"].astype(str) == subject)
        & (history["behavior"].astype(str) == behavior)
        & (history["model_name"].astype(str) == model)
    ].copy()
    if n_channels is not None and "source_history" in rows.columns:
        if n_channels == 128:
            rows = rows[~rows["source_history"].astype(str).str.contains("channel_decimation_EEGNet")]
        else:
            channel_pattern = f"_{n_channels}ch_"
            rows = rows[rows["source_history"].astype(str).str.contains(channel_pattern, regex=False)]
    if "eval_type" in rows.columns:
        rows = rows[rows["eval_type"] == "single"]
    if rows.empty:
        return None
    configs = sorted(rows["config"].dropna().astype(str).unique())
    if prefer_eeg_emg:
        eeg_emg_configs = []
        for config in configs:
            config_path = Path(config)
            if config_path.exists():
                try:
                    args = OmegaConf.load(config_path)
                except Exception:
                    continue
                if str(args.get("decode_from", "")).lower() == "eeg_emg":
                    eeg_emg_configs.append(config)
        configs = eeg_emg_configs
    if expected_decode_from is not None:
        matching_configs = []
        for config in configs:
            config_path = Path(config)
            if config_path.exists():
                try:
                    args = OmegaConf.load(config_path)
                except Exception:
                    continue
                if str(args.get("decode_from", "")).lower() == expected_decode_from:
                    matching_configs.append(config)
        configs = matching_configs
    if not configs:
        return None
    config_path = Path(configs[-1])
    if config_path.name == "config.yaml" and config_path.parent.name == ".hydra":
        return config_path.parent.parent
    return config_path.parent


def channel_decimation_eval_rows(
    row: dict[str, str],
    channels: list[int] | None,
) -> list[dict[str, str]]:
    if channels is None:
        return [row]

    rows = []
    for n_channels in channels:
        out_row = row.copy()
        if n_channels == 128:
            out_row["model"] = "EEGNet"
            out_row["model_alias"] = "EEGNet_128ch"
        else:
            out_row["model"] = "EEGNet_with_mask"
            out_row["model_alias"] = f"EEGNet_with_mask_{n_channels}ch"
        out_row["n_channels"] = str(n_channels)
        rows.append(out_row)
    return rows


def ranked_cvs(
    args,
    run_dir: Path,
    history: pd.DataFrame,
    rank_metric: str,
) -> list[int]:
    n_splits = int(args.n_splits)
    if rank_metric == "cv_order":
        return list(range(n_splits))

    if rank_metric in {"balanced_acc_test", "acc_test"}:
        config_path = str((run_dir / ".hydra" / "config.yaml").resolve())
        if not history.empty and rank_metric in history.columns:
            rows = history.copy()
            if "config" in rows.columns:
                rows = rows[rows["config"].map(lambda x: str(Path(str(x)).resolve())) == config_path]
            if "eval_type" in rows.columns:
                single_rows = rows[rows["eval_type"] == "single"]
                if not single_rows.empty:
                    rows = single_rows
                else:
                    rows = rows[rows["eval_type"] == "transfer_adapted"]
            if not rows.empty and "CV" in rows.columns:
                rows = rows.sort_values(rank_metric, ascending=False)
                return [int(cv) for cv in rows["CV"].tolist()]

    metric_file = run_dir / f"rotating_test_fold_{rank_metric}.npy"
    if metric_file.exists():
        values = np.load(metric_file)
        if values.ndim == 2:
            values = values.max(axis=1)
        return [int(cv) for cv in np.argsort(-values)]

    raise FileNotFoundError(
        f"Could not rank CV models for {run_dir} using {rank_metric}. "
        "Pass --test-history, choose --rank-metric cv_order, or ensure metric npy exists."
    )


def existing_model_paths(args, run_dir: Path, cv_order: list[int], n_models: int) -> tuple[list[int], list[Path]]:
    selected_cvs: list[int] = []
    paths: list[Path] = []
    for cv in cv_order:
        path = run_dir / model_filename(args.config_name, args.model_name, args.n_trial_avg, cv)
        if not path.exists() and args.model_name != "CovTanSVM":
            transfer_path = (
                run_dir
                / f"adapted_model_{args.config_name}_N{args.n_trial_avg}_cv{cv}.pth"
            )
            if transfer_path.exists():
                path = transfer_path
        if path.exists():
            selected_cvs.append(cv)
            paths.append(path)
        if len(paths) == n_models:
            break
    if len(paths) < n_models:
        raise FileNotFoundError(
            f"Only found {len(paths)} model files in {run_dir}; required {n_models}."
        )
    return selected_cvs, paths


def apply_online_data(
    args,
    row: dict[str, str],
    gpu: int | None,
    jitter: float | None,
    jitter_mode: str | None,
):
    condition = row["condition"]
    _, behavior = condition_subject_behavior(condition)
    args.parallel_sets = condition
    args.gmail.behavior = behavior
    args.gmail.csv_dir = row["csv_dir"]
    args.gmail.npy_dir = row["npy_dir"]
    args.gmail.csv_header = row["csv_header"]
    if gpu is not None:
        args.gpu = gpu
    if jitter is not None:
        args.jitter = jitter
    if jitter_mode is not None:
        args.jitter_mode = jitter_mode
    return args


def parse_label(value: str, n_class: int) -> int | None:
    from uhd_eeg.analysis.online_trial_selection import parse_word_label

    return parse_word_label(value, n_class)


def read_valid_online_labels(csv_path: Path, n_class: int, max_trials: int) -> tuple[list[int], list[int], int]:
    from uhd_eeg.analysis.online_trial_selection import read_valid_online_labels as _read

    return _read(csv_path, n_class=n_class, max_trials=max_trials)


def prepare_filtered_online_data(args, row: dict[str, str], temp_root: Path, max_trials: int) -> None:
    source_csv_dir = Path(row["csv_dir"])
    source_npy_dir = Path(row["npy_dir"])
    source_csv = source_csv_dir / f"word_list{row['csv_header']}.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing word-list CSV: {source_csv}")

    original_indices, labels, n_csv_rows = read_valid_online_labels(
        source_csv,
        int(args.n_class),
        max_trials,
    )
    if not labels:
        raise ValueError(f"No valid labels in {source_csv}; valid labels are 0..{int(args.n_class) - 1}")

    filtered_csv_dir = temp_root / "csv"
    filtered_npy_dir = temp_root / "npy"
    filtered_csv_dir.mkdir(parents=True, exist_ok=True)
    filtered_npy_dir.mkdir(parents=True, exist_ok=True)

    for filtered_idx, original_idx in enumerate(original_indices):
        source_npy = source_npy_dir / f"{original_idx}.npy"
        if not source_npy.exists():
            raise FileNotFoundError(
                f"Missing npy for valid CSV row {original_idx}: {source_npy}"
            )
        os.symlink(source_npy.resolve(), filtered_npy_dir / f"{filtered_idx}.npy")

    filtered_csv = filtered_csv_dir / f"word_list{row['csv_header']}.csv"
    with filtered_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        for label in labels:
            writer.writerow([label])

    args.gmail.csv_dir = os.fspath(filtered_csv_dir)
    args.gmail.npy_dir = os.fspath(filtered_npy_dir)
    args.pseudo_online_filter = OmegaConf.create(
        {
            "source_csv": os.fspath(source_csv),
            "source_npy_dir": os.fspath(source_npy_dir),
            "n_csv_rows": n_csv_rows,
            "n_valid_used": len(labels),
            "max_trials": max_trials,
            "original_indices": original_indices,
        }
    )


class EEGEMGInputVariantDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        mode: str,
        pseudo_emg_eeg_channels: list[int],
        n_ch_eeg: int,
        n_ch_noise: int,
    ) -> None:
        self.dataset = dataset
        self.mode = mode
        self.pseudo_emg_eeg_channels = pseudo_emg_eeg_channels
        self.n_ch_eeg = n_ch_eeg
        self.n_ch_noise = n_ch_noise
        self.labels_all = getattr(dataset, "labels_all", None)
        self.window_eegnet = dataset.window_eegnet
        self.device = dataset.device

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        if self.mode == "real":
            return x, y

        x = x.clone()
        emg_slice = slice(self.n_ch_eeg, self.n_ch_eeg + self.n_ch_noise)
        if self.mode == "zero":
            x[:, emg_slice, :] = 0
        elif self.mode == "pseudo_eeg_mean":
            pseudo_emg = x[:, self.pseudo_emg_eeg_channels, :].mean(dim=1, keepdim=True)
            x[:, emg_slice, :] = pseudo_emg.repeat(1, self.n_ch_noise, 1)
        else:
            raise ValueError(f"Unsupported EMG input mode: {self.mode}")
        return x, y


def build_dataset(args):
    decode_from = str(args.decode_from).lower()
    if decode_from == "emg":
        return EMGDataset(args)
    if decode_from == "eeg_emg":
        raise NotImplementedError(
            "decode_from=eeg_emg requires EEGEMGDataset, which is not bundled in this repo."
        )
    return EEGDataset(args)


def wrap_emg_input_variant(
    args,
    dataset,
    mode: str,
    pseudo_emg_eeg_channels: list[int],
):
    if mode == "real":
        return dataset
    if str(args.decode_from).lower() != "eeg_emg":
        raise ValueError(
            f"--emg-input-modes {mode} requires an EEG+EMG model/config "
            f"(decode_from=eeg_emg), got decode_from={args.decode_from}"
        )
    invalid = [ch for ch in pseudo_emg_eeg_channels if ch < 0 or ch >= int(args.n_ch_eeg)]
    if invalid:
        raise ValueError(
            f"Pseudo-EMG EEG channels must be 0..{int(args.n_ch_eeg) - 1}; invalid={invalid}"
        )
    return EEGEMGInputVariantDataset(
        dataset,
        mode,
        pseudo_emg_eeg_channels,
        int(args.n_ch_eeg),
        int(args.n_ch_noise),
    )


def predict_scores(args, model_paths: list[Path], dataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    models = []
    for path in model_paths:
        if args.model_name == "CovTanSVM":
            with path.open("rb") as f:
                model = dill.load(f)
        else:
            model = build_model(args, dataset.window_eegnet)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
        models.append(model)

    loader = DataLoader(dataset, batch_size=1, num_workers=args.n_worekers, pin_memory=False, shuffle=False)
    all_scores = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in loader:
            sample_scores = []
            if args.model_name == "CovTanSVM":
                x = inputs.cpu().detach().numpy()[:, 0, :, :]
                for model in models:
                    sample_scores.append(model.predict_proba(x)[0])
                labels_np = labels.cpu().detach().numpy()
            else:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs, labels = prepare_inputs(args, inputs, labels)
                for model in models:
                    scores = model(inputs).cpu().detach().numpy()
                    sample_scores.append(scores.mean(axis=0) if scores.shape[0] > 1 else scores[0])
                labels_np = labels.cpu().detach().numpy()
                labels_np = labels_np[:1] if labels_np.shape[0] > 1 else labels_np
            all_scores.append(np.array(sample_scores))
            labels_all.append(int(labels_np[0]))
    return np.array(all_scores), np.array(labels_all)


def write_results(
    output_dir: Path,
    row: dict[str, str],
    args,
    run_dir: Path,
    selected_cvs: list[int],
    method: str,
    scores: np.ndarray,
    labels: np.ndarray,
    emg_input_mode: str,
    pseudo_emg_eeg_channels: list[int],
) -> Path:
    pred_scores = np.array([combine_predictions(s, method, int(args.n_class)) for s in scores])
    preds = np.argmax(pred_scores, axis=1)
    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    online_label = row.get("online_label") or "online"
    result_model_name = row.get("model_alias") or row.get("result_model_name") or str(args.model_name)
    n_channels = row.get("n_channels") or row.get("num_channels_used") or ""
    subject, _ = condition_subject_behavior(str(args.parallel_sets))
    stem = "_".join(
        [
            slug(subject),
            slug(result_model_name),
            slug(str(args.parallel_sets)),
            slug(str(online_label)),
            slug(emg_input_mode),
            f"n{len(selected_cvs)}",
            slug(method),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{stem}.csv"
    result = pd.DataFrame(
        {
            "trial": np.arange(len(labels)),
            "label": labels,
            "pred": preds,
            "correct": preds == labels,
            "source_trial": list(args.pseudo_online_filter.original_indices),
            "condition": str(args.parallel_sets),
            "subject": subject,
            "behavior": str(args.gmail.behavior),
            "model_name": result_model_name,
            "training_model_name": str(args.model_name),
            "n_channels": n_channels,
            "online_label": online_label,
            "emg_input_mode": emg_input_mode,
            "pseudo_emg_eeg_channels": ";".join(map(str, pseudo_emg_eeg_channels)),
            "eval_type": "pseudo_online_ensemble",
            "ensemble_method": method,
            "n_models": len(selected_cvs),
            "selected_cvs": ";".join(map(str, selected_cvs)),
            "jitter": float(args.jitter),
            "jitter_mode": str(args.get("jitter_mode", "random")),
            "max_trials": int(args.pseudo_online_filter.max_trials),
            "n_csv_rows": int(args.pseudo_online_filter.n_csv_rows),
            "n_valid_used": int(args.pseudo_online_filter.n_valid_used),
            "acc": round(acc, 6),
            "balanced_acc": round(balanced_acc, 6),
            "run_dir": str(run_dir),
            "csv_dir": row["csv_dir"],
            "npy_dir": row["npy_dir"],
            "csv_header": row["csv_header"],
        }
    )
    result.to_csv(result_path, index=False)
    sidecar = result_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "condition": str(args.parallel_sets),
                "model_name": result_model_name,
                "training_model_name": str(args.model_name),
                "n_channels": n_channels,
                "online_label": online_label,
                "emg_input_mode": emg_input_mode,
                "pseudo_emg_eeg_channels": pseudo_emg_eeg_channels,
                "eval_type": "pseudo_online_ensemble",
                "ensemble_method": method,
                "n_models": len(selected_cvs),
                "selected_cvs": selected_cvs,
                "jitter": float(args.jitter),
                "jitter_mode": str(args.get("jitter_mode", "random")),
                "max_trials": int(args.pseudo_online_filter.max_trials),
                "n_csv_rows": int(args.pseudo_online_filter.n_csv_rows),
                "n_valid_used": int(args.pseudo_online_filter.n_valid_used),
                "source_csv": str(args.pseudo_online_filter.source_csv),
                "source_npy_dir": str(args.pseudo_online_filter.source_npy_dir),
                "acc": acc,
                "balanced_acc": balanced_acc,
                "run_dir": str(run_dir),
            },
            indent=2,
        )
    )
    return result_path


def main() -> None:
    cli = parse_args()
    manifest = load_manifest(cli.manifest)
    history = load_histories(cli.test_history)
    written: list[Path] = []
    prefer_eeg_emg = any(mode != "real" for mode in cli.emg_input_modes)
    default_expected_decode_from = "eeg" if cli.test_history is None and not prefer_eeg_emg else None

    for row in manifest:
        for eval_row in channel_decimation_eval_rows(row, cli.channel_decimation_channels):
            condition = eval_row["condition"]
            models = [eval_row["model"]] if eval_row.get("model") else cli.models
            n_channels = int(eval_row["n_channels"]) if eval_row.get("n_channels") else None
            for model in models:
                if eval_row.get("run_dir"):
                    run_dir = Path(eval_row["run_dir"])
                else:
                    run_dir = discover_run_dir_from_history(
                        condition,
                        model,
                        history,
                        prefer_eeg_emg=prefer_eeg_emg,
                        n_channels=n_channels,
                        expected_decode_from=default_expected_decode_from,
                    )
                    if run_dir is None and n_channels is None:
                        run_dir = discover_run_dir(
                            condition,
                            model,
                            prefer_eeg_emg=prefer_eeg_emg,
                            expected_decode_from=default_expected_decode_from,
                        )
                if run_dir is None:
                    message = f"Missing rotating-CV run for condition={condition}, model={model}"
                    if n_channels is not None:
                        message += f", n_channels={n_channels}"
                    if cli.allow_missing:
                        print(f"SKIP: {message}", flush=True)
                        continue
                    raise FileNotFoundError(message)

                args = load_config(run_dir)
                if cli.disable_dataset_cache:
                    args.dataset_cache = OmegaConf.create({"enabled": False})
                apply_online_data(args, eval_row, cli.gpu, cli.jitter, cli.jitter_mode)
                cv_order = ranked_cvs(args, run_dir, history, cli.rank_metric)
                selected_cvs, model_paths = existing_model_paths(args, run_dir, cv_order, cli.n_models)
                with tempfile.TemporaryDirectory(prefix="pseudo_online_") as tmp_dir:
                    prepare_filtered_online_data(args, eval_row, Path(tmp_dir), cli.max_trials)
                    dataset_base = build_dataset(args)
                    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
                    for emg_input_mode in cli.emg_input_modes:
                        dataset = wrap_emg_input_variant(
                            args,
                            dataset_base,
                            emg_input_mode,
                            cli.pseudo_emg_eeg_channels,
                        )
                        scores, labels = predict_scores(args, model_paths, dataset, device)
                        result_path = write_results(
                            cli.output_dir,
                            eval_row,
                            args,
                            run_dir,
                            selected_cvs,
                            cli.method,
                            scores,
                            labels,
                            emg_input_mode,
                            cli.pseudo_emg_eeg_channels,
                        )
                        written.append(result_path)
                        print(f"Wrote {result_path}", flush=True)

    if not written:
        raise SystemExit("No pseudo-online results were written.")


if __name__ == "__main__":
    main()
