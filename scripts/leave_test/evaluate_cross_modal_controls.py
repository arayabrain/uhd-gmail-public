#!/usr/bin/env python3
"""Cross-modal EEG/EMG controls for Supplementary Table S1.

Evaluates rotating-fold EEGNet checkpoints with mismatched test inputs:

- **Table S1 row**: train EMG (3 ch) → test denoised EEG (MI-top-3 channels).
- Additional control: train EEG (128 ch) → test EMG (3 ch tiled to 128).

Requires GPU, derived per-trial arrays (``configs/paths.yaml``), pre-trained
rotating-fold weights under ``--weights-root``, and MI exports under ``--mi-root``.

CPU-friendly replay from saved fold CSVs::

    uv run python scripts/leave_test/evaluate_cross_modal_controls.py --summarize-only
    uv run python scripts/leave_test/summarize_supplementary_tables_s1_s2.py
"""

from __future__ import annotations

import argparse
import contextlib
import os
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from scripts.leave_test._conditions import (
    behavior_from_run_key,
    default_calibration_run_keys,
    subject_from_run_key,
)
from uhd_eeg.datasets.DatasetUHD import EEGDataset, EMGDataset
from uhd_eeg.trainers.eval_helpers import build_model, get_behavior, prepare_inputs


class SelectChannelsDataset(Dataset):
    def __init__(self, dataset: Dataset, channels: list[int]) -> None:
        self.dataset = dataset
        self.channels = channels
        self.labels_all = dataset.labels_all
        self.window_eegnet = dataset.window_eegnet
        self.device = dataset.device

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x[:, self.channels, :], y


class TileChannelsDataset(Dataset):
    def __init__(self, dataset: Dataset, n_channels: int) -> None:
        self.dataset = dataset
        self.n_channels = n_channels
        self.labels_all = dataset.labels_all
        self.window_eegnet = dataset.window_eegnet
        self.device = dataset.device

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        reps = int(np.ceil(self.n_channels / x.shape[1]))
        tiled = x.repeat(1, reps, 1)[:, : self.n_channels, :]
        return tiled, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-keys", nargs="+", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_modal_controls"))
    parser.add_argument("--weights-root", type=Path, default=Path("outputs"))
    parser.add_argument("--mi-root", type=Path, default=Path("outputs/mutual_information"))
    parser.add_argument(
        "--trainer-config",
        type=Path,
        default=Path("configs/trainer/config_color_within_offline_split.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def ensure_config_defaults(args: DictConfig, gpu: int) -> DictConfig:
    OmegaConf.set_struct(args, False)
    args.use_hydra_savedir = False
    if "jitter_mode" not in args:
        args.jitter_mode = "random"
    args.gpu = gpu
    if "gmail" not in args:
        args.gmail = args[args.parallel_sets]
    OmegaConf.set_struct(args, True)
    return args


def load_config(run_dir: Path, gpu: int) -> DictConfig:
    args = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return ensure_config_defaults(args, gpu)


def complete_run_dir(run_dir: Path, config_name: str, suffix: str) -> bool:
    for cv in range(10):
        if not (run_dir / f"{suffix}_{config_name}_N5_cv{cv}.pth").exists():
            return False
    return True


def find_run_dir(run_key: str, decode_from: str, weights_root: Path) -> Path:
    if decode_from == "emg":
        config_name = f"config_color_rotating_test_fold_emg_only_{run_key}_EEGNet"
    elif decode_from == "eeg":
        config_name = f"config_color_rotating_test_fold_{run_key}_EEGNet"
    else:
        raise ValueError(decode_from)

    paths = sorted(
        weights_root.glob(f"**/model_weight_{config_name}_N5_cv0.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        run_dir = path.parent
        if complete_run_dir(run_dir, config_name, "model_weight"):
            return run_dir
    raise FileNotFoundError(
        f"No complete rotating-fold run for {run_key} ({decode_from}) under {weights_root}"
    )


def condition_index(table: pd.DataFrame, eeg_type: str, emg_name: str) -> int:
    mask = (
        (table["eeg_type"] == eeg_type)
        & (table["emg_type"] == "raw")
        & (table["emg_name"] == emg_name)
        & (table["surrogate_type"] == "none")
    )
    matches = np.where(mask.to_numpy())[0]
    if len(matches) != 1:
        raise ValueError(f"Expected one MI row for {eeg_type}/{emg_name}, got {len(matches)}")
    return int(matches[0])


def mi_top3_channels(run_key: str, behavior: str, mi_root: Path) -> tuple[list[int], list[float]]:
    subject = subject_from_run_key(run_key)
    subject_dir = mi_root / subject / "data"
    mis = np.load(subject_dir / "mis.npy")
    mis_table = pd.read_csv(subject_dir / "mis_table.csv")
    trial_table = pd.read_csv(subject_dir / "eegs_table_all.csv")
    trial_idx = np.where(trial_table["condition"].to_numpy() == behavior)[0]
    if len(trial_idx) == 0:
        raise ValueError(f"No MI trials for {run_key} behavior={behavior}")

    rows = []
    for emg_name in ["EMG_upper", "EMG_lower"]:
        rows.append(mis[condition_index(mis_table, "after_preproc", emg_name), trial_idx, :])
    values = np.concatenate(rows, axis=0).mean(axis=0)
    top3 = np.argsort(-values)[:3].astype(int).tolist()
    return top3, values[top3].astype(float).tolist()


def model_weight_path(run_dir: Path, config_name: str, cv: int) -> Path:
    return run_dir / f"model_weight_{config_name}_N5_cv{cv}.pth"


def evaluate_fold(
    args: DictConfig,
    model_path: Path,
    dataset: Dataset,
    indices: np.ndarray,
    device: torch.device,
):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        num_workers=int(args.n_worekers),
        pin_memory=False,
        shuffle=False,
    )
    model = build_model(args, dataset.window_eegnet)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    scores = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs, labels = prepare_inputs(args, inputs, labels)
            pred = model(inputs).cpu().detach().numpy()
            scores.append(pred.mean(axis=0) if pred.shape[0] > 1 else pred[0])
            labels_np = labels.cpu().detach().numpy()
            labels_all.append(int(labels_np[:1][0]))
    labels_arr = np.array(labels_all)
    pred_arr = np.argmax(np.array(scores), axis=1)
    return (
        accuracy_score(labels_arr, pred_arr),
        balanced_accuracy_score(labels_arr, pred_arr),
        len(labels_arr),
    )


def evaluate_experiment(
    run_key: str,
    experiment: str,
    run_dir: Path,
    args: DictConfig,
    dataset: Dataset,
    device: torch.device,
    output_dir: Path,
    metadata: dict,
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))
    folds = [test_idx for _, test_idx in skf.split(np.arange(len(dataset)), dataset.labels_all)]
    rows = []
    behavior = get_behavior(args) if get_behavior(args) != "unknown" else behavior_from_run_key(run_key)
    subject = subject_from_run_key(run_key)
    for cv, indices in enumerate(folds):
        acc, bacc, n = evaluate_fold(
            args,
            model_weight_path(run_dir, str(args.config_name), cv),
            dataset,
            indices,
            device,
        )
        rows.append(
            {
                "run_key": run_key,
                "subject": subject,
                "behavior": behavior,
                "experiment": experiment,
                "model_name": str(args.model_name),
                "train_decode_from": str(args.decode_from),
                "test_input": metadata["test_input"],
                "input_transform": metadata["input_transform"],
                "CV": cv,
                "n_test": n,
                "acc": round(acc, 6),
                "balanced_acc": round(bacc, 6),
                "run_dir": str(run_dir),
                "config_name": str(args.config_name),
                "mi_top3_channels": metadata.get("mi_top3_channels", ""),
                "mi_top3_values": metadata.get("mi_top3_values", ""),
            }
        )
    df = pd.DataFrame(rows)
    out = output_dir / "folds" / f"{run_key}_{experiment}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def evaluate_run_key(run_key: str, gpu: int, cli: argparse.Namespace) -> pd.DataFrame:
    output_dir = cli.output_dir
    fold_a = output_dir / "folds" / f"{run_key}_emg3_to_denoised_eeg_mi_top3.csv"
    fold_b = output_dir / "folds" / f"{run_key}_eeg128_to_emg3_tiled.csv"
    if not cli.overwrite and fold_a.exists() and fold_b.exists():
        return pd.concat([pd.read_csv(fold_a), pd.read_csv(fold_b)], ignore_index=True)

    emg_run = find_run_dir(run_key, "emg", cli.weights_root)
    eeg_run = find_run_dir(run_key, "eeg", cli.weights_root)
    emg_args = load_config(emg_run, gpu)
    eeg_args = load_config(eeg_run, gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    behavior = behavior_from_run_key(run_key)
    top3, top3_values = mi_top3_channels(run_key, behavior, cli.mi_root)

    eeg_dataset = EEGDataset(emg_args)
    emg3_to_eeg = SelectChannelsDataset(eeg_dataset, top3)
    df_a = evaluate_experiment(
        run_key,
        "emg3_to_denoised_eeg_mi_top3",
        emg_run,
        emg_args,
        emg3_to_eeg,
        device,
        output_dir,
        {
            "test_input": "denoised_eeg_after_preproc_mi_top3",
            "input_transform": "select_mi_top3_channels",
            "mi_top3_channels": ";".join(map(str, top3)),
            "mi_top3_values": ";".join(f"{v:.8f}" for v in top3_values),
        },
    )

    emg_dataset = EMGDataset(eeg_args)
    emg_tiled = TileChannelsDataset(emg_dataset, 128)
    df_b = evaluate_experiment(
        run_key,
        "eeg128_to_emg3_tiled",
        eeg_run,
        eeg_args,
        emg_tiled,
        device,
        output_dir,
        {
            "test_input": "emg3",
            "input_transform": "tile_3_channels_to_128",
        },
    )
    return pd.concat([df_a, df_b], ignore_index=True)


def worker(gpu: int, queue: Queue, cli: argparse.Namespace) -> None:
    while True:
        try:
            run_key = queue.get_nowait()
        except Empty:
            return
        print(f"[gpu {gpu}] evaluating {run_key}", flush=True)
        log_dir = cli.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_key}.log"
        with log_path.open("a") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            df = evaluate_run_key(run_key, gpu, cli)
        print(
            f"[gpu {gpu}] done {run_key}: "
            + ", ".join(
                f"{k}={v:.3f}"
                for k, v in df.groupby("experiment")["balanced_acc"].mean().items()
            ),
            flush=True,
        )


def write_summary(output_dir: Path) -> None:
    fold_paths = sorted((output_dir / "folds").glob("*.csv"))
    if not fold_paths:
        print(f"No fold CSVs under {output_dir / 'folds'}")
        return
    folds = pd.concat([pd.read_csv(path) for path in fold_paths], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "cross_modal_control_folds.csv", index=False)
    summary = (
        folds.groupby(
            [
                "experiment",
                "run_key",
                "subject",
                "behavior",
                "model_name",
                "train_decode_from",
                "test_input",
                "input_transform",
                "mi_top3_channels",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_folds=("CV", "count"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            balanced_acc_mean=("balanced_acc", "mean"),
            balanced_acc_std=("balanced_acc", "std"),
        )
    )
    summary.to_csv(output_dir / "cross_modal_control_summary_by_condition.csv", index=False)
    wide = summary.pivot_table(
        index=["run_key", "subject", "behavior"],
        columns="experiment",
        values="balanced_acc_mean",
    ).reset_index()
    wide.to_csv(output_dir / "cross_modal_control_balanced_acc_table.csv", index=False)


def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    if cli.summarize_only:
        write_summary(cli.output_dir)
        return

    run_keys = cli.run_keys or default_calibration_run_keys()
    gpus = [int(value) for value in cli.gpus.split(",") if value.strip()]
    queue: Queue = Queue()
    for run_key in run_keys:
        queue.put(run_key)
    processes = [Process(target=worker, args=(gpu, queue, cli)) for gpu in gpus]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise SystemExit(process.exitcode)
    write_summary(cli.output_dir)
    print(f"Saved cross-modal control tables to {cli.output_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
