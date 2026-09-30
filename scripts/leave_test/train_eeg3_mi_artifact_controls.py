#!/usr/bin/env python3
"""EEG (MI-top-3) vs artifact cross-modal controls for Supplementary Table S2.

Trains one 3-channel EEGNet per artifact signal (EOG, EMG upper, EMG lower) on
the three denoised EEG electrodes with highest MI to that artifact, then
evaluates with held-out EEG (baseline) and with the artifact tiled across inputs.

**GPU required for training.** The public repo does not ship rotating-fold
training checkpoints; use ``--eval-only`` to score existing weights, or run
training in an environment that provides ``fit_rotating_test_fold`` and derived
arrays under ``configs/paths.yaml``.

CPU-friendly aggregation::

    uv run python scripts/leave_test/train_eeg3_mi_artifact_controls.py --summarize-only
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

from scripts.leave_test._conditions import behavior_from_run_key, default_calibration_run_keys, subject_from_run_key
from uhd_eeg.datasets.DatasetUHD import EEGDataset, EMGDataset
from uhd_eeg.trainers.eval_helpers import build_model, get_behavior, model_path, prepare_inputs

try:
    from uhd_eeg.trainers.trainer_rotating_test_fold import fit_rotating_test_fold
except ImportError:  # pragma: no cover
    fit_rotating_test_fold = None


ARTIFACTS = {
    "EOG": 0,
    "EMG_upper": 1,
    "EMG_lower": 2,
}


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


class SingleArtifactAs3Dataset(Dataset):
    def __init__(self, dataset: Dataset, artifact_index: int) -> None:
        self.dataset = dataset
        self.artifact_index = artifact_index
        self.labels_all = dataset.labels_all
        self.window_eegnet = dataset.window_eegnet
        self.device = dataset.device

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        one = x[:, self.artifact_index : self.artifact_index + 1, :]
        return one.repeat(1, 3, 1), y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-keys", nargs="+", default=None)
    parser.add_argument("--artifacts", nargs="+", default=list(ARTIFACTS))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eeg3_mi_artifact_controls"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trainer/config_color_within_offline_split.yaml"),
    )
    parser.add_argument("--mi-root", type=Path, default=Path("outputs/mutual_information"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def ensure_config(
    args: DictConfig,
    run_key: str,
    artifact: str,
    gpu: int,
    output_dir: Path,
) -> DictConfig:
    OmegaConf.set_struct(args, False)
    args.use_hydra_savedir = False
    args.saved_data_root = os.fspath(output_dir / "weights")
    if "jitter_mode" not in args:
        args.jitter_mode = "random"
    args.parallel_sets = run_key
    if run_key not in args:
        raise KeyError(
            f"{run_key} missing from trainer config; extend "
            "configs/trainer/config_color_within_offline_split.yaml with BIDS npy_dir entries."
        )
    args.gmail = args[run_key]
    args.gpu = gpu
    args.decode_from = "eeg"
    args.model_name = "EEGNet"
    args.num_channels = 3
    if "cBraMod" in args:
        args.cBraMod.num_channels = 3
    safe_artifact = artifact.lower()
    args.config_name = f"config_color_rotating_test_fold_eeg3_mi_{safe_artifact}_{run_key}_EEGNet"
    args.record_history_filepath = os.fspath(
        output_dir / "history" / f"history_eeg3_mi_{safe_artifact}_cv.csv"
    )
    args.test_record_history_filepath = os.fspath(
        output_dir / "history" / f"history_eeg3_mi_{safe_artifact}_test.csv"
    )
    OmegaConf.set_struct(args, True)
    return args


def condition_index(table: pd.DataFrame, eeg_type: str, artifact: str) -> int:
    mask = (
        (table["eeg_type"] == eeg_type)
        & (table["emg_type"] == "raw")
        & (table["emg_name"] == artifact)
        & (table["surrogate_type"] == "none")
    )
    matches = np.where(mask.to_numpy())[0]
    if len(matches) != 1:
        raise ValueError(f"Expected one MI row for {eeg_type}/{artifact}, got {len(matches)}")
    return int(matches[0])


def mi_top3_channels(
    run_key: str,
    behavior: str,
    artifact: str,
    mi_root: Path,
) -> tuple[list[int], list[float]]:
    subject = subject_from_run_key(run_key)
    data_dir = mi_root / subject / "data"
    mis = np.load(data_dir / "mis.npy")
    mis_table = pd.read_csv(data_dir / "mis_table.csv")
    trial_table = pd.read_csv(data_dir / "eegs_table_all.csv")
    trial_idx = np.where(trial_table["condition"].to_numpy() == behavior)[0]
    idx = condition_index(mis_table, "after_preproc", artifact)
    values = mis[idx, trial_idx, :].mean(axis=0)
    top3 = np.argsort(-values)[:3].astype(int).tolist()
    return top3, values[top3].astype(float).tolist()


def all_weights_exist(args: DictConfig) -> bool:
    return all(Path(model_path(args, cv)).exists() for cv in range(int(args.n_splits)))


def train_if_needed(args: DictConfig, dataset: Dataset, overwrite: bool) -> None:
    if fit_rotating_test_fold is None:
        raise RuntimeError(
            "Rotating-fold training is not available in this checkout. "
            "Use --eval-only with exported weights, or add "
            "uhd_eeg.trainers.trainer_rotating_test_fold from the full training stack."
        )
    if all_weights_exist(args) and not overwrite:
        print(f"weights exist, skip training: {args.config_name}", flush=True)
        return
    Path(args.record_history_filepath).parent.mkdir(parents=True, exist_ok=True)
    fit_rotating_test_fold(args, dataset, dataset.window_eegnet, dataset.device)


def evaluate_fold_rows(
    args: DictConfig,
    dataset: Dataset,
    top3: list[int],
    top3_values: list[float],
    *,
    run_key: str,
    artifact: str | None,
    test_input: str,
    output_dir: Path,
    outfile: str,
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))
    folds = [test_idx for _, test_idx in skf.split(np.arange(len(dataset)), dataset.labels_all)]
    device = dataset.device
    behavior = get_behavior(args) if get_behavior(args) != "unknown" else behavior_from_run_key(run_key)
    subject = subject_from_run_key(run_key)
    rows = []
    for cv, indices in enumerate(folds):
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=1,
            num_workers=int(args.n_worekers),
            pin_memory=False,
            shuffle=False,
        )
        model = build_model(args, dataset.window_eegnet)
        model.load_state_dict(torch.load(model_path(args, cv), map_location=device))
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
        rows.append(
            {
                "run_key": run_key,
                "subject": subject,
                "behavior": behavior,
                "artifact": artifact or "",
                "train_input": "denoised_eeg_after_preproc_mi_top3",
                "test_input": test_input,
                "CV": cv,
                "n_test": len(labels_arr),
                "acc": round(accuracy_score(labels_arr, pred_arr), 6),
                "balanced_acc": round(balanced_accuracy_score(labels_arr, pred_arr), 6),
                "config_name": str(args.config_name),
                "mi_top3_channels": ";".join(map(str, top3)),
                "mi_top3_values": ";".join(f"{v:.8f}" for v in top3_values),
            }
        )
    df = pd.DataFrame(rows)
    out = output_dir / "folds" / outfile
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def run_one(run_key: str, artifact: str, gpu: int, cli: argparse.Namespace) -> pd.DataFrame:
    output_dir = cli.output_dir
    fold_path = output_dir / "folds" / f"{run_key}_{artifact}.csv"
    eeg_fold = output_dir / "folds" / f"{run_key}_eeg_test.csv"
    if not cli.overwrite and fold_path.exists() and (cli.eval_only or eeg_fold.exists()):
        parts = [pd.read_csv(fold_path)]
        if eeg_fold.exists():
            parts.append(pd.read_csv(eeg_fold))
        return pd.concat(parts, ignore_index=True)

    args = OmegaConf.load(cli.config)
    args = ensure_config(args, run_key, artifact, gpu, output_dir)
    behavior = behavior_from_run_key(run_key)
    top3, top3_values = mi_top3_channels(run_key, behavior, artifact, cli.mi_root)

    eeg_dataset = EEGDataset(args)
    train_dataset = SelectChannelsDataset(eeg_dataset, top3)
    if not cli.eval_only:
        train_if_needed(args, train_dataset, cli.overwrite)

    if artifact == cli.artifacts[0] and (cli.overwrite or not eeg_fold.exists()):
        evaluate_fold_rows(
            args,
            train_dataset,
            top3,
            top3_values,
            run_key=run_key,
            artifact=None,
            test_input="denoised_eeg_after_preproc_mi_top3",
            output_dir=output_dir,
            outfile=f"{run_key}_eeg_test.csv",
        )

    emg_dataset = EMGDataset(args)
    test_dataset = SingleArtifactAs3Dataset(emg_dataset, ARTIFACTS[artifact])
    df_artifact = evaluate_fold_rows(
        args,
        test_dataset,
        top3,
        top3_values,
        run_key=run_key,
        artifact=artifact,
        test_input=f"{artifact}_single_channel_tiled_to_3",
        output_dir=output_dir,
        outfile=f"{run_key}_{artifact}.csv",
    )
    if eeg_fold.is_file():
        return pd.concat([pd.read_csv(eeg_fold), df_artifact], ignore_index=True)
    return df_artifact


def worker(gpu: int, queue: Queue, cli: argparse.Namespace) -> None:
    while True:
        try:
            run_key, artifact = queue.get_nowait()
        except Empty:
            return
        print(f"[gpu {gpu}] {run_key} {artifact}", flush=True)
        log_dir = cli.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_key}_{artifact}.log"
        with log_path.open("a") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            df = run_one(run_key, artifact, gpu, cli)
        print(
            f"[gpu {gpu}] done {run_key} {artifact}: bacc={df['balanced_acc'].mean():.3f}",
            flush=True,
        )


def write_summary(output_dir: Path) -> None:
    fold_paths = sorted((output_dir / "folds").glob("*.csv"))
    if not fold_paths:
        print(f"No fold CSVs under {output_dir / 'folds'}")
        return
    folds = pd.concat([pd.read_csv(path) for path in fold_paths], ignore_index=True)
    folds.to_csv(output_dir / "eeg3_mi_artifact_control_folds.csv", index=False)
    artifact_folds = folds[folds["artifact"].astype(str).str.len() > 0]
    summary = (
        artifact_folds.groupby(
            [
                "run_key",
                "subject",
                "behavior",
                "artifact",
                "train_input",
                "test_input",
                "config_name",
                "mi_top3_channels",
            ],
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
    summary.to_csv(output_dir / "eeg3_mi_artifact_control_summary_by_condition.csv", index=False)


def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    if cli.summarize_only:
        write_summary(cli.output_dir)
        return

    run_keys = cli.run_keys or default_calibration_run_keys()
    queue: Queue = Queue()
    for run_key in run_keys:
        for artifact in cli.artifacts:
            if artifact not in ARTIFACTS:
                raise ValueError(f"Unknown artifact {artifact}; choose {list(ARTIFACTS)}")
            queue.put((run_key, artifact))
    gpus = [int(value) for value in cli.gpus.split(",") if value.strip()]
    processes = [Process(target=worker, args=(gpu, queue, cli)) for gpu in gpus]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise SystemExit(process.exitcode)
    write_summary(cli.output_dir)
    print(f"Saved EEG3 MI artifact control tables to {cli.output_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
