#!/usr/bin/env python3
"""Rotating CV for EEGNet_with_mask at k-medoids channel densities 4/8/16/32.

Uses ``uhd_eeg.analysis.electrode_subsets_kmedoids.EXPECTED_SUBSETS`` only.

Example::

    uv run python scripts/offline/run_channel_decimation_EEGNet.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.figures._bids_runs import OFFLINE_RUNS
from uhd_eeg.analysis.electrode_subsets_kmedoids import EXPECTED_SUBSETS

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "uhd_eeg" / "trainers" / "trainer_rotating_test_fold.py"
DEFAULT_DENSITIES = (4, 8, 16, 32)
HISTORY_DIR = Path("outputs/offline/channel_decimation_EEGNet")


def default_conditions() -> list[str]:
    return [run.parallel_set for run in OFFLINE_RUNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Hydra parallel_sets keys (default: published OFFLINE_RUNS).",
    )
    parser.add_argument(
        "--densities",
        type=int,
        nargs="*",
        default=list(DEFAULT_DENSITIES),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=HISTORY_DIR,
        help="Directory for per-density CV/test history CSV files.",
    )
    parser.add_argument(
        "--hydra-run-root",
        type=Path,
        default=None,
        help="Optional hydra.run.dir root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_one(
    condition: str,
    density: int,
    channels: list[int],
    gpu: int,
    seed: int,
    history_dir: Path,
    hydra_run_root: Path | None,
    dry_run: bool,
) -> None:
    variant = f"{density}ch"
    cv_history = history_dir / f"history_color_rotating_test_fold_EEGNet_with_mask_{variant}_cv.csv"
    test_history = history_dir / f"history_color_rotating_test_fold_EEGNet_with_mask_{variant}_test.csv"
    config_name = f"config_color_rotating_test_fold_{condition}_EEGNet_with_mask_{variant}"
    channels_json = json.dumps(channels)
    overrides = [
        f"parallel_sets={condition}",
        "model_name=EEGNet_with_mask",
        f"channels_use={channels_json}",
        f"seed={seed}",
        f"config_name={config_name}",
        f"record_history_filepath={cv_history.as_posix()}",
        f"test_record_history_filepath={test_history.as_posix()}",
    ]
    if hydra_run_root is not None:
        overrides.append(
            f"hydra.run.dir={(hydra_run_root / condition / variant).as_posix()}"
        )

    cmd = [
        sys.executable,
        os.fspath(TRAINER),
        "--config-name",
        "config_color_within_offline_split",
        *overrides,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    history_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    args = parse_args()
    conditions = args.conditions or default_conditions()
    history_dir = args.history_dir.resolve()
    for density in args.densities:
        if density not in EXPECTED_SUBSETS:
            raise KeyError(f"No k-medoids subset for density={density}")
        channels = list(EXPECTED_SUBSETS[density])
        if len(channels) != density:
            raise ValueError(f"EXPECTED_SUBSETS[{density}] length mismatch")
        for condition in conditions:
            run_one(
                condition=condition,
                density=density,
                channels=channels,
                gpu=args.gpu,
                seed=args.seed,
                history_dir=history_dir,
                hydra_run_root=args.hydra_run_root,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
