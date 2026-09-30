#!/usr/bin/env python3
"""Train rotating 10-fold CV baselines for published calibration runs.

Example::

    uv run python scripts/offline/run_rotating_cv_baseline.py --gpu 0
    uv run python scripts/offline/run_rotating_cv_baseline.py --gpu 0 --conditions sub-1_task-overt_acq-calibration --models EEGNet cBraMod
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scripts.figures._bids_runs import OFFLINE_RUNS

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "uhd_eeg" / "trainers" / "trainer_rotating_test_fold.py"
DEFAULT_CV_HISTORY = "outputs/offline/baseline/history_color_rotating_test_fold_cv.csv"
DEFAULT_TEST_HISTORY = "outputs/offline/baseline/history_color_rotating_test_fold_test.csv"
DEFAULT_MODELS = ("EEGNet", "RNN", "CovTanSVM", "cBraMod")


def default_conditions() -> list[str]:
    return [run.parallel_set for run in OFFLINE_RUNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Hydra parallel_sets keys (default: all OFFLINE_RUNS calibration keys).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help=f"model_name values (default: {' '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES index.")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--cv-history",
        type=Path,
        default=Path(DEFAULT_CV_HISTORY),
        help="record_history_filepath override.",
    )
    parser.add_argument(
        "--test-history",
        type=Path,
        default=Path(DEFAULT_TEST_HISTORY),
        help="test_record_history_filepath override.",
    )
    parser.add_argument(
        "--hydra-run-root",
        type=Path,
        default=None,
        help="Optional hydra.run.dir root (per condition/model subdirs are appended).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned trainer invocations without running them.",
    )
    return parser.parse_args()


def run_one(
    condition: str,
    model: str,
    gpu: int,
    seed: int,
    cv_history: Path,
    test_history: Path,
    hydra_run_root: Path | None,
    dry_run: bool,
) -> None:
    config_name = f"config_color_rotating_test_fold_{condition}_{model}"
    overrides = [
        f"parallel_sets={condition}",
        f"model_name={model}",
        f"config_name={config_name}",
        f"seed={seed}",
        f"record_history_filepath={cv_history.as_posix()}",
        f"test_record_history_filepath={test_history.as_posix()}",
    ]
    if model == "cBraMod":
        overrides.append('cBraMod.patch_mode="non_overlap"')
        overrides.append("cBraMod.use_backbone_weights=true")
    if hydra_run_root is not None:
        run_dir = hydra_run_root / condition / model
        overrides.append(f"hydra.run.dir={run_dir.as_posix()}")

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
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    args = parse_args()
    conditions = args.conditions or default_conditions()
    cv_history = args.cv_history.resolve()
    test_history = args.test_history.resolve()
    cv_history.parent.mkdir(parents=True, exist_ok=True)
    test_history.parent.mkdir(parents=True, exist_ok=True)

    for condition in conditions:
        for model in args.models:
            run_one(
                condition=condition,
                model=model,
                gpu=args.gpu,
                seed=args.seed,
                cv_history=cv_history,
                test_history=test_history,
                hydra_run_root=args.hydra_run_root,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
