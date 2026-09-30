#!/usr/bin/env python3
"""Train EEGNet rotating CV under fixed / zero training-jitter ablations (Fig. S7 upstream).

Writes one test-history CSV per jitter label under ``outputs/offline/jitter_ablation/``.
GPU required.

Example::

    uv run python scripts/offline/run_jitter_ablation_offline.py --gpu 0
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

# Manuscript Fig. S7 training-jitter conditions (seconds; fixed mode).
JITTER_SPECS: tuple[tuple[str, float], ...] = (
    ("fixed_jitter_m100ms", -0.1),
    ("fixed_jitter_m50ms", -0.05),
    ("fixed_jitter_p50ms", 0.05),
    ("fixed_jitter_p100ms", 0.1),
    ("no_jitter", 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Hydra parallel_sets keys (default: all OFFLINE_RUNS).",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("outputs/offline/jitter_ablation"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = args.conditions or [run.parallel_set for run in OFFLINE_RUNS]
    history_dir = args.history_dir
    history_dir.mkdir(parents=True, exist_ok=True)

    for label, jitter in JITTER_SPECS:
        cv_hist = history_dir / f"history_color_rotating_test_fold_{label}_cv.csv"
        test_hist = history_dir / f"history_color_rotating_test_fold_{label}_test.csv"
        for condition in conditions:
            config_name = f"config_color_rotating_test_fold_{label}_{condition}_EEGNet"
            cmd = [
                sys.executable,
                os.fspath(TRAINER),
                "--config-name",
                "config_color_within_offline_split",
                f"parallel_sets={condition}",
                "model_name=EEGNet",
                "jitter_mode=fixed",
                f"jitter={jitter}",
                f"config_name={config_name}",
                f"seed={args.seed}",
                f"record_history_filepath={cv_hist.as_posix()}",
                f"test_record_history_filepath={test_hist.as_posix()}",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(" ".join(cmd), flush=True)
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    main()
