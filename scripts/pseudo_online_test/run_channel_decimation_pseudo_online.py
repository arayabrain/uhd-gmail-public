#!/usr/bin/env python3
"""Pseudo-online evaluation for EEGNet channel-decimation rotating-CV models.

Example::

    uv run python scripts/pseudo_online_test/run_channel_decimation_pseudo_online.py --gpu 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PSEUDO_ONLINE = REPO_ROOT / "scripts" / "pseudo_online_test" / "run_rotating_cv_pseudo_online.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES index.")
    parser.add_argument("--device", type=int, default=0, help="Config gpu field passed to the evaluator.")
    parser.add_argument("--n-models", type=int, default=4)
    parser.add_argument("--method", default="zscore_mean")
    parser.add_argument("--rank-metric", default="balanced_acc_test")
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scripts/pseudo_online_test/online_data_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/online/channel_decimation_EEGNet"),
    )
    parser.add_argument(
        "--channel-history-dir",
        type=Path,
        default=Path("outputs/offline/channel_decimation_EEGNet"),
    )
    parser.add_argument(
        "--full-history",
        type=Path,
        default=Path("outputs/offline/baseline/history_color_rotating_test_fold_test.csv"),
    )
    parser.add_argument(
        "--channel-decimation-channels",
        type=int,
        nargs="*",
        default=[4, 8, 16, 32, 128],
    )
    parser.add_argument("--disable-dataset-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.full_history.is_file():
        raise FileNotFoundError(f"128-channel history not found: {args.full_history}")

    channel_histories = sorted(args.channel_history_dir.glob("*_test.csv"))
    if not channel_histories:
        raise FileNotFoundError(
            f"No channel-decimation test histories in {args.channel_history_dir}"
        )

    full_resolved = args.full_history.resolve()
    unique_histories = [
        path for path in channel_histories if path.resolve() != full_resolved
    ]
    unique_histories.append(args.full_history)

    cmd = [
        sys.executable,
        os.fspath(PSEUDO_ONLINE),
        "--manifest",
        os.fspath(args.manifest),
        "--n-models",
        str(args.n_models),
        "--method",
        args.method,
        "--rank-metric",
        args.rank_metric,
        "--test-history",
        *[os.fspath(path) for path in unique_histories],
        "--gpu",
        str(args.device),
        "--jitter",
        "0",
        "--jitter-mode",
        "random",
        "--max-trials",
        str(args.max_trials),
        "--output-dir",
        os.fspath(args.output_dir),
        "--channel-decimation-channels",
        *[str(value) for value in args.channel_decimation_channels],
    ]
    if args.disable_dataset_cache:
        cmd.append("--disable-dataset-cache")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    main()
