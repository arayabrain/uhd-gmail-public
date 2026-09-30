#!/usr/bin/env python3
"""Pseudo-online evaluation for rotating-CV models across jitter ablation trains.

Example::

    uv run python scripts/pseudo_online_test/run_jitter_ablation_pseudo_online.py --gpu 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PSEUDO_ONLINE = REPO_ROOT / "scripts" / "pseudo_online_test" / "run_rotating_cv_pseudo_online.py"

JITTER_CONDITIONS = (
    "fixed_jitter_m100ms",
    "fixed_jitter_m50ms",
    "fixed_jitter_p50ms",
    "fixed_jitter_p100ms",
    "no_jitter",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES index.")
    parser.add_argument("--device", type=int, default=0, help="Config gpu field passed to the evaluator.")
    parser.add_argument("--n-models", type=int, default=4)
    parser.add_argument("--method", default="zscore_mean")
    parser.add_argument("--rank-metric", default="balanced_acc_test")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scripts/pseudo_online_test/online_data_manifest.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/online/jitter_ablation"),
    )
    parser.add_argument(
        "--test-history-dir",
        type=Path,
        default=Path("outputs/offline/jitter_ablation"),
    )
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--inference-jitter", type=float, default=0.0)
    parser.add_argument("--inference-jitter-mode", default="random")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model list; default reads unique models from each history CSV.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def models_from_history(history: Path) -> list[str]:
    import pandas as pd

    df = pd.read_csv(history)
    if "model_name" not in df.columns:
        raise ValueError(f"{history} has no model_name column")
    return sorted(df["model_name"].dropna().astype(str).unique())


def main() -> None:
    args = parse_args()
    for jitter_condition in JITTER_CONDITIONS:
        history = args.test_history_dir / f"history_color_rotating_test_fold_{jitter_condition}_test.csv"
        if not history.is_file():
            raise FileNotFoundError(f"Missing test history CSV: {history}")
        model_args = args.models or models_from_history(history)
        if not model_args:
            raise ValueError(f"No models found in {history}")
        output_dir = (
            args.output_root
            / f"{jitter_condition}_infer_{args.inference_jitter_mode}_{args.inference_jitter}"
        )
        cmd = [
            sys.executable,
            os.fspath(PSEUDO_ONLINE),
            "--manifest",
            os.fspath(args.manifest),
            "--models",
            *model_args,
            "--n-models",
            str(args.n_models),
            "--method",
            args.method,
            "--rank-metric",
            args.rank_metric,
            "--test-history",
            os.fspath(history),
            "--gpu",
            str(args.device),
            "--jitter",
            str(args.inference_jitter),
            "--jitter-mode",
            args.inference_jitter_mode,
            "--max-trials",
            str(args.max_trials),
            "--output-dir",
            os.fspath(output_dir),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    main()
