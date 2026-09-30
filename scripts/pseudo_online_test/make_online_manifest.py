#!/usr/bin/env python3
"""Write a BIDS-derived pseudo-online evaluation manifest.

Paths use ``configs/paths.yaml`` ``output_root`` (default ``data/derived``).
Condition keys are Hydra ``parallel_sets`` strings (no ``_run-`` segment).

Example::

    uv run python scripts/pseudo_online_test/make_online_manifest.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.figures._bids_runs import ONLINE_RUNS
from uhd_eeg.paths import get_output_root

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "scripts" / "pseudo_online_test" / "online_data_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paths-file", type=Path, default=None)
    parser.add_argument(
        "--online-label",
        default="online50",
        help="Label stored in the online_label column (default: online50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_root = Path(get_output_root(args.paths_file))
    except (FileNotFoundError, KeyError):
        output_root = REPO_ROOT / "data" / "derived"

    rows = []
    for run in ONLINE_RUNS:
        csv_dir = output_root / run.subject / run.session
        npy_dir = csv_dir / f"task-{run.task}_acq-{run.acq}_run-{run.run}"
        rows.append(
            {
                "condition": run.parallel_set,
                "csv_dir": csv_dir.as_posix(),
                "npy_dir": npy_dir.as_posix(),
                "csv_header": f"_{run.acq}_{run.run}",
                "online_label": args.online_label,
                "model": "",
                "run_dir": "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "csv_dir",
                "npy_dir",
                "csv_header",
                "online_label",
                "model",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
