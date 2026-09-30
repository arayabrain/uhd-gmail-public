#!/usr/bin/env python3
"""Build a pseudo-online manifest for EEGNet channel decimation models."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from scripts.leave_test._conditions import behavior_from_run_key, subject_from_run_key


def condition_subject_behavior(condition: str) -> tuple[str, str]:
    if "_run-" in condition:
        return subject_from_run_key(condition), behavior_from_run_key(condition)
    if condition.startswith("sub-") and "_task-" in condition:
        subject = condition.split("_", 1)[0]
        task = condition.split("_task-", 1)[1].split("_", 1)[0]
        behavior = "minimally_overt" if task == "minimallyovert" else task
        return subject, behavior
    raise ValueError(f"Unrecognized BIDS condition key: {condition!r}")


def infer_n_channels(path: Path) -> int:
    match = re.search(r"with_mask_(\d+)ch", path.name)
    if not match:
        raise ValueError(f"Could not infer channel count from {path}")
    return int(match.group(1))


def load_online_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"condition", "csv_dir", "npy_dir", "csv_header", "online_label"}
    missing = sorted(required - set(rows[0].keys() if rows else []))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return rows


def run_dir_from_config(config: str) -> str:
    config_path = Path(config)
    if config_path.name == "config.yaml" and config_path.parent.name == ".hydra":
        return str(config_path.parent.parent)
    return str(config_path.parent)


def build_history_index(
    channel_history_dir: Path,
    full_history: Path,
    full_model_name: str,
) -> dict[tuple[str, str, int], str]:
    index: dict[tuple[str, str, int], str] = {}

    def record(key: tuple[str, str, int], run_dir: str, source: Path) -> None:
        previous = index.get(key)
        if previous is not None and Path(previous).resolve() != Path(run_dir).resolve():
            raise ValueError(
                f"Conflicting run directories for {key}: {previous} versus "
                f"{run_dir} ({source})"
            )
        index[key] = run_dir

    for path in sorted(channel_history_dir.glob("*_test.csv")):
        n_channels = infer_n_channels(path)
        df = pd.read_csv(path)
        df = df[df["model_name"].astype(str).eq("EEGNet_with_mask")].copy()
        if "eval_type" in df.columns:
            df = df[df["eval_type"].astype(str).eq("single")]
        for (subject, behavior), group in df.groupby(["sbj", "behavior"], observed=False):
            configs = sorted(group["config"].dropna().astype(str).unique())
            if configs:
                record(
                    (str(subject), str(behavior), n_channels),
                    run_dir_from_config(configs[-1]),
                    path,
                )

    full_df = pd.read_csv(full_history)
    full_df = full_df[
        full_df["model_name"].astype(str).eq(full_model_name)
    ].copy()
    if full_df.empty:
        raise ValueError(
            f"{full_history} has no rows for model_name={full_model_name!r}"
        )
    if "eval_type" in full_df.columns:
        full_df = full_df[full_df["eval_type"].astype(str).eq("single")]
    for (subject, behavior), group in full_df.groupby(["sbj", "behavior"], observed=False):
        configs = sorted(group["config"].dropna().astype(str).unique())
        if configs:
            record(
                (str(subject), str(behavior), 128),
                run_dir_from_config(configs[-1]),
                full_history,
            )
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pseudo-online manifest rows for 4/8/16/32/128ch EEGNet models."
    )
    parser.add_argument(
        "--online-manifest",
        type=Path,
        default=Path("scripts/pseudo_online_test/online_data_manifest.csv"),
    )
    parser.add_argument(
        "--channel-history-dir",
        type=Path,
        default=Path("outputs/rotating/channel_decimation_EEGNet"),
    )
    parser.add_argument(
        "--full-history",
        type=Path,
        default=Path("outputs/rotating/baseline/history_color_rotating_test_fold_test.csv"),
    )
    parser.add_argument(
        "--full-model-name",
        default="EEGNet",
        help="Model name stored in --full-history (seeded runs use EEGNet_with_mask).",
    )
    parser.add_argument(
        "--full-model-alias",
        default="EEGNet_128ch",
        help="Result label for the 128-channel model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/pseudo_online_test/channel_decimation_online_data_manifest.csv"),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Restrict output to these condition identifiers.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        choices=(4, 8, 16, 32, 128),
        default=[4, 8, 16, 32, 128],
        help="Restrict output to these channel densities; default is unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    online_rows = load_online_manifest(args.online_manifest)
    if args.conditions is not None:
        requested = set(args.conditions)
        known = {row["condition"] for row in online_rows}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(f"Unknown online conditions: {missing}")
        online_rows = [row for row in online_rows if row["condition"] in requested]
    history_index = build_history_index(
        args.channel_history_dir,
        args.full_history,
        args.full_model_name,
    )
    out_rows: list[dict[str, str]] = []
    channels = args.channels
    if channels != sorted(set(channels)):
        raise ValueError("--channels must be unique and increasing")

    for online_row in online_rows:
        condition = online_row["condition"]
        subject, behavior = condition_subject_behavior(condition)
        for n_channels in channels:
            run_dir = history_index.get((subject, behavior, n_channels))
            if run_dir is None:
                raise FileNotFoundError(
                    f"Missing run_dir for condition={condition}, behavior={behavior}, "
                    f"n_channels={n_channels}"
                )
            model = args.full_model_name if n_channels == 128 else "EEGNet_with_mask"
            model_alias = (
                args.full_model_alias
                if n_channels == 128
                else f"EEGNet_with_mask_{n_channels}ch"
            )
            out_row = {
                "condition": condition,
                "csv_dir": online_row["csv_dir"],
                "npy_dir": online_row["npy_dir"],
                "csv_header": online_row["csv_header"],
                "online_label": online_row.get("online_label", "online50"),
                "model": model,
                "run_dir": run_dir,
                "model_alias": model_alias,
                "n_channels": str(n_channels),
            }
            out_rows.append(out_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "csv_dir",
        "npy_dir",
        "csv_header",
        "online_label",
        "model",
        "run_dir",
        "model_alias",
        "n_channels",
    ]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {args.output} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
