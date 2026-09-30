#!/usr/bin/env python3
"""Compute Integrated Gradients for manuscript decoder runs.

This is the script version of the IG calculation cells in
the contribution analysis notebook, adapted for the manuscript data aliases and
state_dict-based model checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "integrated_gradients"
DEFAULT_BATCH_MANIFEST = REPO_ROOT / "scripts/figures" / "manifests" / "ig_from_online_data_manifest.csv"
DEFAULT_SUMMARY_DIRS = [
    REPO_ROOT / "outputs" / "online" / "baseline",
    REPO_ROOT / "outputs" / "online" / "eeg_wo_adapt_filt",
    REPO_ROOT / "outputs" / "online" / "emg_only",
]
DEFAULT_SUBJECTS = [f"sub-{i}" for i in range(1, 10)]
DEFAULT_MODELS = ["EEGNet", "EEGNet_wo_adapt_filt", "EMG_EEGNet"]


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Hydra run directory containing .hydra/config.yaml and model weights.",
    )
    parser.add_argument(
        "--code-root",
        type=Path,
        default=None,
        help="Repository root providing the uhd_eeg package (defaults to this repo).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV with condition,csv_dir,npy_dir,csv_header. "
            "Use with --condition to compute IG on pseudo-online data."
        ),
    )
    parser.add_argument(
        "--condition",
        default=None,
        help="Manifest condition row to use, e.g. a BIDS condition key.",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help=(
            "Compute every matching manifest row. Requires manifest rows to contain run_dir. "
            "If --manifest is omitted, the IG manifest under scripts/figures/manifests is used."
        ),
    )
    parser.add_argument(
        "--summary-dirs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Pseudo-online summary directories containing JSON files with run_dir/source_csv/"
            "source_npy_dir. Only used when explicitly provided. If used without paths, "
            "the three legacy default summary dirs are used."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=DEFAULT_SUBJECTS,
        help="Subjects to include with --all-conditions (default: sub-1 ... sub-9).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Manifest model names to include with --all-conditions. Defaults to EEGNet.",
    )
    parser.add_argument("--csv-dir", type=Path, default=None, help="Override args.gmail.csv_dir.")
    parser.add_argument("--npy-dir", type=Path, default=None, help="Override args.gmail.npy_dir.")
    parser.add_argument("--csv-header", default=None, help="Override args.gmail.csv_header.")
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU id in loaded config.")
    parser.add_argument("--batch-size", type=int, default=1, help="DataLoader batch size.")
    parser.add_argument("--cvs", nargs="*", type=int, default=None, help="CV indices to process.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where IG tensors and metadata are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even when the output file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and print the planned jobs without computing IG.",
    )
    return parser.parse_args()


def choose_code_root(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    return REPO_ROOT


def setup_imports(code_root: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(code_root))


def condition_subject_behavior(condition: str) -> tuple[str, str]:
    from scripts.leave_test._conditions import behavior_from_run_key, subject_from_run_key

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


def load_manifest_row(path: Path, condition: str) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [row for row in rows if row.get("condition") == condition]
    if len(matches) != 1:
        raise ValueError(f"{path} has {len(matches)} rows for condition={condition!r}")
    return matches[0]


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"condition", "csv_dir", "npy_dir", "csv_header"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def filter_manifest_rows(
    rows: list[dict[str, str]],
    subjects: list[str],
    models: list[str],
) -> list[dict[str, str]]:
    subject_set = set(subjects)
    model_set = set(models)
    selected = []
    for row in rows:
        condition = row.get("condition", "")
        subject, _ = condition_subject_behavior(condition)
        model = row.get("model", "")
        if subject not in subject_set:
            continue
        if model and model not in model_set:
            continue
        if "run_dir" not in row or not row["run_dir"]:
            raise ValueError(
                "Batch IG requires a non-empty run_dir in each selected manifest row; "
                f"condition={condition!r}"
            )
        selected.append(row)
    return selected


def csv_header_from_source_csv(source_csv: str) -> tuple[str, str]:
    path = Path(source_csv)
    stem = path.stem
    prefix = "word_list"
    if not stem.startswith(prefix):
        raise ValueError(f"source_csv must be a word_list*.csv file: {source_csv}")
    return str(path.parent), stem.removeprefix(prefix)


def load_summary_rows(
    summary_dirs: list[Path],
    subjects: list[str],
    models: list[str],
) -> list[dict[str, str]]:
    subject_set = set(subjects)
    model_set = set(models)
    rows = []
    for summary_dir in summary_dirs:
        for path in sorted(summary_dir.glob("*.json")):
            with path.open() as f:
                summary = json.load(f)
            condition = str(summary.get("condition", ""))
            subject, _ = condition_subject_behavior(condition)
            model_name = str(summary.get("model_name", ""))
            if subject not in subject_set or model_name not in model_set:
                continue
            if not summary.get("run_dir"):
                raise ValueError(f"{path} is missing run_dir")
            if not summary.get("source_csv") or not summary.get("source_npy_dir"):
                raise ValueError(f"{path} is missing source_csv/source_npy_dir")
            csv_dir, csv_header = csv_header_from_source_csv(str(summary["source_csv"]))
            rows.append(
                {
                    "condition": condition,
                    "model": model_name,
                    "run_dir": str(summary["run_dir"]),
                    "csv_dir": csv_dir,
                    "npy_dir": str(summary["source_npy_dir"]),
                    "csv_header": csv_header,
                    "summary_json": str(path),
                }
            )
    return rows


def load_config(run_dir: Path):
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing Hydra config: {config_path}")
    args = OmegaConf.load(config_path)
    OmegaConf.set_struct(args, False)
    args.use_hydra_savedir = False
    if "gmail" not in args:
        if "parallel_sets" not in args or args.parallel_sets not in args:
            raise KeyError(
                f"Cannot reconstruct args.gmail from {config_path}; "
                "missing gmail and parallel_sets entry."
            )
        args.gmail = OmegaConf.create(OmegaConf.to_container(args[args.parallel_sets], resolve=True))
    if "behavior" not in args.gmail:
        _, behavior = condition_subject_behavior(str(args.parallel_sets))
        args.gmail.behavior = behavior
    return args


def apply_data_overrides(args, cli: argparse.Namespace) -> None:
    if cli.manifest is not None and cli.condition is not None:
        row = load_manifest_row(cli.manifest, cli.condition)
        apply_manifest_row(args, row, cli.condition)
    apply_runtime_overrides(args, cli)


def apply_manifest_row(args, row: dict[str, str], condition: str) -> None:
    args.gmail.csv_dir = row["csv_dir"]
    args.gmail.npy_dir = row["npy_dir"]
    args.gmail.csv_header = row["csv_header"]
    _, behavior = condition_subject_behavior(condition)
    args.gmail.behavior = behavior
    args.ig_condition = condition
    args.ig_model_label = row.get("model", str(args.model_name))
    if "summary_json" in row:
        args.ig_summary_json = row["summary_json"]


def apply_runtime_overrides(args, cli: argparse.Namespace) -> None:
    if cli.csv_dir is not None:
        args.gmail.csv_dir = str(cli.csv_dir)
    if cli.npy_dir is not None:
        args.gmail.npy_dir = str(cli.npy_dir)
    if cli.csv_header is not None:
        args.gmail.csv_header = cli.csv_header
    if cli.gpu is not None:
        args.gpu = cli.gpu
    elif "gpu" not in args:
        args.gpu = 0
    args.batch_size = cli.batch_size
    args.n_worekers = 0


def model_filename(args, cv: int) -> str:
    if str(args.model_name) == "CovTanSVM":
        return f"CovTanSVM_{args.config_name}_N{args.n_trial_avg}_cv{cv}.dill"
    return f"model_weight_{args.config_name}_N{args.n_trial_avg}_cv{cv}.pth"


def build_dataset(args):
    from uhd_eeg.datasets.DatasetUHD import EEGDataset, EMGDataset

    if str(args.decode_from).lower() == "emg":
        return EMGDataset(args)
    return EEGDataset(args)


def load_model(args, dataset, model_path: Path, device: torch.device):
    import torch

    if str(args.model_name) == "CovTanSVM":
        raise NotImplementedError("Integrated Gradients is not available for CovTanSVM.")

    from uhd_eeg.trainers.trainer_within_offline_split import build_model

    model = build_model(args, dataset.window_eegnet)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def prepare_for_ig(args, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    if args.no_avg:
        inputs = torch.concat([inp for inp in inputs])
        labels = labels.repeat_interleave(args.n_trial_avg)
    if str(args.model_name) == "RNN":
        inputs = inputs[:, 0, :, :].permute(0, 2, 1)
    return inputs, labels


def baseline_like(args, inputs: torch.Tensor) -> torch.Tensor:
    if str(args.model_name) == "RNN":
        return inputs.mean(dim=1, keepdim=True).expand(inputs.shape)
    return inputs.mean(dim=-1, keepdim=True).expand(inputs.shape)


def selected_trial_metadata(args, dataset) -> list[dict[str, object]]:
    labels = np.asarray(dataset.labels_all, dtype=int)
    trial_files = list(getattr(dataset, "trial_files", [None] * len(labels)))
    rows = []
    for dataset_index, (label, trial_file) in enumerate(zip(labels, trial_files)):
        source_trial = dataset_index
        trial_file_text = ""
        if trial_file is not None:
            trial_file = Path(trial_file)
            trial_file_text = str(trial_file)
            try:
                source_trial = int(trial_file.stem)
            except ValueError:
                source_trial = dataset_index
        if args.no_avg:
            for inner_trial in range(int(args.n_trial_avg)):
                rows.append(
                    {
                        "dataset_index": dataset_index,
                        "source_trial": source_trial,
                        "inner_trial": inner_trial,
                        "trial_file": trial_file_text,
                        "label": int(label),
                    }
                )
        else:
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "source_trial": source_trial,
                    "inner_trial": "",
                    "trial_file": trial_file_text,
                    "label": int(label),
                }
            )
    return rows


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    import torch

    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError(f"Expected model output as (batch, n_class), got {tuple(logits.shape)}")
    return torch.softmax(logits, dim=-1)


def write_trial_predictions_csv(
    path: Path,
    rows: list[dict[str, object]],
    labels: torch.Tensor,
    preds: torch.Tensor,
    probabilities: torch.Tensor,
) -> None:
    labels_np = labels.cpu().numpy().astype(int)
    preds_np = preds.cpu().numpy().astype(int)
    probs_np = probabilities.cpu().numpy()
    if len(rows) != len(labels_np):
        raise ValueError(f"metadata rows ({len(rows)}) != labels ({len(labels_np)})")

    fieldnames = [
        "ig_index",
        "dataset_index",
        "source_trial",
        "inner_trial",
        "trial_file",
        "label",
        "pred",
        "correct",
        "pred_probability",
    ] + [f"prob_{idx}" for idx in range(probs_np.shape[1])]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ig_index, row in enumerate(rows):
            label = int(labels_np[ig_index])
            pred = int(preds_np[ig_index])
            out_row = {
                "ig_index": ig_index,
                "dataset_index": row["dataset_index"],
                "source_trial": row["source_trial"],
                "inner_trial": row["inner_trial"],
                "trial_file": row["trial_file"],
                "label": label,
                "pred": pred,
                "correct": pred == label,
                "pred_probability": float(probs_np[ig_index, pred]),
            }
            for prob_index in range(probs_np.shape[1]):
                out_row[f"prob_{prob_index}"] = float(probs_np[ig_index, prob_index])
            writer.writerow(out_row)


def compute_ig_for_cv(args, dataset, model_path: Path, device: torch.device, batch_size: int):
    import torch
    from torch.utils.data import DataLoader

    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError("captum is required to compute Integrated Gradients") from exc

    model = load_model(args, dataset, model_path, device)
    ig = IntegratedGradients(model)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False)

    all_igs = []
    all_labels = []
    all_deltas = []
    all_probabilities = []
    all_preds = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs, labels = prepare_for_ig(args, inputs, labels)
        with torch.no_grad():
            probabilities = logits_to_probabilities(model(inputs))
            preds = probabilities.argmax(dim=-1)
        attributions, delta = ig.attribute(
            inputs,
            target=labels.long(),
            baselines=baseline_like(args, inputs),
            method="gausslegendre",
            return_convergence_delta=True,
        )
        all_igs.append(attributions.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_deltas.append(delta.detach().cpu())
        all_probabilities.append(probabilities.detach().cpu())
        all_preds.append(preds.detach().cpu())

    return (
        torch.cat(all_igs, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_deltas, dim=0),
        torch.cat(all_preds, dim=0),
        torch.cat(all_probabilities, dim=0),
    )


def compute_predictions_for_cv(args, dataset, model_path: Path, device: torch.device, batch_size: int):
    import torch
    from torch.utils.data import DataLoader

    model = load_model(args, dataset, model_path, device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False)
    all_labels = []
    all_probabilities = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs, labels = prepare_for_ig(args, inputs, labels)
            probabilities = logits_to_probabilities(model(inputs))
            preds = probabilities.argmax(dim=-1)
            all_labels.append(labels.detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())
            all_preds.append(preds.detach().cpu())
    return (
        torch.cat(all_labels, dim=0),
        torch.cat(all_preds, dim=0),
        torch.cat(all_probabilities, dim=0),
    )


def output_stem(args, run_dir: Path, cv: int) -> str:
    condition = str(args.get("ig_condition", args.get("parallel_sets", "unknown")))
    behavior = str(args.gmail.get("behavior", "unknown"))
    model_label = str(args.get("ig_model_label", args.model_name))
    return f"{condition}_{model_label}_{run_dir.parent.name}_{run_dir.name}_{behavior}_cv{cv}"


def compute_for_run(
    cli: argparse.Namespace,
    code_root: Path,
    run_dir: Path,
    manifest_row: dict[str, str] | None = None,
) -> None:
    args = load_config(run_dir)
    if manifest_row is not None:
        apply_manifest_row(args, manifest_row, manifest_row["condition"])
        apply_runtime_overrides(args, cli)
    else:
        apply_data_overrides(args, cli)

    cvs = cli.cvs if cli.cvs is not None else list(range(int(args.n_splits)))
    if cli.dry_run:
        device = f"cuda:{args.gpu} if available else cpu"
    else:
        import torch

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"code_root: {code_root}")
    print(f"run_dir: {run_dir}")
    print(f"model: {args.model_name}")
    print(f"data csv_dir: {args.gmail.csv_dir}")
    print(f"data npy_dir: {args.gmail.npy_dir}")
    print(f"device: {device}")
    print(f"cvs: {cvs}")

    if cli.dry_run:
        for cv in cvs:
            print(run_dir / model_filename(args, cv))
        return

    dataset = build_dataset(args)
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    for cv in cvs:
        model_path = run_dir / model_filename(args, cv)
        if not model_path.exists():
            raise FileNotFoundError(f"missing model checkpoint: {model_path}")

        stem = output_stem(args, run_dir, cv)
        ig_path = cli.output_dir / f"{stem}_igs.pt"
        labels_path = cli.output_dir / f"{stem}_labels.pt"
        delta_path = cli.output_dir / f"{stem}_convergence_delta.pt"
        pred_path = cli.output_dir / f"{stem}_pred_labels.pt"
        prob_path = cli.output_dir / f"{stem}_probabilities.pt"
        trial_csv_path = cli.output_dir / f"{stem}_trial_predictions.csv"
        ig_outputs_exist = ig_path.exists() and labels_path.exists() and delta_path.exists()
        prediction_outputs_exist = pred_path.exists() and prob_path.exists() and trial_csv_path.exists()
        if (
            ig_outputs_exist
            and prediction_outputs_exist
            and not cli.overwrite
        ):
            print(f"skip existing: {ig_path}")
            continue
        if ig_outputs_exist and not prediction_outputs_exist and not cli.overwrite:
            print(f"computing missing predictions for cv{cv}: {model_path}")
            labels, preds, probabilities = compute_predictions_for_cv(
                args, dataset, model_path, device, cli.batch_size
            )
            trial_rows = selected_trial_metadata(args, dataset)
            torch.save(preds, pred_path)
            torch.save(probabilities, prob_path)
            write_trial_predictions_csv(trial_csv_path, trial_rows, labels, preds, probabilities)
            print(f"saved: {trial_csv_path}")
            continue

        print(f"computing cv{cv}: {model_path}")
        igs, labels, deltas, preds, probabilities = compute_ig_for_cv(
            args, dataset, model_path, device, cli.batch_size
        )
        trial_rows = selected_trial_metadata(args, dataset)
        torch.save(igs, ig_path)
        torch.save(labels, labels_path)
        torch.save(deltas, delta_path)
        torch.save(preds, pred_path)
        torch.save(probabilities, prob_path)
        write_trial_predictions_csv(trial_csv_path, trial_rows, labels, preds, probabilities)
        print(f"saved: {ig_path}")
        print(f"saved: {trial_csv_path}")


def main() -> None:
    cli = parse_args()
    code_root = choose_code_root(cli.code_root)
    setup_imports(code_root)

    if cli.all_conditions:
        if cli.summary_dirs is None:
            manifest = cli.manifest or DEFAULT_BATCH_MANIFEST
            rows = filter_manifest_rows(load_manifest_rows(manifest), cli.subjects, cli.models)
            summary_dirs = []
        elif cli.summary_dirs:
            summary_dirs = cli.summary_dirs
            rows = load_summary_rows(summary_dirs, cli.subjects, cli.models)
        else:
            summary_dirs = DEFAULT_SUMMARY_DIRS
            rows = load_summary_rows(summary_dirs, cli.subjects, cli.models)
        print(f"code_root: {code_root}")
        if summary_dirs:
            print(f"summary_dirs: {[str(path) for path in summary_dirs]}")
        else:
            print(f"manifest: {manifest}")
        print(f"subjects: {cli.subjects}")
        print(f"models: {cli.models}")
        print(f"jobs: {len(rows)}")
        for row in rows:
            print(f"job: {row['condition']} {row.get('model', '')} {row['run_dir']}")
            compute_for_run(cli, code_root, Path(row["run_dir"]).resolve(), row)
        return

    if cli.run_dir is None:
        raise ValueError("--run-dir is required unless --all-conditions is used")
    if cli.manifest is not None and cli.condition is None:
        raise ValueError("--condition is required when --manifest is provided without --all-conditions")
    compute_for_run(cli, code_root, cli.run_dir.resolve())


if __name__ == "__main__":
    main()
