"""Fig. 3b: publication-style temporal IG paper-trace figure.

Published panel (author-confirmed)::

    sub-6, overt, dissimilar, orange, source trial 33, CV 0, top1,
    averaged mode, emg-highpass 60, ig-steps 32, dpi 600, amp-bar-z 2,
    time-bar-sec 0.1, spacing/height scale 2 (the former ``_spacing2x`` layout)

Integrated Gradients are loaded from ``--ig-dir`` / ``--precomputed-window-ig-dir``
when available. Full per-window recompute requires ``--checkpoint-root`` (not
needed for the averaged paper-trace layout when precomputed IGs exist).

Example
-------
::

    uv run python scripts/figures/make_temporal_ig_paper_trace.py \\
      --ig-dir outputs/integrated_gradients \\
      --precomputed-window-ig-dir outputs/temporal_ig_windows
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.figures._temporal_ig_helpers import (
        CM_TO_INCH,
        DEFAULT_CANDIDATES_CSV,
        TRACE_COLORS,
        CandidateRecord,
        PreprocessArgs,
        TrialPreprocessor,
        candidate_from_cli,
        compute_model_window_igs,
        discover_model_output_records,
        load_candidate_records,
        load_precomputed_window_igs,
        load_trial_averaged_ig_as_windows,
        preprocess_emg,
        split_online_inner_trials,
        temporal_ig_trace_set,
        top_channel_sets,
        trim_to_online_duration,
    )
except ImportError:  # pragma: no cover
    from _temporal_ig_helpers import (  # type: ignore[no-redef]
        CM_TO_INCH,
        DEFAULT_CANDIDATES_CSV,
        TRACE_COLORS,
        CandidateRecord,
        PreprocessArgs,
        TrialPreprocessor,
        candidate_from_cli,
        compute_model_window_igs,
        discover_model_output_records,
        load_candidate_records,
        load_precomputed_window_igs,
        load_trial_averaged_ig_as_windows,
        preprocess_emg,
        split_online_inner_trials,
        temporal_ig_trace_set,
        top_channel_sets,
        trim_to_online_duration,
    )


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "temporal_ig_trial_examples" / "paper_figures"
)
JACCARD_LABELS = {
    "Denoised EEG": "denoised\nEEG",
    "min preprocessed EEG": "min pre-\nprocessed EEG",
    "EOG": "EOG",
    "EMG upper": "EMG upper",
    "EMG lower": "EMG lower",
}
JACCARD_LABEL_COLORS = {
    "Denoised EEG": "#2b8cbe",
    "min preprocessed EEG": "#d8b365",
    "EOG": "#1a9850",
    "EMG upper": "#b2182b",
    "EMG lower": "#7b3294",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "integrated_gradients",
        help="Directory with paired *_igs.pt and *_trial_predictions.csv.",
    )
    parser.add_argument(
        "--ig-dir",
        type=Path,
        default=None,
        help="Alias for --prediction-dir (precomputed IG tensors).",
    )
    parser.add_argument(
        "--precomputed-window-ig-dir",
        type=Path,
        default=None,
        help="Optional directory with per-window IG .npy/.npz for the example trial.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Optional Hydra outputs root for IG recompute (normally omitted).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument(
        "--condition",
        default="sub-6-overt",
        help="Condition key (BIDS-style). Legacy subject6-overt is accepted.",
    )
    parser.add_argument("--task", default="overt")
    parser.add_argument("--selection-type", default="dissimilar")
    parser.add_argument("--color", default="orange")
    parser.add_argument("--source-trial", default="33")
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--trial-file", type=Path, default=None)
    parser.add_argument("--top-name", choices=["top1", "top10"], default="top1")
    parser.add_argument("--mode", choices=["averaged", "inner", "full"], default="averaged")
    parser.add_argument("--inner-index", type=int, default=1)
    parser.add_argument("--emg-highpass", type=float, default=60.0)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--time-bar-sec", type=float, default=0.1)
    parser.add_argument("--amp-bar-z", type=float, default=2.0)
    parser.add_argument(
        "--spacing-scale",
        type=float,
        default=2.0,
        help=(
            "Vertical spacing multiplier between traces. Default 2.0 matches the "
            "published Fig. 3b layout (formerly the _spacing2x export)."
        ),
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=2.0,
        help=(
            "Figure-height multiplier. Default 2.0 matches the published Fig. 3b "
            "layout (formerly the _spacing2x export)."
        ),
    )
    parser.add_argument("--jaccard-threshold", type=float, default=0.0)
    parser.add_argument(
        "--skip-jaccard",
        action="store_true",
        help="Only write the paper-trace figure (skip 5x5 Jaccard matrices).",
    )
    return parser.parse_args()


def select_record(args: argparse.Namespace) -> CandidateRecord:
    if args.candidates_csv.exists():
        for record in load_candidate_records(args.candidates_csv, args.rank):
            if (
                record.condition.replace("subject", "sub-") == args.condition.replace("subject", "sub-")
                or record.condition == args.condition
            ) and (
                record.task == args.task
                and record.selection_type == args.selection_type
                and record.color == args.color
                and record.source_trial == str(args.source_trial)
                and record.cv == args.cv
            ):
                return record
        # Fall through to CLI record if CSV exists but has no match.
    return candidate_from_cli(
        condition=args.condition,
        task=args.task,
        selection_type=args.selection_type,
        color=args.color,
        source_trial=args.source_trial,
        cv=args.cv,
        trial_file=args.trial_file,
        label=args.label,
        rank=args.rank,
    )


def _resolve_window_igs(
    args: argparse.Namespace,
    record: CandidateRecord,
    model: str,
    windows: np.ndarray | None,
) -> np.ndarray:
    prediction_dir = args.ig_dir or args.prediction_dir
    search_dirs = []
    if args.precomputed_window_ig_dir is not None:
        search_dirs.append(args.precomputed_window_ig_dir)
    search_dirs.append(prediction_dir)

    for directory in search_dirs:
        if directory is None or not directory.exists():
            continue
        loaded = load_precomputed_window_igs(directory, record, model)
        if loaded is not None:
            return loaded

    averaged = load_trial_averaged_ig_as_windows(prediction_dir, record, model)
    if averaged is not None:
        if args.mode != "averaged":
            print(
                f"warning: only averaged trial IG found for {model}; "
                f"--mode {args.mode} will approximate from repeated averaged windows"
            )
        return averaged

    if args.checkpoint_root is None or windows is None:
        raise FileNotFoundError(
            f"No precomputed window IG for model={model}, condition={record.condition}, "
            f"task={record.task}, cv={record.cv}, trial={record.source_trial}. "
            "Provide --precomputed-window-ig-dir / --ig-dir with saved attributions, "
            "or --checkpoint-root plus a --trial-file for recompute."
        )

    model_records = discover_model_output_records(prediction_dir)
    key = (record.condition, record.task, record.cv, model)
    model_record = model_records.get(key)
    if model_record is None:
        raise FileNotFoundError(
            f"Missing model output metadata for {key} under {prediction_dir}"
        )
    return compute_model_window_igs(
        model_record,
        windows,
        record.label,
        args.ig_steps,
        args.device,
        checkpoint_root=args.checkpoint_root,
    )


def compute_traces(args: argparse.Namespace, record: CandidateRecord):
    preprocessor = TrialPreprocessor(PreprocessArgs(jitter_buffer_sec=0.0))
    prediction_dir = args.ig_dir or args.prediction_dir

    windows_denoised = windows_min = windows_emg = None
    if record.trial_file and Path(record.trial_file).exists():
        eeg = np.load(record.trial_file)
        plot_eeg = trim_to_online_duration(eeg, preprocessor.args)
        minimally_preprocessed, denoised = preprocessor.preprocess(plot_eeg)
        emg_z = preprocess_emg(plot_eeg, preprocessor.args, args.emg_highpass)
        windows_denoised = split_online_inner_trials(denoised, preprocessor.args)
        windows_min = split_online_inner_trials(minimally_preprocessed, preprocessor.args)
        windows_emg = split_online_inner_trials(emg_z, preprocessor.args)

    eegnet_igs = _resolve_window_igs(args, record, "EEGNet", windows_denoised)
    wo_igs = _resolve_window_igs(args, record, "EEGNet_wo_adapt_filt", windows_min)
    emg_igs = _resolve_window_igs(args, record, "EMG_EEGNet", windows_emg)

    eegnet_channels = top_channel_sets(eegnet_igs)[args.top_name]
    wo_channels = top_channel_sets(wo_igs)[args.top_name]
    inner_index = args.inner_index - 1 if args.mode == "inner" else None
    traces = temporal_ig_trace_set(
        eegnet_igs,
        wo_igs,
        emg_igs,
        eegnet_channels,
        wo_channels,
        mode=args.mode,
        inner_index=inner_index,
        trace_transform="abs-zscore",
    )
    ig_sets = {
        "Denoised EEG": eegnet_igs,
        "min preprocessed EEG": wo_igs,
        "EOG": emg_igs,
        "EMG upper": emg_igs,
        "EMG lower": emg_igs,
    }
    computed_channel_sets = {
        "top1": {
            "Denoised EEG": top_channel_sets(eegnet_igs)["top1"],
            "min preprocessed EEG": top_channel_sets(wo_igs)["top1"],
            "EOG": np.array([0]),
            "EMG upper": np.array([1]),
            "EMG lower": np.array([2]),
        },
        "top10": {
            "Denoised EEG": top_channel_sets(eegnet_igs)["top10"],
            "min preprocessed EEG": top_channel_sets(wo_igs)["top10"],
            "EOG": np.array([0]),
            "EMG upper": np.array([1]),
            "EMG lower": np.array([2]),
        },
    }
    channel_sets = {
        top_name: temporal_plot_channel_set(args, record, top_name, fallback)
        for top_name, fallback in computed_channel_sets.items()
    }
    _ = prediction_dir
    return traces, preprocessor.args.fs, eegnet_channels, wo_channels, ig_sets, channel_sets


def temporal_plot_channel_set(
    args: argparse.Namespace,
    record: CandidateRecord,
    top_name: str,
    fallback: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    filename = (
        f"{record.condition}_{record.task}_{record.selection_type}_"
        f"label{record.label}_{record.color}_trial{record.source_trial}_cv{record.cv}_"
        f"averaged_{top_name}.channels.txt"
    )
    metadata_path = (
        args.output_dir.parent / record.task / record.selection_type / record.color / filename
    )
    if not metadata_path.exists():
        return fallback
    parsed = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    if "eegnet_channels" not in parsed or "eegnet_wo_adapt_filt_channels" not in parsed:
        return fallback
    return {
        "Denoised EEG": np.array(
            [int(value) for value in parsed["eegnet_channels"].split("-") if value],
            dtype=int,
        ),
        "min preprocessed EEG": np.array(
            [
                int(value)
                for value in parsed["eegnet_wo_adapt_filt_channels"].split("-")
                if value
            ],
            dtype=int,
        ),
        "EOG": np.array([0]),
        "EMG upper": np.array([1]),
        "EMG lower": np.array([2]),
    }


def draw_scale_bar(
    ax,
    *,
    x_end: float,
    y0: float,
    time_bar_sec: float,
    amp_bar_z: float,
    linewidth: float,
) -> None:
    x0 = x_end - time_bar_sec
    ax.plot([x0, x_end], [y0, y0], color="black", linewidth=linewidth, solid_capstyle="butt")
    ax.plot(
        [x_end, x_end],
        [y0, y0 + amp_bar_z],
        color="black",
        linewidth=linewidth,
        solid_capstyle="butt",
    )
    ax.text(
        (x0 + x_end) / 2,
        y0 - 0.35,
        f"{int(round(time_bar_sec * 1000))} ms",
        ha="center",
        va="top",
        fontsize=6,
        color="black",
    )
    ax.text(
        x_end + 0.018,
        y0 + amp_bar_z / 2,
        f"{amp_bar_z:g} z",
        ha="left",
        va="center",
        fontsize=6,
        color="black",
        rotation=90,
    )


def save_trace_figure(
    args: argparse.Namespace,
    record: CandidateRecord,
    traces: dict[str, np.ndarray],
    fs: int,
    eegnet_channels: np.ndarray,
    wo_channels: np.ndarray,
    spacing_scale: float = 2.0,
    height_scale: float = 2.0,
    filename_extra: str = "",
) -> list[Path]:
    n_times = next(iter(traces.values())).shape[-1]
    t = np.arange(n_times) / fs
    spans = [float(np.nanmax(trace) - np.nanmin(trace)) for trace in traces.values()]
    max_span = max(spans) if spans else 1.0
    offset_step = max(max_span, 1.0e-12) * 1.32 * spacing_scale
    offsets = np.arange(len(traces) - 1, -1, -1, dtype=float) * offset_step

    fig, ax = plt.subplots(figsize=(8.4 * CM_TO_INCH, 4.2 * height_scale * CM_TO_INCH))
    all_y = []
    for offset, (name, trace) in zip(offsets, traces.items()):
        y = trace + offset
        all_y.append(y)
        ax.plot(
            t,
            y,
            color=TRACE_COLORS.get(name, "black"),
            linewidth=0.85,
            solid_joinstyle="round",
            solid_capstyle="round",
        )
        ax.text(
            -0.045,
            offset,
            name,
            color=TRACE_COLORS.get(name, "black"),
            fontsize=6,
            ha="right",
            va="center",
            clip_on=False,
        )

    y_values = np.concatenate(all_y)
    trace_min = float(np.nanmin(y_values))
    trace_max = float(np.nanmax(y_values))
    bar_y = trace_min - args.amp_bar_z - 0.42 * offset_step
    draw_scale_bar(
        ax,
        x_end=float(t[-1] - 0.04),
        y0=bar_y,
        time_bar_sec=args.time_bar_sec,
        amp_bar_z=args.amp_bar_z,
        linewidth=0.9,
    )

    ax.set_xlim(float(t[0] - 0.06), float(t[-1] + 0.14))
    ax.set_ylim(float(bar_y - 0.62), float(trace_max + 0.45))
    ax.axis("off")

    filename = (
        f"{record.condition}_{record.task}_{record.selection_type}_"
        f"label{record.label}_{record.color}_trial{record.source_trial}_cv{record.cv}_"
        f"{args.mode}_{args.top_name}_paper_trace_scalebar{filename_extra}"
    )
    save_dir = args.output_dir / record.task / record.selection_type / record.color
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{filename}.png"
    pdf_path = save_dir / f"{filename}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    metadata_path = save_dir / f"{filename}.txt"
    metadata_path.write_text(
        "\n".join(
            [
                "figure_style=publication_trace_no_axes",
                "trace_transform=abs(raw_ig) -> zscore(trace), no clip",
                f"spacing_scale={spacing_scale:g}",
                f"height_scale={height_scale:g}",
                f"time_scale_bar_sec={args.time_bar_sec:g}",
                f"amplitude_scale_bar_z={args.amp_bar_z:g}",
                f"sampling_rate_hz={fs}",
                f"eegnet_channels={'-'.join(map(str, eegnet_channels.tolist()))}",
                f"eegnet_wo_adapt_filt_channels={'-'.join(map(str, wo_channels.tolist()))}",
                "emg_eegnet_channels=EOG:0,EMG_upper:1,EMG_lower:2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [png_path, pdf_path, metadata_path]


def temporal_waveforms_for_top(
    ig_sets: dict[str, np.ndarray],
    channel_set: dict[str, np.ndarray],
) -> tuple[list[str], dict[str, np.ndarray]]:
    names = ["Denoised EEG", "min preprocessed EEG", "EOG", "EMG upper", "EMG lower"]
    waveforms = {}
    for name in names:
        igs = np.abs(ig_sets[name])
        if igs.ndim == 4 and igs.shape[1] == 1:
            igs = igs[:, 0]
        if igs.ndim != 3:
            raise ValueError(f"expected IG shape (window, channel, time), got {igs.shape}")
        if igs.shape[0] != 5:
            raise ValueError(f"expected 5 inner windows, got {igs.shape[0]}")
        channels = np.asarray(channel_set[name], dtype=int)
        waveform = igs[:, channels, :].mean(axis=(0, 1))
        std = float(np.nanstd(waveform))
        if std == 0.0 or np.isnan(std):
            waveforms[name] = np.zeros_like(waveform)
        else:
            waveforms[name] = (waveform - float(np.nanmean(waveform))) / std
    return names, waveforms


def jaccard_index(left: np.ndarray, right: np.ndarray, threshold: float) -> float:
    left_mask = left > threshold
    right_mask = right > threshold
    union = int(np.logical_or(left_mask, right_mask).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(left_mask, right_mask).sum() / union)


def jaccard_matrix(
    names: list[str],
    waveforms: dict[str, np.ndarray],
    threshold: float,
) -> np.ndarray:
    matrix = np.eye(len(names), dtype=float)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if i == j:
                continue
            matrix[i, j] = jaccard_index(waveforms[left], waveforms[right], threshold)
    return matrix


def save_jaccard_matrix(
    args: argparse.Namespace,
    record: CandidateRecord,
    top_name: str,
    names: list[str],
    matrix: np.ndarray,
    channel_set: dict[str, np.ndarray],
) -> list[Path]:
    save_dir = args.output_dir / record.task / record.selection_type / record.color
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{record.condition}_{record.task}_{record.selection_type}_"
        f"label{record.label}_{record.color}_trial{record.source_trial}_cv{record.cv}_"
        f"temporal_ig_jaccard_5x5_{top_name}"
    )
    png_path = save_dir / f"{stem}.png"
    pdf_path = save_dir / f"{stem}.pdf"
    csv_path = save_dir / f"{stem}.csv"
    txt_path = save_dir / f"{stem}.txt"

    plot_values = matrix.copy()
    np.fill_diagonal(plot_values, np.nan)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(plot_values, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([JACCARD_LABELS[name] for name in names], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([JACCARD_LABELS[name] for name in names], fontsize=6)
    for tick, name in zip(ax.get_xticklabels(), names):
        tick.set_color(JACCARD_LABEL_COLORS[name])
    for tick, name in zip(ax.get_yticklabels(), names):
        tick.set_color(JACCARD_LABEL_COLORS[name])
    ax.tick_params(length=0)
    ax.set_title(f"{record.task} {record.color} trial{record.source_trial} {top_name}", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard index")
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["signal", *names])
        writer.writeheader()
        for name, row_values in zip(names, matrix):
            writer.writerow(
                {
                    "signal": name,
                    **{other: f"{value:.10g}" for other, value in zip(names, row_values)},
                }
            )

    txt_path.write_text(
        "\n".join(
            [
                "matrix_rows_and_columns=Denoised EEG,min preprocessed EEG,EOG,EMG upper,EMG lower",
                "cell_value=Jaccard index between averaged 1.25 s temporal IG waveforms",
                "waveform=mean over five repetitions, selected channel(s), and abs(raw_ig), then zscore over time",
                f"threshold={args.jaccard_threshold:g}",
                "diagonal_display=white; diagonal_csv=1",
                f"top_name={top_name}",
                "channels="
                + ";".join(
                    f"{name}:{'-'.join(map(str, np.asarray(channels, dtype=int).tolist()))}"
                    for name, channels in channel_set.items()
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [png_path, pdf_path, csv_path, txt_path]


def main() -> None:
    args = parse_args()
    if args.ig_dir is None:
        args.ig_dir = args.prediction_dir
    record = select_record(args)
    traces, fs, eegnet_channels, wo_channels, ig_sets, channel_sets = compute_traces(
        args, record
    )
    # Published Fig. 3b is the spacing/height ×2 layout only (not the 1× draft).
    for path in save_trace_figure(
        args,
        record,
        traces,
        fs,
        eegnet_channels,
        wo_channels,
        spacing_scale=args.spacing_scale,
        height_scale=args.height_scale,
    ):
        print(path)
    if args.skip_jaccard:
        return
    for top_name in ("top1", "top10"):
        names, waveforms = temporal_waveforms_for_top(ig_sets, channel_sets[top_name])
        matrix = jaccard_matrix(names, waveforms, args.jaccard_threshold)
        for path in save_jaccard_matrix(
            args, record, top_name, names, matrix, channel_sets[top_name]
        ):
            print(path)


if __name__ == "__main__":
    main()
