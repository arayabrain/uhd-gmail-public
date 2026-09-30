#!/usr/bin/env python3
"""Fig. S1: adaptive-filter waveforms as stacked traces with scale bars.

Default: sub-7, overt, offline, trial 49.

Data sources (first match wins):
  1. ``--eeg-before-npy`` / ``--eeg-after-npy`` / ``--emg-npy`` precomputed arrays
  2. ``--trial-npy`` raw extracted trial ``(139, n_samples)`` in ADC units
  3. Default OpenNeuro-derived path::

         {output_root}/sub-7/ses-20260520/task-overt_acq-calibration_run-01/trials/049.npy

     where ``output_root`` is set in ``configs/paths.yaml``.

Example
-------
::

    uv run python scripts/figures/plot_adaptive_filter_waveforms_trace.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("MPLBACKEND", "Agg")

import mne
import numpy as np

from scripts.figures._lib.default_plt import cm_to_inch, dark_blue, magenta, plt
from uhd_eeg.bids.constants import SFREQ

# Sibling helper when run as ``python scripts/figures/...``; package import otherwise.
try:
    from scripts.figures._adaptive_filter_waveforms import (
        EMG_COLORS,
        EMG_NAMES,
        load_or_compute_bundle,
        slice_window,
    )
except ImportError:  # pragma: no cover - script invocation
    from _adaptive_filter_waveforms import (  # type: ignore[no-redef]
        EMG_COLORS,
        EMG_NAMES,
        load_or_compute_bundle,
        slice_window,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject",
        default="sub-7",
        help="BIDS subject ID (e.g. sub-7). Default: sub-7.",
    )
    parser.add_argument(
        "--condition",
        default="overt",
        choices=("overt", "minimally_overt", "covert"),
        help="Speech condition. Default: overt.",
    )
    parser.add_argument(
        "--recording-type",
        default="offline",
        choices=("offline", "online"),
        help="Offline maps to acq-calibration; online to acq-online. Default: offline.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="BIDS session id or date (e.g. ses-20260520 or 20260520). "
        "Default: known ses date for --subject/--condition.",
    )
    parser.add_argument(
        "--run",
        default="01",
        help="BIDS run entity (zero-padded). Default: 01.",
    )
    parser.add_argument("--trial", type=int, default=49, help="Trial index to plot.")
    parser.add_argument(
        "--trial-npy",
        type=Path,
        default=None,
        help="Raw extracted trial array (139, n_samples) in ADC units.",
    )
    parser.add_argument(
        "--eeg-before-npy",
        type=Path,
        default=None,
        help="Optional precomputed EEG before adaptive filter (128, n_samples).",
    )
    parser.add_argument(
        "--eeg-after-npy",
        type=Path,
        default=None,
        help="Optional precomputed EEG after adaptive filter (128, n_samples).",
    )
    parser.add_argument(
        "--emg-npy",
        type=Path,
        default=None,
        help="Optional precomputed EMG/EOG array (3, n_samples).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override derived-data root (default: configs/paths.yaml output_root).",
    )
    parser.add_argument(
        "--affected-channel",
        type=int,
        default=64,
        help="0-based EEG channel expected to be strongly affected by filtering.",
    )
    parser.add_argument(
        "--minimal-channel",
        type=int,
        default=47,
        help="0-based EEG channel expected to be minimally affected by filtering.",
    )
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time to plot.")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Duration to plot. Defaults to the full 5-repeat window.",
    )
    parser.add_argument(
        "--scale-bar-sec",
        type=float,
        default=1.0,
        help="Horizontal scale-bar duration in seconds.",
    )
    parser.add_argument(
        "--scale-bar-value",
        type=float,
        default=2.0,
        help="Vertical scale-bar amplitude in plotted data units.",
    )
    parser.add_argument(
        "--scale-bar-label",
        default="z-score",
        help="Vertical scale-bar unit label.",
    )
    parser.add_argument(
        "--output-suffix",
        default="trace_zscore",
        help="Suffix appended to the output file stem.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figures/figS1_adaptive_filter",
        help="Directory for figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        choices=("png", "pdf", "svg"),
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=SFREQ,
        help=f"Sampling rate in Hz. Default: {SFREQ}.",
    )
    return parser.parse_args()


def trace_rows(bundle, affected_channel: int, minimal_channel: int, win: slice):
    rows = [
        *[
            (name, bundle.emg[idx, win], color)
            for idx, (name, color) in enumerate(zip(EMG_NAMES, EMG_COLORS))
        ],
        (f"Ch {affected_channel} before", bundle.eeg_before[affected_channel, win], dark_blue),
        (f"Ch {affected_channel} after", bundle.eeg_after[affected_channel, win], magenta),
        (f"Ch {minimal_channel} before", bundle.eeg_before[minimal_channel, win], dark_blue),
        (f"Ch {minimal_channel} after", bundle.eeg_after[minimal_channel, win], magenta),
    ]
    return rows


def plot_trace_variant(
    subject: str,
    condition: str,
    recording_type: str,
    trial_index: int,
    time: np.ndarray,
    rows: list[tuple[str, np.ndarray, object]],
    scale_bar_sec: float,
    scale_bar_value: float,
    scale_bar_label: str,
    output_suffix: str,
    output_dir: Path,
    formats: tuple[str, ...],
) -> Path:
    all_values = np.concatenate([signal for _, signal, _ in rows])
    amplitude_limit = float(np.nanmax(np.abs(all_values)))
    if not np.isfinite(amplitude_limit) or amplitude_limit == 0:
        amplitude_limit = 1.0
    amplitude_limit *= 1.05
    row_spacing = amplitude_limit * 2.35
    offsets = np.arange(len(rows))[::-1] * row_spacing

    fig, ax = plt.subplots(figsize=(18 * cm_to_inch, 13 * cm_to_inch), constrained_layout=True)
    for offset, (label, signal, color) in zip(offsets, rows):
        ax.plot(time, signal + offset, color=color, linewidth=0.65)
        ax.text(
            -0.015,
            offset,
            label,
            color="0.15",
            fontsize=7,
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
        )

    duration = float(time[-1] - time[0])
    bar_x0 = max(0.0, duration - scale_bar_sec - 0.25)
    bar_y0 = offsets[-1] - row_spacing * 0.72
    ax.plot(
        [bar_x0, bar_x0 + scale_bar_sec],
        [bar_y0, bar_y0],
        color="0.1",
        linewidth=1.2,
        solid_capstyle="butt",
    )
    ax.text(
        bar_x0 + scale_bar_sec / 2,
        bar_y0 - amplitude_limit * 0.35,
        f"{scale_bar_sec:g} s",
        ha="center",
        va="top",
        fontsize=7,
        color="0.1",
    )
    ax.plot(
        [bar_x0, bar_x0],
        [bar_y0, bar_y0 + scale_bar_value],
        color="0.1",
        linewidth=1.2,
        solid_capstyle="butt",
    )
    ax.text(
        bar_x0 - duration * 0.012,
        bar_y0 + scale_bar_value / 2,
        f"{scale_bar_value:g} {scale_bar_label}",
        ha="right",
        va="center",
        fontsize=7,
        color="0.1",
    )

    ax.set_xlim(0, duration)
    ax.set_ylim(bar_y0 - amplitude_limit * 0.85, offsets[0] + amplitude_limit * 1.25)
    ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{subject}_{condition}_{recording_type}_trial{trial_index}_"
        f"adaptive_filter_waveforms_{output_suffix}"
    )
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_dir / f"{stem}.{formats[0]}"


def main() -> int:
    args = parse_args()
    mne.set_log_level("WARNING")

    if args.subject.startswith("subject"):
        print(
            f"Use public BIDS IDs (e.g. sub-7), not legacy names ({args.subject!r}).",
            file=sys.stderr,
        )
        return 2

    try:
        bundle = load_or_compute_bundle(
            trial_npy=args.trial_npy,
            eeg_before_npy=args.eeg_before_npy,
            eeg_after_npy=args.eeg_after_npy,
            emg_npy=args.emg_npy,
            trial_index=args.trial,
            subject=args.subject,
            condition=args.condition,
            recording_type=args.recording_type,
            session=args.session,
            run=args.run,
            output_root=args.output_root,
            fs=args.fs,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(exc, file=sys.stderr)
        return 2

    time, win = slice_window(bundle, args.fs, args.start_sec, args.duration_sec)
    rows = trace_rows(bundle, args.affected_channel, args.minimal_channel, win)
    out = plot_trace_variant(
        subject=args.subject,
        condition=args.condition,
        recording_type=args.recording_type,
        trial_index=args.trial,
        time=time,
        rows=rows,
        scale_bar_sec=args.scale_bar_sec,
        scale_bar_value=args.scale_bar_value,
        scale_bar_label=args.scale_bar_label,
        output_suffix=args.output_suffix,
        output_dir=Path(args.output_dir),
        formats=tuple(args.formats),
    )
    src = bundle.source_path if bundle.source_path is not None else "precomputed arrays"
    print(f"Wrote Fig. S1 trace to {out.parent} (source: {src})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
