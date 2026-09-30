#!/usr/bin/env python3
"""Supplementary Figs S3/S4: inter-subject variability at 3.5-inch print width.

S3 = offline (``--recording-type offline``)
S4 = online  (``--recording-type online``)

The original wide figure is seven inches. Scaling it to one column halves every
font. This version keeps the same 4 x 3 panels and data, but shares the
condition headings and row labels so that all text can be drawn at its final
publication size. Bonferroni-adjusted Spearman P values and significance
markers are drawn above each panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.stats import spearmanr

try:
    from scripts.figures.condition_colors import (
        CONDITION_ORDER,
        condition_color,
        condition_label,
    )
    from scripts.figures.plot_intersubject_variability_supplement import (
        CHANCE,
        ROOT,
        ROWS,
        load,
        panel_overrides,
        place_subject_labels,
        significance_marker,
    )
except ImportError:  # pragma: no cover - script invocation
    from condition_colors import (  # type: ignore[no-redef]
        CONDITION_ORDER,
        condition_color,
        condition_label,
    )
    from plot_intersubject_variability_supplement import (  # type: ignore[no-redef]
        CHANCE,
        ROOT,
        ROWS,
        load,
        panel_overrides,
        place_subject_labels,
        significance_marker,
    )

DEFAULT_OUT = {
    "offline": ROOT / "outputs/figures/figS3_intersubject_offline",
    "online": ROOT / "outputs/figures/figS4_intersubject_online",
}

WIDTH = 3.5
HEIGHT = 8.0
LEFT = 0.43
RIGHT = 0.06
GUTTER = 0.075
AXES_HEIGHT = 0.98
FIRST_AXES_TOP = 7.17
ROW_PITCH = 1.84
COL_WIDTH = (WIDTH - LEFT - RIGHT - 2 * GUTTER) / 3

# Sparse, fixed ticks keep their six-point labels apart at the final width.
X_TICKS = {
    "snr": (
        (0.01, 0.02, 0.03),
        (0.005, 0.015),
        (0.002, 0.005, 0.008),
    ),
    "eeg_emg_mutual_information": (
        (0.02, 0.03, 0.04),
        (0.02, 0.04),
        (0.02, 0.04),
    ),
    "emg_rms": (
        (0.95, 1.05, 1.15),
        (0.8, 1.0, 1.2),
        (0.4, 0.7, 1.0),
    ),
    "age": (
        (20, 40, 60),
        (20, 40, 60),
        (20, 40, 60),
    ),
}

# Online values cover different ranges; these ticks stay inside the data
# margins and leave enough room for their six-point labels at 3.5 inches.
ONLINE_X_TICKS = {
    **X_TICKS,
    "snr": (
        (0.0, 0.02, 0.04),
        (0.0, 0.01, 0.02),
        (0.0, 0.005, 0.01),
    ),
    "eeg_emg_mutual_information": (
        (0.02, 0.03, 0.04),
        (0.02, 0.04),
        (0.015, 0.025),
    ),
    "emg_rms": (
        (0.95, 1.05, 1.15),
        (0.8, 1.0),
        (0.4, 0.7, 1.0),
    ),
}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": None,
    "savefig.pad_inches": 0,
}

FIGURE_IDS = {
    "offline": "S3",
    "online": "S4",
}


def tick_label(value: float, measure: str) -> str:
    if measure == "age":
        return str(int(value))
    if measure == "snr":
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if measure == "eeg_emg_mutual_information":
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2f}" if value in (0.95, 1.05, 1.15) else f"{value:.1f}"


def draw_figure(
    recording_type: str,
    *,
    show_subject_labels: bool = True,
    measures_csv: Path | None = None,
    summary_csv: Path | None = None,
    participants_tsv: Path | None = None,
    ages_csv: Path | None = None,
) -> plt.Figure:
    df = load(
        recording_type,
        measures_csv=measures_csv,
        summary_csv=summary_csv,
        participants_tsv=participants_tsv,
        ages_csv=ages_csv,
    )
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=600, facecolor="white")
    panels = []

    for j, condition in enumerate(CONDITION_ORDER):
        center_x = (LEFT + j * (COL_WIDTH + GUTTER) + COL_WIDTH / 2) / WIDTH
        fig.text(
            center_x, 7.82 / HEIGHT, condition_label(condition),
            ha="center", va="center", fontsize=10,
        )

    fig.text(
        0.09 / WIDTH, 4.02 / HEIGHT, "balanced accuracy",
        ha="center", va="center", rotation=90, fontsize=8,
    )

    for i, (measure, xlabel) in enumerate(ROWS):
        top = FIRST_AXES_TOP - i * ROW_PITCH
        bottom = top - AXES_HEIGHT
        fig.text(
            1.94 / WIDTH, (bottom - 0.34) / HEIGHT, xlabel,
            ha="center", va="center", fontsize=8,
        )

        for j, condition in enumerate(CONDITION_ORDER):
            left = LEFT + j * (COL_WIDTH + GUTTER)
            ax = fig.add_axes((left / WIDTH, bottom / HEIGHT,
                               COL_WIDTH / WIDTH, AXES_HEIGHT / HEIGHT))
            sub = df[df["condition"] == condition].dropna(subset=[measure, "accuracy"])
            x = sub[measure].to_numpy(dtype=float)
            y = sub["accuracy"].to_numpy(dtype=float)
            rho, p_raw = spearmanr(x, y)
            p_adj = min(3 * p_raw, 1.0)
            marker = significance_marker(p_raw, p_adj)

            center_x = (left + COL_WIDTH / 2) / WIDTH
            fig.text(
                center_x, (top + 0.32) / HEIGHT,
                rf"$\rho={rho:.2f}$", ha="center", va="center", fontsize=8,
            )
            fig.text(
                center_x, (top + 0.18) / HEIGHT,
                rf"$P_{{\mathrm{{adj}}}}={p_adj:.3f}$ {marker}",
                ha="center", va="center", fontsize=8,
            )

            ax.axhline(CHANCE, color="0.47", linestyle=(0, (3, 2)),
                       linewidth=0.6, zorder=1)
            ax.scatter(
                x, y, s=26, marker="o", color=condition_color(condition),
                edgecolor="white", linewidth=0.4, alpha=0.92, zorder=3,
            )
            spread = max(x) - min(x)
            ax.set_xlim(min(x) - 0.13 * spread, max(x) + 0.13 * spread)
            ax.set_ylim(0.08, 0.97)
            ticks = (ONLINE_X_TICKS if recording_type == "online"
                     else X_TICKS)[measure][j]
            ax.set_xticks(ticks)
            ax.set_xticklabels([tick_label(value, measure) for value in ticks])
            ax.set_yticks((0.2, 0.4, 0.6, 0.8))
            if j:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="both", labelsize=6, pad=1.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            panels.append((ax, x, y, sub["subject_num"].to_numpy(), condition, measure))

    if show_subject_labels:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, x, y, labels, condition, measure in panels:
            place_subject_labels(
                ax, x, y, labels, renderer,
                panel_overrides(recording_type, condition, measure),
                avoid_crossings=True,
            )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-type",
        default="offline",
        choices=("offline", "online"),
        help="S3 = offline (default); S4 = online",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/figures/figS3_… (offline) or figS4_… (online).",
    )
    parser.add_argument("--measures-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument(
        "--participants-tsv",
        type=Path,
        default=None,
        help="OpenNeuro participants.tsv (default: {bids_root}/participants.tsv).",
    )
    parser.add_argument(
        "--ages-csv",
        type=Path,
        default=None,
        help="Optional subject,age CSV (e.g. synthetic example for offline tests).",
    )
    parser.add_argument(
        "--no-subject-labels", action="store_true",
        help="save a separate version without subject numbers or leader lines",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUT[args.recording_type]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_id = FIGURE_IDS[args.recording_type]
    name = f"fig_{fig_id}_intersubject_variability_{args.recording_type}_3p5in"
    if args.no_subject_labels:
        name += "_no_subject_labels"
    stem = args.output_dir / name

    with plt.rc_context(STYLE):
        fig = draw_figure(
            args.recording_type,
            show_subject_labels=not args.no_subject_labels,
            measures_csv=args.measures_csv,
            summary_csv=args.summary_csv,
            participants_tsv=args.participants_tsv,
            ages_csv=args.ages_csv,
        )
        for suffix in (".pdf", ".svg", ".png"):
            fig.savefig(stem.with_suffix(suffix), dpi=600, bbox_inches=None, pad_inches=0)
        plt.close(fig)
    print(
        f"Saved {stem}.pdf, {stem}.svg, and {stem}.png "
        f"(Fig. {fig_id}, {args.recording_type}, 3.5 × 8.0 in)"
    )


if __name__ == "__main__":
    main()
