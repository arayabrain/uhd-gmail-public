#!/usr/bin/env python3
"""Helper + optional wide layout for inter-subject variability panels.

Rows: per-word across-trial SNR (power ratio), EEG-EMG mutual information,
EMG RMS, age. Columns: overt, min-overt, covert. Each point is one subject
(public IDs ``sub-N``).

Inputs (already computed; nothing is recomputed here):
  data/intersubject/subject_measures.csv
  data/intersubject/subject_condition_summary.csv
  {bids_root}/participants.tsv  (age; path from configs/paths.yaml)

The publication entry point is ``plot_intersubject_variability_publication.py``
(Supplementary Figs S3 offline / S4 online). This module supplies shared loaders
and label placement; ``main()`` optionally draws the wider 7-inch layout.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

try:
    from scripts.figures.condition_colors import (
        CONDITION_ORDER,
        condition_color,
        condition_label,
    )
except ImportError:  # pragma: no cover - script invocation
    from condition_colors import (  # type: ignore[no-redef]
        CONDITION_ORDER,
        condition_color,
        condition_label,
    )

from uhd_eeg.paths import get_bids_root

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEASURES = ROOT / "data/intersubject/subject_measures.csv"
DEFAULT_SUMMARY = ROOT / "data/intersubject/subject_condition_summary.csv"
DEFAULT_AGES_EXAMPLE = ROOT / "data/subject_demographics.example.csv"
DEFAULT_OUT = {
    "offline": ROOT / "outputs/figures/figS3_intersubject_offline",
    "online": ROOT / "outputs/figures/figS4_intersubject_online",
}

CHANCE = 0.2
BOOTSTRAP = 10000
SEED = 0

ROWS = [
    ("snr", "SNR (power ratio)"),
    ("eeg_emg_mutual_information", "EEG-EMG mutual information"),
    ("emg_rms", "EMG RMS (z-scored)"),
    ("age", "age (years)"),
]

_SUBJECT_RE = re.compile(r"^(?:sub-|subject)?(\d+)$", re.IGNORECASE)

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def to_public_subject_id(value: object) -> str:
    """Normalize ``subjectN`` / ``sub-N`` / bare digits to public ``sub-N``."""
    text = str(value).strip().replace(" ", "")
    match = _SUBJECT_RE.match(text)
    if match is None:
        raise ValueError(f"Unrecognized subject id: {value!r}")
    return f"sub-{int(match.group(1))}"


def subject_plot_label(subject_id: object) -> str:
    """Compact panel annotation for ``sub-N`` (numeric index only)."""
    return str(int(to_public_subject_id(subject_id).removeprefix("sub-")))


def _normalize_subject_column(series: pd.Series) -> pd.Series:
    return series.map(to_public_subject_id)


def load_ages(
    *,
    participants_tsv: Path | None = None,
    ages_csv: Path | None = None,
) -> pd.DataFrame:
    """Load ``subject, age`` from OpenNeuro ``participants.tsv`` or a test CSV.

    Production default: ``{bids_root}/participants.tsv`` from ``configs/paths.yaml``.
    Tests may pass ``ages_csv=data/subject_demographics.example.csv`` (synthetic ages).
    """
    if ages_csv is not None:
        path = Path(ages_csv)
        if not path.is_file():
            raise FileNotFoundError(path)
        ages = pd.read_csv(path)
        id_col = "subject" if "subject" in ages.columns else "participant_id"
    else:
        path = (
            Path(participants_tsv)
            if participants_tsv is not None
            else get_bids_root() / "participants.tsv"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing participants table: {path}. "
                "Set bids_root in configs/paths.yaml to the OpenNeuro ds007591 root, "
                f"or pass ages_csv={DEFAULT_AGES_EXAMPLE} for synthetic ages."
            )
        ages = pd.read_csv(path, sep="\t")
        id_col = "participant_id" if "participant_id" in ages.columns else "subject"
    if "age" not in ages.columns:
        raise ValueError(f"{path} must contain an 'age' column")
    out = ages[[id_col, "age"]].rename(columns={id_col: "subject"}).copy()
    out["subject"] = _normalize_subject_column(out["subject"])
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    return out


def load(
    recording_type: str,
    *,
    measures_csv: Path | None = None,
    summary_csv: Path | None = None,
    participants_tsv: Path | None = None,
    ages_csv: Path | None = None,
) -> pd.DataFrame:
    measures_path = Path(measures_csv) if measures_csv else DEFAULT_MEASURES
    summary_path = Path(summary_csv) if summary_csv else DEFAULT_SUMMARY

    measures = pd.read_csv(measures_path).query("recording_type == @recording_type")
    summary = pd.read_csv(summary_path).query("recording_type == @recording_type")
    measures = measures.copy()
    summary = summary.copy()
    measures["subject"] = _normalize_subject_column(measures["subject"])
    summary["subject"] = _normalize_subject_column(summary["subject"])

    df = measures[["subject", "condition", "snr", "accuracy"]].merge(
        summary[
            ["subject", "condition", "emg_rms", "eeg_emg_mutual_information"]
        ],
        on=["subject", "condition"],
        validate="one_to_one",
    )
    ages = load_ages(participants_tsv=participants_tsv, ages_csv=ages_csv)
    df = df.merge(ages, on="subject", validate="many_to_one")
    df["subject_num"] = df["subject"].map(subject_plot_label)
    return df


def bootstrap_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.RandomState(SEED)
    values = []
    for _ in range(BOOTSTRAP):
        idx = rng.randint(0, len(x), len(x))
        if np.unique(x[idx]).size > 1 and np.unique(y[idx]).size > 1:
            values.append(spearmanr(x[idx], y[idx])[0])
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def significance_marker(p_raw: float, p_corrected: float) -> str:
    """Manuscript legend: ``*`` for Bonferroni-adjusted P < 0.05 (Figs. 1b / 2a / 2b)."""
    from uhd_eeg.analysis.stats import manuscript_star

    del p_raw
    return manuscript_star(p_corrected)


# Labels placed at an offset radius of this many points or more get a thin
# leader line, so that every non-adjacent label is visibly tied to its point.
LEADER_MIN_RADIUS = 8

# Hand-tuned label positions for crowded panels, applied before the automatic
# placement of the remaining labels:
# (recording_type, condition, measure, subject number) -> (dx, dy, ha, va) in points.
LABEL_OVERRIDES = {
    ("online", "covert", "emg_rms", "6"): (-5.0, 0.0, "right", "center"),
    ("online", "covert", "emg_rms", "1"): (4.0, 1.5, "left", "bottom"),
    ("online", "minimally_overt", "age", "7"): (-1.0, -4.0, "center", "top"),
    ("online", "minimally_overt", "age", "6"): (3.0, 3.0, "left", "bottom"),
}


def panel_overrides(recording_type: str, condition: str, measure: str) -> dict:
    return {
        subject: position
        for (rec, cond, meas, subject), position in LABEL_OVERRIDES.items()
        if (rec, cond, meas) == (recording_type, condition, measure)
    }


def place_subject_labels(ax, x, y, labels, renderer, overrides=None,
                         avoid_crossings: bool = False) -> None:
    """Place subject numbers clear of markers, optionally untangling leader lines."""
    points_to_pixels = renderer.points_to_pixels
    marker_radius = points_to_pixels(np.sqrt(26) / 2 + 0.45 / 2 + 0.35)
    label_padding = points_to_pixels(0.35)
    edge_padding = points_to_pixels(0.35)
    points = ax.transData.transform(np.column_stack((x, y)))
    directions = (
        (1, 1, "left", "bottom"), (-1, 1, "right", "bottom"),
        (1, -1, "left", "top"), (-1, -1, "right", "top"),
        (1, 0, "left", "center"), (-1, 0, "right", "center"),
        (0, 1, "center", "bottom"), (0, -1, "center", "top"),
    )
    radii = (4, 6, 8, 11, 15, 20, 26, 34, 44, 56)

    def touches_marker(box) -> bool:
        for px, py in points:
            dx = max(box.x0 - px, 0, px - box.x1)
            dy = max(box.y0 - py, 0, py - box.y1)
            if dx * dx + dy * dy < marker_radius * marker_radius:
                return True
        return False

    def inside_axes(box) -> bool:
        bounds = ax.bbox
        return (
            box.x0 >= bounds.x0 + edge_padding
            and box.x1 <= bounds.x1 - edge_padding
            and box.y0 >= bounds.y0 + edge_padding
            and box.y1 <= bounds.y1 - edge_padding
        )

    chance_y = ax.transData.transform((0.0, CHANCE))[1]
    line_padding = points_to_pixels(0.8)

    def crosses_chance_line(box) -> bool:
        return box.y0 - line_padding < chance_y < box.y1 + line_padding

    def overlaps(a, b) -> bool:
        return a.padded(label_padding).overlaps(b.padded(label_padding))

    ambiguity_margin = points_to_pixels(2.0)

    def box_distance(box, px, py) -> float:
        dx = max(box.x0 - px, 0, px - box.x1)
        dy = max(box.y0 - py, 0, py - box.y1)
        return float(np.hypot(dx, dy))

    def unambiguous(box, own_index) -> bool:
        """The label must sit clearly closer to its own point than to any other."""
        own = box_distance(box, *points[own_index])
        return all(
            own + ambiguity_margin < box_distance(box, px, py)
            for other_index, (px, py) in enumerate(points)
            if other_index != own_index
        )

    candidates = []
    all_candidates = []
    overrides = overrides or {}
    for own_index, (xi, yi, label) in enumerate(zip(x, y, labels)):
        choices = []
        every_choice = []
        if str(label) in overrides:
            dx, dy, ha, va = overrides[str(label)]
            annotation = ax.annotate(
                str(label), (xi, yi), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va=va,
                fontsize=6, color="0.3", zorder=4,
            )
            box = annotation.get_window_extent(renderer)
            annotation.remove()
            fixed = ((dx, dy), ha, va, box, float(np.hypot(dx, dy)))
            candidates.append([fixed])
            all_candidates.append([fixed])
            continue
        for radius in radii:
            for horizontal, vertical, ha, va in directions:
                offset = (horizontal * radius, vertical * radius)
                annotation = ax.annotate(
                    str(label), (xi, yi), xytext=offset,
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=6, color="0.3", zorder=4,
                )
                box = annotation.get_window_extent(renderer)
                annotation.remove()
                choice = (offset, ha, va, box, radius)
                every_choice.append(choice)
                if (inside_axes(box) and not touches_marker(box)
                        and not crosses_chance_line(box)
                        and (radius >= LEADER_MIN_RADIUS or unambiguous(box, own_index))):
                    choices.append(choice)
        candidates.append(choices)
        all_candidates.append(every_choice)

    order = sorted(range(len(labels)), key=lambda index: (len(candidates[index]), index))
    min_radius = [min((choice[4] for choice in candidates[index]), default=np.inf)
                  for index in range(len(labels))]
    best = {"cost": np.inf, "placed": None}
    nodes = [0]

    def search(position, selected, cost):
        nodes[0] += 1
        if nodes[0] > 200000:
            return
        if position == len(order):
            if cost < best["cost"]:
                best["cost"] = cost
                best["placed"] = selected.copy()
            return
        bound = cost + sum(min_radius[index] for index in order[position:])
        if bound >= best["cost"]:
            return
        index = order[position]
        for choice in candidates[index]:
            if all(not overlaps(choice[3], other[3]) for other in selected.values()):
                selected[index] = choice
                search(position + 1, selected, cost + choice[4])
                del selected[index]

    search(0, {}, 0.0)
    placed = best["placed"] or {}

    if len(placed) != len(labels):
        placed = {}
        for index, choices in enumerate(candidates):
            pool = choices or all_candidates[index]
            placed[index] = min(
                pool,
                key=lambda choice: (
                    int(not inside_axes(choice[3]))
                    + int(touches_marker(choice[3]))
                    + sum(overlaps(choice[3], other[3])
                          for other in placed.values()),
                    choice[4],
                ),
            )

    if avoid_crossings and len(placed) == len(labels) and all(candidates):
        def leader_segment(index, choice, assignment):
            box, radius = choice[3], choice[4]
            if (radius < LEADER_MIN_RADIUS and inside_axes(box)
                    and not touches_marker(box)
                    and not any(overlaps(box, assignment[other][3])
                                for other in assignment if other != index)):
                return None
            px, py = points[index]
            return ((px, py),
                    (np.clip(px, box.x0, box.x1), np.clip(py, box.y0, box.y1)))

        def segments_cross(first, second):
            if first is None or second is None:
                return False
            a, b = first
            c, d = second

            def orient(p, q, r):
                return ((q[0] - p[0]) * (r[1] - p[1])
                        - (q[1] - p[1]) * (r[0] - p[0]))

            return (orient(a, b, c) * orient(a, b, d) < 0
                    and orient(c, d, a) * orient(c, d, b) < 0)

        def segment_hits_box(segment, box):
            if segment is None:
                return False
            (x0, y0), (x1, y1) = segment
            dx, dy = x1 - x0, y1 - y0
            near, far = 0.0, 1.0
            for direction, edge in (
                (-dx, x0 - box.x0), (dx, box.x1 - x0),
                (-dy, y0 - box.y0), (dy, box.y1 - y0),
            ):
                if abs(direction) < 1e-9:
                    if edge < 0:
                        return False
                elif direction < 0:
                    near = max(near, edge / direction)
                else:
                    far = min(far, edge / direction)
            return near <= far

        def placement_score(assignment):
            segments = [leader_segment(index, assignment[index], assignment)
                        for index in range(len(labels))]
            crossings = sum(
                segments_cross(segments[first], segments[second])
                for first in range(len(labels))
                for second in range(first + 1, len(labels))
            )
            label_hits = sum(
                segment_hits_box(segments[first],
                                 assignment[second][3].padded(label_padding))
                for first in range(len(labels))
                for second in range(len(labels))
                if first != second
            )
            return (crossings, label_hits,
                    sum(choice[4] for choice in assignment.values()))

        score = placement_score(placed)
        while True:
            best_move = None
            best_score = score
            for index, choices in enumerate(candidates):
                for choice in choices:
                    if choice is placed[index] or any(
                        overlaps(choice[3], placed[other][3])
                        for other in placed if other != index
                    ):
                        continue
                    trial = {**placed, index: choice}
                    trial_score = placement_score(trial)
                    if trial_score < best_score:
                        best_score = trial_score
                        best_move = (index, choice)
            if best_move is None:
                break
            placed[best_move[0]] = best_move[1]
            score = best_score

    for index, (xi, yi, label) in enumerate(zip(x, y, labels)):
        offset, ha, va, box, radius = placed[index]
        ax.annotate(
            str(label), (xi, yi), xytext=offset,
            textcoords="offset points", ha=ha, va=va,
            fontsize=6, color="0.3", zorder=4,
        )
        if (radius >= LEADER_MIN_RADIUS or not inside_axes(box) or touches_marker(box) or any(
            overlaps(box, other[3]) for other_index, other in placed.items()
            if other_index != index
        )):
            px, py = points[index]
            end = ax.transData.inverted().transform(
                (np.clip(px, box.x0, box.x1), np.clip(py, box.y0, box.y1))
            )
            ax.add_artist(Line2D(
                (xi, end[0]), (yi, end[1]), transform=ax.transData,
                color="0.55", linewidth=0.35, zorder=2,
            ))


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-type", default="offline", choices=("offline", "online"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--measures-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--participants-tsv", type=Path, default=None)
    parser.add_argument("--ages-csv", type=Path, default=None)
    parser.add_argument("--lowres-dpi", type=int, default=110)
    parser.add_argument(
        "--split-panels",
        action="store_true",
        help="also save each panel as its own PDF/PNG under panels/",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or DEFAULT_OUT[args.recording_type]

    df = load(
        args.recording_type,
        measures_csv=args.measures_csv,
        summary_csv=args.summary_csv,
        participants_tsv=args.participants_tsv,
        ages_csv=args.ages_csv,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(
            len(ROWS), len(CONDITION_ORDER), figsize=(7.0, 8.4), sharey="row"
        )
        stats = []
        combined_panels = []
        n_family = len(CONDITION_ORDER)
        for i, (column, xlabel) in enumerate(ROWS):
            cells = []
            for condition in CONDITION_ORDER:
                sub = df[df["condition"] == condition].dropna(subset=[column, "accuracy"])
                x = sub[column].to_numpy(dtype=float)
                y = sub["accuracy"].to_numpy(dtype=float)
                rho, p = spearmanr(x, y)
                ci_low, ci_high = bootstrap_ci(x, y)
                cells.append((condition, sub, x, y, rho, p, ci_low, ci_high))
            for j, (condition, sub, x, y, rho, p, ci_low, ci_high) in enumerate(cells):
                ax = axes[i, j]
                p_corr = min(p * n_family, 1.0)
                marker_text = significance_marker(p, p_corr)
                stats.append(
                    {
                        "recording_type": args.recording_type,
                        "condition": condition,
                        "measure": column,
                        "n": len(sub),
                        "rho": rho,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_raw": p,
                        "p_bonferroni": p_corr,
                        "n_family": n_family,
                        "marker": marker_text,
                    }
                )

                ax.axhline(CHANCE, color="0.45", linestyle="--", linewidth=0.8, zorder=1)
                ax.scatter(
                    x, y, s=26, marker="o", color=condition_color(condition),
                    edgecolor="white", linewidth=0.45, alpha=0.88, zorder=3,
                )
                combined_panels.append(
                    (ax, x, y, sub["subject_num"].to_numpy(), (condition, column))
                )
                stats_line = (
                    rf"$\rho$ = {rho:.2f}, $P_{{\mathrm{{adj}}}}$ = {p_corr:.3f} {marker_text}"
                )
                ax.set_title(
                    f"{condition_label(condition)}\n{stats_line}" if i == 0 else stats_line
                )
                ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel("balanced accuracy")
                ax.set_ylim(0.1, 0.95)
                ax.grid(False)
                _despine(ax)

        layout_labels = [
            ax.annotate(
                str(label), (xi, yi), xytext=(2.5, 2.5),
                textcoords="offset points", fontsize=6, color="0.3", zorder=4,
            )
            for ax, x, y, labels, _ in combined_panels
            for xi, yi, label in zip(x, y, labels)
        ]
        fig.tight_layout()
        for annotation in layout_labels:
            annotation.remove()
        fig.set_dpi(600)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, x, y, labels, (condition, column) in combined_panels:
            place_subject_labels(
                ax, x, y, labels, renderer,
                panel_overrides(args.recording_type, condition, column),
            )
        base = output_dir / "fig_intersubject_variability"
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".png"))
        fig.savefig(base.parent / f"{base.name}_lowres.png", dpi=args.lowres_dpi)
        plt.close(fig)

        if args.split_panels:
            panel_dir = output_dir / "panels"
            panel_dir.mkdir(parents=True, exist_ok=True)
            for row in stats:
                column = row["measure"]
                condition = row["condition"]
                sub = df[df["condition"] == condition].dropna(subset=[column, "accuracy"])
                x = sub[column].to_numpy(dtype=float)
                y = sub["accuracy"].to_numpy(dtype=float)
                xlabel = dict(ROWS)[column]
                fig1, ax = plt.subplots(figsize=(2.5, 2.3))
                ax.axhline(CHANCE, color="0.45", linestyle="--", linewidth=0.8, zorder=1)
                ax.scatter(
                    x, y, s=26, marker="o", color=condition_color(condition),
                    edgecolor="white", linewidth=0.45, alpha=0.88, zorder=3,
                )
                ax.set_title(
                    f"{condition_label(condition)}\n"
                    rf"$\rho$ = {row['rho']:.2f}, "
                    rf"$P_{{\mathrm{{adj}}}}$ = {row['p_bonferroni']:.3f} {row['marker']}"
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel("balanced accuracy")
                ax.set_ylim(0.1, 0.95)
                ax.grid(False)
                _despine(ax)
                layout_labels = [
                    ax.annotate(
                        str(label), (xi, yi), xytext=(2.5, 2.5),
                        textcoords="offset points", fontsize=6, color="0.3", zorder=4,
                    )
                    for xi, yi, label in zip(x, y, sub["subject_num"])
                ]
                fig1.tight_layout()
                for annotation in layout_labels:
                    annotation.remove()
                fig1.set_dpi(600)
                fig1.canvas.draw()
                place_subject_labels(
                    ax, x, y, sub["subject_num"].to_numpy(), fig1.canvas.get_renderer(),
                    panel_overrides(args.recording_type, condition, column),
                )
                stem = panel_dir / f"{column}_{condition}"
                fig1.savefig(stem.with_suffix(".pdf"))
                fig1.savefig(stem.with_suffix(".png"), dpi=600)
                plt.close(fig1)
            print(f"panels saved to {panel_dir}")

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / "intersubject_variability_stats.csv", index=False)
    print(stats_df.round(3).to_string(index=False))
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
