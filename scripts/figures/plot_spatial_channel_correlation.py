#!/usr/bin/env python3
"""Channel-level spatial contribution Pearson correlations (Figs. 4–5 / S5–S6).

Computes Pearson correlations across the 128 EEG channels between speech
conditions (and optional MI maps). Subject-level / mixed-effects analyses are
intentionally not included.

Input: a directory of ``.npy`` files named::

    {task}_{word}_spatial.npy   # shape (128,)

or a single NPZ with keys ``{task}_{word}``.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from scripts.figures.condition_colors import CONDITION_ORDER

DEFAULT_WORDS = ("green", "magenta", "orange", "violet", "yellow")
TASK_LABELS = {
    "overt": "overt",
    "minimally_overt": "min overt",
    "covert": "covert",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spatial-dir",
        type=Path,
        required=True,
        help="Directory of {task}_{word}_spatial.npy vectors (length 128).",
    )
    parser.add_argument("--tasks", nargs="*", default=list(CONDITION_ORDER))
    parser.add_argument("--words", nargs="*", default=list(DEFAULT_WORDS))
    parser.add_argument("--plot-vmin", type=float, default=-1.0)
    parser.add_argument("--plot-vmax", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/fig4_5_spatial_channel_corr"),
    )
    return parser.parse_args()


def load_vectors(spatial_dir: Path, tasks: list[str], words: list[str]) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for task, word in itertools.product(tasks, words):
        path = spatial_dir / f"{task}_{word}_spatial.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        vec = np.asarray(np.load(path), dtype=float).reshape(-1)
        if vec.size != 128:
            raise ValueError(f"{path} has length {vec.size}, expected 128")
        vectors[f"{task}_{word}"] = vec
    return vectors


def channel_level_matrix(
    vectors: dict[str, np.ndarray],
    tasks: list[str],
    words: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean Pearson r / p across words for each task pair (channel-level)."""
    n = len(tasks)
    corr = np.eye(n, dtype=float)
    pvals = np.ones((n, n), dtype=float)
    for i, j in itertools.product(range(n), repeat=2):
        if i == j:
            continue
        rs = []
        ps = []
        for word in words:
            left = vectors[f"{tasks[i]}_{word}"]
            right = vectors[f"{tasks[j]}_{word}"]
            r, p = pearsonr(left, right)
            rs.append(r)
            ps.append(p)
        corr[i, j] = float(np.mean(rs))
        # Conservative: report max p across words (all must be considered).
        pvals[i, j] = float(np.max(ps))
    return corr, pvals


def plot_matrix(
    matrix: np.ndarray,
    pvals: np.ndarray,
    tasks: list[str],
    *,
    vmin: float,
    vmax: float,
    alpha: float,
    output_dir: Path,
) -> Path:
    labels = [TASK_LABELS.get(task, task) for task in tasks]
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i, j in itertools.product(range(len(tasks)), repeat=2):
        if i == j:
            continue
        mark = "*" if pvals[i, j] < alpha else ""
        ax.text(
            j,
            i,
            f"{matrix[i, j]:.2f}{mark}",
            ha="center",
            va="center",
            fontsize=6,
            color="black",
        )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("channel-level Pearson r", fontsize=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "spatial_channel_correlation_between_tasks"
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    np.save(out.with_name(out.name + "_r.npy"), matrix)
    np.save(out.with_name(out.name + "_p.npy"), pvals)
    plt.close(fig)
    return out.with_suffix(".pdf")


def main() -> None:
    args = parse_args()
    vectors = load_vectors(args.spatial_dir, args.tasks, args.words)
    corr, pvals = channel_level_matrix(vectors, args.tasks, args.words)
    path = plot_matrix(
        corr,
        pvals,
        args.tasks,
        vmin=args.plot_vmin,
        vmax=args.plot_vmax,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
