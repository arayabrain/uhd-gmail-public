"""Supplementary Fig. S2a: k-medoids electrode subsets on the montage.

Uses the manuscript subsets from ``uhd_eeg.analysis.electrode_subsets_kmedoids``
(Supplementary Fig. S2). Hand-picked subsets are not included.

Example::

    uv run python scripts/figures/plot_fig_s2a_electrode_montage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.image import imread

from scripts.figures._lib.default_plt import cm_to_inch, plt
from uhd_eeg.analysis.electrode_subsets_kmedoids import DENSITIES, EXPECTED_SUBSETS

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"


def load_subsets() -> dict[int, list[int]]:
    return {density: list(EXPECTED_SUBSETS[density]) for density in DENSITIES}


def show_montage(channels_to_show: list[int], save_stem: Path) -> None:
    img = imread(ASSETS / "montage_colorless.png")
    coordinates = np.load(ASSETS / "coordinates_colorless.npy")
    plt.figure(figsize=(7.5 * cm_to_inch, 7.5 * cm_to_inch))
    plt.imshow(img)
    plt.scatter(
        coordinates[channels_to_show, 0],
        coordinates[channels_to_show, 1],
        s=30.0,
        c="r",
        linewidths=0,
    )
    plt.axis("off")
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        plt.savefig(save_stem.with_suffix(suffix), dpi=600)
    plt.clf()
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "figS2a_electrode_montage",
    )
    parser.add_argument(
        "--densities",
        type=int,
        nargs="*",
        default=list(DENSITIES) + [128],
        help="Channel counts to draw (default: 4 8 16 32 128).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsets = {str(k): list(v) for k, v in EXPECTED_SUBSETS.items()}
    subsets["128"] = list(range(128))
    for n in args.densities:
        key = str(int(n))
        if key not in subsets:
            raise KeyError(f"No subset defined for n_channels={n}")
        channels = subsets[key]
        assert len(channels) == int(key)
        show_montage(channels, args.output_dir / f"montage_{key}")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
