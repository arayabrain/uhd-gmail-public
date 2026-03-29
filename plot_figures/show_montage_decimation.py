"""show montage of electrode decimation (Fig. S1)"""

from pathlib import Path
from typing import List

import numpy as np
from matplotlib.image import imread
from termcolor import cprint

from plot_figures.src.default_plt import cm_to_inch, plt


def show_montage(channels_to_show: List[int], save_path: Path) -> None:
    """show_montage

    Args:
        channels_to_show (List[int]): channels to show
        save_path (Path): save path
    """
    img = imread("plot_figures/montage_colorless.png")
    coordinates = np.load("plot_figures/coordinates_colorless.npy")
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
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def main():
    channels_to_shows = [
        [7, 48, 87, 106],
        [5, 21, 37, 53, 69, 85, 105, 117],
        [1, 12, 17, 28, 33, 44, 49, 60, 65, 76, 81, 92, 97, 108, 113, 124],
        [
            1,
            7,
            9,
            12,
            17,
            23,
            25,
            28,
            33,
            39,
            41,
            44,
            49,
            51,
            55,
            57,
            60,
            65,
            73,
            76,
            81,
            87,
            89,
            92,
            97,
            103,
            105,
            108,
            113,
            119,
            121,
            124,
        ],
        list(range(128)),
    ]
    save_dir = Path("figures/figS1/decimation_montage")
    save_dir.mkdir(parents=True, exist_ok=True)
    for channels_to_show in channels_to_shows:
        cprint(f"len(channels_to_show): {len(channels_to_show)}", "cyan")
        save_path = save_dir / f"montage_{len(channels_to_show)}.png"
        show_montage(channels_to_show, save_path)


if __name__ == "__main__":
    main()
