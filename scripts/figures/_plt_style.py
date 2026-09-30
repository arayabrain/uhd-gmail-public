"""Shared matplotlib colors for public figure scripts."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

cm_to_inch = 1 / 2.54

green = tuple(np.array([0, 176, 80]) / 255)
magenta = tuple(np.array([208, 0, 149]) / 255)
orange = tuple(np.array([237, 125, 49]) / 255)
violet = tuple(np.array([112, 48, 160]) / 255)
yellow = tuple(np.array([255, 192, 0]) / 255)
dark_blue = tuple(np.array([47, 85, 151]) / 255)

def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def apply_rc(rc: dict) -> None:
    plt.rcParams.update(rc)


__all__ = [
    "apply_rc",
    "cm_to_inch",
    "dark_blue",
    "despine",
    "green",
    "magenta",
    "orange",
    "plt",
    "violet",
    "yellow",
]
