#!/usr/bin/env python3
"""Fig. S5: channel-level spatial correlations for EEGNet_wo_adapt_filt.

Thin wrapper around ``plot_spatial_correlation_between_tasks.py`` with
``--model EEGNet_wo_adapt_filt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_spatial_correlation_between_tasks as base


def main() -> None:
    argv = list(sys.argv[1:])
    if "--model" not in argv:
        argv = ["--model", "EEGNet_wo_adapt_filt", *argv]
    # Default save dir for the wo-adapt analysis when caller did not set one.
    if "--save-dir" not in argv:
        save_dir = (
            base.REPO_ROOT
            / "outputs"
            / "IG_spatial"
            / "eegnet_wo_adapt_filt_spatial_contribution_correlation"
        )
        argv = ["--save-dir", str(save_dir), *argv]
    sys.argv = [sys.argv[0], *argv]
    base.main()


if __name__ == "__main__":
    main()
