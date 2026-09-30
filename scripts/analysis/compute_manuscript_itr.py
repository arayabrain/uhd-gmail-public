#!/usr/bin/env python3
"""Print manuscript ITR values derived from Table 2 EEGNet accuracies.

Reproduces the main-text rates (bits/min):

- overt / minimally overt / covert at T = 12.4 s → 1.8 / 0.9 / 0.15
- overt at T = 6.25 s → 3.6

Run::

    uv run python scripts/analysis/compute_manuscript_itr.py
"""

from __future__ import annotations

from uhd_eeg.analysis.itr import TABLE2_EEGNET, manuscript_itr_summary


def main() -> None:
    summary = manuscript_itr_summary(TABLE2_EEGNET)
    mean_12_4 = summary[12.4]
    mean_overt_6_25 = summary[6.25]["overt"]

    print("Manuscript ITR from Table 2 EEGNet (online balanced accuracy, N=9 subjects)")
    print(f"  T = 12.4 s  overt:           {mean_12_4['overt']:.4f}  (rounded {round(mean_12_4['overt'], 1)})")
    print(
        f"  T = 12.4 s  minimally overt: {mean_12_4['minimally_overt']:.4f}  "
        f"(rounded {round(mean_12_4['minimally_overt'], 1)})"
    )
    print(
        f"  T = 12.4 s  covert:          {mean_12_4['covert']:.4f}  "
        f"(rounded {round(mean_12_4['covert'], 2)})"
    )
    print(
        f"  T = 6.25 s  overt:           {mean_overt_6_25:.4f}  "
        f"(rounded {round(mean_overt_6_25, 1)})"
    )


if __name__ == "__main__":
    main()
