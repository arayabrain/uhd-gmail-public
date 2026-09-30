"""Voice-volume analysis for manuscript Fig. 1b.

Subject-level values ship in ``data/voice_volume/voice_volume_subject.csv``.
Statistics use subject-centered dB only (comparable across MIC and private-audio
sources). Omnibus: Friedman; pairwise: Wilcoxon signed-rank with Bonferroni ×3.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from uhd_eeg.analysis.stats import bonferroni, manuscript_star

CONDITIONS = ("overt", "minimally_overt", "covert")
CONDITION_LABELS = {
    "overt": "overt",
    "minimally_overt": "min overt",
    "covert": "covert",
}
DEFAULT_SUBJECT_CSV = Path("data/voice_volume/voice_volume_subject.csv")

# Manuscript-facing rounded targets (subject-centered means and stats).
MANUSCRIPT_CONDITION_MEANS = {
    "overt": 12.24,
    "minimally_overt": -5.23,
    "covert": -7.01,
}
MANUSCRIPT_FRIEDMAN_CHI2 = 14.89
MANUSCRIPT_FRIEDMAN_P = 0.000585
MANUSCRIPT_PAIRWISE_P_ADJ = {
    ("overt", "minimally_overt"): 0.0117,
    ("overt", "covert"): 0.0117,
    ("minimally_overt", "covert"): 0.293,
}


def load_subject_voice_volume(path: Path | None = None) -> pd.DataFrame:
    """Load the published subject-level voice-volume table."""
    csv_path = Path(path) if path is not None else DEFAULT_SUBJECT_CSV
    df = pd.read_csv(csv_path, float_precision="round_trip")
    required = {
        "subject",
        "condition",
        "volume_db_subject_centered",
        "volume_db",
        "n_trials",
        "source",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    return df


def subject_centered_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows=subject, columns=condition, values=subject-centered dB."""
    pivot = df.pivot(index="subject", columns="condition", values="volume_db_subject_centered")
    missing = [c for c in CONDITIONS if c not in pivot.columns]
    if missing:
        raise ValueError(f"Missing conditions in voice-volume table: {missing}")
    return pivot.loc[:, list(CONDITIONS)].sort_index(
        key=lambda idx: idx.map(lambda s: int(str(s).split("-", 1)[1]))
    )


def condition_means(df: pd.DataFrame) -> dict[str, float]:
    pivot = subject_centered_pivot(df)
    return {condition: float(pivot[condition].mean()) for condition in CONDITIONS}


def voice_volume_stats(df: pd.DataFrame) -> dict:
    """Friedman + pairwise Wilcoxon (Bonferroni ×3) on subject-centered dB."""
    pivot = subject_centered_pivot(df)
    friedman = friedmanchisquare(*(pivot[c].to_numpy() for c in CONDITIONS))
    pair_rows = []
    raw_p: list[float] = []
    for left, right in combinations(CONDITIONS, 2):
        result = wilcoxon(pivot[left], pivot[right], alternative="two-sided")
        raw_p.append(float(result.pvalue))
        pair_rows.append(
            {
                "condition1": left,
                "condition2": right,
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "n_subjects": int(len(pivot)),
            }
        )
    adjusted = bonferroni(raw_p, n_comparisons=3)
    for row, p_adj in zip(pair_rows, adjusted):
        row["p_value_corrected"] = float(p_adj)
        row["reject"] = bool(p_adj < 0.05)
        row["star"] = manuscript_star(float(p_adj))
    return {
        "n_subjects": int(len(pivot)),
        "means": condition_means(df),
        "friedman": {
            "statistic": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
        },
        "pairwise": pair_rows,
    }


def assert_matches_manuscript(stats: Mapping, *, atol_mean: float = 0.005) -> None:
    """Raise AssertionError if stats disagree with the printed manuscript values."""
    for condition, expected in MANUSCRIPT_CONDITION_MEANS.items():
        observed = float(stats["means"][condition])
        if abs(observed - expected) > atol_mean:
            raise AssertionError(
                f"{condition} mean {observed:.4f} != manuscript {expected}"
            )
    friedman = stats["friedman"]
    if round(friedman["statistic"], 2) != MANUSCRIPT_FRIEDMAN_CHI2:
        raise AssertionError(
            f"Friedman chi2 {friedman['statistic']} != {MANUSCRIPT_FRIEDMAN_CHI2}"
        )
    if abs(friedman["p_value"] - MANUSCRIPT_FRIEDMAN_P) > 5e-7:
        raise AssertionError(
            f"Friedman p {friedman['p_value']} != {MANUSCRIPT_FRIEDMAN_P}"
        )
    by_pair = {
        (row["condition1"], row["condition2"]): row for row in stats["pairwise"]
    }
    for pair, expected in MANUSCRIPT_PAIRWISE_P_ADJ.items():
        observed = float(by_pair[pair]["p_value_corrected"])
        if abs(observed - expected) > 5e-4:
            raise AssertionError(f"{pair} P_adj {observed} != {expected}")
