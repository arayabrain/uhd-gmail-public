"""Bonferroni correction and manuscript significance markers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def bonferroni(p_values: Iterable[float], n_comparisons: int | None = None) -> np.ndarray:
    """Return Bonferroni-adjusted p-values, clipped to 1."""
    values = np.asarray(list(p_values), dtype=float)
    if n_comparisons is None:
        n_comparisons = len(values)
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1")
    return np.clip(values * n_comparisons, 0.0, 1.0)


def manuscript_star(p_corrected: float, alpha: float = 0.05) -> str:
    """Manuscript legend mark for Figs. 1b / 2a / 2b: ``*`` when ``P_adj < alpha``."""
    if not np.isfinite(p_corrected):
        return ""
    return "*" if float(p_corrected) < alpha else ""


def significance_marker(p_raw: float, p_corrected: float) -> str:
    """Compatibility wrapper; manuscript panels use a single ``*`` (P_adj < 0.05).

    ``p_raw`` is accepted for call-site compatibility and ignored.
    """
    del p_raw  # unused; kept so existing call sites stay valid
    return manuscript_star(p_corrected)
