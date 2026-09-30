"""Weighted Pearson correlation with permutation p-value (channel-level)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class WeightedCorr:
    """Weighted Pearson correlation coefficient with a shuffle p-value."""

    def __init__(
        self,
        w: Optional[np.ndarray] = None,
        num_shuffle: int = 9999,
        seed: int = 0,
    ) -> None:
        self.w = w
        self.num_shuffle = num_shuffle
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def cov(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        return float(
            np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w)))
            / np.sum(w)
        )

    def corr(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        denom = np.sqrt(self.cov(x, x, w) * self.cov(y, y, w))
        if denom == 0:
            return float("nan")
        return float(self.cov(x, y, w) / denom)

    def __call__(
        self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
    ) -> tuple[float, float]:
        if w is not None:
            self.w = w
        elif self.w is None:
            self.w = np.ones_like(x, dtype=np.float64)

        weights = np.asarray(self.w, dtype=np.float64)
        corr = self.corr(x, y, weights)
        corr_shuffled = np.zeros(self.num_shuffle, dtype=np.float64)
        for i in range(self.num_shuffle):
            corr_shuffled[i] = self.corr(x, self.rng.permutation(y), weights)
        p_value = np.sum(corr_shuffled > corr) / (self.num_shuffle + 1)
        return corr, float(p_value)
