"""Weighted Correlation Coefficient with p-value"""

from typing import Optional, Tuple

import numpy as np


class WeightedCorr:
    """weighted correlation coefficient with p-value
    ref: https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas
    definition: https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Weighted_correlation_coefficient
    p_valuie: https://stats.stackexchange.com/questions/94569/p-value-for-weighted-pearson-correlation-coefficient
    """

    def __init__(
        self, w: Optional[np.ndarray] = None, num_shuffle: int = 9999, seed: int = 0
    ) -> None:
        """__init__

        Args:
            w (np.ndarray): weight
            num_shuffle (int, optional): num of the shuffle repeat. Defaults to 9999.
            seed (int, optional): random seed. Defaults to 0.
        """
        self.w = w
        self.num_shuffle = num_shuffle
        np.random.seed(seed)

    def cov(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Weighted Covariance

        Args:
            x (np.ndarray): x data
            y (np.ndarray): y data
            w (np.ndarray): weight

        Returns:
            float: weighted covariance
        """
        return np.sum(
            w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))
        ) / np.sum(w)

    def corr(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Weighted Correlation Coefficient

        Args:
            x (np.ndarray): x data
            y (np.ndarray): y data
            w (np.ndarray): weight

        Returns:
            float: weighted correlation coefficient
        """
        return self.cov(x, y, w) / np.sqrt(self.cov(x, x, w) * self.cov(y, y, w))

    def __call__(
        self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """__call__

        Args:
            x (np.ndarray): x data
            y (np.ndarray): y data
            w (np.ndarray): weight

        Returns:
            Tuple[float, float]: correlation coefficient and p-value
        """
        if w is not None:
            self.w = w
        else:
            if self.w is None:
                self.w = np.ones_like(x)
        corr = self.corr(x, y, self.w)
        corr_shuffled = np.zeros(self.num_shuffle)
        for i in range(self.num_shuffle):
            corr_shuffled[i] = self.corr(x, np.random.permutation(y), self.w)
        p = np.sum(corr_shuffled > corr) / (self.num_shuffle + 1)
        return corr, p
