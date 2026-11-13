"""Summary statistics used across the experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

Z_95 = 1.96


def ci95(x: ArrayLike) -> float:
    """Half-width of the normal-approximation 95% confidence interval of the mean.

    Returns 0.0 when fewer than two values are given.
    """
    arr = np.asarray(x, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(Z_95 * (np.std(arr, ddof=1) / np.sqrt(len(arr))))


def cohens_d(group1: Sequence[float] | np.ndarray, group2: Sequence[float] | np.ndarray) -> float:
    """Cohen's d with pooled standard deviation: ``(mean1 - mean2) / s_pooled``.

    ``group1`` is the experimental condition (e.g. PAG) and ``group2`` the
    baseline, so a positive value means group1 scored higher. Returns 0.0 if
    either group has fewer than two values or the pooled SD is zero.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)
