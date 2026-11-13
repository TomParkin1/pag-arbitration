"""Confidence intervals and effect sizes."""

from __future__ import annotations

import numpy as np
import pytest

from pag_arbitration.stats import bootstrap_cohens_d_ci, ci95, cohens_d


def test_ci95_known_value() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]  # sample SD = sqrt(2.5)
    assert ci95(x) == pytest.approx(1.96 * np.sqrt(2.5) / np.sqrt(5))


def test_ci95_needs_two_values() -> None:
    assert ci95([3.0]) == 0.0


def test_cohens_d_known_value_and_sign() -> None:
    a, b = np.array([2.0, 4.0, 6.0]), np.array([1.0, 3.0, 5.0])  # pooled SD = 2
    assert cohens_d(a, b) == pytest.approx(0.5)
    assert cohens_d(b, a) == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("a", "b"),
    [([1.0], [1.0, 2.0]), ([2.0, 2.0], [2.0, 2.0])],  # too few values; zero variance
)
def test_cohens_d_degenerate_cases(a: list[float], b: list[float]) -> None:
    assert cohens_d(np.array(a), np.array(b)) == 0.0


def test_bootstrap_ci_brackets_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 1.0, 80), rng.normal(0.0, 1.0, 80)
    low, high = bootstrap_cohens_d_ci(a, b, n_resamples=500, seed=3)
    assert low < cohens_d(a, b) < high
    assert (low, high) == bootstrap_cohens_d_ci(a, b, n_resamples=500, seed=3)
