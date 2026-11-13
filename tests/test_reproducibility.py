"""Golden-master check: episodes reproduce the dissertation's submitted results.

Values are copied from factorial_master_data.csv as generated for the
dissertation. A small sample of seeds is enough to catch any change to the
environment dynamics, the controllers or the RNG call order.
"""

from __future__ import annotations

import pytest

from pag_arbitration.simulation import EpisodeSpec, run_episode

# (controller, threat, noise, delay, seed, survival steps, normalised AUC)
GOLDEN = [
    ("Blend", "medium", 0.0, 0, 1000, 81, 0.0),
    ("Blend", "medium", 0.0, 0, 1001, 1090, 8.446096420288086),
    ("Switch", "medium", 0.0, 0, 1000, 122, 0.4146927297115326),
    ("Switch", "medium", 0.0, 0, 1001, 731, 4.221476078033447),
    ("PAG", "medium", 0.0, 0, 1000, 189, 1.4306736478887148),
    ("PAG", "medium", 0.0, 0, 1001, 3047, 1.298916422030593),
    ("Blend", "high", 5.0, 2, 1000, 366, 0.4856656491756439),
    ("Blend", "high", 5.0, 2, 1001, 449, 1.1867045164108276),
    ("Switch", "high", 5.0, 2, 1000, 263, 0.0912359580397605),
    ("Switch", "high", 5.0, 2, 1001, 194, 0.2164600789546966),
    ("PAG", "high", 5.0, 2, 1000, 405, 0.0),
    ("PAG", "high", 5.0, 2, 1001, 189, 0.0928407698396646),
]


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: f"{c[0]}-{c[1]}-s{c[2]:g}-d{c[3]}-{c[4]}")
def test_matches_dissertation(case: tuple[str, str, float, int, int, int, float]) -> None:
    controller, threat, noise, delay, seed, steps, auc = case
    spec = EpisodeSpec(controller=controller, threat=threat, noise=noise, delay=delay, seed=seed)
    result = run_episode(spec)
    assert result.survival_steps == steps
    assert result.normalised_auc == pytest.approx(auc, rel=1e-9, abs=1e-12)


def test_same_seed_same_result() -> None:
    spec = EpisodeSpec(controller="PAG", threat="low", noise=5.0, delay=1, seed=7)
    assert run_episode(spec) == run_episode(spec)
