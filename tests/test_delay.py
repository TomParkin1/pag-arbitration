"""Sensorimotor delay: the controller sees observations delta + 1 steps stale."""

from __future__ import annotations

import numpy as np
import pytest

from pag_arbitration.environment import PAGArbitrationEnv


@pytest.mark.parametrize("delta", [0, 1, 2, 3, 4, 5])
def test_observation_delay(delta: int) -> None:
    # One implicit step of latency comes from sampling after the move, plus delta
    # steps of FIFO buffering, so seen[k] == truth[k - delta - 1].
    env = PAGArbitrationEnv(config={"threat_level": "none", "delay": delta})
    obs, _ = env.reset(seed=1234)
    truth = [env._get_obs()]
    seen = [obs]
    for _ in range(delta + 3):
        obs, _r, _term, _trunc, _info = env.step(np.array([0.5, 0.5]))
        seen.append(obs)
        truth.append(env._get_obs())

    k = delta + 2
    np.testing.assert_allclose(seen[k], truth[k - delta - 1], atol=1e-4)
    # Before the buffer fills, the controller keeps seeing the initial observation.
    np.testing.assert_allclose(seen[delta], truth[0], atol=1e-4)


def test_negative_delay_rejected() -> None:
    with pytest.raises(ValueError, match="delay"):
        PAGArbitrationEnv(config={"delay": -1})
