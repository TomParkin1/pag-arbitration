"""Environment dynamics, termination and validation."""

from __future__ import annotations

import numpy as np
import pytest

from pag_arbitration.environment import (
    MAX_ENERGY,
    METABOLIC_DECAY,
    MOVEMENT_COST_FACTOR,
    PAGArbitrationEnv,
)


def test_reset_returns_valid_observation() -> None:
    env = PAGArbitrationEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert obs[0] == MAX_ENERGY
    assert info == {}


def test_energy_cost_of_standing_still_and_moving() -> None:
    env = PAGArbitrationEnv(config={"threat_level": "none"})
    env.reset(seed=0)
    env.food_pos[:] = -1000  # out of reach
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["energy"] == pytest.approx(MAX_ENERGY - METABOLIC_DECAY)
    _, _, _, _, info = env.step(np.array([1.0, 0.0]))
    expected = MAX_ENERGY - 2 * METABOLIC_DECAY - 1.5 * MOVEMENT_COST_FACTOR
    assert info["energy"] == pytest.approx(expected)


def test_starvation_terminates_episode() -> None:
    env = PAGArbitrationEnv(config={"threat_level": "none"})
    env.reset(seed=0)
    env.food_pos[:] = -1000
    terminated, steps = False, 0
    while not terminated:
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        assert not truncated
        steps += 1
    assert steps == pytest.approx(MAX_ENERGY / METABOLIC_DECAY, abs=1)
    assert info["survival_step"] == steps
    assert info["time_outside_band"] > 0


def test_capture_terminates_episode() -> None:
    env = PAGArbitrationEnv(config={"threat_level": "low"})  # predator speed 1.2
    env.reset(seed=0)
    env.agent_pos = np.array([100.0, 100.0], dtype=np.float32)
    # A still agent is seen within 20 units; one chase step closes 1.6 to 0.4 (< 1.0).
    env.predator_pos = np.array([101.6, 100.0], dtype=np.float32)
    _, _, terminated, _, _ = env.step(np.zeros(2))
    assert terminated


def test_eating_restores_energy_and_respawns_food() -> None:
    env = PAGArbitrationEnv(config={"threat_level": "none"})
    env.reset(seed=0)
    env.energy = 50.0
    env.food_pos = env.agent_pos.copy()
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["energy"] == pytest.approx(70.0 - METABOLIC_DECAY)
    assert not np.allclose(env.food_pos, env.agent_pos)


def test_unknown_threat_level_rejected() -> None:
    with pytest.raises(ValueError, match="threat_level"):
        PAGArbitrationEnv(config={"threat_level": "extreme"})
