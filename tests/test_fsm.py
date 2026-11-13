"""PAG state machine transitions (dissertation Table 3.1 / Algorithm 3.1)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pag_arbitration.controller import PAGSwitchController


def make_obs(dist_pred: float, energy: float = 100.0) -> NDArray[np.float32]:
    """Observation with a chosen predator distance; predator and food both at angle 0."""
    return np.array([energy, dist_pred, 0.0, 50.0, 0.0], dtype=np.float32)


# (current state, predator distance, expected next state)
HYSTERESIS_TRANSITIONS = [
    ("DEFAULT", 70.0, "DEFAULT"),  # no threat
    ("DEFAULT", 50.0, "DEFAULT"),  # outside freeze entry
    ("DEFAULT", 30.0, "FREEZE"),  # inside freeze entry, outside flight entry
    ("DEFAULT", 10.0, "FLIGHT"),  # inside flight entry
    ("FREEZE", 10.0, "FLIGHT"),  # escalate
    ("FREEZE", 30.0, "FREEZE"),  # hold inside freeze band
    ("FREEZE", 50.0, "FREEZE"),  # between entry and exit: hysteresis holds
    ("FREEZE", 70.0, "DEFAULT"),  # beyond freeze exit: release
    ("FLIGHT", 10.0, "FLIGHT"),  # hold inside flight band
    ("FLIGHT", 25.0, "FLIGHT"),  # between flight thresholds: hysteresis holds
    ("FLIGHT", 40.0, "FREEZE"),  # above flight exit, below freeze exit: de-escalate
    ("FLIGHT", 70.0, "DEFAULT"),  # above freeze exit: release fully
]


@pytest.mark.parametrize(("state", "dist", "expected"), HYSTERESIS_TRANSITIONS)
def test_transitions_with_hysteresis(state: str, dist: float, expected: str) -> None:
    ctrl = PAGSwitchController(use_hysteresis=True)
    ctrl.current_state = state
    ctrl.act(make_obs(dist))
    assert ctrl.current_state == expected


@pytest.mark.parametrize(
    ("state", "dist", "expected"),
    [
        ("FREEZE", 50.0, "DEFAULT"),  # with hysteresis this would hold FREEZE
        ("FLIGHT", 25.0, "FREEZE"),  # with hysteresis this would hold FLIGHT
        ("FLIGHT", 45.0, "DEFAULT"),
    ],
)
def test_transitions_without_hysteresis(state: str, dist: float, expected: str) -> None:
    ctrl = PAGSwitchController(use_hysteresis=False)
    ctrl.current_state = state
    ctrl.act(make_obs(dist))
    assert ctrl.current_state == expected


def test_state_actions() -> None:
    ctrl = PAGSwitchController()
    np.testing.assert_allclose(ctrl.act(make_obs(30.0)), [0.0, 0.0])  # FREEZE: stand still
    np.testing.assert_allclose(ctrl.act(make_obs(10.0)), [-1.0, 0.0])  # FLIGHT: away (angle 0)


def test_default_state_delegates_to_base_controller() -> None:
    ctrl = PAGSwitchController()
    # Low energy, no threat: the SWITCH base controller forages towards the food (angle 0).
    np.testing.assert_allclose(ctrl.act(make_obs(100.0, energy=10.0)), [1.0, 0.0])


def test_switch_count_and_reset() -> None:
    ctrl = PAGSwitchController()
    for dist in (100.0, 30.0, 10.0, 100.0):  # DEFAULT -> FREEZE -> FLIGHT -> DEFAULT
        ctrl.act(make_obs(dist))
    assert ctrl.switch_count == 3
    ctrl.reset_tracking()
    assert (ctrl.current_state, ctrl.switch_count) == ("DEFAULT", 0)
