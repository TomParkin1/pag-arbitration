"""BLEND and SWITCH arbitration and the controller factory."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pag_arbitration.controller import ContinuousBlendController, UtilitySwitchController
from pag_arbitration.simulation import make_controller

FOOD_EAST, PREDATOR_WEST = 0.0, np.pi


def obs(energy: float, dist_pred: float) -> NDArray[np.float32]:
    """Food due east, predator due west."""
    return np.array([energy, dist_pred, PREDATOR_WEST, 30.0, FOOD_EAST], dtype=np.float32)


def test_switch_forages_when_hungry_and_safe() -> None:
    ctrl = UtilitySwitchController()
    np.testing.assert_allclose(ctrl.act(obs(energy=20.0, dist_pred=100.0)), [1.0, 0.0], atol=1e-6)
    assert ctrl.mode == "FORAGE"


def test_switch_avoids_when_threat_dominates() -> None:
    ctrl = UtilitySwitchController()
    # u_avoid = 1 - 4/40 = 0.9 > u_forage = 0.1: flee east, away from the predator.
    np.testing.assert_allclose(ctrl.act(obs(energy=90.0, dist_pred=4.0)), [1.0, 0.0], atol=1e-6)
    assert ctrl.mode == "AVOID"


def test_switch_tie_goes_to_forage() -> None:
    ctrl = UtilitySwitchController()
    ctrl.act(obs(energy=50.0, dist_pred=20.0))  # u_forage = u_avoid = 0.5
    assert ctrl.mode == "FORAGE"


def test_blend_weights_follow_softmax() -> None:
    ctrl = ContinuousBlendController(beta=2.0)
    # Food east, predator east: forage pulls +x, avoid pushes -x.
    o = np.array([50.0, 20.0, 0.0, 30.0, 0.0], dtype=np.float32)  # u_f = u_a = 0.5
    np.testing.assert_allclose(ctrl.act(o), [0.0, 0.0], atol=1e-6)  # equal weights cancel

    o_hungry = np.array([0.0, 40.0, 0.0, 30.0, 0.0], dtype=np.float32)  # u_f = 1, u_a = 0
    w_forage = np.exp(2.0) / (np.exp(2.0) + 1.0)
    np.testing.assert_allclose(ctrl.act(o_hungry), [2 * w_forage - 1, 0.0], atol=1e-6)


def test_blend_counts_mode_switches() -> None:
    ctrl = ContinuousBlendController(beta=2.0)
    for dist in (100.0, 2.0, 100.0):
        ctrl.act(obs(energy=90.0, dist_pred=dist))
    assert ctrl.switch_count == 2
    ctrl.reset_tracking()
    assert (ctrl.mode, ctrl.switch_count) == (None, 0)


@pytest.mark.parametrize("name", ["Blend", "Switch", "PAG"])
def test_actions_are_bounded(name: str) -> None:
    ctrl = make_controller(name)
    rng = np.random.default_rng(0)
    for _ in range(200):
        o = np.array(
            [rng.uniform(0, 100), rng.uniform(-10, 150), rng.uniform(-np.pi, np.pi), 10.0, 1.0],
            dtype=np.float32,
        )
        assert np.all(np.abs(ctrl.act(o)) <= 1.0 + 1e-9)


def test_factory_uses_study_beta_and_rejects_unknown_names() -> None:
    blend = make_controller("Blend")
    assert isinstance(blend, ContinuousBlendController)
    assert blend.beta == 2.0
    with pytest.raises(ValueError, match="Unknown controller"):
        make_controller("Fight")
