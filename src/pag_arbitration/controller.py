"""The three arbitration controllers compared in the study.

All controllers share the same utility functions and action primitives, so
the arbitration rule is the only thing that differs between them:

* ``ContinuousBlendController`` — softmax-weighted command fusion (BLEND).
* ``UtilitySwitchController`` — winner-takes-all argmax over utilities (SWITCH).
* ``PAGSwitchController`` — a three-state FSM with asymmetric hysteresis,
  inspired by the midbrain periaqueductal grey (PAG), layered over SWITCH.

Every controller exposes ``act(obs)``, ``reset_tracking()``, ``mode`` and
``switch_count`` (the number of changes of dominant mode in the episode).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

Action = NDArray[np.floating[Any]]

MAX_ENERGY = 100.0
THREAT_RADIUS = 40.0  # predator distance at which the avoid utility becomes non-zero
DEFAULT_BETA = 1.0  # note: every experiment uses beta=2.0 (see simulation.BLEND_BETA)


class Controller(Protocol):
    """Interface shared by all controllers."""

    mode: str | None
    switch_count: int

    def act(self, obs: NDArray[np.float32]) -> Action: ...

    def reset_tracking(self) -> None: ...


def _utilities(energy: Any, dist_pred: Any, threat_radius: float) -> tuple[Any, Any]:
    """Return ``(u_forage, u_avoid)``, both in ``[0, 1]``."""
    u_forage = (MAX_ENERGY - energy) / MAX_ENERGY
    u_avoid = np.clip(1.0 - (dist_pred / threat_radius), 0.0, 1.0)
    return u_forage, u_avoid


def _towards(angle: Any) -> Action:
    return np.array([np.cos(angle), np.sin(angle)])


def _away_from(angle: Any) -> Action:
    return np.array([-np.cos(angle), -np.sin(angle)])


class ContinuousBlendController:
    """Cooperative command fusion: a softmax over utilities weights the two actions.

    Args:
        beta: Softmax inverse temperature. Higher values approach winner-takes-all.
        threat_radius: Distance at which the avoid utility starts to rise.
    """

    def __init__(self, beta: float = DEFAULT_BETA, threat_radius: float = THREAT_RADIUS) -> None:
        self.beta = beta
        self.threat_radius = threat_radius
        self.mode: str | None = None
        self.switch_count = 0

    def reset_tracking(self) -> None:
        """Clear mode and switch-count tracking for a new episode."""
        self.mode = None
        self.switch_count = 0

    def act(self, obs: NDArray[np.float32]) -> Action:
        """Return the blended velocity command, clipped to ``[-1, 1]``."""
        energy, dist_pred, angle_pred, _dist_food, angle_food = obs
        u_forage, u_avoid = _utilities(energy, dist_pred, self.threat_radius)

        exp_f = np.exp(self.beta * u_forage)
        exp_a = np.exp(self.beta * u_avoid)
        sum_exp = exp_f + exp_a

        w_forage = exp_f / sum_exp
        w_avoid = exp_a / sum_exp

        # Track the dominant behavioural mode for chattering analysis.
        new_mode = "AVOID" if w_avoid > w_forage else "FORAGE"
        if self.mode is not None and new_mode != self.mode:
            self.switch_count += 1
        self.mode = new_mode

        # The blended vector can be shorter than 1, so blending also slows the agent.
        blended_action: Action = (w_forage * _towards(angle_food)) + (
            w_avoid * _away_from(angle_pred)
        )
        return np.clip(blended_action, -1.0, 1.0)


class UtilitySwitchController:
    """Winner-takes-all arbitration: execute only the higher-utility action.

    This is the non-biological baseline for discrete action selection. Ties go
    to foraging.
    """

    def __init__(self, threat_radius: float = THREAT_RADIUS) -> None:
        self.threat_radius = threat_radius
        self.mode: str | None = None
        self.switch_count = 0

    def reset_tracking(self) -> None:
        """Clear mode and switch-count tracking for a new episode."""
        self.mode = None
        self.switch_count = 0

    def act(self, obs: NDArray[np.float32]) -> Action:
        """Return a unit vector towards food or away from the predator."""
        energy, dist_pred, angle_pred, _dist_food, angle_food = obs
        u_forage, u_avoid = _utilities(energy, dist_pred, self.threat_radius)

        if u_avoid > u_forage:
            new_mode = "AVOID"
            action = _away_from(angle_pred)
        else:
            new_mode = "FORAGE"
            action = _towards(angle_food)

        if self.mode is not None and new_mode != self.mode:
            self.switch_count += 1
        self.mode = new_mode

        return action


class PAGSwitchController:
    """PAG-inspired subsumption controller: a DEFAULT/FREEZE/FLIGHT state machine.

    In DEFAULT the base controller (SWITCH by default) acts. When the predator
    comes within ``freeze_entry`` the agent freezes (zero velocity, which shrinks
    the predator's vision radius); within ``flight_entry`` it flees directly
    away. Exit thresholds are larger than entry thresholds, creating a deadband
    (hysteresis) that prevents rapid oscillation between states. The FSM reacts
    to predator distance only; energy is ignored.

    Args:
        base_controller: Controller used in the DEFAULT state.
        use_hysteresis: If False, exit thresholds equal entry thresholds (ablation).
        freeze_entry, freeze_exit, flight_entry, flight_exit: Distance thresholds.
            The exit values apply only when ``use_hysteresis`` is True.
    """

    def __init__(
        self,
        base_controller: Controller | None = None,
        use_hysteresis: bool = True,
        *,
        freeze_entry: float = 40.0,
        freeze_exit: float = 55.0,
        flight_entry: float = 20.0,
        flight_exit: float = 30.0,
    ) -> None:
        self.base_controller: Controller = base_controller or UtilitySwitchController()
        self.use_hysteresis = use_hysteresis

        self.freeze_entry = freeze_entry
        self.flight_entry = flight_entry
        if use_hysteresis:
            # Asymmetric thresholds: the exit band sits beyond the predator's vision
            # radius, so a freshly released agent is not instantly re-acquired.
            self.freeze_exit = freeze_exit
            self.flight_exit = flight_exit
        else:
            # Collapsed thresholds: entry and exit coincide, removing the deadband.
            self.freeze_exit = freeze_entry
            self.flight_exit = flight_entry

        self.current_state = "DEFAULT"
        self.mode: str | None = "DEFAULT"
        self.switch_count = 0

    def reset_tracking(self) -> None:
        """Return to DEFAULT and clear the switch count for a new episode."""
        self.current_state = "DEFAULT"
        self.mode = "DEFAULT"
        self.switch_count = 0

    def _next_state(self, dist_pred: float) -> str:
        state = self.current_state
        if state == "DEFAULT":
            if dist_pred < self.flight_entry:
                return "FLIGHT"
            if dist_pred < self.freeze_entry:
                return "FREEZE"
        elif state == "FREEZE":
            if dist_pred < self.flight_entry:
                return "FLIGHT"
            if dist_pred > self.freeze_exit:
                return "DEFAULT"
        elif state == "FLIGHT" and dist_pred > self.flight_exit:
            return "FREEZE" if dist_pred < self.freeze_exit else "DEFAULT"
        return state

    def act(self, obs: NDArray[np.float32]) -> Action:
        """Update the FSM state from predator distance and return its action."""
        _energy, dist_pred, angle_pred, _dist_food, _angle_food = obs

        new_state = self._next_state(dist_pred)
        if new_state != self.current_state:
            self.switch_count += 1
        self.current_state = new_state
        self.mode = new_state

        if new_state == "FREEZE":
            return np.array([0.0, 0.0])
        if new_state == "FLIGHT":
            return _away_from(angle_pred)
        return self.base_controller.act(obs)
