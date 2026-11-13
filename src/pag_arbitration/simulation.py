"""Running episodes and parallel sweeps.

Every experiment is a grid of :class:`EpisodeSpec` objects. Each spec is run
independently (one fresh environment and controller per episode, seeded via
``env.reset(seed=...)``), so results are deterministic per seed and do not
depend on how episodes are scheduled across worker processes.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from pag_arbitration.controller import (
    ContinuousBlendController,
    Controller,
    PAGSwitchController,
    UtilitySwitchController,
)
from pag_arbitration.environment import SAFE_ENERGY_THRESHOLD, PAGArbitrationEnv

CONTROLLER_NAMES = ("Blend", "Switch", "PAG")
BLEND_BETA = 2.0  # softmax inverse temperature used throughout the study
SEED_BASE = 1000  # episode i of every condition uses seed SEED_BASE + i


def make_controller(
    name: str, *, beta: float = BLEND_BETA, use_hysteresis: bool = True
) -> Controller:
    """Build a controller by name ("Blend", "Switch" or "PAG")."""
    if name == "Blend":
        return ContinuousBlendController(beta=beta)
    if name == "Switch":
        return UtilitySwitchController()
    if name == "PAG":
        return PAGSwitchController(use_hysteresis=use_hysteresis)
    raise ValueError(f"Unknown controller {name!r}; expected one of {CONTROLLER_NAMES}.")


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything needed to run one episode reproducibly."""

    controller: str
    seed: int
    threat: str = "medium"
    noise: float = 0.0
    delay: int = 0
    beta: float = BLEND_BETA
    use_hysteresis: bool = True


@dataclass(frozen=True)
class EpisodeResult:
    """Per-episode metrics.

    Attributes:
        survival_steps: Steps until starvation or capture.
        normalised_auc: Mean energy deficit below the viability band per step.
        time_outside_band: Steps spent with energy below the band.
        max_recovery_time, avg_recovery_time: Deficit durations as reported by
            the environment (an open deficit at death counts as a recovery).
        switch_count: Changes of the controller's dominant mode.
        true_recoveries: Durations of deficits that closed while alive.
        terminal_deficit: Length of a deficit still open at death (0 if none).
    """

    survival_steps: int
    normalised_auc: float
    time_outside_band: int
    max_recovery_time: float
    avg_recovery_time: float
    switch_count: int
    true_recoveries: tuple[int, ...]
    terminal_deficit: int


def run_episode(spec: EpisodeSpec) -> EpisodeResult:
    """Run a single episode to termination and return its metrics."""
    agent = make_controller(spec.controller, beta=spec.beta, use_hysteresis=spec.use_hysteresis)
    env = PAGArbitrationEnv(
        config={"threat_level": spec.threat, "obs_noise": spec.noise, "delay": spec.delay}
    )
    obs, info = env.reset(seed=spec.seed)
    # Kept from the original scripts; actions are never sampled, so this has no effect.
    env.action_space.seed(spec.seed)

    total_auc = 0.0
    recoveries: list[int] = []
    open_deficit = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _reward, terminated, truncated, info = env.step(agent.act(obs))
        total_auc += info["deviation"]

        # Separate deficits that recover from the one (if any) still open at death.
        if info["energy"] < SAFE_ENERGY_THRESHOLD:
            open_deficit += 1
        elif open_deficit > 0:
            recoveries.append(open_deficit)
            open_deficit = 0
    env.close()

    survival = info["survival_step"]
    return EpisodeResult(
        survival_steps=survival,
        # Normalise by survival time so short episodes are not rewarded for their brevity.
        normalised_auc=total_auc / survival if survival > 0 else 0.0,
        time_outside_band=info["time_outside_band"],
        max_recovery_time=info["max_recovery_time"],
        avg_recovery_time=info["avg_recovery_time"],
        switch_count=agent.switch_count,
        true_recoveries=tuple(recoveries),
        terminal_deficit=open_deficit,
    )


def default_workers() -> int:
    """Use all but one CPU core so the machine stays responsive."""
    return max(1, (os.cpu_count() or 1) - 1)


def run_sweep(
    specs: Sequence[EpisodeSpec],
    *,
    workers: int | None = None,
    label: str = "sweep",
    progress: Callable[[str], None] | None = print,
) -> list[EpisodeResult]:
    """Run episodes in parallel; results are returned in the same order as ``specs``."""
    workers = default_workers() if workers is None else workers
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}.")
    if progress:
        progress(f"[{label}] {len(specs)} episodes on {workers} worker(s)...")
    if workers == 1:
        results = [run_episode(s) for s in specs]
    else:
        chunksize = max(1, len(specs) // (workers * 8))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_episode, specs, chunksize=chunksize))
    if progress:
        progress(f"[{label}] done.")
    return results
