"""Watch a controller in the environment, or run a few headless episodes.

This is a sanity-check and demonstration tool; it is not used to produce any
of the dissertation's results. Run it with ``pag-arbitration demo``.
"""

from __future__ import annotations

import pandas as pd

from pag_arbitration.environment import PAGArbitrationEnv
from pag_arbitration.simulation import CONTROLLER_NAMES, make_controller


def run_demo(
    controller: str = "PAG",
    seed: int = 42,
    episodes: int = 1,
    render: bool = True,
    threat: str = "medium",
) -> pd.DataFrame:
    """Run ``episodes`` episodes with seeds ``seed, seed + 1, ...``.

    Args:
        controller: One of "Blend", "Switch" or "PAG".
        seed: Seed of the first episode.
        episodes: Number of episodes to run.
        render: If True, draw each step in a pygame window. Closing the window
            stops the demo early.
        threat: Predator threat level ("low", "medium" or "high").

    Returns:
        One row per completed episode with survival steps, final energy and AUC.
    """
    if controller not in CONTROLLER_NAMES:
        raise ValueError(f"Unknown controller {controller!r}; expected one of {CONTROLLER_NAMES}.")

    env = PAGArbitrationEnv(
        config={"threat_level": threat, "render_mode": "human" if render else None}
    )
    agent = make_controller(controller)
    results: list[dict[str, float]] = []

    for episode_seed in range(seed, seed + episodes):
        obs, info = env.reset(seed=episode_seed)
        agent.reset_tracking()
        total_auc = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            if render and _window_closed():
                env.close()
                return pd.DataFrame(results)
            obs, _reward, terminated, truncated, info = env.step(agent.act(obs))
            total_auc += info["deviation"]
            if render:
                env.render()

        survival = info["survival_step"]
        results.append(
            {
                "episode_seed": episode_seed,
                "survival_steps": survival,
                "final_energy": info["energy"],
                "total_auc": total_auc,
                "normalised_auc": total_auc / survival if survival > 0 else 0.0,
            }
        )

    env.close()
    return pd.DataFrame(results)


def _window_closed() -> bool:
    import pygame

    return any(event.type == pygame.QUIT for event in pygame.event.get())
