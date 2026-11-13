"""Recovery-time decomposition over the factorial grid.

Splits each episode's energy deficits into "true" recoveries (closed while
the agent is alive) and a "terminal" deficit still open at death. The
environment's recovery metric counts both, which can flatter controllers that
die quickly; this decomposition separates the two (dissertation Section 6).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pag_arbitration.experiments._common import THREAT_LEVELS, seeds, write_csv
from pag_arbitration.experiments.factorial import DELAY_LEVELS, NOISE_LEVELS
from pag_arbitration.simulation import CONTROLLER_NAMES, SEED_BASE, EpisodeSpec, run_sweep


def run(
    out_dir: Path, episodes: int = 100, seed_base: int = SEED_BASE, workers: int | None = None
) -> pd.DataFrame:
    """Run the factorial grid and write ``recovery_decomposition.csv`` to ``out_dir``."""
    specs = [
        EpisodeSpec(controller=ctrl, threat=threat, noise=noise, delay=delay, seed=seed)
        for threat in THREAT_LEVELS
        for noise in NOISE_LEVELS
        for delay in DELAY_LEVELS
        for ctrl in CONTROLLER_NAMES
        for seed in seeds(episodes, seed_base)
    ]
    results = run_sweep(specs, workers=workers, label="recovery decomposition")
    rows = []
    for s, r in zip(specs, results, strict=True):
        n_true = len(r.true_recoveries)
        sum_true = sum(r.true_recoveries)
        rows.append(
            {
                "Controller": s.controller,
                "Threat": s.threat,
                "Noise": s.noise,
                "Delay": s.delay,
                "Seed": s.seed,
                "Survival_Steps": r.survival_steps,
                "Num_True_Recoveries": n_true,
                "Avg_True_Recovery": sum_true / n_true if n_true else 0.0,
                "Sum_True_Recovery": sum_true,
                "Has_Terminal_Deficit": int(r.terminal_deficit > 0),
                "Terminal_Recovery_Duration": r.terminal_deficit,
            }
        )
    df = pd.DataFrame(rows)
    write_csv(df, out_dir, "recovery_decomposition.csv")
    return df
