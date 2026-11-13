"""Joint observation noise x delay sweep: BLEND versus PAG at medium threat.

Repeats the delay sweep at sigma = 0 and sigma = 5 to test whether noise moves
the point at which PAG's advantage collapses (dissertation Section 5.4).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pag_arbitration.experiments._common import seeds, write_csv
from pag_arbitration.simulation import SEED_BASE, EpisodeSpec, run_sweep

DELAY_LEVELS = [0, 1, 2, 3, 4, 5]
NOISE_LEVELS = [0.0, 5.0]
CONTROLLERS = ["Blend", "PAG"]
THREAT = "medium"


def run(
    out_dir: Path, episodes: int = 100, seed_base: int = SEED_BASE, workers: int | None = None
) -> pd.DataFrame:
    """Run the joint sweep and write ``ablation_delay_noise_data.csv`` to ``out_dir``."""
    specs = [
        EpisodeSpec(controller=ctrl, threat=THREAT, noise=noise, delay=delay, seed=seed)
        for ctrl in CONTROLLERS
        for noise in NOISE_LEVELS
        for delay in DELAY_LEVELS
        for seed in seeds(episodes, seed_base)
    ]
    results = run_sweep(specs, workers=workers, label="noise x delay")
    df = pd.DataFrame(
        [
            {
                "Controller": s.controller,
                "Threat": s.threat,
                "Noise": s.noise,
                "Delay": s.delay,
                "Seed": s.seed,
                "Survival_Steps": r.survival_steps,
                "Normalised_AUC": r.normalised_auc,
            }
            for s, r in zip(specs, results, strict=True)
        ]
    )
    write_csv(df, out_dir, "ablation_delay_noise_data.csv")
    return df
