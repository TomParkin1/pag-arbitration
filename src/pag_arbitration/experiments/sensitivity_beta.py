"""Sensitivity of the BLEND baseline to its softmax inverse temperature (beta).

The main study fixes beta = 2. This sweep runs BLEND at medium threat for
beta in {0.5, 1, 2, 4, 8} to check that the comparison is not an artefact of
that choice (dissertation Appendix A).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pag_arbitration.experiments._common import seeds, write_csv
from pag_arbitration.simulation import SEED_BASE, EpisodeSpec, run_sweep

BETAS = [0.5, 1.0, 2.0, 4.0, 8.0]


def run(
    out_dir: Path, episodes: int = 100, seed_base: int = SEED_BASE, workers: int | None = None
) -> pd.DataFrame:
    """Run the beta sweep and write ``sensitivity_beta_data.csv`` to ``out_dir``."""
    specs = [
        EpisodeSpec(controller="Blend", beta=beta, seed=seed)
        for beta in BETAS
        for seed in seeds(episodes, seed_base)
    ]
    results = run_sweep(specs, workers=workers, label="beta sensitivity")
    df = pd.DataFrame(
        [
            {
                "Beta": s.beta,
                "Seed": s.seed,
                "Survival_Steps": r.survival_steps,
                "Normalised_AUC": r.normalised_auc,
                "Switch_Count": r.switch_count,
            }
            for s, r in zip(specs, results, strict=True)
        ]
    )
    write_csv(df, out_dir, "sensitivity_beta_data.csv")
    return df
