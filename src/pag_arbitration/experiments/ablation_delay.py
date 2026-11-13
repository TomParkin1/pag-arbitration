"""Sensorimotor delay sweep: BLEND versus PAG at medium threat.

Sweeps the observation delay from 0 to 5 steps with no noise (dissertation
Section 5.4) to test how command fusion and discrete mode switching degrade
when the controller acts on stale observations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pag_arbitration.experiments._common import seeds, write_csv
from pag_arbitration.plotting import CONTROLLER_COLOURS, save_figure, save_table_png
from pag_arbitration.simulation import SEED_BASE, EpisodeSpec, run_sweep
from pag_arbitration.stats import ci95

DELAY_LEVELS = [0, 1, 2, 3, 4, 5]
CONTROLLERS = ["Blend", "PAG"]
LABELS = {"Blend": "ContinuousBlend", "PAG": "PAG (Hysteresis ON)"}


def run_sweep_df(
    episodes: int, seed_base: int = SEED_BASE, workers: int | None = None
) -> pd.DataFrame:
    """Run each controller at each delay level; one row per episode."""
    specs = [
        EpisodeSpec(controller=ctrl, delay=delay, seed=seed)
        for ctrl in CONTROLLERS
        for delay in DELAY_LEVELS
        for seed in seeds(episodes, seed_base)
    ]
    results = run_sweep(specs, workers=workers, label="delay ablation")
    return pd.DataFrame(
        [
            {
                "Controller": s.controller,
                "Delay": s.delay,
                "Seed": s.seed,
                "Survival_Steps": r.survival_steps,
                "Normalised_AUC": r.normalised_auc,
                "State_Switches": r.switch_count,
            }
            for s, r in zip(specs, results, strict=True)
        ]
    )


def plot_results(df: pd.DataFrame, path: Path) -> None:
    """Plot survival and AUC against delay for each controller, with 95% CIs."""
    agg = (
        df.groupby(["Controller", "Delay"])
        .agg(
            Mean_Survival=("Survival_Steps", "mean"),
            CI_Survival=("Survival_Steps", ci95),
            Mean_AUC=("Normalised_AUC", "mean"),
            CI_AUC=("Normalised_AUC", ci95),
        )
        .reset_index()
    )
    panels = [
        ("Survival", "Survival vs Sensorimotor Delay (95% CI)", "Mean Survival Steps"),
        ("AUC", "Normalised AUC vs Sensorimotor Delay (95% CI)", "Energy Deviation per Step"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for ctrl in CONTROLLERS:
            subset = agg[agg["Controller"] == ctrl].sort_values("Delay")
            ax.errorbar(
                subset["Delay"],
                subset[f"Mean_{metric}"],
                yerr=subset[f"CI_{metric}"],
                label=LABELS[ctrl],
                color=CONTROLLER_COLOURS[ctrl],
                marker="o",
                capsize=5,
                linewidth=2,
            )
        ax.set_title(title)
        ax.set_xlabel("Observation Delay (steps)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(DELAY_LEVELS)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    fig.suptitle(
        "Delay Sensitivity: Command Fusion vs Discrete Mode-Switching (Medium Threat)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, path)


def run(
    out_dir: Path, episodes: int = 100, seed_base: int = SEED_BASE, workers: int | None = None
) -> pd.DataFrame:
    """Run the delay sweep and write its CSV and figures to ``out_dir``."""
    df = run_sweep_df(episodes, seed_base, workers)
    write_csv(df, out_dir, "ablation_delay_data.csv")

    summary = (
        df.groupby(["Controller", "Delay"])
        .agg(Mean_Survival=("Survival_Steps", "mean"), Mean_AUC=("Normalised_AUC", "mean"))
        .round(3)
    )
    print(summary.to_string())
    save_table_png(
        summary,
        out_dir / "ablation_delay_summary.png",
        title="Delay Sensitivity Summary (Medium Threat)",
    )
    plot_results(df, out_dir / "ablation_delay_degradation.png")
    return df


if __name__ == "__main__":
    run(Path("results"))
