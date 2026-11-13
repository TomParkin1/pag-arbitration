"""Hysteresis ablation: PAG with and without its asymmetric deadband.

At medium threat with no noise or delay, compares the PAG controller with its
asymmetric entry/exit thresholds against a variant where entry and exit
coincide (dissertation Section 5.3). Removing the deadband causes "chattering"
between states and roughly halves survival.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pag_arbitration.experiments._common import seeds, write_csv
from pag_arbitration.plotting import save_figure, save_table_png
from pag_arbitration.simulation import SEED_BASE, EpisodeSpec, run_sweep
from pag_arbitration.stats import ci95

LABELS = {True: "PAG (Hysteresis ON)", False: "PAG (Hysteresis OFF)"}
COLOURS = ["#2ca02c", "#d62728"]


def run_sweep_df(
    episodes: int, seed_base: int = SEED_BASE, workers: int | None = None
) -> pd.DataFrame:
    """Run PAG with hysteresis on and off; one row per episode."""
    specs = [
        EpisodeSpec(controller="PAG", use_hysteresis=use_h, seed=seed)
        for use_h in (True, False)
        for seed in seeds(episodes, seed_base)
    ]
    results = run_sweep(specs, workers=workers, label="hysteresis ablation")
    return pd.DataFrame(
        [
            {
                "Controller": LABELS[s.use_hysteresis],
                "Hysteresis": s.use_hysteresis,
                "Seed": s.seed,
                "Survival_Steps": r.survival_steps,
                "Normalised_AUC": r.normalised_auc,
                "State_Switches": r.switch_count,
            }
            for s, r in zip(specs, results, strict=True)
        ]
    )


def plot_results(df: pd.DataFrame, path: Path) -> None:
    """Bar charts of state switches, survival and AUC for both variants."""
    agg = (
        df.groupby("Controller")
        .agg(
            Mean_Switches=("State_Switches", "mean"),
            CI_Switches=("State_Switches", ci95),
            Mean_Survival=("Survival_Steps", "mean"),
            CI_Survival=("Survival_Steps", ci95),
            Mean_AUC=("Normalised_AUC", "mean"),
            CI_AUC=("Normalised_AUC", ci95),
        )
        .reset_index()
    )
    agg["Controller"] = pd.Categorical(
        agg["Controller"], categories=list(LABELS.values()), ordered=True
    )
    agg = agg.sort_values("Controller")

    panels = [
        ("Switches", "Total State Switches per Episode (95% CI)", "Mean State Switches"),
        ("Survival", "Survival Steps (95% CI)", "Mean Survival Steps"),
        ("AUC", "Normalised AUC (95% CI)", "Energy Deviation per Step"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (metric, title, ylabel) in zip(axes, panels, strict=True):
        ax.bar(
            agg["Controller"].astype(str),
            agg[f"Mean_{metric}"],
            yerr=agg[f"CI_{metric}"],
            color=COLOURS,
            capsize=8,
            edgecolor="black",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.tick_params(axis="x", labelrotation=10)

    fig.suptitle(
        "Hysteresis Ablation: Chattering vs. Stability (Medium Threat)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, path)


def run(
    out_dir: Path, episodes: int = 100, seed_base: int = SEED_BASE, workers: int | None = None
) -> pd.DataFrame:
    """Run the hysteresis ablation and write its CSV and figures to ``out_dir``."""
    df = run_sweep_df(episodes, seed_base, workers)
    write_csv(df, out_dir, "ablation_hysteresis_data.csv")

    summary = (
        df.groupby("Controller")
        .agg(
            Mean_Switches=("State_Switches", "mean"),
            Mean_Survival=("Survival_Steps", "mean"),
            Mean_AUC=("Normalised_AUC", "mean"),
        )
        .round(3)
    )
    print(summary.to_string())
    save_table_png(
        summary,
        out_dir / "ablation_hysteresis_summary.png",
        title="Hysteresis Ablation Summary (Medium Threat)",
    )
    plot_results(df, out_dir / "ablation_chattering.png")
    return df
