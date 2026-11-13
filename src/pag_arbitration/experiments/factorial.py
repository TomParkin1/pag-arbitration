"""Full factorial sweep: controller x threat x observation noise x delay.

Produces the headline results (dissertation Sections 5.1-5.2): survival, metabolic
cost (normalised AUC) and recovery time per controller across threat levels,
plus Cohen's d effect sizes for PAG versus SWITCH.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pag_arbitration.experiments._common import THREAT_LEVELS, seeds, write_csv
from pag_arbitration.plotting import CONTROLLER_COLOURS, save_figure, save_table_png
from pag_arbitration.simulation import CONTROLLER_NAMES, SEED_BASE, EpisodeSpec, run_sweep
from pag_arbitration.stats import ci95, cohens_d

NOISE_LEVELS = [0.0, 5.0]
DELAY_LEVELS = [0, 2]
EFFECT_SIZE_METRICS = ["Survival_Steps", "Normalised_AUC", "Avg_Recovery_Time"]


def run_factorial_sweep(
    episodes: int, seed_base: int = SEED_BASE, workers: int | None = None
) -> pd.DataFrame:
    """Run every (threat, noise, delay, controller) cell and return one row per episode."""
    specs = [
        EpisodeSpec(controller=ctrl, threat=threat, noise=noise, delay=delay, seed=seed)
        for threat in THREAT_LEVELS
        for noise in NOISE_LEVELS
        for delay in DELAY_LEVELS
        for ctrl in CONTROLLER_NAMES
        for seed in seeds(episodes, seed_base)
    ]
    results = run_sweep(specs, workers=workers, label="factorial")
    return pd.DataFrame(
        [
            {
                "Controller": s.controller,
                "Threat": s.threat,
                "Noise": s.noise,
                "Delay": s.delay,
                "Seed": s.seed,
                "Survival_Steps": r.survival_steps,
                "Normalised_AUC": r.normalised_auc,
                "Time_Outside_Band": r.time_outside_band,
                "Max_Recovery_Time": r.max_recovery_time,
                "Avg_Recovery_Time": r.avg_recovery_time,
            }
            for s, r in zip(specs, results, strict=True)
        ]
    )


def calculate_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Cohen's d of PAG versus SWITCH for each metric and threat level.

    PAG is the experimental group, so a positive d means PAG scored higher.
    Results are pooled over the noise and delay conditions.
    """
    rows = []
    for threat in THREAT_LEVELS:
        threat_df = df[df["Threat"] == threat]
        pag = threat_df[threat_df["Controller"] == "PAG"]
        switch = threat_df[threat_df["Controller"] == "Switch"]
        row: dict[str, object] = {"Threat_Level": threat}
        for metric in EFFECT_SIZE_METRICS:
            d = cohens_d(pag[metric].to_numpy(), switch[metric].to_numpy())
            row[f"{metric}_Cohen_d"] = round(d, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_factorial_results(df: pd.DataFrame, path: Path) -> None:
    """Plot mean survival, AUC and recovery time against threat, with 95% CIs."""
    grouped = (
        df.groupby(["Controller", "Threat"])
        .agg(
            Mean_Survival=("Survival_Steps", "mean"),
            CI_Survival=("Survival_Steps", ci95),
            Mean_AUC=("Normalised_AUC", "mean"),
            CI_AUC=("Normalised_AUC", ci95),
            Mean_Recovery=("Avg_Recovery_Time", "mean"),
            CI_Recovery=("Avg_Recovery_Time", ci95),
        )
        .reset_index()
    )
    # Force low -> medium -> high on the x-axis rather than alphabetical order.
    grouped["Threat"] = pd.Categorical(grouped["Threat"], categories=THREAT_LEVELS, ordered=True)
    grouped = grouped.sort_values("Threat")

    panels = [
        ("Survival", "Mean Survival by Threat Level (95% CI)", "Survival Steps"),
        ("AUC", "Metabolic Cost (AUC) by Threat Level (95% CI)", "Normalised Energy Deviation"),
        ("Recovery", "Deficit Recovery Time (95% CI)", "Average Steps to Recover"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for ctrl in CONTROLLER_NAMES:
            subset = grouped[grouped["Controller"] == ctrl]
            ax.errorbar(
                subset["Threat"],
                subset[f"Mean_{metric}"],
                yerr=subset[f"CI_{metric}"],
                label=ctrl,
                color=CONTROLLER_COLOURS[ctrl],
                marker="o",
                capsize=5,
                linewidth=2,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Threat Intensity")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    fig.suptitle("Factorial Sweep Results: Controller Robustness", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, path)


def run(
    out_dir: Path, episodes: int = 100, seed_base: int = SEED_BASE, workers: int | None = None
) -> pd.DataFrame:
    """Run the factorial sweep and write its CSVs and figures to ``out_dir``."""
    df = run_factorial_sweep(episodes, seed_base, workers)
    write_csv(df, out_dir, "factorial_master_data.csv")
    plot_factorial_results(df, out_dir / "factorial_results.png")

    effects = calculate_effect_sizes(df)
    print(effects.to_string(index=False))
    write_csv(effects, out_dir, "effect_sizes_pag_vs_switch.csv")
    save_table_png(
        effects,
        out_dir / "effect_sizes_pag_vs_switch.png",
        title="Cohen's d: PAG vs Switch by Threat Level",
    )
    return df
