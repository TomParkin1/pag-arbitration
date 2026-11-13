"""Summary tables and extra figures built from the experiment CSVs.

Added after the dissertation was submitted. In the submitted artefact these
results were produced by ad-hoc steps that were not bundled with the code:
the median/IQR survival table, the bootstrap confidence intervals on Cohen's d,
the switch-rate analysis of the hysteresis ablation, the low-threat survival
distribution figure and the noise x delay figure. This module regenerates them
from the experiment outputs (bootstrap: 2,000 percentile resamples, seed 42).
Run on the full-configuration data, every point estimate equals the value in
the dissertation; because the original resampling stream was not recorded, a
few bootstrap interval bounds differ from it by up to 0.02.

It reads whichever CSVs exist in ``out_dir`` and skips the rest, then writes
``summary.md`` with every headline table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pag_arbitration.experiments._common import THREAT_LEVELS, write_csv
from pag_arbitration.plotting import CONTROLLER_COLOURS, save_figure
from pag_arbitration.simulation import CONTROLLER_NAMES
from pag_arbitration.stats import bootstrap_cohens_d_ci, ci95, cohens_d

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
INPUT_FILES = (
    "factorial_master_data.csv",
    "ablation_hysteresis_data.csv",
    "ablation_delay_noise_data.csv",
    "recovery_decomposition.csv",
    "sensitivity_beta_data.csv",
)
EFFECT_SIZE_METRICS = {
    "Survival_Steps": "Survival steps",
    "Normalised_AUC": "Normalised AUC",
    "Avg_Recovery_Time": "Avg. recovery time",
}


def survival_summary(factorial: pd.DataFrame) -> pd.DataFrame:
    """Mean, median and interquartile range of survival per threat and controller.

    Pooled over the noise and delay conditions (N = 400 per cell at full scale).
    """
    rows = []
    for threat in THREAT_LEVELS:
        for ctrl in CONTROLLER_NAMES:
            x = factorial.loc[
                (factorial["Threat"] == threat) & (factorial["Controller"] == ctrl),
                "Survival_Steps",
            ].to_numpy()
            q1, median, q3 = np.percentile(x, [25, 50, 75])
            rows.append(
                {
                    "Threat": threat,
                    "Controller": ctrl,
                    "N": len(x),
                    "Mean": x.mean(),
                    "Median": median,
                    "Q1": q1,
                    "Q3": q3,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_effect_sizes(factorial: pd.DataFrame) -> pd.DataFrame:
    """Cohen's d (PAG vs SWITCH) with 95% percentile-bootstrap CIs per threat and metric."""
    rows = []
    for threat in THREAT_LEVELS:
        cell = factorial[factorial["Threat"] == threat]
        pag = cell[cell["Controller"] == "PAG"]
        switch = cell[cell["Controller"] == "Switch"]
        for metric in EFFECT_SIZE_METRICS:
            a, b = pag[metric].to_numpy(), switch[metric].to_numpy()
            low, high = bootstrap_cohens_d_ci(
                a, b, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
            )
            rows.append(
                {
                    "Threat": threat,
                    "Metric": metric,
                    "Cohen_d": cohens_d(a, b),
                    "CI_Low": low,
                    "CI_High": high,
                }
            )
    return pd.DataFrame(rows)


def hysteresis_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Survival, per-step switch rate and AUC with and without hysteresis."""
    df = df.assign(Switch_Rate=df["State_Switches"] / df["Survival_Steps"])
    on, off = df[df["Hysteresis"]], df[~df["Hysteresis"]]
    rows = []
    for metric in ("Survival_Steps", "Switch_Rate", "Normalised_AUC"):
        rows.append(
            {
                "Metric": metric,
                "Hysteresis_On": on[metric].mean(),
                "Hysteresis_Off": off[metric].mean(),
                "Cohen_d_On_vs_Off": cohens_d(on[metric].to_numpy(), off[metric].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def delay_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Mean survival and Cohen's d (PAG minus BLEND) at each noise and delay level."""
    rows = []
    for (noise, delay), cell in df.groupby(["Noise", "Delay"]):
        pag = cell.loc[cell["Controller"] == "PAG", "Survival_Steps"].to_numpy()
        blend = cell.loc[cell["Controller"] == "Blend", "Survival_Steps"].to_numpy()
        rows.append(
            {
                "Noise": noise,
                "Delay": delay,
                "PAG_Mean": pag.mean(),
                "PAG_CI95": ci95(pag),
                "Blend_Mean": blend.mean(),
                "Blend_CI95": ci95(blend),
                "Cohen_d_PAG_vs_Blend": cohens_d(pag, blend),
            }
        )
    return pd.DataFrame(rows)


def plot_survival_distribution(factorial: pd.DataFrame, path: Path, threat: str = "low") -> None:
    """Violin plot with jittered points of per-episode survival at one threat level."""
    cell = factorial[factorial["Threat"] == threat]
    data = [
        cell.loc[cell["Controller"] == c, "Survival_Steps"].to_numpy() for c in CONTROLLER_NAMES
    ]
    positions = np.arange(1, len(CONTROLLER_NAMES) + 1)
    rng = np.random.default_rng(0)  # jitter only

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, positions=positions, showextrema=False, widths=0.7)
    bodies = cast(list[Any], parts["bodies"])
    for body, ctrl in zip(bodies, CONTROLLER_NAMES, strict=True):
        body.set_facecolor(CONTROLLER_COLOURS[ctrl])
        body.set_edgecolor("grey")
        body.set_alpha(0.35)
    for pos, x, ctrl in zip(positions, data, CONTROLLER_NAMES, strict=True):
        ax.scatter(
            pos + rng.uniform(-0.12, 0.12, len(x)),
            x,
            s=8,
            alpha=0.4,
            color=CONTROLLER_COLOURS[ctrl],
            linewidths=0,
        )
        ax.hlines(x.mean(), pos - 0.28, pos + 0.28, colors="black", linewidth=2)
        ax.hlines(np.median(x), pos - 0.28, pos + 0.28, colors="black", linestyles="--", lw=2)
    ax.plot([], [], color="black", lw=2, label="mean")
    ax.plot([], [], color="black", lw=2, ls="--", label="median")
    ax.set_xticks(positions, CONTROLLER_NAMES)
    ax.set_ylabel("Survival steps (single episode)")
    ax.set_title(
        f"{threat.capitalize()}-threat survival per controller (n={len(data[0])} episodes)"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, path)


def plot_delay_noise(df: pd.DataFrame, effects: pd.DataFrame, path: Path) -> None:
    """Survival against delay at each noise level, and PAG-vs-BLEND effect sizes."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    styles = {0.0: "-", 5.0: "--"}
    for ctrl in ("Blend", "PAG"):
        for noise, style in styles.items():
            cell = df[(df["Controller"] == ctrl) & (df["Noise"] == noise)]
            agg = cell.groupby("Delay")["Survival_Steps"].agg(["mean", ci95])
            ax_l.errorbar(
                agg.index,
                agg["mean"],
                yerr=agg["ci95"],
                color=CONTROLLER_COLOURS[ctrl],
                linestyle=style,
                marker="o",
                capsize=4,
                linewidth=2,
                label=rf"{ctrl}, $\sigma = {noise:g}$",
            )
    ax_l.set_title(r"Survival vs delay at $\sigma = 0$ (solid) and $\sigma = 5$ (dashed)")
    ax_l.set_ylabel("Mean survival steps (95% CI)")

    for noise, style in styles.items():
        cell = effects[effects["Noise"] == noise]
        ax_r.plot(
            cell["Delay"],
            cell["Cohen_d_PAG_vs_Blend"],
            color="#444444",
            linestyle=style,
            marker="o",
            linewidth=2,
            label=rf"$\sigma = {noise:g}$",
        )
    ax_r.axhline(0, color="red", linestyle=":", linewidth=1)
    ax_r.set_title("PAG vs Blend effect size by delay and noise")
    ax_r.set_ylabel("Cohen's d (PAG - Blend)")

    for ax in (ax_l, ax_r):
        ax.set_xlabel(r"Sensorimotor delay $\delta$ (steps)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, path)


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    def fmt(v: object) -> str:
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def _read(out_dir: Path, name: str) -> pd.DataFrame | None:
    path = out_dir / name
    if not path.exists():
        print(f"Skipping analyses that need {path} (not found).")
        return None
    return pd.read_csv(path)


def run(out_dir: Path) -> bool:
    """Build every available table and figure from the CSVs in ``out_dir``.

    Returns False (writing nothing) if none of the experiment CSVs exist.
    """
    if not any((out_dir / name).exists() for name in INPUT_FILES):
        return False
    sections: list[str] = ["# Results summary", ""]

    factorial = _read(out_dir, "factorial_master_data.csv")
    if factorial is not None:
        surv = survival_summary(factorial)
        write_csv(surv, out_dir, "survival_summary.csv")
        shown = surv.assign(
            **{c: surv[c].round().astype(int) for c in ("Mean", "Median", "Q1", "Q3")}
        )
        sections += ["## Survival steps (pooled over noise and delay)", _markdown_table(shown), ""]

        effects = bootstrap_effect_sizes(factorial)
        write_csv(effects, out_dir, "effect_sizes_bootstrap.csv")
        sections += [
            "## Cohen's d, PAG vs Switch (95% bootstrap CI)",
            _markdown_table(effects),
            "",
        ]
        plot_survival_distribution(factorial, out_dir / "low_threat_survival_distribution.png")

    hysteresis = _read(out_dir, "ablation_hysteresis_data.csv")
    if hysteresis is not None:
        hyst = hysteresis_summary(hysteresis)
        write_csv(hyst, out_dir, "hysteresis_summary.csv")
        sections += ["## Hysteresis ablation (medium threat)", _markdown_table(hyst, ".4g"), ""]

    delay_noise = _read(out_dir, "ablation_delay_noise_data.csv")
    if delay_noise is not None:
        dn = delay_effect_sizes(delay_noise)
        write_csv(dn, out_dir, "delay_noise_effect_sizes.csv")
        sections += ["## Delay x noise (medium threat)", _markdown_table(dn), ""]
        plot_delay_noise(delay_noise, dn, out_dir / "ablation_delay_noise_degradation.png")

    recovery = _read(out_dir, "recovery_decomposition.csv")
    if recovery is not None:
        rec = (
            recovery.groupby(["Threat", "Controller"], sort=False)[
                ["Num_True_Recoveries", "Avg_True_Recovery", "Has_Terminal_Deficit"]
            ]
            .mean()
            .reset_index()
        )
        sections += [
            "## Recovery decomposition (means per episode)",
            _markdown_table(rec, ".3f"),
            "",
        ]

    beta = _read(out_dir, "sensitivity_beta_data.csv")
    if beta is not None:
        b = beta.groupby("Beta")[["Survival_Steps", "Normalised_AUC"]].mean().reset_index()
        sections += ["## Blend sensitivity to beta (medium threat)", _markdown_table(b), ""]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved {summary_path}")
    return True
