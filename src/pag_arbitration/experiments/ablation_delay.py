"""
W6 Sensitivity Analysis: Sensorimotor delay robustness.

Sweeps sensorimotor delay over [0..5] at medium threat for the ContinuousBlend
and PAG (hysteresis ON) controllers to probe how command fusion degrades
relative to discrete mode-switching under outdated observations.
"""

import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pag_arbitration.environment import PAGArbitrationEnv
from pag_arbitration.controller import ContinuousBlendController, PAGSwitchController


DELAY_LEVELS = [0, 1, 2, 3, 4, 5]
CONTROLLERS = ["Blend", "PAG"]


def _build_agent(ctrl_name):
    if ctrl_name == "Blend":
        return ContinuousBlendController(beta=2.0)
    return PAGSwitchController(use_hysteresis=True)


def _run_single_episode(args):
    """Worker: single episode at a specified delay for the given controller."""
    ctrl_name, delay, seed = args

    agent = _build_agent(ctrl_name)
    env = PAGArbitrationEnv(config={
        "threat_level": "medium",
        "obs_noise": 0.0,
        "delay": delay,
        "render_mode": None,
    })

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    terminated = False
    truncated = False
    total_auc = 0.0

    while not (terminated or truncated):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_auc += info["deviation"]

    info["mode_switches"] = agent.switch_count

    survival_time = info["survival_step"]
    normalised_auc = total_auc / survival_time if survival_time > 0 else 0.0

    env.close()

    return {
        "Controller": ctrl_name,
        "Delay": delay,
        "Seed": seed,
        "Survival_Steps": survival_time,
        "Normalised_AUC": normalised_auc,
        "State_Switches": info["mode_switches"],
    }


def run_sweep(episodes=100):
    tasks = []
    for ctrl in CONTROLLERS:
        for delay in DELAY_LEVELS:
            for i in range(episodes):
                tasks.append((ctrl, delay, 1000 + i))

    print(f"Delay Sensitivity Sweep: {episodes} episodes x {len(CONTROLLERS)} controllers "
          f"x {len(DELAY_LEVELS)} delay levels = {len(tasks)} simulations.")
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Distributing workload across {max_workers} CPU cores...\n")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_episode, t) for t in tasks]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"Progress: {completed} / {len(tasks)} simulations completed...")

    print("\nSweep finished successfully.")
    return pd.DataFrame(results)


def compute_ci(x):
    x = np.asarray(x)
    if len(x) < 2:
        return 0.0
    return 1.96 * (np.std(x, ddof=1) / np.sqrt(len(x)))


def _save_table_png(df, filename, title=""):
    """Render a DataFrame as a presentation-ready PNG table.

    Long underscore-separated column names are wrapped onto two lines so headers
    do not overflow their cells; the *_Cohen_d suffix is rewritten as 'Cohen's d'
    on its own line, matching how the metric reads in the dissertation.
    """
    display_df = df.reset_index() if df.index.name is not None or isinstance(df.index, pd.MultiIndex) else df.copy()

    def _wrap_header(name, max_len=14):
        name_str = str(name)
        if name_str.endswith("_Cohen_d"):
            prefix = name_str[:-len("_Cohen_d")].replace("_", " ")
            return f"{prefix}\nCohen's d"
        cleaned = name_str.replace("_", " ")
        if len(cleaned) <= max_len:
            return cleaned
        words = cleaned.split(" ")
        if len(words) == 1:
            return cleaned
        best_split = min(
            range(1, len(words)),
            key=lambda i: max(len(" ".join(words[:i])), len(" ".join(words[i:]))),
        )
        return " ".join(words[:best_split]) + "\n" + " ".join(words[best_split:])

    wrapped_columns = [_wrap_header(c) for c in display_df.columns]

    n_cols = len(display_df.columns)
    n_rows = len(display_df)

    fig_w = max(9.0, 2.6 * n_cols)
    fig_h = max(3.2, 0.55 * n_rows + 2.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", color="#1a1a1a", pad=22)

    cell_text = [
        [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        for row in display_df.values
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=wrapped_columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.4)

    header_color = "#2c3e50"
    stripe_light = "#ffffff"
    stripe_dark = "#eef2f5"
    border_color = "#cccccc"

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color="white")
        else:
            cell.set_facecolor(stripe_light if row % 2 == 1 else stripe_dark)
            cell.set_text_props(color="#222222")

    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Table image saved to '{filename}'.")


def plot_results(df):
    agg = df.groupby(["Controller", "Delay"]).agg(
        Mean_Survival=("Survival_Steps", "mean"),
        CI_Survival=("Survival_Steps", compute_ci),
        Mean_AUC=("Normalised_AUC", "mean"),
        CI_AUC=("Normalised_AUC", compute_ci),
    ).reset_index()

    colours = {"Blend": "#1f77b4", "PAG": "#2ca02c"}
    labels = {"Blend": "ContinuousBlend", "PAG": "PAG (Hysteresis ON)"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ctrl in CONTROLLERS:
        subset = agg[agg["Controller"] == ctrl].sort_values("Delay")
        axes[0].errorbar(subset["Delay"], subset["Mean_Survival"], yerr=subset["CI_Survival"],
                         label=labels[ctrl], color=colours[ctrl], marker="o",
                         capsize=5, linewidth=2)
    axes[0].set_title("Survival vs Sensorimotor Delay (95% CI)")
    axes[0].set_xlabel("Observation Delay (steps)")
    axes[0].set_ylabel("Mean Survival Steps")
    axes[0].set_xticks(DELAY_LEVELS)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    for ctrl in CONTROLLERS:
        subset = agg[agg["Controller"] == ctrl].sort_values("Delay")
        axes[1].errorbar(subset["Delay"], subset["Mean_AUC"], yerr=subset["CI_AUC"],
                         label=labels[ctrl], color=colours[ctrl], marker="o",
                         capsize=5, linewidth=2)
    axes[1].set_title("Normalised AUC vs Sensorimotor Delay (95% CI)")
    axes[1].set_xlabel("Observation Delay (steps)")
    axes[1].set_ylabel("Energy Deviation per Step")
    axes[1].set_xticks(DELAY_LEVELS)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.suptitle("Delay Sensitivity: Command Fusion vs Discrete Mode-Switching (Medium Threat)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("ablation_delay_degradation.png", dpi=300)
    print("Plot saved to 'ablation_delay_degradation.png'.")
    plt.show()


if __name__ == "__main__":
    df = run_sweep(episodes=100)
    df.to_csv("ablation_delay_data.csv", index=False)
    print("Raw data saved to 'ablation_delay_data.csv'.\n")

    summary = df.groupby(["Controller", "Delay"]).agg(
        Mean_Survival=("Survival_Steps", "mean"),
        Mean_AUC=("Normalised_AUC", "mean"),
    ).round(3)
    print("Summary:")
    print(summary.to_string())

    _save_table_png(
        summary,
        "ablation_delay_summary.png",
        title="Delay Sensitivity Summary (Medium Threat)",
    )

    plot_results(df)
