import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
from pag_arbitration.environment import PAGArbitrationEnv
from pag_arbitration.controller import ContinuousBlendController, UtilitySwitchController, PAGSwitchController

def _run_single_episode(args):
    """
    Worker function to execute a single simulation episode.
    Instantiates its own environment and controller to ensure process-safety.
    """
    threat, noise, delay, ctrl_name, seed = args

    if ctrl_name == "Blend":
        agent = ContinuousBlendController(beta=2.0)
    elif ctrl_name == "Switch":
        agent = UtilitySwitchController()
    elif ctrl_name == "PAG":
        agent = PAGSwitchController()

    # Environment is built inside the worker so each process owns its own Pygame/NumPy state.
    env = PAGArbitrationEnv(config={
        "threat_level": threat,
        "obs_noise": noise,
        "delay": delay,
        "render_mode": None
    })

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    terminated = False
    truncated = False
    total_auc = 0.0

    while not (terminated or truncated):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_auc += info['deviation']

    survival_time = info["survival_step"]
    normalised_auc = total_auc / survival_time if survival_time > 0 else 0.0

    time_outside_band = info.get("time_outside_band", 0)
    max_recovery_time = info.get("max_recovery_time", 0.0)
    avg_recovery_time = info.get("avg_recovery_time", 0.0)

    env.close()

    return {
        "Controller": ctrl_name,
        "Threat": threat,
        "Noise": noise,
        "Delay": delay,
        "Seed": seed,
        "Survival_Steps": survival_time,
        "Normalised_AUC": normalised_auc,
        "Time_Outside_Band": time_outside_band,
        "Max_Recovery_Time": max_recovery_time,
        "Avg_Recovery_Time": avg_recovery_time
    }

def run_factorial_sweep(episodes=30):
    """
    Executes a full factorial experimental sweep using multiprocessing.
    Variables: Controllers (3) x Threat (3) x Noise (2) x Delay (2)
    """
    threat_levels = ["low", "medium", "high"]
    noise_levels = [0.0, 5.0]  
    delay_levels = [0, 2]      
    controllers = ["Blend", "Switch", "PAG"]

    tasks = []
    for threat in threat_levels:
        for noise in noise_levels:
            for delay in delay_levels:
                for ctrl_name in controllers:
                    for i in range(episodes):
                        seed = 1000 + i 
                        tasks.append((threat, noise, delay, ctrl_name, seed))

    total_tasks = len(tasks)
    print(f"Starting Full Factorial Sweep: {episodes} episodes per condition.")
    print(f"Total simulations to execute: {total_tasks}")

    # Leave one core free so the host OS remains responsive during long sweeps.
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Distributing workload across {max_workers} CPU cores...\n")

    all_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_episode, t) for t in tasks]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())
            completed += 1
            if completed % 100 == 0 or completed == total_tasks:
                print(f"Progress: {completed} / {total_tasks} simulations completed...")

    print("\nSweep finished successfully.")
    return pd.DataFrame(all_results)

def compute_ci(x):
    """Calculates the 95% Confidence Interval for a given array of data."""
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

def compute_cohens_d(group1, group2):
    """
    Calculates Cohen's d effect size to measure the standardised difference 
    between two means.
    group1: The experimental condition (e.g., PAG)
    group2: The control condition (e.g., Switch)
    """
    n1, n2 = len(group1), len(group2)

    # Sample variance (ddof=1) is undefined for n < 2.
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
        
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def calculate_effect_sizes(df):
    """
    Computes the pairwise Cohen's d between the PAG and Switch controllers
    across all threat levels to quantify the magnitude of performance differences.
    """
    print("\n--- Effect Size Analysis: PAG vs Switch ---")
    metrics = ["Survival_Steps", "Normalised_AUC", "Avg_Recovery_Time"]
    threat_levels = ["low", "medium", "high"]
    
    effect_sizes = []
    
    for threat in threat_levels:
        threat_df = df[df["Threat"] == threat]

        pag_data = threat_df[threat_df["Controller"] == "PAG"]
        switch_data = threat_df[threat_df["Controller"] == "Switch"]

        row = {"Threat_Level": threat}

        for metric in metrics:
            # Order matters: PAG is the experimental group, Switch is the baseline,
            # so a positive d reads as "PAG improves over Switch".
            d_value = compute_cohens_d(pag_data[metric].values, switch_data[metric].values)
            row[f"{metric}_Cohen_d"] = round(d_value, 4)

        effect_sizes.append(row)

    ef_df = pd.DataFrame(effect_sizes)
    print(ef_df.to_string(index=False))

    file_name = "effect_sizes_pag_vs_switch.csv"
    ef_df.to_csv(file_name, index=False)
    print(f"\nEffect size data successfully saved to '{file_name}'.")

    _save_table_png(
        ef_df,
        "effect_sizes_pag_vs_switch.png",
        title="Cohen's d: PAG vs Switch by Threat Level",
    )

    return ef_df

def plot_factorial_results(df):
    """
    Generates a matrix of line plots showing Threat Intensity vs Metrics,
    averaged across the noise and delay conditions for top-level trends.
    """
    grouped = df.groupby(["Controller", "Threat"]).agg(
        Mean_Survival=("Survival_Steps", "mean"),
        CI_Survival=("Survival_Steps", compute_ci),
        Mean_AUC=("Normalised_AUC", "mean"),
        CI_AUC=("Normalised_AUC", compute_ci),
        Mean_Recovery=("Avg_Recovery_Time", "mean"),
        CI_Recovery=("Avg_Recovery_Time", compute_ci)
    ).reset_index()

    # Force low → medium → high on the x-axis; the default alphabetical order would mislead the reader.
    threat_order = ["low", "medium", "high"]
    grouped['Threat'] = pd.Categorical(grouped['Threat'], categories=threat_order, ordered=True)
    grouped = grouped.sort_values("Threat")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colours = {"Blend": "#1f77b4", "Switch": "#ff7f0e", "PAG": "#2ca02c"}

    for ctrl in ["Blend", "Switch", "PAG"]:
        subset = grouped[grouped["Controller"] == ctrl]
        axes[0].errorbar(subset["Threat"], subset["Mean_Survival"], yerr=subset["CI_Survival"],
                         label=ctrl, color=colours[ctrl], marker='o', capsize=5, linewidth=2)
    axes[0].set_title("Mean Survival by Threat Level (95% CI)")
    axes[0].set_ylabel("Survival Steps")
    axes[0].set_xlabel("Threat Intensity")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    for ctrl in ["Blend", "Switch", "PAG"]:
        subset = grouped[grouped["Controller"] == ctrl]
        axes[1].errorbar(subset["Threat"], subset["Mean_AUC"], yerr=subset["CI_AUC"],
                         label=ctrl, color=colours[ctrl], marker='o', capsize=5, linewidth=2)
    axes[1].set_title("Metabolic Cost (AUC) by Threat Level (95% CI)")
    axes[1].set_ylabel("Normalised Energy Deviation")
    axes[1].set_xlabel("Threat Intensity")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    for ctrl in ["Blend", "Switch", "PAG"]:
        subset = grouped[grouped["Controller"] == ctrl]
        axes[2].errorbar(subset["Threat"], subset["Mean_Recovery"], yerr=subset["CI_Recovery"],
                         label=ctrl, color=colours[ctrl], marker='o', capsize=5, linewidth=2)
    axes[2].set_title("Deficit Recovery Time (95% CI)")
    axes[2].set_ylabel("Average Steps to Recover")
    axes[2].set_xlabel("Threat Intensity")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()

    plt.suptitle("Factorial Sweep Results: Controller Robustness", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("factorial_results.png", dpi=300)
    print("Visualisation saved to 'factorial_results.png'.")
    plt.show()

if __name__ == "__main__":
    results_df = run_factorial_sweep(episodes=100)

    results_df.to_csv("factorial_master_data.csv", index=False)
    print("Master dataset successfully saved to 'factorial_master_data.csv'.")

    plot_factorial_results(results_df)
    calculate_effect_sizes(results_df)