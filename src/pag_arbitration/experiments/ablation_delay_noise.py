"""
Joint (noise, delay) sensitivity sweep at medium threat.

The §5.4 delay ablation in the dissertation sweeps delta in [0..5] at zero
noise and predicts that the PAG cliff at delta=5 will shift earlier as
observation noise rises. This script tests that prediction directly by
re-running the same delay grid at sigma=0 alongside sigma=5, for both PAG
(hysteretic) and Blend, with 100 matched-seed episodes per cell.

Output: ablation_delay_noise_data.csv with one row per episode.
"""
import concurrent.futures
import multiprocessing
import pandas as pd
from pag_arbitration.environment import PAGArbitrationEnv
from pag_arbitration.controller import ContinuousBlendController, PAGSwitchController


DELAY_LEVELS = [0, 1, 2, 3, 4, 5]
NOISE_LEVELS = [0.0, 5.0]
N_EPISODES = 100
THREAT = "medium"


def _make_agent(ctrl_name):
    if ctrl_name == "Blend":
        return ContinuousBlendController(beta=2.0)
    return PAGSwitchController(use_hysteresis=True)


def _run_one(args):
    ctrl, noise, delay, seed = args
    env = PAGArbitrationEnv(config={
        "threat_level": THREAT,
        "obs_noise": noise,
        "delay": delay,
        "render_mode": None,
    })
    agent = _make_agent(ctrl)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    terminated = truncated = False
    total_auc = 0.0
    while not (terminated or truncated):
        action = agent.act(obs)
        obs, _r, terminated, truncated, info = env.step(action)
        total_auc += info["deviation"]

    survival = info["survival_step"]
    norm_auc = total_auc / survival if survival > 0 else 0.0
    env.close()
    return {
        "Controller": ctrl,
        "Threat": THREAT,
        "Noise": noise,
        "Delay": delay,
        "Seed": seed,
        "Survival_Steps": survival,
        "Normalised_AUC": norm_auc,
    }


def main():
    tasks = []
    for ctrl in ["Blend", "PAG"]:
        for noise in NOISE_LEVELS:
            for delay in DELAY_LEVELS:
                for i in range(N_EPISODES):
                    tasks.append((ctrl, noise, delay, 1000 + i))

    workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running {len(tasks)} episodes across {workers} workers...")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, t) for t in tasks]
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(fut.result())
            if n % 200 == 0 or n == len(tasks):
                print(f"  {n} / {len(tasks)}")

    df = pd.DataFrame(results)
    df.to_csv("ablation_delay_noise_data.csv", index=False)
    print("Saved ablation_delay_noise_data.csv")


if __name__ == "__main__":
    main()
