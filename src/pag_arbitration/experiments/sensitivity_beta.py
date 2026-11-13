"""
Sensitivity sweep on the Blend controller's inverse temperature beta.

The main study fixes beta=2; this script tests whether the Blend baseline
is robust to that choice, by running medium-threat episodes at
beta in {0.5, 1, 2, 4, 8} with matched seeds.

Output: sensitivity_beta_data.csv
"""
import concurrent.futures
import multiprocessing
import pandas as pd
from pag_arbitration.environment import PAGArbitrationEnv
from pag_arbitration.controller import ContinuousBlendController


BETAS = [0.5, 1.0, 2.0, 4.0, 8.0]
N_EPISODES = 100
THREAT = "medium"


def _run_one(args):
    beta, seed = args
    env = PAGArbitrationEnv(config={
        "threat_level": THREAT,
        "obs_noise": 0.0,
        "delay": 0,
        "render_mode": None,
    })
    agent = ContinuousBlendController(beta=beta)
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
        "Beta": beta,
        "Seed": seed,
        "Survival_Steps": survival,
        "Normalised_AUC": norm_auc,
        "Switch_Count": agent.switch_count,
    }


def main():
    tasks = [(b, 1000 + i) for b in BETAS for i in range(N_EPISODES)]
    workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running {len(tasks)} episodes across {workers} workers...")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, t) for t in tasks]
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(fut.result())
            if n % 100 == 0 or n == len(tasks):
                print(f"  {n} / {len(tasks)}")

    df = pd.DataFrame(results)
    df.to_csv("sensitivity_beta_data.csv", index=False)
    print("Saved sensitivity_beta_data.csv")


if __name__ == "__main__":
    main()
