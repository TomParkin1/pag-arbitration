"""
Recovery-time decomposition: split each episode's deficit episodes into
"true" recoveries (closed while the agent is alive) and "terminal" hunger
spirals (deficit still open at termination, closed out for honesty).

Produces recovery_decomposition.csv, one row per episode.
"""
import concurrent.futures
import multiprocessing
import pandas as pd
from pag_arbitration.environment import PAGArbitrationEnv
from pag_arbitration.controller import (
    ContinuousBlendController,
    UtilitySwitchController,
    PAGSwitchController,
)


SAFE_ENERGY = 80.0


def _run_single_episode(args):
    threat, noise, delay, ctrl_name, seed = args

    if ctrl_name == "Blend":
        agent = ContinuousBlendController(beta=2.0)
    elif ctrl_name == "Switch":
        agent = UtilitySwitchController()
    elif ctrl_name == "PAG":
        agent = PAGSwitchController()

    env = PAGArbitrationEnv(config={
        "threat_level": threat,
        "obs_noise": noise,
        "delay": delay,
        "render_mode": None,
    })

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    completed = []       # durations of recoveries closed before terminal step
    open_duration = 0    # currently-open deficit length
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.act(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        energy = info["energy"]

        if energy < SAFE_ENERGY:
            open_duration += 1
        else:
            if open_duration > 0:
                completed.append(open_duration)
                open_duration = 0

    terminal_duration = open_duration if open_duration > 0 else 0
    has_terminal = terminal_duration > 0

    n_completed = len(completed)
    sum_completed = sum(completed) if completed else 0
    avg_completed = (sum_completed / n_completed) if n_completed else 0.0

    env.close()

    return {
        "Controller": ctrl_name,
        "Threat": threat,
        "Noise": noise,
        "Delay": delay,
        "Seed": seed,
        "Survival_Steps": info["survival_step"],
        "Num_True_Recoveries": n_completed,
        "Avg_True_Recovery": avg_completed,
        "Sum_True_Recovery": sum_completed,
        "Has_Terminal_Deficit": int(has_terminal),
        "Terminal_Recovery_Duration": terminal_duration,
    }


def main():
    threat_levels = ["low", "medium", "high"]
    noise_levels = [0.0, 5.0]
    delay_levels = [0, 2]
    controllers = ["Blend", "Switch", "PAG"]
    episodes = 100

    tasks = []
    for threat in threat_levels:
        for noise in noise_levels:
            for delay in delay_levels:
                for ctrl in controllers:
                    for i in range(episodes):
                        tasks.append((threat, noise, delay, ctrl, 1000 + i))

    workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running {len(tasks)} episodes across {workers} workers...")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_single_episode, t) for t in tasks]
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(fut.result())
            if n % 200 == 0 or n == len(tasks):
                print(f"  {n} / {len(tasks)}")

    df = pd.DataFrame(results)
    df.to_csv("recovery_decomposition.csv", index=False)
    print("Saved recovery_decomposition.csv")


if __name__ == "__main__":
    main()
