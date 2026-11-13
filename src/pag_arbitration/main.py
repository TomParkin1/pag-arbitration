import sys
import pandas as pd
import pygame
from pag_arbitration.environment import PAGArbitrationEnv
from pag_arbitration.controller import ContinuousBlendController, UtilitySwitchController, PAGSwitchController

def run_simulation_batch(seed, episodes=1, render=False, controller_type="blend"):
    """
    Executes a batch of simulation episodes with a specified controller.
    
    Args:
        seed (int): Base seed for deterministic pseudo-random generation (NFR2).
        episodes (int): Number of episodes to execute.
        render (bool): Toggles Pygame visualisation.
        controller_type (str): 'blend' for Command Fusion, 'switch' for Utility Switching, 'pag' for Discrete FSM.
        
    Returns:
        pd.DataFrame: Contains survival steps, final energy, and AUC for each episode.
    """
    mode = "human" if render else None
    env = PAGArbitrationEnv(config={"threat_level": "medium", "render_mode": mode})

    if controller_type == "pag":
        agent = PAGSwitchController()
    elif controller_type == "switch":
        agent = UtilitySwitchController()
    else:
        agent = ContinuousBlendController(beta=2.0)
        
    results = []

    for i in range(episodes):
        obs, info = env.reset(seed=seed + i)
        env.action_space.seed(seed + i) 
        
        terminated = False
        truncated = False
        total_auc = 0.0
        
        while not (terminated or truncated):
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return pd.DataFrame(results)

            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_auc += info['deviation']

            if render:
                env.render()

        # Normalise AUC by survival time so short episodes are not rewarded for their brevity.
        survival_time = info["survival_step"]
        normalised_auc = total_auc / survival_time if survival_time > 0 else 0.0

        results.append({
            "episode_seed": seed + i,
            "survival_steps": survival_time,
            "final_energy": info["energy"],
            "total_auc": total_auc,
            "normalised_auc": normalised_auc
        })
        
    env.close()
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("--- Dissertation Simulation: Arbitration Comparison ---")

    base_seed = 42
    num_episodes = 3
    enable_render = True
    ctrl_type = "blend"

    if len(sys.argv) > 1:
        try: 
            base_seed = int(sys.argv[1])
        except ValueError: 
            pass
            
    if len(sys.argv) > 2:
        try: 
            num_episodes = int(sys.argv[2])
        except ValueError: 
            pass
            
    if len(sys.argv) > 3:
        enable_render = sys.argv[3].lower() == 'true'

    if len(sys.argv) > 4:
        parsed_ctrl = sys.argv[4].lower()
        if parsed_ctrl in ["blend", "switch", "pag"]:
            ctrl_type = parsed_ctrl

    print(f"\nConfiguration -> Seed: {base_seed} | Episodes: {num_episodes} | Render: {enable_render} | Controller: {ctrl_type.upper()}")
    
    df = run_simulation_batch(seed=base_seed, episodes=num_episodes, render=enable_render, controller_type=ctrl_type)
    
    print(f"\nResults ({ctrl_type.upper()}):")
    print(df)