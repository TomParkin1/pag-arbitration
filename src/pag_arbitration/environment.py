import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class PAGArbitrationEnv(gym.Env):
    """
    2D Kinematic simulation environment for testing arbitration mechanisms.
    Features metabolic decay, resource foraging, and a motion-sensitive predator.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config=None):
        super().__init__()
        
        self.config = config or {}
        self.arena_size = self.config.get("arena_size", 200.0) 
        self.threat_level = self.config.get("threat_level", "medium") 
        self.obs_noise = self.config.get("obs_noise", 0.0)
        self.delay_steps = self.config.get("delay", 0)
        self.render_mode = self.config.get("render_mode", None)

        self.window_size = 600
        self.scale = self.window_size / self.arena_size

        self.max_speed = 1.5
        self.safe_energy_threshold = 80.0
        self.metabolic_decay = 0.05
        self.movement_cost_factor = 0.1
        self.capture_radius = 2.0 
        self.energy_gain = 20.0

        self.time_outside_band = 0
        self.current_deficit_duration = 0
        self.recovery_times = []

        # Predator out-paces the agent at every threat level, so stealth is the only viable strategy.
        speed_map = {"low": 1.2, "medium": 1.6, "high": 2.0}
        self.predator_speed = speed_map.get(self.threat_level, 1.6)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.agent_pos = None
        self.predator_pos = None
        self.food_pos = None
        self.energy = None
        self.steps = 0
        
        self.predator_wander_vec = None
        self.predator_wander_timer = 0
        self.obs_buffer = []

        self.window = None
        self.clock = None
        
        if self.render_mode == "human":
            self._init_render()

    def _init_render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("PAG Arbitration Simulation")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.energy = 100.0

        self.time_outside_band = 0
        self.current_deficit_duration = 0
        self.recovery_times = []
        
        self.agent_pos = self.np_random.uniform(0, self.arena_size, size=2).astype(np.float32)
        self.predator_pos = self.np_random.uniform(0, self.arena_size, size=2).astype(np.float32)
        self.food_pos = self.np_random.uniform(0, self.arena_size, size=2).astype(np.float32)

        self.predator_wander_vec = self.np_random.uniform(-1.0, 1.0, size=2)
        self.predator_wander_vec /= np.linalg.norm(self.predator_wander_vec)
        self.predator_wander_timer = 0

        initial_obs = self._get_obs()
        self.obs_buffer = [initial_obs] * (self.delay_steps + 1)
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        velocity = action * self.max_speed
        self.agent_pos += velocity
        self.agent_pos = np.clip(self.agent_pos, 0, self.arena_size)

        dist_to_food = np.linalg.norm(self.agent_pos - self.food_pos)
        if dist_to_food < self.capture_radius:
            self.energy += self.energy_gain
            self.energy = min(self.energy, 100.0) 
            self.food_pos = self.np_random.uniform(0, self.arena_size, size=2).astype(np.float32)

        if self.threat_level != "none":
            pred_vec = self.agent_pos - self.predator_pos
            pred_dist = np.linalg.norm(pred_vec)
            agent_speed = np.linalg.norm(velocity)

            # Motion-sensitive vision: a stationary agent is harder for the predator to detect,
            # so a FREEZE response can genuinely break line-of-sight.
            vision_radius = 50.0 if agent_speed > 0.01 else 20.0

            if 0 < pred_dist < vision_radius:
                chase_vec = pred_vec / pred_dist
                self.predator_pos += chase_vec * self.predator_speed
                self.predator_wander_vec = chase_vec
                self.predator_wander_timer = 0
            else:
                if self.predator_wander_timer <= 0:
                    # Commit to a random heading for 60 steps so the predator does not jitter in place.
                    self.predator_wander_vec = self.np_random.uniform(-1.0, 1.0, size=2)
                    self.predator_wander_vec /= np.linalg.norm(self.predator_wander_vec)
                    self.predator_wander_timer = 60

                self.predator_pos += self.predator_wander_vec * (self.predator_speed * 0.6)
                self.predator_wander_timer -= 1

                # Reflect off arena walls rather than clipping, so patrol momentum is preserved.
                if self.predator_pos[0] <= 0 or self.predator_pos[0] >= self.arena_size:
                    self.predator_wander_vec[0] *= -1
                if self.predator_pos[1] <= 0 or self.predator_pos[1] >= self.arena_size:
                    self.predator_wander_vec[1] *= -1
            
            self.predator_pos = np.clip(self.predator_pos, 0, self.arena_size)

        move_cost = np.linalg.norm(velocity) * self.movement_cost_factor
        self.energy -= (self.metabolic_decay + move_cost)

        deviation = max(0, self.safe_energy_threshold - self.energy)
        
        if self.energy < self.safe_energy_threshold:
            self.time_outside_band += 1
            self.current_deficit_duration += 1
        else:
            if self.current_deficit_duration > 0:
                self.recovery_times.append(self.current_deficit_duration)
                self.current_deficit_duration = 0

        terminated = False
        if self.energy <= 0 or np.linalg.norm(self.agent_pos - self.predator_pos) < 1.0:
            terminated = True
            # Terminal deficits never recover, so close them out here to keep the metric honest.
            if self.current_deficit_duration > 0:
                self.recovery_times.append(self.current_deficit_duration)

        max_recovery = max(self.recovery_times) if self.recovery_times else 0.0
        avg_recovery = sum(self.recovery_times) / len(self.recovery_times) if self.recovery_times else 0.0

        info = {
            "energy": self.energy,
            "deviation": deviation,
            "survival_step": self.steps,
            "time_outside_band": self.time_outside_band,
            "max_recovery_time": max_recovery,
            "avg_recovery_time": avg_recovery
        }

        # FIFO buffer simulates sensorimotor delay: the controller always sees a stale observation.
        self.obs_buffer.append(self._get_obs())
        delayed_obs = self.obs_buffer.pop(0)

        return delayed_obs, 1.0, terminated, False, info

    def render(self):
        self._init_render() 
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) 

        food_coords = (int(self.food_pos[0] * self.scale), int(self.food_pos[1] * self.scale))
        pred_coords = (int(self.predator_pos[0] * self.scale), int(self.predator_pos[1] * self.scale))
        agent_coords = (int(self.agent_pos[0] * self.scale), int(self.agent_pos[1] * self.scale))

        pygame.draw.circle(canvas, (0, 255, 0), food_coords, 8)    
        pygame.draw.circle(canvas, (255, 0, 0), pred_coords, 10)   
        pygame.draw.circle(canvas, (0, 0, 255), agent_coords, 12)  

        bar_width = int((max(0, self.energy) / 100) * 150)
        pygame.draw.rect(canvas, (0, 0, 0), (10, 10, 150, 20), 2)         
        pygame.draw.rect(canvas, (0, 200, 0), (10, 10, bar_width, 20))    

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _get_obs(self):
        to_pred = self.predator_pos - self.agent_pos
        to_food = self.food_pos - self.agent_pos
        
        dist_pred = np.linalg.norm(to_pred)
        dist_food = np.linalg.norm(to_food)

        angle_pred = np.arctan2(to_pred[1], to_pred[0]) 
        angle_food = np.arctan2(to_food[1], to_food[0])
        
        if self.obs_noise > 0:
            dist_pred += self.np_random.normal(0, self.obs_noise)
        
        return np.array([self.energy, dist_pred, angle_pred, dist_food, angle_food], dtype=np.float32)