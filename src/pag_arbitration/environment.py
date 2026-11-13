"""Gymnasium environment for the foraging-under-predation task.

A single agent forages for food in a bounded 2D arena while a motion-sensitive
predator patrols and chases. Energy decays every step, so the agent must keep
eating while avoiding capture. Observation noise and sensorimotor delay can be
injected to stress-test arbitration controllers.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

# --- Agent and energy dynamics ------------------------------------------------
ARENA_SIZE = 200.0
MAX_SPEED = 1.5
MAX_ENERGY = 100.0
SAFE_ENERGY_THRESHOLD = 80.0  # lower edge of the viability band
METABOLIC_DECAY = 0.05  # energy lost per step regardless of movement
MOVEMENT_COST_FACTOR = 0.1  # extra energy lost per unit of speed
FOOD_CAPTURE_RADIUS = 2.0
FOOD_ENERGY_GAIN = 20.0

# --- Predator -----------------------------------------------------------------
# The predator out-paces the agent at every threat level except "low", so
# stealth (freezing) is often the only viable strategy.
PREDATOR_SPEEDS = {"low": 1.2, "medium": 1.6, "high": 2.0}
THREAT_LEVELS = (*PREDATOR_SPEEDS, "none")
PREDATOR_KILL_RADIUS = 1.0
VISION_RADIUS_MOVING = 50.0
VISION_RADIUS_STILL = 20.0
STILLNESS_SPEED = 0.01  # agent speeds at or below this count as "still"
WANDER_SPEED_FRACTION = 0.6
WANDER_HEADING_STEPS = 60

# --- Rendering ----------------------------------------------------------------
WINDOW_SIZE = 600

Observation = NDArray[np.float32]


class PAGArbitrationEnv(gym.Env[Observation, NDArray[np.floating[Any]]]):
    """2D kinematic environment for comparing arbitration mechanisms.

    Observation: ``[energy, dist_pred, angle_pred, dist_food, angle_food]``.
    Action: a 2D velocity command in ``[-1, 1]^2``, scaled by ``MAX_SPEED``.

    Episodes end only when the agent starves (energy <= 0) or is caught; there
    is no step limit. The reward is a constant 1.0 per step and is not used by
    any controller; the metrics of interest are returned in ``info``.

    Args:
        config: Optional overrides. Supported keys: ``arena_size`` (float),
            ``threat_level`` ("low", "medium", "high" or "none"), ``obs_noise``
            (standard deviation of Gaussian noise added to predator distance),
            ``delay`` (observation delay in steps) and ``render_mode``.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}  # noqa: RUF012 (gym.Env API)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()

        self.config = config or {}
        self.arena_size: float = self.config.get("arena_size", ARENA_SIZE)
        self.threat_level: str = self.config.get("threat_level", "medium")
        self.obs_noise: float = self.config.get("obs_noise", 0.0)
        self.delay_steps: int = self.config.get("delay", 0)
        self.render_mode = self.config.get("render_mode", None)

        if self.threat_level not in THREAT_LEVELS:
            raise ValueError(
                f"Unknown threat_level {self.threat_level!r}; expected one of {THREAT_LEVELS}."
            )
        if self.delay_steps < 0:
            raise ValueError(f"delay must be non-negative, got {self.delay_steps}.")

        self.window_size = WINDOW_SIZE
        self.scale = self.window_size / self.arena_size

        self.max_speed = MAX_SPEED
        self.safe_energy_threshold = SAFE_ENERGY_THRESHOLD
        self.metabolic_decay = METABOLIC_DECAY
        self.movement_cost_factor = MOVEMENT_COST_FACTOR
        self.capture_radius = FOOD_CAPTURE_RADIUS
        self.energy_gain = FOOD_ENERGY_GAIN

        self.time_outside_band = 0
        self.current_deficit_duration = 0
        self.recovery_times: list[int] = []

        # "none" disables the predator; its speed is then irrelevant.
        self.predator_speed = PREDATOR_SPEEDS.get(self.threat_level, PREDATOR_SPEEDS["medium"])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.agent_pos: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
        self.predator_pos: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
        self.food_pos: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
        self.energy: float | np.floating[Any] = MAX_ENERGY
        self.steps = 0

        self.predator_wander_vec: NDArray[np.float64] = np.zeros(2)
        self.predator_wander_timer = 0
        self.obs_buffer: list[Observation] = []

        self.window: Any = None
        self.clock: Any = None

        if self.render_mode == "human":
            self._init_render()

    def _init_render(self) -> None:
        # Imported lazily so headless experiment runs never initialise pygame.
        import pygame

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("PAG Arbitration Simulation")
            self.clock = pygame.time.Clock()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the episode, placing agent, predator and food uniformly at random."""
        super().reset(seed=seed)

        self.steps = 0
        self.energy = MAX_ENERGY

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

        # Note: this samples a second observation (and, with noise enabled, a
        # second noise draw). Kept as-is so that results reproduce exactly.
        return self._get_obs(), {}

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Advance the world by one step and return the (possibly delayed) observation."""
        self.steps += 1

        velocity = action * self.max_speed
        self.agent_pos += velocity
        self.agent_pos = np.clip(self.agent_pos, 0, self.arena_size)

        dist_to_food = np.linalg.norm(self.agent_pos - self.food_pos)
        if dist_to_food < self.capture_radius:
            self.energy += self.energy_gain
            self.energy = min(self.energy, MAX_ENERGY)
            self.food_pos = self.np_random.uniform(0, self.arena_size, size=2).astype(np.float32)

        if self.threat_level != "none":
            self._move_predator(velocity)

        move_cost = np.linalg.norm(velocity) * self.movement_cost_factor
        self.energy -= self.metabolic_decay + move_cost

        deviation = max(0, self.safe_energy_threshold - self.energy)

        if self.energy < self.safe_energy_threshold:
            self.time_outside_band += 1
            self.current_deficit_duration += 1
        elif self.current_deficit_duration > 0:
            self.recovery_times.append(self.current_deficit_duration)
            self.current_deficit_duration = 0

        terminated = False
        caught = np.linalg.norm(self.agent_pos - self.predator_pos) < PREDATOR_KILL_RADIUS
        if self.energy <= 0 or caught:
            terminated = True
            # Terminal deficits never recover, so close them out here to keep the metric honest.
            if self.current_deficit_duration > 0:
                self.recovery_times.append(self.current_deficit_duration)

        max_recovery = max(self.recovery_times) if self.recovery_times else 0.0
        avg_recovery = (
            sum(self.recovery_times) / len(self.recovery_times) if self.recovery_times else 0.0
        )

        info = {
            "energy": self.energy,
            "deviation": deviation,
            "survival_step": self.steps,
            "time_outside_band": self.time_outside_band,
            "max_recovery_time": max_recovery,
            "avg_recovery_time": avg_recovery,
        }

        # FIFO buffer simulates sensorimotor delay: the controller always sees a stale observation.
        self.obs_buffer.append(self._get_obs())
        delayed_obs = self.obs_buffer.pop(0)

        return delayed_obs, 1.0, terminated, False, info

    def _move_predator(self, velocity: NDArray[np.floating[Any]]) -> None:
        """Chase the agent if it is visible, otherwise patrol on a held heading."""
        pred_vec = self.agent_pos - self.predator_pos
        pred_dist = np.linalg.norm(pred_vec)
        agent_speed = np.linalg.norm(velocity)

        # Motion-sensitive vision: a stationary agent is harder for the predator to detect,
        # so a FREEZE response can genuinely break line-of-sight.
        vision_radius = (
            VISION_RADIUS_MOVING if agent_speed > STILLNESS_SPEED else VISION_RADIUS_STILL
        )

        if 0 < pred_dist < vision_radius:
            chase_vec = pred_vec / pred_dist
            self.predator_pos += chase_vec * self.predator_speed
            self.predator_wander_vec = chase_vec
            self.predator_wander_timer = 0
        else:
            if self.predator_wander_timer <= 0:
                # Commit to a random heading so the predator does not jitter in place.
                self.predator_wander_vec = self.np_random.uniform(-1.0, 1.0, size=2)
                self.predator_wander_vec /= np.linalg.norm(self.predator_wander_vec)
                self.predator_wander_timer = WANDER_HEADING_STEPS

            self.predator_pos += self.predator_wander_vec * (
                self.predator_speed * WANDER_SPEED_FRACTION
            )
            self.predator_wander_timer -= 1

            # Reflect off arena walls rather than clipping, so patrol momentum is preserved.
            if self.predator_pos[0] <= 0 or self.predator_pos[0] >= self.arena_size:
                self.predator_wander_vec[0] *= -1
            if self.predator_pos[1] <= 0 or self.predator_pos[1] >= self.arena_size:
                self.predator_wander_vec[1] *= -1

        self.predator_pos = np.clip(self.predator_pos, 0, self.arena_size)

    def render(self) -> None:
        """Draw the arena in a pygame window (green food, red predator, blue agent)."""
        import pygame

        self._init_render()
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        def to_px(pos: NDArray[np.float32]) -> tuple[int, int]:
            return int(pos[0] * self.scale), int(pos[1] * self.scale)

        pygame.draw.circle(canvas, (0, 255, 0), to_px(self.food_pos), 8)
        pygame.draw.circle(canvas, (255, 0, 0), to_px(self.predator_pos), 10)
        pygame.draw.circle(canvas, (0, 0, 255), to_px(self.agent_pos), 12)

        bar_width = int((max(0, self.energy) / MAX_ENERGY) * 150)
        pygame.draw.rect(canvas, (0, 0, 0), (10, 10, 150, 20), 2)
        pygame.draw.rect(canvas, (0, 200, 0), (10, 10, bar_width, 20))

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        """Close the pygame window if one was opened."""
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _get_obs(self) -> Observation:
        to_pred = self.predator_pos - self.agent_pos
        to_food = self.food_pos - self.agent_pos

        dist_pred = np.linalg.norm(to_pred)
        dist_food = np.linalg.norm(to_food)

        angle_pred = np.arctan2(to_pred[1], to_pred[0])
        angle_food = np.arctan2(to_food[1], to_food[0])

        if self.obs_noise > 0:
            # Noise is applied to predator distance only and may make it negative.
            dist_pred += self.np_random.normal(0, self.obs_noise)

        return np.array(
            [self.energy, dist_pred, angle_pred, dist_food, angle_food], dtype=np.float32
        )
