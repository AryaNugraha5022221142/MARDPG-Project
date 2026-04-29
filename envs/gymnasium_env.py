"""
envs/gymnasium_env.py

Gymnasium-compatible wrapper for the QuadcopterEnv.
Registers as 'MultiUAVNav-v0' for use with SB3 / standard RL libraries.

Usage:
    import gymnasium as gym
    import envs.gymnasium_env   # triggers registration
    env = gym.make('MultiUAVNav-v0', num_agents=3, scenario='urban_canyon')
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from .quadcopter_env import QuadcopterEnv


class MultiUAVGymEnv(gym.Env):
    """
    Gymnasium wrapper for multi-agent UAV navigation.

    Observation space: Flattened (num_agents × 33) — all agents concatenated.
    Action space:      Box(num_agents × 4) — all agents concatenated.

    For MARL use, split obs/action by num_agents after stepping.
    For single-policy benchmarking (e.g. PPO), the environment appears as one agent
    controlling all UAVs simultaneously.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        num_agents: int = 3,
        scenario: str = 'urban_canyon',
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.render_mode = render_mode

        # Underlying environment
        self._env = QuadcopterEnv(
            num_agents=num_agents,
            config=config,
            render_mode=render_mode,
            scenario=scenario,
        )

        # Obs dim per agent: 25 (rangefinder) + 5 (goal) + 3 (vel) + 1 (is_sat) + 4 (prev_act) = 38
        self.obs_dim_per_agent = 38
        total_obs_dim = self.obs_dim_per_agent * num_agents

        # Single-agent interface: all agents' observations concatenated
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )

        # Actions: 4 dims per agent, all in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_agents * 4,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)
        return obs.flatten(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Reshape flat action to (num_agents, 4)
        actions = action.reshape(self.num_agents, 4)
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        # Flatten obs; sum rewards for single-policy training
        obs_flat = obs.flatten()
        total_reward = float(np.sum(rewards))

        return obs_flat, total_reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


# ──────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────
def register_env():
    """Call once at import time to register with gymnasium."""
    if 'MultiUAVNav-v0' not in gym.envs.registry:
        gym.register(
            id='MultiUAVNav-v0',
            entry_point='envs.gymnasium_env:MultiUAVGymEnv',
            kwargs={'num_agents': 3, 'scenario': 'urban_canyon'},
            max_episode_steps=2000,
        )


register_env()
