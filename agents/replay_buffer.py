# agents/replay_buffer.py
import numpy as np
from collections import deque
import random
from typing import Tuple

class ReplayBuffer:
    """
    Experience replay buffer for multi-agent reinforcement learning.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
             next_obs: np.ndarray, dones: np.ndarray):
        """
        Stores a transition in the buffer.
        obs: (num_agents, obs_dim)
        actions: (num_agents,)
        rewards: (num_agents,)
        next_obs: (num_agents, obs_dim)
        dones: (num_agents,)
        """
        self.buffer.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples a batch of transitions.
        Returns stacked numpy arrays of shape (batch_size, num_agents, ...)
        """
        batch = random.sample(self.buffer, batch_size)
        
        obs_batch = np.stack([item[0] for item in batch])
        actions_batch = np.stack([item[1] for item in batch])
        rewards_batch = np.stack([item[2] for item in batch])
        next_obs_batch = np.stack([item[3] for item in batch])
        dones_batch = np.stack([item[4] for item in batch])
        
        return obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch

    def __len__(self) -> int:
        return len(self.buffer)
