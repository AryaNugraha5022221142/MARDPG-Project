# agents/replay_buffer.py
import numpy as np
from collections import deque
import random
from typing import Tuple, List

class ReplayBuffer:
    """
    Standard experience replay buffer for non-recurrent MARL.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, rewards, next_obs, dones):
        self.buffer.append((obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*transitions)
        return (np.array(obs), np.array(actions), np.array(rewards), 
                np.array(next_obs), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class SequenceReplayBuffer:
    """
    Sequence-based experience replay buffer for recurrent MARL.
    Stores full episodes and samples sequences of a fixed length.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.current_episode = []

    def push(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
             next_obs: np.ndarray, dones: np.ndarray, episode_done: bool):
        """
        Adds a transition to the current episode.
        episode_done: True if the entire episode is over
        """
        self.current_episode.append((obs, actions, rewards, next_obs, dones))
        
        if episode_done:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def clear(self):
        """Clears the buffer for curriculum transitions."""
        self.buffer.clear()
        self.current_episode = []

    def sample(self, batch_size: int, seq_len: int):
        """
        Samples a batch of sequences.
        Returns stacked numpy arrays and padding masks.
        """
        valid_episodes = list(self.buffer)
        if not valid_episodes:
            return None
            
        sampled_episodes = random.choices(valid_episodes, k=batch_size)
        
        obs_batch, actions_batch, rewards_batch = [], [], []
        next_obs_batch, dones_batch, mask_batch = [], [], []
        
        for ep in sampled_episodes:
            if len(ep) > seq_len:
                start_idx = random.randint(0, len(ep) - seq_len)
                seq = ep[start_idx : start_idx + seq_len]
                pad_len = 0
            else:
                seq = ep
                pad_len = seq_len - len(ep)
            
            obs_seq = np.stack([item[0] for item in seq])
            act_seq = np.stack([item[1] for item in seq])
            rew_seq = np.stack([item[2] for item in seq])
            nobs_seq = np.stack([item[3] for item in seq])
            done_seq = np.stack([item[4] for item in seq])
            
            mask_seq = np.ones((len(seq), 1), dtype=np.float32)
            
            if pad_len > 0:
                obs_seq = np.pad(obs_seq, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                act_seq = np.pad(act_seq, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                rew_seq = np.pad(rew_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
                nobs_seq = np.pad(nobs_seq, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                done_seq = np.pad(done_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=1.0)
                mask_seq = np.pad(mask_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.0)
                
            obs_batch.append(obs_seq)
            actions_batch.append(act_seq)
            rewards_batch.append(rew_seq)
            next_obs_batch.append(nobs_seq)
            dones_batch.append(done_seq)
            mask_batch.append(mask_seq)
            
        return (np.stack(obs_batch), np.stack(actions_batch), np.stack(rewards_batch), 
                np.stack(next_obs_batch), np.stack(dones_batch), np.stack(mask_batch))

    def __len__(self) -> int:
        return len(self.buffer)
