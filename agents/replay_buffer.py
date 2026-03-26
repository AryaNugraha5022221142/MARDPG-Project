# agents/replay_buffer.py
import numpy as np
from collections import deque
import random
from typing import Tuple

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
             next_obs: np.ndarray, dones: np.ndarray):
        """
        Adds a transition to the current episode.
        """
        self.current_episode.append((obs, actions, rewards, next_obs, dones))
        
        # If any agent is done, the episode ends
        if np.any(dones):
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample(self, batch_size: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples a batch of sequences.
        Returns stacked numpy arrays of shape (batch_size, seq_len, num_agents, ...)
        """
        # Filter out episodes that are too short
        valid_episodes = [ep for ep in self.buffer if len(ep) >= seq_len]
        if not valid_episodes:
            # Fallback: return shorter sequences if necessary, but ideally we have enough data
            valid_episodes = list(self.buffer)
            
        sampled_episodes = random.choices(valid_episodes, k=batch_size)
        
        obs_batch = []
        actions_batch = []
        rewards_batch = []
        next_obs_batch = []
        dones_batch = []
        
        for ep in sampled_episodes:
            # Randomly pick a starting point for the sequence
            if len(ep) > seq_len:
                start_idx = random.randint(0, len(ep) - seq_len)
            else:
                start_idx = 0
            
            seq = ep[start_idx : start_idx + seq_len]
            
            # If sequence is shorter than seq_len, pad it (though we try to avoid this)
            obs_seq = np.stack([item[0] for item in seq])
            act_seq = np.stack([item[1] for item in seq])
            rew_seq = np.stack([item[2] for item in seq])
            nobs_seq = np.stack([item[3] for item in seq])
            done_seq = np.stack([item[4] for item in seq])
            
            # Padding if necessary
            if len(seq) < seq_len:
                pad_len = seq_len - len(seq)
                obs_seq = np.pad(obs_seq, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                act_seq = np.pad(act_seq, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                rew_seq = np.pad(rew_seq, ((0, pad_len), (0, 0)), mode='edge')
                nobs_seq = np.pad(nobs_seq, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                done_seq = np.pad(done_seq, ((0, pad_len), (0, 0)), mode='edge')
                
            obs_batch.append(obs_seq)
            actions_batch.append(act_seq)
            rewards_batch.append(rew_seq)
            next_obs_batch.append(nobs_seq)
            dones_batch.append(done_seq)
            
        return (np.stack(obs_batch), np.stack(actions_batch), np.stack(rewards_batch), 
                np.stack(next_obs_batch), np.stack(dones_batch))

    def __len__(self) -> int:
        return len(self.buffer)
