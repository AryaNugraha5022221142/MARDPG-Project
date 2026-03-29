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
             next_obs: np.ndarray, dones: np.ndarray, hidden: List[Tuple[np.ndarray, np.ndarray]], 
             episode_done: bool):
        """
        Adds a transition to the current episode.
        hidden: list of (h, c) for each agent at the START of this transition
        episode_done: True if the entire episode is over (all agents done or max steps)
        """
        self.current_episode.append((obs, actions, rewards, next_obs, dones, hidden))
        
        if episode_done:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample(self, batch_size: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Samples a batch of sequences.
        Returns stacked numpy arrays of shape (batch_size, seq_len, num_agents, ...)
        And the initial hidden states for the sequence: (num_agents, num_layers, batch_size, hidden_dim)
        """
        # Filter out episodes that are too short
        valid_episodes = [ep for ep in self.buffer if len(ep) >= seq_len]
        if not valid_episodes:
            valid_episodes = list(self.buffer)
            
        sampled_episodes = random.choices(valid_episodes, k=batch_size)
        
        obs_batch = []
        actions_batch = []
        rewards_batch = []
        next_obs_batch = []
        dones_batch = []
        init_hidden_batch = [] # List of (num_agents, (h, c))
        
        for ep in sampled_episodes:
            # Randomly pick a starting point for the sequence
            if len(ep) > seq_len:
                start_idx = random.randint(0, len(ep) - seq_len)
            else:
                start_idx = 0
            
            seq = ep[start_idx : start_idx + seq_len]
            
            # Initial hidden state for this sequence
            init_hidden_batch.append(seq[0][5])
            
            # If sequence is shorter than seq_len, pad it
            obs_seq = np.stack([item[0] for item in seq])
            act_seq = np.stack([item[1] for item in seq])
            rew_seq = np.stack([item[2] for item in seq])
            nobs_seq = np.stack([item[3] for item in seq])
            done_seq = np.stack([item[4] for item in seq])
            
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
            
        # Reshape init_hidden_batch to (num_agents, num_layers, batch_size, hidden_dim)
        # init_hidden_batch is list of length batch_size, each item is list of length num_agents, each item is (h, c)
        num_agents = len(init_hidden_batch[0])
        h_init = []
        c_init = []
        for i in range(num_agents):
            h_agent = np.stack([hb[i][0] for hb in init_hidden_batch]) # (batch, num_layers, 1, hidden_dim) -> wait, select_actions returns (num_layers, 1, hidden_dim)
            c_agent = np.stack([hb[i][1] for hb in init_hidden_batch])
            
            # We want (num_layers, batch, hidden_dim) for LSTM
            h_agent = np.transpose(h_agent.squeeze(2), (1, 0, 2))
            c_agent = np.transpose(c_agent.squeeze(2), (1, 0, 2))
            
            h_init.append(h_agent)
            c_init.append(c_agent)
            
        return (np.stack(obs_batch), np.stack(actions_batch), np.stack(rewards_batch), 
                np.stack(next_obs_batch), np.stack(dones_batch), (h_init, c_init))

    def __len__(self) -> int:
        return len(self.buffer)
