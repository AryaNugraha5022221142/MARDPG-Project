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
             next_obs: np.ndarray, dones: np.ndarray, 
             actor_hidden: List[Tuple[np.ndarray, np.ndarray]], 
             critic_hidden: List[Tuple[np.ndarray, np.ndarray]],
             episode_done: bool):
        """
        Adds a transition to the current episode.
        actor_hidden: list of (h, c) for each agent's actor at the START of this transition
        critic_hidden: list of (h, c) for each agent's critic at the START of this transition
        episode_done: True if the entire episode is over
        """
        self.current_episode.append((obs, actions, rewards, next_obs, dones, actor_hidden, critic_hidden))
        
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
        Returns stacked numpy arrays and initial hidden states.
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
        actor_h_batch = [] # List of (num_agents, (h, c))
        critic_h_batch = [] # List of (num_agents, (h, c))
        
        for ep in sampled_episodes:
            if len(ep) > seq_len:
                start_idx = random.randint(0, len(ep) - seq_len)
            else:
                start_idx = 0
            
            seq = ep[start_idx : start_idx + seq_len]
            
            # Initial hidden states for this sequence
            actor_h_batch.append(seq[0][5])
            critic_h_batch.append(seq[0][6])
            
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
            
        def process_hidden(h_batch):
            num_agents = len(h_batch[0])
            h_init = []
            c_init = []
            for i in range(num_agents):
                # h_agent_batch: (batch, num_layers, 1, hidden_dim)
                h_agent_batch = np.stack([hb[i][0] for hb in h_batch])
                c_agent_batch = np.stack([hb[i][1] for hb in h_batch])
                
                # Reshape to (num_layers, batch, hidden_dim)
                # First, remove the singleton dimension at index 2
                h_agent_batch = np.squeeze(h_agent_batch, axis=2)
                c_agent_batch = np.squeeze(c_agent_batch, axis=2)
                
                # Then transpose from (batch, num_layers, hidden_dim) to (num_layers, batch, hidden_dim)
                h_agent_batch = np.transpose(h_agent_batch, (1, 0, 2))
                c_agent_batch = np.transpose(c_agent_batch, (1, 0, 2))
                
                h_init.append(h_agent_batch)
                c_init.append(c_agent_batch)
            return h_init, c_init

        actor_h_init, actor_c_init = process_hidden(actor_h_batch)
        critic_h_init, critic_c_init = process_hidden(critic_h_batch)
            
        return (np.stack(obs_batch), np.stack(actions_batch), np.stack(rewards_batch), 
                np.stack(next_obs_batch), np.stack(dones_batch), 
                (actor_h_init, actor_c_init), (critic_h_init, critic_c_init))

    def __len__(self) -> int:
        return len(self.buffer)
