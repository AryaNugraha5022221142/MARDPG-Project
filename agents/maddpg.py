# agents/maddpg.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
import copy

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer

class GaussianNoise:
    def __init__(self, action_dim, mu=0.0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma
        self.action_dim = action_dim

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.action_dim)

class MADDPG:
    """
    Multi-Agent Deterministic Policy Gradient (MADDPG) - Non-Recurrent Baseline
    """
    def __init__(self, obs_dim: int = 33, action_dim: int = 4, num_agents: int = 3, 
                 config: Dict[str, Any] = None, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = device
        
        if config is None:
            config = {
                'network': {'actor': {'hidden_dim': 256}},
                'learning': {'actor_lr': 1e-4, 'critic_lr': 1e-3, 'max_grad_norm': 1.0},
                'memory': {'buffer_size': 100000, 'batch_size': 128},
                'targets': {'update_rate': 0.01}
            }
        self.config = config
        
        self.gamma = config['learning'].get('gamma', 0.99)
        self.tau = config['targets'].get('update_rate', 0.01)
        self.batch_size = config['memory'].get('batch_size', 128)
        self.max_grad_norm = config['learning'].get('max_grad_norm', 1.0)
        
        # Shared Actor
        hidden_dim = config['network']['actor'].get('hidden_dim', 256)
        dropout = config['network'].get('dropout', 0.2)
        
        self.actor = Actor(obs_dim, hidden_dim, action_dim, dropout=dropout).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        
        # Critics (one per agent)
        self.critics = [Critic(obs_dim, action_dim, num_agents, dropout=dropout).to(self.device) for _ in range(num_agents)]
        self.critics_target = copy.deepcopy(self.critics)
        
        # Noise
        self.noise = GaussianNoise(action_dim, sigma=config.get('noise_sigma', 0.1))
        
        # Optimizers
        actor_lr = config['learning'].get('actor_lr', 1e-4)
        critic_lr = config['learning'].get('critic_lr', 1e-3)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]
        
        # Replay Buffer
        buffer_size = config['memory'].get('buffer_size', 100000)
        self.memory = ReplayBuffer(buffer_size)

    def select_actions(self, obs: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """
        Selects actions for all agents.
        obs: (num_agents, obs_dim)
        """
        actions = []
        
        self.actor.eval()
        with torch.no_grad():
            for i in range(self.num_agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device) # (1, obs_dim)
                action = self.actor(obs_tensor, agent_idx=i).cpu().numpy().squeeze(0)
                
                if noise_scale > 0:
                    noise = self.noise.sample() * noise_scale
                    action = np.clip(action + noise, -3.5, 3.5)
                
                actions.append(action)
                
        self.actor.train()
        return np.array(actions)

    def update(self) -> Dict[str, float]:
        if len(self.memory) < self.batch_size:
            return {}
            
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.memory.sample(self.batch_size)
        
        obs = torch.FloatTensor(obs_batch).to(self.device) # (batch, num_agents, obs_dim)
        actions = torch.FloatTensor(actions_batch).to(self.device) # (batch, num_agents, action_dim)
        rewards = torch.FloatTensor(rewards_batch).to(self.device) # (batch, num_agents)
        next_obs = torch.FloatTensor(next_obs_batch).to(self.device) # (batch, num_agents, obs_dim)
        dones = torch.FloatTensor(dones_batch).to(self.device) # (batch, num_agents)
        
        # Flatten obs and actions for centralized critic
        obs_full = obs.view(self.batch_size, -1)
        next_obs_full = next_obs.view(self.batch_size, -1)
        
        # 1. Update Critics
        critic_losses = []
        with torch.no_grad():
            next_actions = []
            for i in range(self.num_agents):
                agent_next_obs = next_obs[:, i, :]
                next_act = self.actor_target(agent_next_obs, agent_idx=i)
                next_actions.append(next_act)
                
            next_actions = torch.stack(next_actions, dim=1) # (batch, num_agents, action_dim)
            next_actions_full = next_actions.view(self.batch_size, -1)
            
        for i in range(self.num_agents):
            with torch.no_grad():
                target_q = self.critics_target[i](next_obs_full, next_actions_full).squeeze(-1)
                target_q = rewards[:, i] + self.gamma * target_q * (1 - dones[:, i])
                
            actions_full = actions.view(self.batch_size, -1)
            current_q = self.critics[i](obs_full, actions_full).squeeze(-1)
            critic_loss = F.mse_loss(current_q, target_q)
            critic_losses.append(critic_loss.item())
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
            self.critic_optimizers[i].step()
            
        actor_loss = 0
        for i in range(self.num_agents):
            # Recompute agent i's action
            agent_i_obs = obs[:, i, :]
            agent_i_act = self.actor(agent_i_obs, agent_idx=i)
            
            # Build joint action with other agents' actions detached
            other_acts = []
            for j in range(self.num_agents):
                if j == i:
                    other_acts.append(agent_i_act)
                else:
                    agent_j_obs = obs[:, j, :]
                    a_j = self.actor(agent_j_obs, agent_idx=j)
                    other_acts.append(a_j.detach())
            
            joint_actions = torch.stack(other_acts, dim=1)
            joint_actions_full = joint_actions.view(self.batch_size, -1)
            
            q_values = self.critics[i](obs_full, joint_actions_full)
            actor_loss += -q_values.mean()
            
        actor_loss /= self.num_agents
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        self._soft_update(self.actor, self.actor_target)
        for i in range(self.num_agents):
            self._soft_update(self.critics[i], self.critics_target[i])
            
        return {'actor_loss': actor_loss.item(), 'critic_loss': np.mean(critic_losses)}

    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str, epsilon: float, episode: int):
        state = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'epsilon': epsilon,
            'episode': episode
        }
        for i in range(self.num_agents):
            state[f'critic_{i}'] = self.critics[i].state_dict()
            state[f'critic_target_{i}'] = self.critics_target[i].state_dict()
            state[f'critic_optimizer_{i}'] = self.critic_optimizers[i].state_dict()
        torch.save(state, filepath)

    def load(self, filepath: str) -> Tuple[float, int]:
        state = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        for i in range(self.num_agents):
            self.critics[i].load_state_dict(state[f'critic_{i}'])
            self.critics_target[i].load_state_dict(state[f'critic_target_{i}'])
            self.critic_optimizers[i].load_state_dict(state[f'critic_optimizer_{i}'])
        return state.get('epsilon', 0.0), state.get('episode', 0)
