# agents/mardpg.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
import copy

from .networks import ActorLSTM, CriticLSTM
from .replay_buffer import SequenceReplayBuffer

class OrnsteinUhlenbeckNoise:
    """
    OU Noise for continuous action exploration.
    """
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class MARDPG:
    """
    Multi-Agent Recurrent Deterministic Policy Gradient (MARDPG)
    """
    def __init__(self, obs_dim: int = 33, action_dim: int = 4, num_agents: int = 3, 
                 config: Dict[str, Any] = None, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = device
        
        if config is None:
            config = {
                'network': {'actor': {'hidden_dim': 128, 'lstm_layers': 1}},
                'learning': {'actor_lr': 1e-4, 'critic_lr': 1e-3, 'max_grad_norm': 1.0},
                'memory': {'buffer_size': 1000, 'batch_size': 32, 'seq_len': 16},
                'targets': {'update_rate': 0.01}
            }
        self.config = config
        
        self.gamma = 0.99
        self.tau = config['targets'].get('update_rate', 0.01)
        self.batch_size = config['memory'].get('batch_size', 32)
        self.seq_len = config['memory'].get('seq_len', 16)
        self.max_grad_norm = config['learning'].get('max_grad_norm', 1.0)
        
        # Shared Actor
        hidden_dim = config['network']['actor'].get('hidden_dim', 128)
        lstm_layers = config['network']['actor'].get('lstm_layers', 1)
        
        self.actor = ActorLSTM(obs_dim, hidden_dim, lstm_layers, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        
        # Centralized Critics (one per agent, but each sees all agents)
        self.critics = [CriticLSTM(obs_dim, action_dim, num_agents, hidden_dim, lstm_layers).to(self.device) for _ in range(num_agents)]
        self.critics_target = [copy.deepcopy(c).to(self.device) for c in self.critics]
        
        # Noise (OU for continuous)
        self.noise = [OrnsteinUhlenbeckNoise(action_dim) for _ in range(num_agents)]
        
        # Optimizers
        actor_lr = config['learning'].get('actor_lr', 1e-4)
        critic_lr = config['learning'].get('critic_lr', 1e-3)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]
        
        # Replay Buffer
        buffer_size = config['memory'].get('buffer_size', 1000)
        self.memory = SequenceReplayBuffer(buffer_size)

    def select_actions(self, obs: np.ndarray, hidden: List[Tuple[torch.Tensor, torch.Tensor]], 
                       noise_scale: float = 0.1) -> Tuple[np.ndarray, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Selects continuous actions for all agents.
        obs: (num_agents, obs_dim)
        hidden: list of (h, c) for each agent
        """
        actions = []
        new_hidden = []
        
        self.actor.eval()
        with torch.no_grad():
            for i in range(self.num_agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device) # (1, obs_dim)
                h, c = hidden[i]
                
                # Actor returns (batch, 1, action_dim) for single step
                action, (new_h, new_c) = self.actor(obs_tensor, (h, c))
                new_hidden.append((new_h, new_c))
                
                action = action.cpu().numpy().flatten()
                
                # Add OU Noise
                if noise_scale > 0:
                    action += self.noise[i].sample() * noise_scale
                
                action = np.clip(action, -1.0, 1.0)
                actions.append(action)
                
        self.actor.train()
        return np.array(actions), new_hidden

    def update(self) -> Dict[str, float]:
        """
        Performs one training step using BPTT on sequences.
        """
        if len(self.memory) < self.batch_size:
            return {}
            
        # Sample batch of sequences: (batch, seq_len, num_agents, dim)
        obs_seq, act_seq, rew_seq, nobs_seq, done_seq = self.memory.sample(self.batch_size, self.seq_len)
        
        # Convert to tensors
        obs = torch.FloatTensor(obs_seq).to(self.device)
        actions = torch.FloatTensor(act_seq).to(self.device)
        rewards = torch.FloatTensor(rew_seq).to(self.device)
        next_obs = torch.FloatTensor(nobs_seq).to(self.device)
        dones = torch.FloatTensor(done_seq).to(self.device)
        
        # 1. Update Critics
        critic_losses = []
        
        # Target actions for next_obs
        with torch.no_grad():
            next_actions_all = []
            for i in range(self.num_agents):
                # Actor target needs (batch, seq_len, obs_dim)
                agent_next_obs = next_obs[:, :, i, :]
                next_act, _ = self.actor_target(agent_next_obs, None)
                next_actions_all.append(next_act)
            next_actions_all = torch.stack(next_actions_all, dim=2) # (batch, seq_len, num_agents, action_dim)
            
        for i in range(self.num_agents):
            # Target Q
            with torch.no_grad():
                target_q_seq, _ = self.critics_target[i](next_obs, next_actions_all, None)
                target_q_seq = target_q_seq.squeeze(-1) # (batch, seq_len)
                target_q = rewards[:, :, i] + self.gamma * target_q_seq * (1 - dones[:, :, i])
                
            # Current Q
            current_q_seq, _ = self.critics[i](obs, actions, None)
            current_q = current_q_seq.squeeze(-1) # (batch, seq_len)
            
            # Critic loss (MSE over sequence)
            critic_loss = F.mse_loss(current_q, target_q)
            critic_losses.append(critic_loss.item())
            
            # Optimize critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
            self.critic_optimizers[i].step()
            
        # 2. Update Actor
        # We need to compute gradients through the critic
        actor_actions_all = []
        for i in range(self.num_agents):
            agent_obs = obs[:, :, i, :]
            act, _ = self.actor(agent_obs, None)
            actor_actions_all.append(act)
        actor_actions_all = torch.stack(actor_actions_all, dim=2) # (batch, seq_len, num_agents, action_dim)
        
        actor_loss = 0
        for i in range(self.num_agents):
            # Centralized critic evaluation
            q_values, _ = self.critics[i](obs, actor_actions_all, None)
            actor_loss += -q_values.mean()
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # 3. Soft Update Targets
        self._soft_update(self.actor, self.actor_target)
        for i in range(self.num_agents):
            self._soft_update(self.critics[i], self.critics_target[i])
            
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': np.mean(critic_losses)
        }

    def _soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str, epsilon: float, episode: int):
        """Saves models and state."""
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
        """Loads models and state."""
        state = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        
        for i in range(self.num_agents):
            self.critics[i].load_state_dict(state[f'critic_{i}'])
            self.critics_target[i].load_state_dict(state[f'critic_target_{i}'])
            self.critic_optimizers[i].load_state_dict(state[f'critic_optimizer_{i}'])
            
        return state.get('epsilon', 0.0), state.get('episode', 0)
