# agents/mardpg.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
import copy

from .networks import ActorLSTM, Critic
from .replay_buffer import ReplayBuffer

class GaussianNoise:
    def __init__(self, action_dim, mu=0.0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma
        self.action_dim = action_dim

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.action_dim)

class MARDPG:
    """
    Multi-Agent Recurrent Deterministic Policy Gradient (MARDPG)
    """
    def __init__(self, obs_dim: int = 28, action_dim: int = 6, num_agents: int = 3, 
                 config: Dict[str, Any] = None, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = device
        
        if config is None:
            config = {
                'network': {'actor': {'hidden_dim': 128, 'lstm_layers': 1}},
                'learning': {'actor_lr': 1e-4, 'critic_lr': 1e-3, 'max_grad_norm': 1.0},
                'memory': {'buffer_size': 100000, 'batch_size': 128},
                'targets': {'update_rate': 0.01}
            }
        self.config = config
        
        self.gamma = 0.99
        self.tau = config['targets'].get('update_rate', 0.01)
        self.batch_size = config['memory'].get('batch_size', 128)
        self.max_grad_norm = config['learning'].get('max_grad_norm', 1.0)
        
        # Shared Actor
        hidden_dim = config['network']['actor'].get('hidden_dim', 128)
        lstm_layers = config['network']['actor'].get('lstm_layers', 1)
        dropout = config['network'].get('dropout', 0.2)
        
        self.actor = ActorLSTM(obs_dim, hidden_dim, lstm_layers, action_dim, dropout=dropout).to(self.device)
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

    def select_actions(self, obs: np.ndarray, hidden: List[Tuple[torch.Tensor, torch.Tensor]], 
                       epsilon: float = 0.0) -> Tuple[List[int], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Selects actions for all agents using epsilon-greedy policy.
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
                
                logits, (new_h, new_c) = self.actor(obs_tensor, (h, c))
                new_hidden.append((new_h, new_c))
                
                # Apply Gaussian Noise to Logits for exploration
                if epsilon > 0:
                    noise = torch.FloatTensor(self.noise.sample()).to(self.device) * epsilon
                    logits = logits + noise
                
                action = torch.argmax(logits, dim=1).item()
                actions.append(action)
                
        self.actor.train()
        return actions, new_hidden

    def update(self) -> Dict[str, float]:
        """
        Performs one training step.
        """
        if len(self.memory) < self.batch_size:
            return {}
            
        # Sample batch
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        obs = torch.FloatTensor(obs_batch).to(self.device)           # (batch, num_agents, obs_dim)
        actions = torch.LongTensor(actions_batch).to(self.device)    # (batch, num_agents)
        rewards = torch.FloatTensor(rewards_batch).to(self.device)   # (batch, num_agents)
        next_obs = torch.FloatTensor(next_obs_batch).to(self.device) # (batch, num_agents, obs_dim)
        dones = torch.FloatTensor(dones_batch).to(self.device)       # (batch, num_agents)
        
        # Actions one-hot
        actions_onehot = F.one_hot(actions, self.action_dim).float() # (batch, num_agents, action_dim)
        
        # 1. Update Critics
        critic_losses = []
        with torch.no_grad():
            # Get next actions from target actor
            next_actions_logits = []
            for i in range(self.num_agents):
                # We treat each agent's sequence independently for the actor target
                # Since we don't store sequences in this basic buffer, we just pass single steps
                # and init hidden states to zero. For a true recurrent buffer, we'd sample sequences.
                # Here we simplify by passing the batch of next_obs for agent i.
                agent_next_obs = next_obs[:, i, :] # (batch, obs_dim)
                logits, _ = self.actor_target(agent_next_obs, None)
                next_actions_logits.append(logits)
                
            next_actions_logits = torch.stack(next_actions_logits, dim=1) # (batch, num_agents, action_dim)
            next_actions = torch.argmax(next_actions_logits, dim=-1) # (batch, num_agents)
            next_actions_onehot = F.one_hot(next_actions, self.action_dim).float()
            
        for i in range(self.num_agents):
            # Target Q
            with torch.no_grad():
                target_q = self.critics_target[i](next_obs, next_actions_onehot).squeeze(-1) # (batch,)
                target_q = rewards[:, i] + self.gamma * target_q * (1 - dones[:, i])
                
            # Current Q
            current_q = self.critics[i](obs, actions_onehot).squeeze(-1) # (batch,)
            
            # Critic loss
            critic_loss = F.mse_loss(current_q, target_q)
            critic_losses.append(critic_loss.item())
            
            # Optimize critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
            self.critic_optimizers[i].step()
            
        # 2. Update Actor
        # Forward pass for all agents
        actor_logits = []
        for i in range(self.num_agents):
            agent_obs = obs[:, i, :]
            logits, _ = self.actor(agent_obs, None)
            actor_logits.append(logits)
            
        actor_logits = torch.stack(actor_logits, dim=1) # (batch, num_agents, action_dim)
        actor_probs = F.softmax(actor_logits, dim=-1) # (batch, num_agents, action_dim)
        
        actor_loss = 0
        for i in range(self.num_agents):
            # The critic expects one-hot actions, we pass probabilities for differentiability (Gumbel-Softmax alternative)
            # In MADDPG, we can pass the continuous relaxation
            q_values = self.critics[i](obs, actor_probs)
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
