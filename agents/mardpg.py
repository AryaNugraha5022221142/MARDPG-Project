# agents/mardpg.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
import copy

from .networks import ActorLSTM, CriticLSTM
from .replay_buffer import SequenceReplayBuffer

class AdaptiveGaussianNoise:
    """
    Gaussian Noise for continuous action exploration with annealing.
    """
    def __init__(self, action_dim, sigma_start=0.5, sigma_end=0.05, total_steps=500000):
        self.sigma = sigma_start
        self.sigma_end = sigma_end
        self.decay = (sigma_start - sigma_end) / total_steps
        self.action_dim = action_dim

    def sample(self):
        return np.random.normal(0, self.sigma, self.action_dim)
        
    def step(self):
        self.sigma = max(self.sigma_end, self.sigma - self.decay)

class MARDPG:
    """
    Multi-Agent Recurrent Deterministic Policy Gradient (MARDPG)
    """
    def __init__(self, obs_dim: int = 34, action_dim: int = 4, num_agents: int = 3, 
                 config: Dict[str, Any] = None, device: str = 'cpu', independent_critics: bool = False):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = device
        self.independent_critics = independent_critics
        
        if config is None:
            config = {
                'network': {'actor': {'hidden_dim': 128, 'lstm_layers': 1}},
                'learning': {'actor_lr': 1e-4, 'critic_lr': 1e-3, 'max_grad_norm': 1.0},
                'memory': {'buffer_size': 1000, 'batch_size': 32, 'seq_len': 16},
                'targets': {'update_rate': 0.01}
            }
        self.config = config
        
        self.gamma = config['learning'].get('gamma', 0.99)
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
        self.critics = [CriticLSTM(obs_dim, action_dim, num_agents, hidden_dim, lstm_layers, independent=independent_critics).to(self.device) for _ in range(num_agents)]
        self.critics_target = [copy.deepcopy(c).to(self.device) for c in self.critics]
        
        # Noise (Gaussian for continuous)
        # Set total steps to a reasonable estimate: e.g. 600 max steps/ep * 5000 episodes = 3,000,000 steps.
        # This allows exploration to organically decay properly over the full 5000 episodes.
        self.noise = [AdaptiveGaussianNoise(action_dim,
                                            sigma_start=1.2,
                                            sigma_end=0.15,
                                            total_steps=3_000_000) for _ in range(num_agents)]
        
        # Optimizers
        actor_lr = config['learning'].get('actor_lr', 1e-4)
        critic_lr = config['learning'].get('critic_lr', 1e-3)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]
        
        # Replay Buffer
        buffer_size = config['memory'].get('buffer_size', 1000)
        self.memory = SequenceReplayBuffer(buffer_size)

    def select_actions(self, obs: np.ndarray, 
                       actor_hidden: List[Tuple[torch.Tensor, torch.Tensor]], 
                       critic_hidden: List[Tuple[torch.Tensor, torch.Tensor]],
                       explore: bool = True) -> Tuple[np.ndarray, List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Selects continuous actions for all agents and updates both actor and critic hidden states.
        obs: (num_agents, obs_dim)
        actor_hidden: list of (h, c) for each agent's actor
        critic_hidden: list of (h, c) for each agent's critic
        explore: whether to add exploration noise
        """
        actions = []
        new_actor_hidden = []
        new_critic_hidden = []
        
        self.actor.eval()
        for c in self.critics: c.eval()
        
        with torch.no_grad():
            # 1. Select actions using Actor
            for i in range(self.num_agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device) # (1, obs_dim)
                h, c = actor_hidden[i]
                
                # Actor returns (batch, 1, action_dim) for single step
                action, (new_h, new_c), h_seq = self.actor(obs_tensor, (h, c), agent_idx=i)
                new_actor_hidden.append((new_h, new_c))
                
                action = action.cpu().numpy().flatten()
                
                # Add Gaussian Noise
                if explore:
                    action += self.noise[i].sample()
                    self.noise[i].step() # anneal after each call
                
                action = np.clip(action, -3.5, 3.5)
                actions.append(action)
                
                # Store hidden state for critic
                if i == 0:
                    actor_hiddens = []
                actor_hiddens.append(h_seq)
            
            actions_np = np.array(actions)
            actions_tensor = torch.FloatTensor(actions_np).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, num_agents, action_dim)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, num_agents, obs_dim)
            
            # 2. Update Critic hidden states (even if we don't use the Q-values)
            for i in range(self.num_agents):
                h, c = critic_hidden[i]
                _, (new_h, new_c) = self.critics[i](obs_tensor, actions_tensor, (h, c), agent_idx=i)
                new_critic_hidden.append((new_h, new_c))
                
        self.actor.train()
        for c in self.critics: c.train()
        return actions_np, new_actor_hidden, new_critic_hidden

    def update(self) -> Dict[str, float]:
        """
        Performs one training step using BPTT on sequences.
        """
        if len(self.memory) < self.batch_size:
            return {}
            
        sampled = self.memory.sample(self.batch_size, self.seq_len)
        if sampled is None:
            return {}
            
        obs_seq, act_seq, rew_seq, nobs_seq, done_seq, mask_seq = sampled
        
        # Convert to tensors
        obs = torch.FloatTensor(obs_seq).to(self.device)
        actions = torch.FloatTensor(act_seq).to(self.device)
        rewards = torch.FloatTensor(rew_seq).to(self.device)
        # rewards = torch.clamp(rewards, -5.0, 10.0)
        next_obs = torch.FloatTensor(nobs_seq).to(self.device)
        dones = torch.FloatTensor(done_seq).to(self.device)
        masks = torch.FloatTensor(mask_seq).to(self.device).squeeze(-1) # (batch, seq)
        
        burn_in = min(8, self.seq_len // 2)
        masks[:, :burn_in] = 0.0 # Set burn_in steps to 0
        
        actor_hidden = [self.actor.init_hidden(self.batch_size, self.device) for _ in range(self.num_agents)]
        critic_hidden = [self.critics[i].init_hidden(self.batch_size, self.device) for i in range(self.num_agents)]
        
        # 1. Update Critics
        critic_losses = []
        
        # Target actions
        with torch.no_grad():
            next_actions_all = []
            for i in range(self.num_agents):
                agent_next_obs = next_obs[:, :, i, :]
                next_act, _, _ = self.actor_target(agent_next_obs, actor_hidden[i], agent_idx=i)
                next_actions_all.append(next_act)
            next_actions_all = torch.stack(next_actions_all, dim=2) # (batch, seq_len, num_agents, action_dim)
            
        for i in range(self.num_agents):
            # Target Q
            with torch.no_grad():
                target_q_seq, _ = self.critics_target[i](next_obs, next_actions_all, critic_hidden[i], agent_idx=i)
                target_q_seq = target_q_seq.squeeze(-1) # (batch, seq_len)
                target_q = rewards[:, :, i] + self.gamma * target_q_seq * (1 - dones[:, :, i])
                
            # Current Q
            current_q_seq, _ = self.critics[i](obs, actions, critic_hidden[i], agent_idx=i)
            current_q = current_q_seq.squeeze(-1) # (batch, seq_len)
            
            # Critic loss (MSE over sequence with mask)
            loss = F.mse_loss(current_q, target_q, reduction='none')
            critic_loss = (loss * masks).sum() / (masks.sum() + 1e-8)
            critic_losses.append(critic_loss.item())
            
            # Optimize critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
            self.critic_optimizers[i].step()
            
        # 2. Update Actor
        actor_loss = 0
        for i in range(self.num_agents):
            # Only recompute agent i's action from current policy
            agent_i_obs = obs[:, :, i, :]
            agent_i_act, _, _ = self.actor(agent_i_obs, actor_hidden[i], agent_idx=i)
            
            # Build joint action: agent i from current policy, others from dict
            other_acts = []
            for j in range(self.num_agents):
                if j == i:
                    other_acts.append(agent_i_act)
                else:
                    agent_j_obs = obs[:, :, j, :]
                    with torch.no_grad():
                        a_j, _, _ = self.actor(agent_j_obs, actor_hidden[j], agent_idx=j)
                    other_acts.append(a_j.detach())  # detach — no grad for j!=i
            
            # Stack: (batch, seq, N, dim)
            joint_actions = torch.stack(other_acts, dim=2)
            
            # Use stored critic state for actor gradient calculation
            q_values, _ = self.critics[i](obs, joint_actions, critic_hidden[i], agent_idx=i)
            q_values = q_values.squeeze(-1) # (batch, seq)
            
            # Actor loss with mask
            a_loss = (-q_values * masks).sum() / (masks.sum() + 1e-8)
            actor_loss += a_loss
            
        actor_loss /= self.num_agents
            
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
