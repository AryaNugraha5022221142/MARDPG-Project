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
    Multi-Agent Recurrent Deterministic Policy Gradient (MARDPG).

    FIX 4: This class was originally designed around a 4-D velocity-command
    action space (vx, vy, vz, yaw_rate) but the paper and the kinematic
    environment use a 2-D steering-angle action space (rho, tau).
    The training script correctly passes action_dim=2; the default here has
    been changed to 2 to prevent silent misconfiguration when the class is
    instantiated directly.  If you genuinely need 4-D control (e.g. with
    QuadcopterEnv), pass action_dim=4 explicitly.
    """
    def __init__(self, obs_dim: int = 34, action_dim: int = 2,  # FIX 4: default 4→2
                 num_agents: int = 3,
                 config: Dict[str, Any] = None, device: str = 'cpu',
                 independent_critics: bool = False):
        # FIX 4: Explicit guard so mismatches surface at construction time.
        assert action_dim in (2, 4), (
            f"action_dim must be 2 (kinematic steering) or 4 (velocity control), got {action_dim}"
        )
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
        self.action_bound = float(config.get('environment', {}).get('action_bound', np.pi / 6.0))
        
        # Shared Actor
        hidden_dim = config['network']['actor'].get('hidden_dim', 128)
        lstm_layers = config['network']['actor'].get('lstm_layers', 1)
        
        self.actor = ActorLSTM(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            output_dim=action_dim,
            num_agents=num_agents,
            action_limit=self.action_bound,
        ).to(self.device)
        
        assert len(self.actor.output_heads) == self.num_agents, (
            f"Actor has {len(self.actor.output_heads)} heads, "
            f"but num_agents={self.num_agents}"
        )
        
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        
        # Centralized Critics (one per agent, but each sees all agents)
        self.critics = [CriticLSTM(obs_dim, action_dim, num_agents, hidden_dim, lstm_layers, independent=independent_critics).to(self.device) for _ in range(num_agents)]
        self.critics_target = [copy.deepcopy(c).to(self.device) for c in self.critics]
        
        # Noise (Gaussian for continuous)
        # Set total steps to a reasonable estimate: e.g. 600 max steps/ep * 5000 episodes = 3,000,000 steps.
        # This allows exploration to organically decay properly over the full 5000 episodes.
        self.noise = AdaptiveGaussianNoise(action_dim,
                                            sigma_start=0.5,
                                            sigma_end=0.10,
                                            total_steps=10_000_000)
        
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
                    action += self.noise.sample()
                
                action = np.clip(action, -self.action_bound, self.action_bound)
                actions.append(action)
                
                # Store hidden state for critic
                if i == 0:
                    actor_hiddens = []
                actor_hiddens.append(h_seq)
            
            if explore:
                self.noise.step()
            
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
        rewards = torch.clamp(rewards, -500.0, 500.0)
        next_obs = torch.FloatTensor(nobs_seq).to(self.device)
        dones = torch.FloatTensor(done_seq).to(self.device)
        masks = torch.FloatTensor(mask_seq).to(self.device).squeeze(-1) # (batch, seq)
        
        burn_in = self.seq_len // 4
        masks[:, :burn_in] = 0.0 # Set burn_in steps to 0
        
        # FIX 6d: Dedicated hidden-state tensors per forward-pass phase.
        # Prevents accidental reuse when new components are added (see audit Risk 5).
        actor_hidden_target  = [self.actor.init_hidden(self.batch_size, self.device) for _ in range(self.num_agents)]
        actor_hidden_critic  = [self.actor.init_hidden(self.batch_size, self.device) for _ in range(self.num_agents)]
        actor_hidden_actor   = [self.actor.init_hidden(self.batch_size, self.device) for _ in range(self.num_agents)]
        critic_hidden_update = [self.critics[i].init_hidden(self.batch_size, self.device) for i in range(self.num_agents)]
        critic_hidden_target = [self.critics_target[i].init_hidden(self.batch_size, self.device) for i in range(self.num_agents)]
        critic_hidden_actor  = [self.critics[i].init_hidden(self.batch_size, self.device) for i in range(self.num_agents)]
        
        # 1. Update Critics
        critic_losses = []
        q_value_mags = []
        
        # Target actions
        with torch.no_grad():
            next_actions_all = []
            for i in range(self.num_agents):
                agent_next_obs = next_obs[:, :, i, :]
                next_act, _, _ = self.actor_target(agent_next_obs, actor_hidden_target[i], agent_idx=i)
                next_actions_all.append(next_act)
            next_actions_all = torch.stack(next_actions_all, dim=2)
            
        for i in range(self.num_agents):
            # Target Q
            with torch.no_grad():
                target_q_seq, _ = self.critics_target[i](next_obs, next_actions_all, critic_hidden_target[i], agent_idx=i)
                target_q_seq = target_q_seq.squeeze(-1)
                target_q = rewards[:, :, i] + self.gamma * target_q_seq * (1 - dones[:, :, i])
                
            # Current Q
            current_q_seq, _ = self.critics[i](obs, actions, critic_hidden_update[i], agent_idx=i)
            current_q = current_q_seq.squeeze(-1)
            
            # Critic loss (MSE over sequence with mask)
            loss = F.mse_loss(current_q, target_q, reduction='none')
            critic_loss = (loss * masks).sum() / (masks.sum() + 1e-8)
            critic_losses.append(critic_loss.item())
            
            # Track average Q-value magnitude (applied mask)
            q_mag = (current_q.abs() * masks).sum() / (masks.sum() + 1e-8)
            q_value_mags.append(q_mag.item())
            
            # Optimize critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
            self.critic_optimizers[i].step()
            
        # 2. Update Actor
        actor_loss = 0

        # FIX 3c: Freeze critic parameters before actor backward to prevent
        # gradient leakage into critic.parameters().grad (see audit Risk 4).
        for i in range(self.num_agents):
            for p in self.critics[i].parameters():
                p.requires_grad_(False)

        for i in range(self.num_agents):
            # Only recompute agent i's action from current policy
            agent_i_obs = obs[:, :, i, :]
            # FIX 6f: gradient-carrying forward uses actor_hidden_actor[i]
            agent_i_act, _, _ = self.actor(agent_i_obs, actor_hidden_actor[i], agent_idx=i)
            
            # Build joint action: agent i from current policy, others detached
            other_acts = []
            for j in range(self.num_agents):
                if j == i:
                    other_acts.append(agent_i_act)
                else:
                    agent_j_obs = obs[:, :, j, :]
                    with torch.no_grad():
                        # FIX 6f: non-gradient agents use actor_hidden_critic[j]
                        a_j, _, _ = self.actor(agent_j_obs, actor_hidden_critic[j], agent_idx=j)
                    other_acts.append(a_j.detach())
            
            # Stack: (batch, seq, N, dim)
            joint_actions = torch.stack(other_acts, dim=2)
            
            # FIX 6f: dedicated critic hidden for actor Q-value computation
            q_values, _ = self.critics[i](obs, joint_actions, critic_hidden_actor[i], agent_idx=i)
            q_values = q_values.squeeze(-1) # (batch, seq)
            
            # Actor loss with mask
            a_loss = (-q_values * masks).sum() / (masks.sum() + 1e-8)
            actor_loss += a_loss
            
        actor_loss /= self.num_agents
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # FIX 3d: Re-enable critic gradients for the next update cycle.
        for i in range(self.num_agents):
            for p in self.critics[i].parameters():
                p.requires_grad_(True)
        
        # 3. Soft Update Targets
        self._soft_update(self.actor, self.actor_target)
        for i in range(self.num_agents):
            self._soft_update(self.critics[i], self.critics_target[i])
            
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': np.mean(critic_losses),
            'q_value_mag': np.mean(q_value_mags)
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
