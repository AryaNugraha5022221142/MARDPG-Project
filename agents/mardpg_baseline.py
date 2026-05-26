import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import List, Tuple, Dict, Any

from .replay_buffer import SequenceReplayBuffer

class AnnealedGaussianNoise:
    def __init__(self, action_dim: int, sigma_start: float = 0.3, sigma_end: float = 0.05, total_steps: int = 5000000):
        self.action_dim = action_dim
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.total_steps = total_steps
        self.current_step = 0
        self.sigma = sigma_start

    def sample(self) -> np.ndarray:
        fraction = min(1.0, float(self.current_step) / max(1.0, float(self.total_steps)))
        self.sigma = self.sigma_start - fraction * (self.sigma_start - self.sigma_end)
        return np.random.normal(0, self.sigma, size=self.action_dim)

    def step(self):
        self.current_step += 1

class MARDPGBaseNetwork(nn.Module):
    def __init__(self, obs_structure: dict):
        super().__init__()
        self.obs_structure = obs_structure
        
        angles_dim = obs_structure['angles'][1] - obs_structure['angles'][0]
        goal_dim = obs_structure['goal'][1] - obs_structure['goal'][0]
        
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1)
        self.fc_angles = nn.Linear(angles_dim, 8)
        self.fc_goal = nn.Linear(goal_dim, 8)
        self.fusion_dim = 32 * 4 * 4 + 8 + 8  # 4x4 output spatial size * 32 channels + 8 (angles) + 8 (goal)

    def forward(self, x: torch.Tensor):
        b, s, _ = x.shape
        
        angles = x[:, :, self.obs_structure['angles'][0]:self.obs_structure['angles'][1]]
        ranges = x[:, :, self.obs_structure['ranges'][0]:self.obs_structure['ranges'][1]]
        goal = x[:, :, self.obs_structure['goal'][0]:self.obs_structure['goal'][1]]
        
        ranges_2d = ranges.contiguous().view(b * s, 1, 5, 5)
        c_out = F.relu(self.conv(ranges_2d))
        c_out = c_out.view(b * s, -1)

        angles_flat = angles.contiguous().view(b * s, -1)
        goal_flat = goal.contiguous().view(b * s, -1)

        angles_out = F.relu(self.fc_angles(angles_flat))
        goal_out = F.relu(self.fc_goal(goal_flat))

        fused = torch.cat([c_out, angles_out, goal_out], dim=1).view(b, s, -1)
        return fused

class ActorLSTMAgentHead(nn.Module):
    def __init__(self, fusion_dim: int, hidden_dim: int, action_bound: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_bound = action_bound
        self.lstm = nn.LSTM(fusion_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, 2)

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, fused: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        out, hidden = self.lstm(fused, hidden)
        logits = self.output_head(out)
        norm_out = torch.tanh(logits)
        actions = self.action_bound * norm_out
        rho = actions[:, :, 0]
        tau = actions[:, :, 1]
        return actions, hidden, out, (rho, tau)

class MultiActorLSTMBaseline(nn.Module):
    """
    Implements a shared base network with per-agent LSTM and output heads.
    """
    def __init__(self, obs_dim_total: int, num_agents: int = 3, hidden_dim: int = 128, device: str = 'cpu', action_bound: float = 0.5235987756, obs_structure: dict = None):
        super().__init__()
        assert action_bound < 1.0, "Action bound must be < 1.0 for kinematic control to prevent instability."
        if obs_structure is None:
            obs_structure = {'angles': [0, 2], 'ranges': [2, 27], 'goal': [27, 30]}
        self.num_agents = num_agents
        self.shared_base = MARDPGBaseNetwork(obs_structure)
        self.heads = nn.ModuleList([ActorLSTMAgentHead(self.shared_base.fusion_dim, hidden_dim, action_bound) for _ in range(num_agents)])

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        return self.heads[0].init_hidden(batch_size, device)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None, agent_idx: int = 0):
        is_single_step = x.dim() == 2
        if is_single_step:
            x = x.unsqueeze(1) # (batch, 1, input_dim)

        fused = self.shared_base(x)

        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        actions, hidden_out, lstm_out, (rho, tau) = self.heads[agent_idx](fused, hidden)
        
        if is_single_step:
            actions = actions.squeeze(1)
            rho = rho.squeeze(1)
            tau = tau.squeeze(1)

        return actions, hidden_out, lstm_out, (rho, tau)

class AttentionCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, 
                 hidden_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # Per-agent encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head self-attention over agents
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor, 
                hidden: Tuple[torch.Tensor, torch.Tensor] = None,
                agent_idx: int = None):
        """
        obs: (batch_size, seq_len, num_agents, obs_dim)
        actions: (batch_size, seq_len, num_agents, action_dim)
        """
        batch_size, seq_len, N, _ = obs.shape
        
        # Encode each agent's (obs, action) pair
        x = torch.cat([obs, actions], dim=-1)           # (B, T, N, obs+act)
        x = self.encoder(x)                            # (B, T, N, hidden_dim)
        
        # Reshape for attention: (B*T, N, hidden_dim)
        x_flat = x.view(batch_size * seq_len, N, self.hidden_dim)
        
        # Self-attention: each agent attends to all agents
        attn_out, attn_weights = self.attention(x_flat, x_flat, x_flat)
        # attn_out: (B*T, N, hidden_dim)
        # attn_weights: (B*T, N, N) - interpretable interaction weights!
        
        # Residual + LayerNorm
        x_flat = self.norm1(x_flat + attn_out)
        
        # Feed-forward
        ff_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + ff_out)
        
        # Pool across agents (mean pooling for centralized Q)
        # Alternative: use agent_idx to select specific agent's embedding
        if agent_idx is not None and agent_idx < x_flat.shape[1]:
            pooled = x_flat[:, agent_idx, :]    # agent-specific embedding (B*T, hidden_dim)
        else:
            pooled = x_flat.mean(dim=1)         # fallback
        
        # Reshape for LSTM: (B, T, hidden_dim)
        pooled = pooled.view(batch_size, seq_len, self.hidden_dim)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, pooled.device)
            
        lstm_out, new_hidden = self.lstm(pooled, hidden)
        q_value = self.fc_out(lstm_out)  # (B, T, 1)
        
        return q_value, new_hidden

class MARDPG_Baseline:
    """
    MARDPG Baseline Implementation (as specified)
    """
    def __init__(self, obs_dim: int = 30, action_dim: int = 2, num_agents: int = 3, 
                 config: Dict[str, Any] = None, device: str = 'cpu', independent_critics: bool = False):
        self.obs_dim = obs_dim
        self.action_dim = action_dim # Kinematic env action dim is 2: [rho, tau]
        self.num_agents = num_agents
        self.device = device
        
        if config is None:
            config = {
                'network': {'actor': {'hidden_dim': 128}},
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
        
        hidden_dim = config['network']['actor'].get('hidden_dim', 128)
        obs_structure = config.get('obs_structure', {'angles': [0, 2], 'ranges': [2, 27], 'goal': [27, 30]})
        
        self.actor = MultiActorLSTMBaseline(obs_dim, num_agents, hidden_dim, device, self.action_bound, obs_structure).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        
        critic_hidden_dim = config['network'].get('critic', {}).get('hidden_dim', hidden_dim)
        
        self.critics = [
            AttentionCritic(obs_dim, action_dim, num_agents, critic_hidden_dim, n_heads=4).to(self.device)
            for _ in range(num_agents)
        ]
        self.critics_target = [copy.deepcopy(c).to(self.device) for c in self.critics]
        
        actor_lr = config['learning'].get('actor_lr', 1e-4)
        critic_lr = config['learning'].get('critic_lr', 1e-3)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]
        
        buffer_size = config['memory'].get('buffer_size', 1000)
        self.memory = SequenceReplayBuffer(buffer_size)

        sigma_start = config.get('exploration', {}).get('sigma_start', 0.3)
        sigma_end = config.get('exploration', {}).get('sigma_end', 0.05)
        total_steps = config.get('exploration', {}).get('total_steps', 5000000)
        self.noise = AnnealedGaussianNoise(action_dim, sigma_start, sigma_end, total_steps)

    def select_actions(self, obs: np.ndarray, 
                       actor_hidden: List[Tuple[torch.Tensor, torch.Tensor]], 
                       critic_hidden: List[Tuple[torch.Tensor, torch.Tensor]],
                       explore: bool = True):
        actions = []
        new_actor_hidden = []
        new_critic_hidden = []
        
        self.actor.eval()
        for c in self.critics: c.eval()
        
        with torch.no_grad():
            for i in range(self.num_agents):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
                h, c = actor_hidden[i]
                
                action, (new_h, new_c), _, _ = self.actor(obs_tensor, (h, c), agent_idx=i)
                new_actor_hidden.append((new_h, new_c))
                
                action_np = action.cpu().numpy().flatten()
                
                if explore:
                    action_np += self.noise.sample()
                    
                action_np = np.clip(action_np, -self.action_bound, self.action_bound)
                actions.append(action_np.astype(np.float32))
            
            if explore:
                self.noise.step()
            
            actions_np = np.array(actions)
            for i in range(self.num_agents):
                h, c = critic_hidden[i]
                new_critic_hidden.append((h, c))
                
        self.actor.train()
        for c in self.critics: c.train()
        return actions_np, new_actor_hidden, new_critic_hidden

    def update(self) -> Dict[str, float]:
        if len(self.memory) < self.batch_size:
            return {}
            
        sampled = self.memory.sample(self.batch_size, self.seq_len)
        if sampled is None:
            return {}
            
        obs_seq, act_seq, rew_seq, nobs_seq, done_seq, mask_seq = sampled
        
        obs = torch.FloatTensor(obs_seq).to(self.device)
        actions = torch.FloatTensor(act_seq).to(self.device)
        rewards = torch.FloatTensor(rew_seq).to(self.device)
        next_obs = torch.FloatTensor(nobs_seq).to(self.device)
        dones = torch.FloatTensor(done_seq).to(self.device)
        masks = torch.FloatTensor(mask_seq).to(self.device).squeeze(-1)
        
        burn_in = self.seq_len // 4
        masks[:, :burn_in] = 0.0
        
        agent_masks = masks.unsqueeze(-1).repeat(1, 1, self.num_agents)
        prev_dones = torch.cat([torch.zeros_like(dones[:, 0:1, :]), dones[:, :-1, :]], dim=1)
        agent_masks = agent_masks * (1.0 - prev_dones)
        
        actor_hidden_target  = [self.actor.init_hidden(self.batch_size, self.device) for _ in range(self.num_agents)]
        actor_hidden_critic  = [self.actor.init_hidden(self.batch_size, self.device) for _ in range(self.num_agents)]
        actor_hidden_actor   = [self.actor.init_hidden(self.batch_size, self.device) for _ in range(self.num_agents)]
        critic_hidden_update = [self.critics[i].init_hidden(self.batch_size, self.device) for i in range(self.num_agents)]
        critic_hidden_target = [self.critics_target[i].init_hidden(self.batch_size, self.device) for i in range(self.num_agents)]
        critic_hidden_actor  = [self.critics[i].init_hidden(self.batch_size, self.device) for i in range(self.num_agents)]
        
        critic_losses = []
        q_value_mags = []
        
        with torch.no_grad():
            next_actions_all = []
            for i in range(self.num_agents):
                agent_next_obs = next_obs[:, :, i, :]
                next_act, _, _, _ = self.actor_target(agent_next_obs, actor_hidden_target[i], agent_idx=i)
                next_actions_all.append(next_act)
            next_actions_all = torch.stack(next_actions_all, dim=2)
            
        for i in range(self.num_agents):
            with torch.no_grad():
                target_q_seq, _ = self.critics_target[i](next_obs, next_actions_all, critic_hidden_target[i], agent_idx=i)
                target_q_seq = target_q_seq.squeeze(-1)
                target_q = rewards[:, :, i] + self.gamma * target_q_seq * (1 - dones[:, :, i])
                
            current_q_seq, _ = self.critics[i](obs, actions, critic_hidden_update[i], agent_idx=i)
            current_q = current_q_seq.squeeze(-1)
            
            agent_mask = agent_masks[:, :, i]
            loss = F.mse_loss(current_q, target_q, reduction='none')
            critic_loss = (loss * agent_mask).sum() / (agent_mask.sum() + 1e-8)
            critic_losses.append(critic_loss.item())
            
            q_mag = (current_q.abs() * agent_mask).sum() / (agent_mask.sum() + 1e-8)
            q_value_mags.append(q_mag.item())
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
            self.critic_optimizers[i].step()
            
        for i in range(self.num_agents):
            for p in self.critics[i].parameters():
                p.requires_grad_(False)

        actor_loss = 0
        for i in range(self.num_agents):
            agent_i_obs = obs[:, :, i, :]
            agent_i_act, _, _, _ = self.actor(agent_i_obs, actor_hidden_actor[i], agent_idx=i)
            
            other_acts = []
            for j in range(self.num_agents):
                if j == i:
                    other_acts.append(agent_i_act)
                else:
                    agent_j_obs = obs[:, :, j, :]
                    with torch.no_grad():
                        a_j, _, _, _ = self.actor(agent_j_obs, actor_hidden_critic[j], agent_idx=j)
                    other_acts.append(a_j.detach())
            
            joint_actions = torch.stack(other_acts, dim=2)
            
            q_values, _ = self.critics[i](obs, joint_actions, critic_hidden_actor[i], agent_idx=i)
            q_values = q_values.squeeze(-1)
            
            agent_mask = agent_masks[:, :, i]
            a_loss = (-q_values * agent_mask).sum() / (agent_mask.sum() + 1e-8)
            actor_loss += a_loss
            
        actor_loss /= self.num_agents
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        for i in range(self.num_agents):
            for p in self.critics[i].parameters():
                p.requires_grad_(True)
        
        self._soft_update(self.actor, self.actor_target)
        for i in range(self.num_agents):
            self._soft_update(self.critics[i], self.critics_target[i])
            
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': np.mean(critic_losses),
            'q_value_mag': np.mean(q_value_mags)
        }

    def _soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module):
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
        self.actor.load_state_dict(state['actor'], strict=False)
        if 'actor_target' in state:
            self.actor_target.load_state_dict(state['actor_target'], strict=False)
            
        try:
            self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        except Exception:
            pass
        
        for i in range(self.num_agents):
            if f'critic_{i}' in state:
                self.critics[i].load_state_dict(state[f'critic_{i}'], strict=False)
            if f'critic_target_{i}' in state:
                self.critics_target[i].load_state_dict(state[f'critic_target_{i}'], strict=False)
            if f'critic_optimizer_{i}' in state:
                try:
                    self.critic_optimizers[i].load_state_dict(state[f'critic_optimizer_{i}'])
                except Exception:
                    pass
            
        return state.get('epsilon', 0.0), state.get('episode', 0)
