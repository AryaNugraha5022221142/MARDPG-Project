import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import List, Tuple, Dict, Any

from .replay_buffer import SequenceReplayBuffer

class ActorLSTMBaseline(nn.Module):
    """
    LSTM-based Actor applying Baseline specifications:
    - CNN for 25 rangefinders
    - FC for state vector
    - Fusion into LSTM
    - Tanh output to produce 2 continuous steering signals [rho, tau]
    - Actions are scaled to [-action_bound, action_bound] for the kinematic env
    """
    def __init__(self, obs_dim_total: int, hidden_dim: int = 128, device: str = 'cpu', action_bound: float = 2.5):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.action_bound = action_bound
        
        # Rangefinders (25 values) -> CNN
        # Input shape: (batch*seq, 1, 25)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # State vector
        self.state_dim = obs_dim_total - 25
        self.fc_state1 = nn.Linear(self.state_dim, 32)
        self.fc_state2 = nn.Linear(32, 8)
        
        # Fusion: 25 * 16 = 400 from CNN + 8 from State = 408
        self.fusion_dim = 400 + 8
        self.lstm = nn.LSTM(self.fusion_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # Output layer with tanh
        # We output 2 steering signals: rho (yaw) and tau (pitch)
        self.output_head = nn.Linear(hidden_dim, 2)

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        is_single_step = x.dim() == 2
        if is_single_step:
            x = x.unsqueeze(1) # (batch, 1, input_dim)
            
        b, s, d = x.shape
        ranges = x[:, :, :25].contiguous().view(b * s, 1, 25)
        state_vec = x[:, :, 25:].contiguous().view(b * s, -1)
        
        # CNN Branch
        c_out = F.relu(self.conv1(ranges))
        c_out = F.relu(self.conv2(c_out))
        c_out = c_out.view(b * s, -1) # 400
        
        # State Branch
        st_out = F.relu(self.fc_state1(state_vec))
        st_out = F.relu(self.fc_state2(st_out)) # 8
        
        # Fusion
        fused = torch.cat([c_out, st_out], dim=1).view(b, s, -1)
        
        if hidden is None:
            hidden = self.init_hidden(b, x.device)
            
        out, hidden = self.lstm(fused, hidden)
        
        # Tanh activation to produce normalized action signals [-1, 1]
        logits = self.output_head(out)
        norm_out = torch.tanh(logits) # (batch, seq, 2)
        
        actions = self.action_bound * norm_out
        rho = actions[:, :, 0]
        tau = actions[:, :, 1]
        
        if is_single_step:
            actions = actions.squeeze(1)
            rho = rho.squeeze(1)
            tau = tau.squeeze(1)
            
        return actions, hidden, out, (rho, tau)


class MultiActorLSTMBaseline(nn.Module):
    """
    Wraps multiple independent ActorLSTMBaseline modules to maintain compatibility
    with external scripts that expect a single 'agent.actor' object.
    """
    def __init__(self, obs_dim_total: int, num_agents: int = 3, hidden_dim: int = 128, device: str = 'cpu', action_bound: float = 2.5):
        super().__init__()
        self.num_agents = num_agents
        self.models = nn.ModuleList([ActorLSTMBaseline(obs_dim_total, hidden_dim, device, action_bound) for _ in range(num_agents)])
        
    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        return self.models[0].init_hidden(batch_size, device)
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None, agent_idx: int = 0):
        return self.models[agent_idx](x, hidden)

class CriticLSTMBaseline(nn.Module):
    """
    Centralized Critic network.
    Takes joint observations and joint actions directly.
    """
    def __init__(self, obs_dim: int, action_dim: int = 2, num_agents: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.num_layers = 1
        self.hidden_dim = hidden_dim
        
        input_dim = (obs_dim * num_agents) + (action_dim * num_agents)
            
        self.fc_embed = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None, agent_idx: int = None):
        batch_size = obs.size(0)
        seq_len = obs.size(1)
        
        o_flat = obs.view(batch_size, seq_len, -1)
        a_flat = actions.view(batch_size, seq_len, -1)
        x = torch.cat([o_flat, a_flat], dim=2)
            
        x = F.relu(self.fc_embed(x))
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
            
        y, new_h = self.lstm(x, hidden)
        return self.fc_out(y), new_h


class MARDPG_Baseline:
    """
    MARDPG Baseline Implementation (as specified)
    """
    def __init__(self, obs_dim: int = 41, action_dim: int = 2, num_agents: int = 3, 
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
        
        self.actor = MultiActorLSTMBaseline(obs_dim, num_agents, hidden_dim, device, self.action_bound).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        
        critic_hidden_dim = config['network'].get('critic', {}).get('hidden_dim', hidden_dim)
        
        self.critics = [CriticLSTMBaseline(obs_dim, action_dim, num_agents, critic_hidden_dim).to(self.device) for _ in range(num_agents)]
        self.critics_target = [copy.deepcopy(c).to(self.device) for c in self.critics]
        
        actor_lr = config['learning'].get('actor_lr', 1e-4)
        critic_lr = config['learning'].get('critic_lr', 1e-3)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]
        
        buffer_size = config['memory'].get('buffer_size', 1000)
        self.memory = SequenceReplayBuffer(buffer_size)

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
                    action_np += np.random.normal(0, 0.1, size=self.action_dim)
                    
                action_np = np.clip(action_np, -self.action_bound, self.action_bound)
                actions.append(action_np.astype(np.float32))
            
            actions_np = np.array(actions)
            actions_tensor = torch.FloatTensor(actions_np).unsqueeze(0).unsqueeze(0).to(self.device)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            
            for i in range(self.num_agents):
                h, c = critic_hidden[i]
                _, (new_h, new_c) = self.critics[i](obs_tensor, actions_tensor, (h, c), agent_idx=i)
                new_critic_hidden.append((new_h, new_c))
                
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
        
        actor_hidden = [self.actor.init_hidden(self.batch_size, self.device) for _ in range(self.num_agents)]
        critic_hidden = [self.critics[i].init_hidden(self.batch_size, self.device) for i in range(self.num_agents)]
        
        critic_losses = []
        q_value_mags = []
        
        with torch.no_grad():
            next_actions_all = []
            for i in range(self.num_agents):
                agent_next_obs = next_obs[:, :, i, :]
                next_act, _, _, _ = self.actor_target(agent_next_obs, actor_hidden[i], agent_idx=i)
                next_actions_all.append(next_act)
            next_actions_all = torch.stack(next_actions_all, dim=2)
            
        for i in range(self.num_agents):
            with torch.no_grad():
                target_q_seq, _ = self.critics_target[i](next_obs, next_actions_all, critic_hidden[i], agent_idx=i)
                target_q_seq = target_q_seq.squeeze(-1)
                target_q = rewards[:, :, i] + self.gamma * target_q_seq * (1 - dones[:, :, i])
                
            current_q_seq, _ = self.critics[i](obs, actions, critic_hidden[i], agent_idx=i)
            current_q = current_q_seq.squeeze(-1)
            
            loss = F.mse_loss(current_q, target_q, reduction='none')
            critic_loss = (loss * masks).sum() / (masks.sum() + 1e-8)
            critic_losses.append(critic_loss.item())
            
            q_mag = (current_q.abs() * masks).sum() / (masks.sum() + 1e-8)
            q_value_mags.append(q_mag.item())
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
            self.critic_optimizers[i].step()
            
        actor_loss = 0
        for i in range(self.num_agents):
            agent_i_obs = obs[:, :, i, :]
            agent_i_act, _, _, _ = self.actor(agent_i_obs, actor_hidden[i], agent_idx=i)
            
            other_acts = []
            for j in range(self.num_agents):
                if j == i:
                    other_acts.append(agent_i_act)
                else:
                    agent_j_obs = obs[:, :, j, :]
                    with torch.no_grad():
                        a_j, _, _, _ = self.actor(agent_j_obs, actor_hidden[j], agent_idx=j)
                    other_acts.append(a_j.detach())
            
            joint_actions = torch.stack(other_acts, dim=2)
            
            q_values, _ = self.critics[i](obs, joint_actions, critic_hidden[i], agent_idx=i)
            q_values = q_values.squeeze(-1)
            
            a_loss = (-q_values * masks).sum() / (masks.sum() + 1e-8)
            actor_loss += a_loss
            
        actor_loss /= self.num_agents
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
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
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        
        for i in range(self.num_agents):
            self.critics[i].load_state_dict(state[f'critic_{i}'])
            self.critics_target[i].load_state_dict(state[f'critic_target_{i}'])
            self.critic_optimizers[i].load_state_dict(state[f'critic_optimizer_{i}'])
            
        return state.get('epsilon', 0.0), state.get('episode', 0)
