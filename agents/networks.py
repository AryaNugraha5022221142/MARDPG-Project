# agents/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ActorLSTM(nn.Module):
    """
    LSTM-based Actor network shared across all agents.
    Outputs continuous actions in [-action_limit, action_limit].
    Implements a shared base network (Conv2d + State FC) with per-agent LSTM and output heads.
    """
    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 1,
        output_dim: int = 4, # Used as action_dim
        dropout: float = 0.1,
        num_agents: int = 3,
        action_limit: float = 2.5,
    ):
        super(ActorLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_agents = num_agents
        self.action_limit = float(action_limit)
        
        # Shared lower layers
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1)
        state_dim = input_dim - 25
        self.fc_state1 = nn.Linear(state_dim, 32)
        self.fc_state2 = nn.Linear(32, 8)
        self.fusion_dim = 16 * 16 + 8
        
        # Per-agent upper layers
        self.lstms = nn.ModuleList([nn.LSTM(self.fusion_dim, hidden_dim, num_layers, batch_first=True) for _ in range(num_agents)])
        self.fc1s = nn.ModuleList([nn.Linear(hidden_dim, 256) for _ in range(num_agents)])
        self.fc2s = nn.ModuleList([nn.Linear(256, 128) for _ in range(num_agents)])
        self.output_heads = nn.ModuleList([nn.Linear(128, output_dim) for _ in range(num_agents)])

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes hidden and cell states for LSTM."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None, agent_idx: int = 0) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        x: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        """
        is_single_step = x.dim() == 2
        if is_single_step:
            x = x.unsqueeze(1) # (batch, 1, input_dim)
            
        b, s, _ = x.shape
        ranges = x[:, :, :25].contiguous().view(b * s, 1, 5, 5)
        state_vec = x[:, :, 25:].contiguous().view(b * s, -1)
        
        c_out = F.relu(self.conv(ranges))
        c_out = c_out.view(b * s, -1)
        
        st_out = F.relu(self.fc_state1(state_vec))
        st_out = F.relu(self.fc_state2(st_out))
        
        fused = torch.cat([c_out, st_out], dim=1).view(b, s, -1)

        if hidden is None:
            hidden = self.init_hidden(b, x.device)
            
        out, hidden = self.lstms[agent_idx](fused, hidden)
        
        lstm_out = out
        out = F.relu(self.fc1s[agent_idx](out))
        out = F.relu(self.fc2s[agent_idx](out))
        actions = self.action_limit * torch.tanh(self.output_heads[agent_idx](out))
        
        if is_single_step:
            actions = actions.squeeze(1)
            
        return actions, hidden, lstm_out

class CriticLSTM(nn.Module):
    """
    Centralized Critic network (MARDPG style).
    Takes joint observations and joint actions directly.
    """
    def __init__(self, obs_dim: int = 34, action_dim: int = 4, num_agents: int = 3, hidden_dim: int = 128, num_layers: int = 1, independent: bool = False, critic_hidden_dim: int = 128):
        super(CriticLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.num_layers = num_layers
        self.independent = independent
        
        if independent:
            input_dim = obs_dim + action_dim
        else:
            input_dim = (obs_dim * num_agents) + (action_dim * num_agents)
            
        self.fc_embed = nn.Linear(input_dim, critic_hidden_dim)
        self.lstm = nn.LSTM(critic_hidden_dim, critic_hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(critic_hidden_dim, 1)

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.critic_hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.critic_hidden_dim).to(device)
        return (h0, c0)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None, agent_idx: int = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        obs: (batch_size, seq_len, num_agents, obs_dim)
        actions: (batch_size, seq_len, num_agents, action_dim)
        """
        batch_size = obs.size(0)
        seq_len = obs.size(1)
        
        if self.independent:
            o_in = obs[:, :, agent_idx, :]
            a_in = actions[:, :, agent_idx, :]
            x = torch.cat([o_in, a_in], dim=2)
        else:
            o_flat = obs.view(batch_size, seq_len, -1)
            a_flat = actions.view(batch_size, seq_len, -1)
            x = torch.cat([o_flat, a_flat], dim=2)
            
        x = F.relu(self.fc_embed(x))
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
            
        y, new_h = self.lstm(x, hidden)
        return self.fc_out(y), new_h

class Actor(nn.Module):
    """
    Standard MLP-based Actor network shared across all agents.
    Outputs continuous actions in [-1, 1].
    """
    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 256,
        output_dim: int = 4,
        dropout: float = 0.2,
        num_agents: int = 3,
        action_limit: float = 2.5,
    ):
        super(Actor, self).__init__()
        self.num_agents = num_agents
        self.action_limit = float(action_limit)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(num_agents)])

    def forward(self, x: torch.Tensor, agent_idx: int = 0) -> torch.Tensor:
        """
        Forward pass.
        x: (batch_size, input_dim)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        actions = self.action_limit * torch.tanh(self.output_heads[agent_idx](x))
        
        return actions

class Critic(nn.Module):
    """
    Centralized Critic network for a single agent.
    """
    def __init__(self, obs_dim: int = 38, action_dim: int = 4, num_agents: int = 3, dropout: float = 0.2):
        super(Critic, self).__init__()
        
        input_dim = (obs_dim * num_agents) + (action_dim * num_agents)
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        obs: (batch_size, num_agents, obs_dim)
        actions: (batch_size, num_agents, action_dim)
        """
        batch_size = obs.size(0)
        
        # Flatten num_agents and dims
        obs_flat = obs.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)
        
        x = torch.cat([obs_flat, actions_flat], dim=1)
        
        x = self.fc1(x)
        if batch_size > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        
        return q_value
