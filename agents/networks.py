# agents/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ActorLSTM(nn.Module):
    """
    LSTM-based Actor network shared across all agents.
    Outputs continuous actions in [-1, 1].
    """
    def __init__(self, input_dim: int = 33, hidden_dim: int = 128, num_layers: int = 1, output_dim: int = 4, dropout: float = 0.1):
        super(ActorLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes hidden and cell states for LSTM."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        x: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        """
        is_single_step = x.dim() == 2
        if is_single_step:
            x = x.unsqueeze(1) # (batch, 1, input_dim)
            
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
            
        out, hidden = self.lstm(x, hidden)
        
        # Process all time steps for sequence training
        x = F.relu(self.fc1(out))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.fc3(x))
        
        if is_single_step:
            actions = actions.squeeze(1)
            
        return actions, hidden

class CriticLSTM(nn.Module):
    """
    Centralized LSTM-based Critic network.
    """
    def __init__(self, obs_dim: int = 33, action_dim: int = 4, num_agents: int = 3, hidden_dim: int = 128, num_layers: int = 1, independent: bool = False):
        super(CriticLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.independent = independent
        
        # Centralized input: all observations + all actions
        # Independent input: only one agent's observation + action
        if independent:
            input_dim = obs_dim + action_dim
        else:
            input_dim = (obs_dim * num_agents) + (action_dim * num_agents)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None, agent_idx: int = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        obs: (batch_size, seq_len, num_agents, obs_dim) or (batch_size, seq_len, obs_dim)
        actions: (batch_size, seq_len, num_agents, action_dim) or (batch_size, seq_len, action_dim)
        """
        batch_size = obs.size(0)
        seq_len = obs.size(1)
        
        if self.independent:
            # If independent, we expect obs and actions to be already sliced or we slice them here
            if obs.dim() == 4:
                obs = obs[:, :, agent_idx, :]
                actions = actions[:, :, agent_idx, :]
            x = torch.cat([obs, actions], dim=2)
        else:
            # Flatten num_agents and dims
            obs_flat = obs.view(batch_size, seq_len, -1)
            actions_flat = actions.view(batch_size, seq_len, -1)
            x = torch.cat([obs_flat, actions_flat], dim=2)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
            
        out, hidden = self.lstm(x, hidden)
        
        x = F.relu(self.fc1(out))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values, hidden

class Actor(nn.Module):
    """
    Standard MLP-based Actor network shared across all agents.
    Outputs continuous actions in [-1, 1].
    """
    def __init__(self, input_dim: int = 33, hidden_dim: int = 256, output_dim: int = 4, dropout: float = 0.2):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch_size, input_dim)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        actions = torch.tanh(self.fc3(x))
        
        return actions

class Critic(nn.Module):
    """
    Centralized Critic network for a single agent.
    """
    def __init__(self, obs_dim: int = 33, action_dim: int = 4, num_agents: int = 3, dropout: float = 0.2):
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
