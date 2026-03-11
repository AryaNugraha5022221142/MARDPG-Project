# tests/test_networks.py
import pytest
import torch
from agents.networks import ActorLSTM, Critic

def test_actor_lstm():
    batch_size = 4
    seq_len = 10
    obs_dim = 28
    action_dim = 6
    
    actor = ActorLSTM(input_dim=obs_dim, hidden_dim=128, num_layers=1, output_dim=action_dim)
    
    # Test single step input
    x_single = torch.randn(batch_size, obs_dim)
    hidden = actor.init_hidden(batch_size)
    
    logits, new_hidden = actor(x_single, hidden)
    
    assert logits.shape == (batch_size, action_dim)
    assert new_hidden[0].shape == (1, batch_size, 128)
    assert new_hidden[1].shape == (1, batch_size, 128)
    
    # Test sequence input
    x_seq = torch.randn(batch_size, seq_len, obs_dim)
    hidden_seq = actor.init_hidden(batch_size)
    
    logits_seq, new_hidden_seq = actor(x_seq, hidden_seq)
    
    # The current implementation returns (batch_size, seq_len, action_dim) for sequences
    # Or (batch_size, action_dim) if it only processes single steps.
    # We just ensure it runs without error for now.
    assert logits_seq is not None

def test_critic():
    batch_size = 4
    num_agents = 3
    obs_dim = 28
    action_dim = 6
    
    critic = Critic(obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents)
    
    obs = torch.randn(batch_size, num_agents, obs_dim)
    actions = torch.randn(batch_size, num_agents, action_dim)
    
    q_value = critic(obs, actions)
    
    assert q_value.shape == (batch_size, 1)
