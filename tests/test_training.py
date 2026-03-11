# tests/test_training.py
import pytest
import numpy as np
import torch
from agents import MARDPG
from envs import QuadcopterEnv

def test_buffer():
    from agents.replay_buffer import ReplayBuffer
    
    buffer = ReplayBuffer(capacity=100)
    
    obs = np.random.randn(3, 28)
    actions = np.array([0, 1, 2])
    rewards = np.array([0.1, 0.2, 0.3])
    next_obs = np.random.randn(3, 28)
    dones = np.array([False, False, False])
    
    buffer.push(obs, actions, rewards, next_obs, dones)
    
    assert len(buffer) == 1
    
    # Push more to sample
    for _ in range(10):
        buffer.push(obs, actions, rewards, next_obs, dones)
        
    b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(batch_size=5)
    
    assert b_obs.shape == (5, 3, 28)
    assert b_actions.shape == (5, 3)
    assert b_rewards.shape == (5, 3)
    assert b_next_obs.shape == (5, 3, 28)
    assert b_dones.shape == (5, 3)

def test_action_selection():
    agent = MARDPG(obs_dim=28, action_dim=6, num_agents=3, device='cpu')
    
    obs = np.random.randn(3, 28).astype(np.float32)
    hidden = [agent.actor.init_hidden(1, 'cpu') for _ in range(3)]
    
    actions, new_hidden = agent.select_actions(obs, hidden, epsilon=0.0)
    
    assert len(actions) == 3
    assert all(isinstance(a, int) for a in actions)
    assert len(new_hidden) == 3

def test_network_update():
    agent = MARDPG(obs_dim=28, action_dim=6, num_agents=3, device='cpu')
    
    # Fill buffer
    for _ in range(agent.batch_size + 10):
        obs = np.random.randn(3, 28).astype(np.float32)
        actions = np.random.randint(0, 6, size=(3,))
        rewards = np.random.randn(3).astype(np.float32)
        next_obs = np.random.randn(3, 28).astype(np.float32)
        dones = np.zeros(3, dtype=np.float32)
        
        agent.memory.push(obs, actions, rewards, next_obs, dones)
        
    loss_dict = agent.update()
    
    assert 'actor_loss' in loss_dict
    assert 'critic_loss' in loss_dict
    assert isinstance(loss_dict['actor_loss'], float)
    assert isinstance(loss_dict['critic_loss'], float)
