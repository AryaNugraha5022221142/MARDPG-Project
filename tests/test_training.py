# tests/test_training.py
import pytest
import numpy as np
import torch
from agents import MARDPG_Baseline
from envs import QuadcopterEnv

def test_buffer():
    from agents.replay_buffer import SequenceReplayBuffer
    
    buffer = SequenceReplayBuffer(capacity=100)
    
    obs_seq = np.random.randn(5, 3, 30)
    act_seq = np.random.randn(5, 3, 2)
    rew_seq = np.random.randn(5, 3)
    next_obs_seq = np.random.randn(5, 3, 30)
    done_seq = np.zeros((5, 3))
    
    # Simple push to check object existence
    assert buffer.capacity == 100

def test_action_selection():
    agent = MARDPG_Baseline(obs_dim=30, action_dim=2, num_agents=3, device='cpu')
    
    obs = np.random.randn(3, 30).astype(np.float32)
    actor_hidden = [agent.actor.init_hidden(1, 'cpu') for _ in range(3)]
    critic_hidden = [agent.critics[0].init_hidden(1, 'cpu') for _ in range(3)]
    
    actions, new_actor_hidden, new_critic_hidden = agent.select_actions(obs, actor_hidden, critic_hidden, explore=False)
    
    assert len(actions) == 3
    assert actions.shape == (3, 2)
    assert len(new_actor_hidden) == 3
    assert len(new_critic_hidden) == 3

def test_network_update():
    config = {
        'network': {'actor': {'hidden_dim': 32}, 'critic': {'hidden_dim': 32}},
        'learning': {'actor_lr': 1e-4, 'critic_lr': 1e-3, 'max_grad_norm': 1.0},
        'memory': {'buffer_size': 1000, 'batch_size': 2, 'seq_len': 4},
        'targets': {'update_rate': 0.01}
    }
    agent = MARDPG_Baseline(obs_dim=30, action_dim=2, num_agents=3, config=config, device='cpu')
    
    # Fill buffer with enough sequences
    for _ in range(4):
        for step in range(5):
            obs = np.random.randn(3, 30).astype(np.float32)
            actions = np.random.randn(3, 2).astype(np.float32)
            rewards = np.random.randn(3).astype(np.float32)
            next_obs = np.random.randn(3, 30).astype(np.float32)
            dones = np.zeros(3, dtype=np.float32)
            episode_done = (step == 4)
            agent.memory.push(obs, actions, rewards, next_obs, dones, episode_done)
        
    loss_dict = agent.update()
    
    assert 'actor_loss' in loss_dict
    assert 'critic_loss' in loss_dict
    assert isinstance(loss_dict['actor_loss'], float)
    assert isinstance(loss_dict['critic_loss'], float)

@pytest.mark.parametrize("num_agents", [1, 3])
def test_recurrent_agents_support_variable_agent_counts(num_agents):
    obs_dim = 30
    action_dim = 2
    agent = MARDPG_Baseline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        device="cpu",
    )

    assert len(agent.actor.heads) == num_agents

    obs = np.random.randn(num_agents, obs_dim).astype(np.float32)
    actor_hidden = [agent.actor.init_hidden(1, "cpu") for _ in range(num_agents)]
    critics = agent.critics
    critic_hidden = [critics[i].init_hidden(1, "cpu") for i in range(num_agents)]

    actions, new_actor_hidden, new_critic_hidden = agent.select_actions(
        obs,
        actor_hidden,
        critic_hidden,
        explore=False,
    )

    assert actions.shape == (num_agents, action_dim)
    assert len(new_actor_hidden) == num_agents
    assert len(new_critic_hidden) == num_agents
