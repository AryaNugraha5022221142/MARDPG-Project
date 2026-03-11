# tests/test_environment.py
import pytest
import numpy as np
from envs import QuadcopterEnv

def test_environment_reset():
    env = QuadcopterEnv(num_agents=3)
    obs, info = env.reset()
    
    assert obs.shape == (3, 28), f"Expected obs shape (3, 28), got {obs.shape}"
    assert isinstance(info, dict)

def test_environment_step():
    env = QuadcopterEnv(num_agents=3)
    env.reset()
    
    actions = [0, 1, 2] # Forward, Left, Right
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    assert obs.shape == (3, 28)
    assert rewards.shape == (3,)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_environment_run_100_steps():
    env = QuadcopterEnv(num_agents=3)
    env.reset()
    
    for _ in range(100):
        actions = [np.random.randint(6) for _ in range(3)]
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
            
    assert True # If we reached here without error, test passes
