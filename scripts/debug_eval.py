import numpy as np
import torch
import sys
from evaluate import main, evaluate_model
from agents.mardpg_baseline import MARDPG_Baseline
from envs.benchmark_wrapped_env import BenchmarkWrappedEnv
import yaml

if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    env = BenchmarkWrappedEnv('urban', 'hard', 3, config)
    obs, info = env.reset()
    for i in range(3):
        print(f"Agent {i} pos:", env.agents[i].state[:3])
        print(f"d_min:", env._get_min_distance(i))
        print(f"collision_dist:", env.collision_dist)
