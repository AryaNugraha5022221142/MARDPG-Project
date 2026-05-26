import torch
import yaml
from agents.mardpg_baseline import MARDPG_Baseline
from envs.base_env import EnvironmentConfig

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cpu')
agent = MARDPG_Baseline(obs_dim=30, action_dim=2, num_agents=3, config=config, device=device)
agent.load('checkpoints/mardpg_baseline_final.pt')

import numpy as np
dummy_obs = np.random.randn(3, 30).astype(np.float32)

actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(3)]
critic_hidden = [agent.critics[i].init_hidden(1, device) for i in range(3)]

actions, _, _ = agent.select_actions(dummy_obs, actor_hidden, critic_hidden, explore=False)
print("Dummy Actions:")
print(actions)
