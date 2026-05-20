import yaml
import numpy as np
import torch
from scripts.evaluate import BENCHMARK_SCENES
from agents.mardpg.agent import MARDPG
from envs.benchmark_wrapped_env import BenchmarkWrappedEnv

def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    env_config = config['environment'].copy()
    env_config['seed'] = 42
    
    num_agents = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dummy_obs_dim = 38
    print("Init agent")
    # Actually we can just forward random actions if we don't have the checkpoint
    
    # Use benchmark wrapped env
    env = BenchmarkWrappedEnv('dynamic', level=5, num_agents=num_agents, config=env_config)
    
    obs, _ = env.reset()
    
    steps = 0
    print("Arena Size:", getattr(env, 'arena_size', None))
    print("Agent init pos:")
    for i, a in enumerate(env.agents):
        print(f"  Agent {i}: {a.state[0:3]}")
    print("Goals:", env.goals)
    
    while steps < 10:
        action = [np.zeros(3) for _ in range(num_agents)]
        obs, rewards, terminated, truncated, info = env.step(action)
        steps += 1
        
        coll = info.get('agent_collision', np.zeros(num_agents))
        for i, c in enumerate(coll):
            if c: # just print on first few steps
                print(f"Step {steps}: Agent {i} COLLIDED! pos={env.agents[i].state[0:3]}")
        
        if terminated or truncated:
            break
main()
