import numpy as np
import torch
from scripts.evaluate import BENCHMARK_SCENES
from envs.benchmark_wrapped_env import BenchmarkWrappedEnv

def main():
    env_config = {
        'action_bound': 0.5235987756,
        'arena_size': [100, 100, 60],
        'num_obstacles': 25,
        'max_velocity': 5.0,
        'rangefinder_max_range': 10.0,
        'collision_distance': 0.8,
        'goal_distance': 2.0,
        'dt': 0.01,
        'dynamic_ratio': 0.0,
        'max_steps': 2000,
        'obs_dim': 38,
        'reward_type': "baseline",
        'rate_limit_per_step': 1.0,
        'randomize_layouts': True,
        'seed': 42
    }
    
    num_agents = 5
    print("Init agent")
    
    env = BenchmarkWrappedEnv('dynamic', level=5, num_agents=num_agents, config=env_config)
    obs, _ = env.reset()
    
    steps = 0
    print("Arena Size:", getattr(env, 'arena_size', None))
    print("Collision Distance:", env.collision_dist)
    for i, a in enumerate(env.agents):
        print(f"  Agent {i}: {a.state[0:3]}")
        
        # Manually compute distances
        pos = a.state[0:3]
        min_dist = float('inf')
        col_source = None
        
        d_walls = min(np.min(pos), np.min(env.arena_size - pos))
        if d_walls < min_dist:
            min_dist = d_walls
            col_source = 'WALL'
            
        for obs_dict in env.obstacles:
            if obs_dict['type'] == 'sphere':
                d = np.linalg.norm(pos - obs_dict['pos']) - obs_dict['radius']
            else:
                d = np.linalg.norm(np.maximum(0, np.abs(pos - obs_dict['pos']) - obs_dict['size']/2))
            if d < min_dist:
                min_dist = d
                col_source = obs_dict
                
        print(f"  Debug dist for Agent {i}: min_dist={min_dist}, source={col_source}")

    while steps < 2:
        action = [np.zeros(3) for _ in range(num_agents)]
        obs, rewards, terminated, truncated, info = env.step(action)
        steps += 1
        
        coll = info.get('agent_collision', np.zeros(num_agents))
        for i, c in enumerate(coll):
            if c:
                print(f"Step {steps}: Agent {i} COLLIDED! pos={env.agents[i].state[0:3]}")
                d_min = env._get_min_distance(i)
                print(f"  -> d_min: {d_min}")

main()
