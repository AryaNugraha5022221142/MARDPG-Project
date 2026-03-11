# scripts/evaluate.py
import argparse
import yaml
import os
import sys
import numpy as np
import torch

# Add the project root to the Python path so it can find 'envs' and 'agents'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import QuadcopterEnv
from agents import MARDPG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--render', action='store_true', help='Enable 3D visualization')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config['training'].get('device') == 'cuda' else "cpu")
    
    env = QuadcopterEnv(
        num_agents=config['training']['num_agents'], 
        config=config['environment'],
        render_mode='human' if args.render else None
    )
    
    agent = MARDPG(
        obs_dim=28, 
        action_dim=6, 
        num_agents=config['training']['num_agents'], 
        config=config, 
        device=device
    )
    
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    successes = 0
    collisions = 0
    times_to_goal = []
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
        
        done = False
        steps = 0
        
        while not done:
            if args.render:
                env.render()
            actions, hidden = agent.select_actions(obs, hidden, epsilon=0.0) # Greedy
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            steps += 1
            
        if args.render:
            env.render()
            
        if info.get('success', False):
            successes += 1
            times_to_goal.append(steps)
        elif info.get('collision', False):
            collisions += 1
            
    success_rate = successes / args.episodes * 100
    collision_rate = collisions / args.episodes * 100
    trapped_rate = 100 - success_rate - collision_rate
    avg_time = np.mean(times_to_goal) if times_to_goal else 0
    
    print("=== Evaluation Results ===")
    print(f"Episodes: {args.episodes}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    print(f"Trapped Rate: {trapped_rate:.1f}%")
    print(f"Avg Time to Goal: {avg_time:.1f} steps")
    
    env.close()

if __name__ == '__main__':
    main()
