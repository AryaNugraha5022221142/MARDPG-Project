import argparse
import yaml
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import QuadcopterEnv
from agents import MARDPG, MADDPG, MARTD3, MARDPG_Gaussian

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to agent checkpoint')
    parser.add_argument('--agent', type=str, default='mardpg', choices=['mardpg', 'maddpg', 'iddpg', 'martd3', 'mardpg_g'])
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per level')
    parser.add_argument('--levels', type=int, default=4, help='Number of curriculum levels to evaluate (e.g., 4 means levels 0,1,2,3)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config['training'].get('device') == 'cuda' else "cpu")
    
    # Init Env (dummy first to initialize)
    env = QuadcopterEnv(
        num_agents=config['training']['num_agents'], 
        config=config['environment'],
        render_mode=None
    )
    
    obs_dim = config['environment'].get('obs_dim', 34)
    
    # Initialize Agent
    if args.agent in ['mardpg', 'iddpg']:
        agent = MARDPG(obs_dim=obs_dim, action_dim=4, num_agents=env.num_agents, config=config, device=device, independent_critics=(args.agent == 'iddpg'))
    elif args.agent == 'martd3':
        agent = MARTD3(obs_dim=obs_dim, action_dim=4, num_agents=env.num_agents, config=config, device=device, independent_critics=False)
    elif args.agent == 'mardpg_g':
        agent = MARDPG_Gaussian(obs_dim=obs_dim, action_dim=4, num_agents=env.num_agents, config=config, device=device, independent_critics=False)
    else:
        agent = MADDPG(obs_dim=obs_dim, action_dim=4, num_agents=env.num_agents, config=config, device=device)
        
    print(f"Loading checkpoint {args.checkpoint}...")
    agent.load(args.checkpoint)
    
    success_rates = []
    collision_rates = []
    avg_times = []

    print(f"=== Starting Curriculum Evaluation for {args.agent.upper()} ===")
    
    for level in range(args.levels):
        print(f"\n--- Evaluating Level {level} ---")
        env.set_curriculum_level(level)
        
        successes = 0
        collisions = 0
        times_to_goal = []
        
        for ep in range(args.episodes):
            obs, _ = env.reset()
            if args.agent in ['mardpg', 'iddpg', 'mardpg_g']:
                actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
                critic_hidden = [agent.critics[i].init_hidden(1, device) for i in range(env.num_agents)]
            elif args.agent == 'martd3':
                actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
                critic_hidden = [agent.critics_1[i].init_hidden(1, device) for i in range(env.num_agents)]
            
            done = False
            steps = 0
            
            while not done:
                if args.agent in ['mardpg', 'iddpg', 'martd3', 'mardpg_g']:
                    actions, actor_hidden, critic_hidden = agent.select_actions(obs, actor_hidden, critic_hidden, explore=False)
                else:
                    actions = agent.select_actions(obs, explore=False)
                    
                obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                steps += 1
                
            if info.get('success', False):
                successes += 1
                times_to_goal.append(steps)
            elif info.get('collision', False):
                collisions += 1
                
            out_str = f"L{level} | Ep {ep+1}/{args.episodes} | Steps: {steps} | Success: {info.get('success', False)} | Collision: {info.get('collision', False)}"
            print(f"{out_str:<80}", end='\r')
            
        sr = (successes / args.episodes) * 100
        cr = (collisions / args.episodes) * 100
        at = np.mean(times_to_goal) if times_to_goal else 0
        
        success_rates.append(sr)
        collision_rates.append(cr)
        avg_times.append(at)
        
        print(f"\nLevel {level} Results: Success: {sr:.1f}%, Collision: {cr:.1f}%, Avg Time: {at:.1f} steps")

    env.close()
    
    # Plotting Curriculum Performance Progression
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    x = np.arange(args.levels)
    
    plt.plot(x, success_rates, marker='o', linewidth=2.5, color='#27ae60', label='Success Rate (%)')
    plt.plot(x, collision_rates, marker='s', linewidth=2.5, color='#c0392b', label='Collision Rate (%)')
    
    plt.title(f'Agent Performance Across Curriculum Levels ({args.agent.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel('Curriculum Level (Difficulty)', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(x, [f'Level {i}' for i in x])
    plt.ylim(-5, 105)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = os.path.join(config['logging']['log_dir'], f'{args.agent}_curriculum_eval.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved curriculum performance plot to {output_path}")

if __name__ == '__main__':
    main()
