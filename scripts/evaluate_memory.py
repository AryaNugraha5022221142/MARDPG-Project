import argparse
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from envs.quadcopter_env import QuadcopterEnv
from agents.mardpg import MARDPG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--num-episodes', type=int, default=50)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    env = QuadcopterEnv(num_agents=config['training']['num_agents'], config=config['environment'])
    obs_dim = config['environment'].get('obs_dim', 34)
    
    agent = MARDPG(obs_dim=obs_dim, action_dim=4, num_agents=config['training']['num_agents'], config=config, device=device)
    agent.load(args.model_path)
    
    modes = ['Memory ON', 'Memory Reset (No Memory)']
    success_rates = {'Memory ON': [], 'Memory Reset (No Memory)': []}
    
    for mode in modes:
        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
            critic_hidden = [agent.critics[i].init_hidden(1, device) for _ in range(env.num_agents)]
            
            done = False
            while not done:
                if mode == 'Memory Reset (No Memory)':
                    actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
                    critic_hidden = [agent.critics[i].init_hidden(1, device) for _ in range(env.num_agents)]
                    
                actions, actor_hidden, critic_hidden = agent.select_actions(obs, actor_hidden, critic_hidden)
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                obs = next_obs
                done = terminated or truncated

            success_rates[mode].append(info.get('individual_success_rate', 0.0))

    # Calculate means and errors
    means = [np.mean(success_rates[m]) for m in modes]
    stds = [np.std(success_rates[m])/np.sqrt(args.num_episodes) for m in modes] # standard error
    
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(modes, means, yerr=stds, capsize=10, color=['#3498db', '#e74c3c'], alpha=0.8)
    plt.title(f'Performance vs History Length (N={args.num_episodes})', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 1.05)
    
    # Add labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.1%}', ha='center', va='bottom', fontweight='bold')
        
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(config['logging']['log_dir'], 'eval_memory_ablation.png'), dpi=300)
    print(f"Memory ablation plot saved to {config['logging']['log_dir']}")

if __name__ == '__main__':
    main()
