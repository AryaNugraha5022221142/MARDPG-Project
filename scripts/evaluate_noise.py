import argparse
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from envs.quadcopter_env import QuadcopterEnv
from agents.mardpg import MARDPG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--num-episodes', type=int, default=30)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    obs_dim = config['environment'].get('obs_dim', 34)
    agent = MARDPG(obs_dim=obs_dim, action_dim=4, num_agents=config['training']['num_agents'], config=config, device=device)
    agent.load(args.model_path)
    
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
    success_means = []
    success_stds = []
    
    for noise in noise_levels:
        env_config = config['environment'].copy()
        env_config['sensor_noise_std'] = noise
        env = QuadcopterEnv(num_agents=config['training']['num_agents'], config=env_config)
        
        noise_successes = []
        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
            critic_hidden = [agent.critics[i].init_hidden(1, device) for _ in range(env.num_agents)]
            
            done = False
            while not done:
                actions, actor_hidden, critic_hidden = agent.select_actions(obs, actor_hidden, critic_hidden)
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                obs = next_obs
                done = terminated or truncated

            noise_successes.append(info.get('individual_success_rate', 0.0))
            
        success_means.append(np.mean(noise_successes))
        success_stds.append(np.std(noise_successes) / np.sqrt(args.num_episodes))
        print(f"Noise: {noise:.2f} -> Success Rate: {success_means[-1]:.2f}")

    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(noise_levels, success_means, yerr=success_stds, fmt='-o', color='#8e44ad', linewidth=2, capsize=5, markersize=8)
    plt.title(f'Robustness: Performance vs Sensor Noise Level (N={args.num_episodes})', fontsize=14, fontweight='bold')
    plt.xlabel('Sensor Noise STD (std as % of max range)', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['logging']['log_dir'], 'eval_noise_robustness.png'), dpi=300)
    print(f"Noise robustness plot saved to {config['logging']['log_dir']}")

if __name__ == '__main__':
    main()
