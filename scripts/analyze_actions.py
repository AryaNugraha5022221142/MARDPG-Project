import argparse
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from envs.quadcopter_kinematic_env import QuadcopterKinematicEnv
from agents import MARDPG_Baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--num-episodes', type=int, default=10)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    env = QuadcopterKinematicEnv(num_agents=config['training']['num_agents'], config=config['environment'])
    obs_dim = 30
    
    agent = MARDPG_Baseline(obs_dim=obs_dim, action_dim=2, num_agents=config['training']['num_agents'], config=config, device=device)
    agent.load(args.model_path)
    
    raw_actions = []
    tracking_errors = []
    
    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
        critic_hidden = [agent.critics[i].init_hidden(1, device) for _ in range(env.num_agents)]
        
        done = False
        while not done:
            # We don't want exploration noise for evaluation
            actions, actor_hidden, critic_hidden = agent.select_actions(obs, actor_hidden, critic_hidden, explore=False)
            raw_actions.append(actions)
            
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            tracking_errors.append(info.get('tracking_error', 0.0))
            
            obs = next_obs
            done = terminated or truncated

    raw_actions = np.array(raw_actions).reshape(-1, 2) # (N, 2)
    
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Plot Action Distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    action_names = ['rho (thrust/pitch proxy)', 'tau (yaw proxy)']
    for i in range(2):
        ax = axes[i]
        sns.histplot(raw_actions[:, i], bins=50, kde=True, ax=ax, color='skyblue')
        ax.set_title(f'Distribution of {action_names[i]}')
        ax.set_xlim(-1.0, 1.0) # network output range
        
    plt.tight_layout()
    plt.savefig(os.path.join(config['logging']['log_dir'], 'eval_action_distribution.png'), dpi=300)
    
    # Tracking Error histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(tracking_errors, bins=50, kde=True, color='lightgreen')
    plt.title('Velocity Tracking Error Distribution over Episodes')
    plt.xlabel('Mean Absolute Error (|v_{ref} - v_actual|)')
    plt.savefig(os.path.join(config['logging']['log_dir'], 'eval_tracking_error_hist.png'), dpi=300)
    
    print(f"Action analysis plots saved to {config['logging']['log_dir']}")

if __name__ == '__main__':
    main()
