# scripts/train.py
import argparse
import yaml
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

# Add the project root to the Python path so it can find 'envs' and 'agents'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import QuadcopterEnv
from agents import MARDPG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--run-name', type=str, default='mardpg_run', help='Name of the run')
    parser.add_argument('--render', action='store_true', help='Enable 3D visualization during training')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds
    seed = config['training'].get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and config['training'].get('device') == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # Environment
    env = QuadcopterEnv(
        num_agents=config['training']['num_agents'], 
        config=config['environment'],
        render_mode='human' if args.render else None
    )

    # Agent
    agent = MARDPG(
        obs_dim=28, 
        action_dim=6, 
        num_agents=config['training']['num_agents'], 
        config=config, 
        device=device
    )

    # Logging
    use_wandb = config['logging'].get('use_wandb', False)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=config['logging'].get('wandb_project', 'mardpg-thesis'), name=args.run_name, config=config)
        except ImportError:
            print("wandb not installed, skipping wandb logging.")
            use_wandb = False
        except Exception as e:
            print(f"wandb initialization failed: {e}. Skipping wandb logging.")
            use_wandb = False

    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # Training Loop
    num_episodes = config['training']['num_episodes']
    epsilon = config['learning']['epsilon_start']
    epsilon_end = config['learning']['epsilon_end']
    epsilon_decay = config['learning']['epsilon_decay']
    
    recent_rewards = deque(maxlen=100)
    recent_success = deque(maxlen=100)
    
    # Academic Data Tracking
    reward_history = []
    success_history = []
    
    print(f"Starting training for {num_episodes} episodes...")

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
        
        episode_reward = 0
        done = False
        
        while not done:
            if args.render:
                env.render()
                
            actions, hidden = agent.select_actions(obs, hidden, epsilon)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            done = terminated or truncated
            dones = np.array([done] * env.num_agents, dtype=np.float32)
            
            agent.memory.push(obs, np.array(actions), rewards, next_obs, dones)
            
            loss_dict = agent.update()
            
            obs = next_obs
            episode_reward += np.sum(rewards)
            
        recent_rewards.append(episode_reward)
        recent_success.append(1.0 if info.get('success', False) else 0.0)
        
        reward_history.append(episode_reward)
        success_history.append(1.0 if info.get('success', False) else 0.0)
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % config['logging']['log_interval'] == 0:
            avg_reward = np.mean(recent_rewards)
            success_rate = np.mean(recent_success)
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.2f} | Epsilon: {epsilon:.3f}")
            
            if use_wandb:
                wandb.log({
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'success_rate': success_rate,
                    'epsilon': epsilon
                })
                
        if episode % config['logging']['save_interval'] == 0:
            save_path = os.path.join(config['logging']['checkpoint_dir'], f"mardpg_ep{episode}.pt")
            agent.save(save_path, epsilon, episode)
            
    if args.render:
        env.close()
            
    # Final save
    final_path = os.path.join(config['logging']['checkpoint_dir'], "mardpg_final.pt")
    agent.save(final_path, epsilon, num_episodes)
    
    # --- ACADEMIC PLOTTING (Fig 8 Style) ---
    print("Generating Academic Learning Curves...")
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Smooth the reward curve
    window_size = 100
    rewards_series = pd.Series(reward_history)
    smooth_rewards = rewards_series.rolling(window=window_size).mean()
    std_rewards = rewards_series.rolling(window=window_size).std()
    
    plt.plot(smooth_rewards, label='MARDPG (Mean Reward)', color='#c0392b', linewidth=2)
    plt.fill_between(range(len(smooth_rewards)), 
                     smooth_rewards - std_rewards, 
                     smooth_rewards + std_rewards, 
                     color='#c0392b', alpha=0.2, label='Std Dev')
    
    plt.title('Reward during all the training episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Number of history trajectories (Episodes)', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(config['logging']['log_dir'], 'learning_curve_reward.png'), dpi=300)
    
    # Success Rate Plot
    plt.figure(figsize=(10, 6))
    success_series = pd.Series(success_history)
    smooth_success = success_series.rolling(window=window_size).mean()
    plt.plot(smooth_success, color='#2ecc71', linewidth=2)
    plt.title('Average Success Rate during Training', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(config['logging']['log_dir'], 'learning_curve_success.png'), dpi=300)
    
    print(f"Training complete! Plots saved to {config['logging']['log_dir']}")

if __name__ == '__main__':
    main()
