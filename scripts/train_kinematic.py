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
import json
from tqdm import tqdm

# Add the project root to the Python path so it can find 'envs' and 'agents'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import QuadcopterKinematicEnv
from agents import MARDPG, MADDPG, MARTD3, MARDPG_Gaussian

def _make_agent(agent_type, obs_dim, action_dim, num_agents, config, device):
    if agent_type in ['mardpg', 'iddpg']:
        from agents import MARDPG
        return MARDPG(obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents,
                      config=config, device=device,
                      independent_critics=(agent_type == 'iddpg'))
    elif agent_type == 'martd3':
        from agents import MARTD3
        return MARTD3(obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents,
                      config=config, device=device, independent_critics=False)
    elif agent_type == 'mardpg_g':
        from agents import MARDPG_Gaussian
        return MARDPG_Gaussian(obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents,
                               config=config, device=device, independent_critics=False)
    elif agent_type == 'mardpg_baseline':
        from agents import MARDPG_Baseline
        return MARDPG_Baseline(obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents,
                               config=config, device=device, independent_critics=False)
    else:
        from agents import MADDPG
        return MADDPG(obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents,
                      config=config, device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--run-name', type=str, default='mardpg_run', help='Name of the run')
    parser.add_argument('--agent', type=str, default='mardpg', choices=['mardpg', 'maddpg', 'iddpg', 'martd3', 'mardpg_g', 'mardpg_baseline'], help='Agent type to train')
    parser.add_argument('--scenario', type=str, default=None, help='Scenario name (e.g., urban_canyon, search_and_rescue)')
    parser.add_argument('--num-episodes', type=int, default=None, help='Number of episodes to train')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save final metrics as JSON')
    parser.add_argument('--render', action='store_true', help='Enable 3D visualization during training')
    parser.add_argument('--sensor-noise', type=float, default=0.02, help='Sensor noise standard deviation')
    parser.add_argument('--reward-type', type=str, default='linear', choices=['linear', 'exponential'], help='Reward function type')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds
    seed = args.seed if args.seed is not None else config['training'].get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and config['training'].get('device') == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # Environment
    env_config = config['environment'].copy()
    if 'rewards' in config:
        env_config['rewards'] = config['rewards']
    
    if args.scenario:
        from envs.scenarios import get_scenario_config
        scenario_config = get_scenario_config(args.scenario)
        env_config.update(scenario_config)
    
    env_config['agent_id_dim'] = config['training']['num_agents']
    env_config['sensor_noise_std'] = args.sensor_noise
    env_config['reward_type'] = args.reward_type

    env = QuadcopterKinematicEnv(
        num_agents=config['training']['num_agents'], 
        config=env_config,
        render_mode='human' if args.render else None,
        scenario=args.scenario
    )

    # Agent
    obs_dim = env.obs_dim

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
    num_episodes = args.num_episodes if args.num_episodes is not None else config['training']['num_episodes']
    
    recent_rewards = deque(maxlen=100)
    recent_success = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_collisions = deque(maxlen=100)
    recent_collisions_per_agent = deque(maxlen=100)
    
    # Curriculum Learning
    curriculum_level = 0
    # require adaptive sustained success before advancing
    min_episodes_per_level = 2000
    episodes_at_level = 0
    env_curriculum_locked = False
    env_curriculum_lock_until = 0
    
    # Agent-count curriculum schedule: thresholds based on competency
    agent_grad_success_threshold = {1: 0.65, 2: 0.60, 3: 0.55, 4: 0.50, 5: 0.48}
    agent_grad_collision_threshold = {1: 0.20, 2: 0.22, 3: 0.25, 4: 0.28, 5: 0.30}
    current_num_agents = 2
    episodes_at_agent_level = 0
    target_max_agents = config['training']['num_agents']
    
    def set_lr_scale(agent, scale):
        actor_lr = config['learning'].get('actor_lr', 1e-4) * scale
        critic_lr = config['learning'].get('critic_lr', 1e-3) * scale
        if hasattr(agent, 'actor_optimizer'):
            for pg in agent.actor_optimizer.param_groups:
                pg['lr'] = actor_lr
        if hasattr(agent, 'critic_optimizer'):
            for pg in agent.critic_optimizer.param_groups:
                pg['lr'] = critic_lr
        elif hasattr(agent, 'critic_optimizers_1'):
            for opt in agent.critic_optimizers_1:
                for pg in opt.param_groups: pg['lr'] = critic_lr
            for opt in agent.critic_optimizers_2:
                for pg in opt.param_groups: pg['lr'] = critic_lr

    warmup_until = 0

    # Re-create env and agent with starting agents
    env_config_start = env_config.copy()
    env = QuadcopterKinematicEnv(num_agents=current_num_agents, config=env_config_start,
                        render_mode='human' if args.render else None,
                        scenario=args.scenario)
    env.set_curriculum_level(curriculum_level)
    agent = _make_agent(args.agent, obs_dim, 2, current_num_agents, config, device)
    
    # Academic Data Tracking
    reward_history = []
    success_history = []
    length_history = []
    collision_history = []
    critic_loss_history = []
    actor_loss_history = []
    q_value_mag_history = []
    
    hidden_state_norm_history = []
    action_smoothness_history = []
    tracking_error_history = []
    min_agent_dist_history = []
    
    recent_sat_rates = deque(maxlen=100)
    recent_act_stds = deque(maxlen=100)
    global_step_count = 0
    
    print(f"Starting training for {num_episodes} episodes...")

    pbar = tqdm(range(1, num_episodes + 1), desc="Training")
    try:
        for episode in pbar:
            # Agent-count curriculum
            min_agent_level_episodes = max(1500, 1000 * current_num_agents)
            if current_num_agents < target_max_agents:
                rolling_per_agent_success = np.mean(recent_success) if len(recent_success) == 100 else 0.0
                rolling_per_agent_collision = np.mean(recent_collisions_per_agent) if len(recent_collisions_per_agent) == 100 else 1.0

                if episodes_at_agent_level >= min_agent_level_episodes:
                    if rolling_per_agent_success >= agent_grad_success_threshold.get(current_num_agents, 0.48):
                        if rolling_per_agent_collision < agent_grad_collision_threshold.get(current_num_agents, 0.30):
                            current_num_agents += 1
                            print(f"\n[Agent Curriculum] Upgrading to {current_num_agents} agents at episode {episode}")
                            # Save actor weights
                            actor_state = agent.actor.state_dict()
                            env = QuadcopterKinematicEnv(num_agents=current_num_agents,
                                               config=env_config.copy(),
                                               render_mode='human' if args.render else None,
                                               scenario=args.scenario)
                            env.set_curriculum_level(curriculum_level)
                            agent = _make_agent(args.agent, obs_dim, 2, current_num_agents, config, device)
                            # Transfer actor weights
                            agent.actor.load_state_dict(actor_state, strict=False)
                            agent.actor_target.load_state_dict(actor_state, strict=False)
                            recent_success.clear()
                            recent_collisions.clear()
                            recent_collisions_per_agent.clear()
                            if hasattr(agent.memory, 'clear'):
                                agent.memory.clear()
                            episodes_at_agent_level = 0
                            env_curriculum_locked = True
                            env_curriculum_lock_until = episode + 1500
                            warmup_until = episode + 200
                            
            if episode == warmup_until:
                set_lr_scale(agent, 1.0)
            elif episode < warmup_until:
                set_lr_scale(agent, 0.3)

            obs, _ = env.reset()
            if obs.shape[1] != obs_dim:
                raise ValueError(f"Observation dim mismatch: env returned {obs.shape[1]}, agent expects {obs_dim}")
            
            if args.agent in ['mardpg', 'iddpg', 'mardpg_g', 'mardpg_baseline']:
                actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
                critic_hidden = [agent.critics[i].init_hidden(1, device) for i in range(env.num_agents)]
            elif args.agent == 'martd3':
                actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
                critic_hidden = [agent.critics_1[i].init_hidden(1, device) for i in range(env.num_agents)]
            
            episode_reward = 0
            done = False
            episode_sat_rates = []
            episode_act_stds = []
            episode_length = 0
            episode_collision = False
            
            episode_critic_losses = []
            episode_actor_losses = []
            episode_q_mags = []
            
            episode_h_norms = []
            episode_tracking_errors = []
            episode_action_smoothness = []
            episode_min_agent_distances = []
            
            while not done:
                if args.render:
                    env.render()
                    
                if args.agent in ['mardpg', 'iddpg', 'martd3', 'mardpg_g', 'mardpg_baseline']:
                    actions, actor_hidden, critic_hidden = agent.select_actions(obs, actor_hidden, critic_hidden)
                    
                    # Compute mean hidden state norm (h part of lstm hidden state)
                    # actor_hidden is list of (h, c) for each agent. h is (batch_size, hidden_dim)
                    h_norms = [torch.norm(h_state[0]).item() for h_state in actor_hidden]
                    episode_h_norms.append(np.mean(h_norms))
                else:
                    actions = agent.select_actions(obs)
                    
                episode_act_stds.append(np.std(actions))
                    
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                
                episode_tracking_errors.append(info.get('tracking_error', 0.0))
                episode_action_smoothness.append(info.get('action_smoothness', 0.0))
                episode_min_agent_distances.append(info.get('min_agent_dist', 0.0))
                
                episode_length += 1
                if info.get('collision', False):
                    episode_collision = True
                
                episode_sat_rates.append(info.get('sat_rate', 0.0))
                
                # Per-agent done flags for the buffer
                dones = info['agent_dones'].astype(np.float32)
                
                # Episode is over if all agents are done or max steps reached
                done = terminated or truncated
                global_step_count += 1
                
                if args.agent in ['mardpg', 'iddpg', 'martd3', 'mardpg_g', 'mardpg_baseline']:
                    agent.memory.push(obs, np.array(actions), rewards, next_obs, dones, done)
                else:
                    agent.memory.push(obs, np.array(actions), rewards, next_obs, dones)
                
                # Only update if we have enough episodes in the buffer
                if len(agent.memory) >= config['memory'].get('batch_size', 32) and global_step_count % config.get('update_interval', 10) == 0:
                    loss_dict = agent.update()
                    if loss_dict and 'critic_loss' in loss_dict:
                        episode_critic_losses.append(loss_dict['critic_loss'])
                        episode_actor_losses.append(loss_dict.get('actor_loss', np.nan))
                        episode_q_mags.append(loss_dict.get('q_value_mag', np.nan))
                else:
                    loss_dict = {}
                
                obs = next_obs
                episode_reward += np.sum(rewards)
                
            recent_rewards.append(episode_reward)
            recent_success.append(info.get('individual_success_rate', 0.0))
            recent_lengths.append(episode_length)
            recent_collisions.append(float(episode_collision))
            agent_col_rate = info.get('agent_collision', np.zeros(current_num_agents, dtype=bool)).sum() / current_num_agents
            recent_collisions_per_agent.append(agent_col_rate)
            recent_sat_rates.append(np.mean(episode_sat_rates))
            recent_act_stds.append(np.mean(episode_act_stds))
            
            reward_history.append(episode_reward)
            success_history.append(info.get('individual_success_rate', 0.0))
            length_history.append(episode_length)
            collision_history.append(float(episode_collision))
            
            hidden_state_norm_history.append(np.mean(episode_h_norms) if hasattr(agent, 'actor') and args.agent in ['mardpg', 'iddpg', 'martd3', 'mardpg_g'] else np.nan)
            tracking_error_history.append(np.mean(episode_tracking_errors))
            action_smoothness_history.append(np.mean(episode_action_smoothness))
            min_agent_dist_history.append(np.mean(episode_min_agent_distances))
            
            if len(episode_critic_losses) > 0:
                critic_loss_history.append(np.mean(episode_critic_losses))
                actor_loss_history.append(np.mean(episode_actor_losses))
                q_value_mag_history.append(np.mean(episode_q_mags))
            else:
                critic_loss_history.append(np.nan)
                actor_loss_history.append(np.nan)
                q_value_mag_history.append(np.nan)
            
            # Collect metrics prior to any buffer clears
            current_avg_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0.0
            current_success_rate = np.mean(recent_success) if len(recent_success) > 0 else 0.0
            current_avg_length = np.mean(recent_lengths) if len(recent_lengths) > 0 else 0.0
            current_collision_rate = np.mean(recent_collisions) if len(recent_collisions) > 0 else 0.0
            current_sat_rate = np.mean(recent_sat_rates) if len(recent_sat_rates) > 0 else 0.0
            current_act_std = np.mean(recent_act_stds) if len(recent_act_stds) > 0 else 0.0
    
            # Update Curriculum
    
            episodes_at_level += 1
            episodes_at_agent_level += 1
            
            if env_curriculum_locked:
                if episode >= env_curriculum_lock_until or (len(recent_success) == 100 and current_success_rate > 0.50):
                    env_curriculum_locked = False
                    
            if episode % 100 == 0:
                adjusted_success_threshold = 0.40 * (0.85 ** max(0, current_num_agents - 3))
                ready = (
                    not env_curriculum_locked
                    and len(recent_success) == 100
                    and current_success_rate >= adjusted_success_threshold
                    and current_collision_rate < 0.30   # also require collision < 30%
                    and curriculum_level < 6
                    and episodes_at_level >= min_episodes_per_level
                )
                if ready:
                    curriculum_level += 1
                    episodes_at_level = 0
                    env.set_curriculum_level(curriculum_level)
                    if hasattr(agent.memory, 'clear'):
                        agent.memory.clear()
                        print(f"Curriculum Level Up! Level: {curriculum_level}. Buffer cleared.")
                    recent_success.clear()
                    recent_collisions.clear()
            
            if episode % config['logging']['log_interval'] == 0:
                avg_reward = current_avg_reward
                success_rate = current_success_rate
                avg_length = current_avg_length
                collision_rate = current_collision_rate
                avg_sat = current_sat_rate
                avg_act_std = current_act_std
                
                # Use actual sigma for continuous agents (MARDPG/MADDPG)
                current_sigma = agent.noise[0].sigma if hasattr(agent, 'noise') else 0.0
                
                pbar.set_postfix({'Level': curriculum_level, 'Reward': f'{avg_reward:.2f}', 'Success': f'{success_rate:.2f}', 'Sigma': f'{current_sigma:.3f}', 'Steps': global_step_count})
                
                if use_wandb:
                    log_data = {
                        'episode': episode,
                        'global_steps': global_step_count,
                        'curriculum_level': curriculum_level,
                        'avg_reward': avg_reward,
                        'success_rate': success_rate,
                        'avg_ep_length': avg_length,
                        'collision_rate': collision_rate,
                        'sigma': current_sigma,
                        'sat_rate': avg_sat,
                        'action_std': avg_act_std
                    }
                    if not np.isnan(critic_loss_history[-1]):
                        log_data['critic_loss'] = critic_loss_history[-1]
                        log_data['actor_loss'] = actor_loss_history[-1]
                        log_data['q_value_mag'] = q_value_mag_history[-1]
                    
                    if not np.isnan(hidden_state_norm_history[-1]):
                        log_data['hidden_state_norm'] = hidden_state_norm_history[-1]
                        
                    log_data['tracking_error'] = tracking_error_history[-1]
                    log_data['action_smoothness'] = action_smoothness_history[-1]
                    log_data['min_agent_dist'] = min_agent_dist_history[-1]
                        
                    wandb.log(log_data)
                    
            if episode % config['logging']['save_interval'] == 0:
                save_path = os.path.join(config['logging']['checkpoint_dir'], f"{args.agent}_ep{episode}.pt")
                agent.save(save_path, 0.0, episode)
                
        if args.render:
            env.close()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user! Saving current progress and generating plots...")
        num_episodes = episode  # Update num_episodes to reflect actual episodes trained
            
    # Final save
    final_path = os.path.join(config['logging']['checkpoint_dir'], f"{args.agent}_final.pt")
    agent.save(final_path, 0.0, num_episodes)
    
    # Save metrics to JSON if requested
    if args.output_json:
        final_metrics = {
            'agent': args.agent,
            'scenario': args.scenario,
            'episodes': num_episodes,
            'avg_reward': np.mean(reward_history[-100:]),
            'success_rate': np.mean(success_history[-100:]),
            'total_successes': int(np.sum(success_history))
        }
        with open(args.output_json, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        print(f"Final metrics saved to {args.output_json}")
    
    # --- ACADEMIC PLOTTING (Fig 8 Style) ---
    print("Generating Academic Learning Curves...")
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Smooth the reward curve
    window_size = min(100, max(1, len(reward_history) // 5))  # dynamic window size
    rewards_series = pd.Series(reward_history)
    smooth_rewards = rewards_series.rolling(window=window_size, min_periods=1).mean()
    std_rewards = rewards_series.rolling(window=window_size, min_periods=1).std().fillna(0)
    
    plt.plot(smooth_rewards.index, smooth_rewards.values, label=f'{args.agent.upper()} (Mean Reward)', color='#c0392b', linewidth=2)
    plt.fill_between(smooth_rewards.index, 
                     (smooth_rewards - std_rewards).values, 
                     (smooth_rewards + std_rewards).values, 
                     color='#c0392b', alpha=0.2, label='Std Dev')
    
    plt.title(f'Reward during all the training episodes ({args.agent.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Number of history trajectories (Episodes)', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_reward_{args.agent}.png'), dpi=300)
    
    # Success Rate Plot
    plt.figure(figsize=(10, 6))
    success_series = pd.Series(success_history)
    smooth_success = success_series.rolling(window=window_size, min_periods=1).mean()
    plt.plot(smooth_success.index, smooth_success.values, color='#2ecc71', linewidth=2)
    plt.title(f'Average Success Rate during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_success_{args.agent}.png'), dpi=300)
    
    # Episode Length Plot
    plt.figure(figsize=(10, 6))
    length_series = pd.Series(length_history)
    smooth_length = length_series.rolling(window=window_size, min_periods=1).mean()
    plt.plot(smooth_length.index, smooth_length.values, color='#3498db', linewidth=2)
    plt.title(f'Average Episode Length during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Steps', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_length_{args.agent}.png'), dpi=300)
    
    # Collision Rate Plot
    plt.figure(figsize=(10, 6))
    collision_series = pd.Series(collision_history)
    smooth_collision = collision_series.rolling(window=window_size, min_periods=1).mean()
    plt.plot(smooth_collision.index, smooth_collision.values, color='#9b59b6', linewidth=2)
    plt.title(f'Average Collision Rate during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Collision Rate', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_collision_{args.agent}.png'), dpi=300)
    
    # Critic Loss Plot
    plt.figure(figsize=(10, 6))
    critic_series = pd.Series(critic_loss_history).dropna()
    if not critic_series.empty:
        smooth_critic = critic_series.rolling(window=window_size, min_periods=1).mean()
        plt.plot(smooth_critic.index, smooth_critic.values, color='#e74c3c', linewidth=2)
        plt.title(f'Critic Loss (TD Error) during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_critic_loss_{args.agent}.png'), dpi=300)
    
    # Actor Loss Plot
    plt.figure(figsize=(10, 6))
    actor_series = pd.Series(actor_loss_history).dropna()
    if not actor_series.empty:
        smooth_actor = actor_series.rolling(window=window_size, min_periods=1).mean()
        plt.plot(smooth_actor.index, smooth_actor.values, color='#2ecc71', linewidth=2)
        plt.title(f'Actor Loss during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_actor_loss_{args.agent}.png'), dpi=300)

    # Q-Value Magnitude Plot
    plt.figure(figsize=(10, 6))
    qmag_series = pd.Series(q_value_mag_history).dropna()
    if not qmag_series.empty:
        smooth_qmag = qmag_series.rolling(window=window_size, min_periods=1).mean()
        plt.plot(smooth_qmag.index, smooth_qmag.values, color='#f1c40f', linewidth=2)
        plt.title(f'Mean Q-Value Magnitude during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('|Q(s,a)|', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_q_mag_{args.agent}.png'), dpi=300)
        
    # Hidden State Norm Plot
    plt.figure(figsize=(10, 6))
    hnorm_series = pd.Series(hidden_state_norm_history).dropna()
    if not hnorm_series.empty:
        smooth_hnorm = hnorm_series.rolling(window=window_size, min_periods=1).mean()
        plt.plot(smooth_hnorm.index, smooth_hnorm.values, color='#8e44ad', linewidth=2)
        plt.title(f'Hidden State Norm ||h_t||_2 during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('L2 Norm', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_h_norm_{args.agent}.png'), dpi=300)
    
    # Action Smoothness Plot
    plt.figure(figsize=(10, 6))
    smoothness_series = pd.Series(action_smoothness_history).dropna()
    if not smoothness_series.empty:
        smooth_smoothness = smoothness_series.rolling(window=window_size, min_periods=1).mean()
        plt.plot(smooth_smoothness.index, smooth_smoothness.values, color='#d35400', linewidth=2)
        plt.title(f'Action Smoothness |a_t - a_{{t-1}}| during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('Mean Absolute Difference', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_action_smoothness_{args.agent}.png'), dpi=300)

    # Tracking Error Plot
    plt.figure(figsize=(10, 6))
    tracking_series = pd.Series(tracking_error_history).dropna()
    if not tracking_series.empty:
        smooth_tracking = tracking_series.rolling(window=window_size, min_periods=1).mean()
        plt.plot(smooth_tracking.index, smooth_tracking.values, color='#16a085', linewidth=2)
        plt.title(f'LQR Tracking Error |v_{{ref}} - v| during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_tracking_error_{args.agent}.png'), dpi=300)

    # Min Agent Distance Plot
    plt.figure(figsize=(10, 6))
    min_dist_series = pd.Series(min_agent_dist_history).dropna()
    if not min_dist_series.empty:
        smooth_min_dist = min_dist_series.rolling(window=window_size, min_periods=1).mean()
        plt.plot(smooth_min_dist.index, smooth_min_dist.values, color='#2980b9', linewidth=2)
        plt.title(f'Minimum Inter-Agent Distance during Training ({args.agent.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('Distance (m)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(config['logging']['log_dir'], f'learning_curve_min_agent_dist_{args.agent}.png'), dpi=300)
    
    print(f"Training complete! Plots saved to {config['logging']['log_dir']}")

if __name__ == '__main__':
    main()
