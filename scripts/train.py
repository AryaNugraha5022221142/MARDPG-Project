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
from envs.benchmark_wrapped_env import BenchmarkWrappedEnv
from envs.base_env import DifficultyLevel
from agents import MARDPG_Baseline

def _make_agent(agent_type, obs_dim, action_dim, num_agents, config, device):
    if agent_type == 'mardpg_baseline':
        return MARDPG_Baseline(obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents,
                               config=config, device=device, independent_critics=False)
    else:
        raise ValueError(f"Unknown agent type {agent_type}")

def save_learning_curve(data, title, ylabel, filename, color, ylim=None, log_dir=None):
    """Generic learning curve plotter."""
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    series = pd.Series(data).dropna()
    if series.empty:
        plt.close()
        return
    
    window_size = min(100, max(1, len(data) // 5))
    smooth = series.rolling(window=window_size, min_periods=1).mean()
    std = series.rolling(window=window_size, min_periods=1).std().fillna(0)
    
    plt.plot(smooth.index, smooth.values, label=f'Mean', color=color, linewidth=2)
    plt.fill_between(smooth.index, (smooth - std).values, (smooth + std).values,
                     color=color, alpha=0.2, label='Std Dev')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    filepath = os.path.join(log_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved: {filepath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--run-name', type=str, default='mardpg_baseline_run', help='Name of the run')
    parser.add_argument('--agent', type=str, default='mardpg_baseline', choices=['mardpg_baseline'], help='Agent type to train')
    parser.add_argument('--scenario', type=str, default=None, help='Scenario name (e.g., urban_canyon, search_and_rescue)')
    parser.add_argument('--num-episodes', type=int, default=None, help='Number of episodes to train')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save final metrics as JSON')
    parser.add_argument('--render', action='store_true', help='Enable 3D visualization during training')
    parser.add_argument('--sensor-noise', type=float, default=0.02, help='Sensor noise standard deviation')
    parser.add_argument('--reward-type', type=str, default='baseline', choices=['baseline', 'linear', 'exponential'], help='Reward function type')
    parser.add_argument('--resume', type=str, nargs='?', const='latest', default=None, help='Resume from checkpoint. Provide path to .pt file or "latest"')
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

    def create_env(num_agents, scenario, config, render_mode, seed, level=None):
        """Single factory function for environment creation."""
        env_config = config['environment'].copy()
        env_config['seed'] = seed
        env_config['sensor_noise_std'] = config.get('sensor_noise_std', 0.02)
        env_config['reward_type'] = config.get('reward_type', 'baseline')
        if 'rewards' in config:
            env_config['rewards'] = config['rewards']
            
        if scenario:
            from envs.scenarios import get_scenario_config
            scenario_config = get_scenario_config(scenario)
            env_config.update(scenario_config)
            
        env_config['agent_id_dim'] = num_agents
        
        env_level = level if level is not None else config['environment'].get('level', 3)
        
        if scenario is None or scenario in ["urban", "forest", "terrain", "structured", "dynamic"]:
            scenario_name = scenario if scenario else "urban"
            env = BenchmarkWrappedEnv(
                benchmark_name=scenario_name,
                level=env_level,
                num_agents=num_agents,
                config=env_config
            )
        else:
            env = QuadcopterKinematicEnv(
                num_agents=num_agents,
                config=env_config,
                render_mode='human' if render_mode else None,
                scenario=scenario
            )
        
        if render_mode and hasattr(env, 'render_mode'):
            env.render_mode = 'human'
        
        return env

    current_num_agents = config['training']['num_agents']
    
    curriculum_to_level = {3: 1, 5: 2, 7: 3, 10: 4}
    initial_level = curriculum_to_level.get(current_num_agents, 1)

    # Environment
    env = create_env(
        num_agents=current_num_agents,
        scenario=args.scenario,
        config=config,
        render_mode=args.render,
        seed=seed,
        level=initial_level
    )

    # Agent
    obs_dim = env.obs_dim
    action_dim = getattr(env, 'action_dim', 2)

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
    recent_trapped = deque(maxlen=100)
    recent_path_eff = deque(maxlen=100)
    
    agent = _make_agent(args.agent, obs_dim, action_dim, current_num_agents, config, device)
    
    def set_lr_scale(agent, scale):
        actor_lr = config['learning'].get('actor_lr', 1e-4) * scale
        critic_lr = config['learning'].get('critic_lr', 1e-3) * scale
        if hasattr(agent, 'actor_optimizer'):
            for pg in agent.actor_optimizer.param_groups:
                pg['lr'] = actor_lr
        if hasattr(agent, 'critic_optimizers'):
            for opt in agent.critic_optimizers:
                for pg in opt.param_groups:
                    pg['lr'] = critic_lr

    warmup_until = config['training'].get('warmup_episodes', 100)
    
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
    
    start_episode = 1
    if args.resume:
        checkpoint_path = args.resume
        if checkpoint_path == 'latest':
            import glob
            # First check if the final save exists (which happens on keyboard interrupt)
            final_path = os.path.join(config['logging']['checkpoint_dir'], f"{args.agent}_final.pt")
            checkpoints = glob.glob(os.path.join(config['logging']['checkpoint_dir'], f"{args.agent}_ep*.pt"))
            
            latest_path = None
            latest_ep = -1
            if checkpoints:
                def extract_ep(f):
                    basename = os.path.basename(f)
                    try:
                        return int(basename.split('_ep')[-1].split('.pt')[0])
                    except (ValueError, IndexError):
                        return -1
                checkpoints.sort(key=extract_ep)
                latest_path = checkpoints[-1]
                latest_ep = extract_ep(latest_path)
                
            if os.path.exists(final_path):
                # The final path might not encode the episode in its name, but if it exists 
                # after a crash/interrupt, it's usually the latest. 
                # We can load it and determine the episode. But for simplicity, let's prefer final_path 
                # if its modification time is newer.
                if latest_path:
                    if os.path.getmtime(final_path) > os.path.getmtime(latest_path):
                        checkpoint_path = final_path
                    else:
                        checkpoint_path = latest_path
                else:
                    checkpoint_path = final_path
            elif latest_path:
                checkpoint_path = latest_path
            else:
                print("No checkpoints found. Starting from scratch.")
                checkpoint_path = None
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            _, ep = agent.load(checkpoint_path)
            start_episode = ep + 1
            print(f"Resuming from episode {start_episode}")
        elif checkpoint_path:
            print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

    def apply_curriculum(agent, env, config, episode, device):
        """Check if curriculum transition needed and reinitialize."""
        schedule = config.get('curriculum', {}).get('schedule', {})
        
        # Find current target agent count
        target_n = None
        for ep_threshold, n_agents in sorted([(int(k), v) for k, v in schedule.items()]):
            if episode >= ep_threshold:
                target_n = n_agents
        
        if target_n is None or target_n == agent.num_agents:
            return agent, env  # No change needed
        
        print(f"[Curriculum] Scaling from {agent.num_agents} to {target_n} agents at episode {episode}")
        
        curriculum_to_level = {3: 1, 5: 2, 7: 3, 10: 4}
        env_level = curriculum_to_level.get(target_n, 1)

        # Create new env
        new_env = create_env(
            num_agents=target_n,
            scenario=args.scenario,
            config=config,
            render_mode=args.render,
            seed=seed,
            level=env_level
        )
        
        # Create new agent
        new_agent = _make_agent(
            args.agent,
            new_env.obs_dim,
            getattr(new_env, 'action_dim', 2),
            target_n,
            config,
            device
        )
        
        # Transfer weights for compatible parts
        if config.get('curriculum', {}).get('transfer_weights', True):
            # Transfer shared base network
            new_agent.actor.shared_base.load_state_dict(agent.actor.shared_base.state_dict())
            new_agent.actor_target.shared_base.load_state_dict(agent.actor_target.shared_base.state_dict())
            
            # Transfer existing agent heads (up to min of old/new count)
            n_transfer = min(agent.num_agents, new_agent.num_agents)
            for i in range(n_transfer):
                new_agent.actor.heads[i].load_state_dict(agent.actor.heads[i].state_dict())
                new_agent.actor_target.heads[i].load_state_dict(agent.actor_target.heads[i].state_dict())
            
            # Transfer critic base (if attention-based, attention weights transfer)
            n_old = len(agent.critics)
            n_new = len(new_agent.critics)
            for i in range(n_new):
                src_idx = i % n_old
                # BUG FIX: Safely load critic encoder weights (skip mismatched linear layers)
                old_enc_state = agent.critics[src_idx].encoder.state_dict()
                new_enc_state = new_agent.critics[i].encoder.state_dict()
                for name, param in old_enc_state.items():
                    if param.shape == new_enc_state[name].shape:
                        new_enc_state[name].copy_(param)
                new_agent.critics[i].encoder.load_state_dict(new_enc_state)
                
                new_agent.critics[i].attention.load_state_dict(agent.critics[src_idx].attention.state_dict())
                new_agent.critics[i].ffn.load_state_dict(agent.critics[src_idx].ffn.state_dict())
                new_agent.critics[i].lstm.load_state_dict(agent.critics[src_idx].lstm.state_dict())
                
                # Fix: transfer the missing normalization and output layers
                new_agent.critics[i].norm1.load_state_dict(agent.critics[src_idx].norm1.state_dict())
                new_agent.critics[i].norm2.load_state_dict(agent.critics[src_idx].norm2.state_dict())
                new_agent.critics[i].fc_out.load_state_dict(agent.critics[src_idx].fc_out.state_dict())
                
                old_tgt_enc_state = agent.critics_target[src_idx].encoder.state_dict()
                new_tgt_enc_state = new_agent.critics_target[i].encoder.state_dict()
                for name, param in old_tgt_enc_state.items():
                    if param.shape == new_tgt_enc_state[name].shape:
                        new_tgt_enc_state[name].copy_(param)
                new_agent.critics_target[i].encoder.load_state_dict(new_tgt_enc_state)
                
                new_agent.critics_target[i].attention.load_state_dict(agent.critics_target[src_idx].attention.state_dict())
                new_agent.critics_target[i].ffn.load_state_dict(agent.critics_target[src_idx].ffn.state_dict())
                new_agent.critics_target[i].lstm.load_state_dict(agent.critics_target[src_idx].lstm.state_dict())
                
                new_agent.critics_target[i].norm1.load_state_dict(agent.critics_target[src_idx].norm1.state_dict())
                new_agent.critics_target[i].norm2.load_state_dict(agent.critics_target[src_idx].norm2.state_dict())
                new_agent.critics_target[i].fc_out.load_state_dict(agent.critics_target[src_idx].fc_out.state_dict())
            
        # Post-curriculum LR warmup (50 episodes)
        new_agent._curriculum_warmup_remaining = 50

        # Clear replay buffer to avoid dimension mismatch
        if config.get('curriculum', {}).get('reset_buffer', True):
            new_agent.memory.clear()
        
        return new_agent, new_env

    print(f"Starting training for {num_episodes} episodes...")

    # FIX 2d: Paper trains on five benchmark scene types randomly selected each episode.
    # Define the rotation list here so it can be extended easily.
    PAPER_SCENES = ["urban", "forest", "terrain", "structured", "dynamic"]

    pbar = tqdm(range(start_episode, num_episodes + 1), desc="Training", initial=start_episode - 1, total=num_episodes)
    try:
        for episode in pbar:
            if hasattr(env, 'set_episode'):
                env.set_episode(episode, num_episodes)
                
            if config.get('curriculum', {}).get('enabled', False):
                agent, env = apply_curriculum(agent, env, config, episode, device)
                current_num_agents = agent.num_agents

            if hasattr(agent, '_curriculum_warmup_remaining') and agent._curriculum_warmup_remaining > 0:
                warmup_len = 50
                scale = 0.3 + 0.7 * (1.0 - agent._curriculum_warmup_remaining / warmup_len)
                set_lr_scale(agent, scale)
                agent._curriculum_warmup_remaining -= 1
            elif episode <= warmup_until:
                # Linear warmup from 0.3 to 1.0
                scale = 0.3 + 0.7 * (episode / warmup_until)
                set_lr_scale(agent, scale)
            else:
                set_lr_scale(agent, 1.0)

            # FIX 2e: rotate scene type each episode to match paper's multi-environment
            # training (Section VI-A).  Only applied when no explicit --scenario is given.
            if args.scenario is None and hasattr(env, 'set_scene_type'):
                scene = PAPER_SCENES[episode % len(PAPER_SCENES)]
                env.set_scene_type(scene)

            obs, _ = env.reset()
            if obs.shape[1] != obs_dim:
                raise ValueError(f"Observation dim mismatch: env returned {obs.shape[1]}, agent expects {obs_dim}")
            
            actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
            critic_hidden = [agent.critics[i].init_hidden(1, device) for i in range(env.num_agents)]
            
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
            
            straight_line_dists = [np.linalg.norm(env.agents[i].state[0:3] - env.goals[i]) for i in range(current_num_agents)]
            trajectory_lengths = [0.0 for _ in range(current_num_agents)]
            prev_positions = [env.agents[i].state[0:3].copy() for i in range(current_num_agents)]
            
            while not done:
                if args.render:
                    env.render()
                    
                actions, actor_hidden, critic_hidden = agent.select_actions(obs, actor_hidden, critic_hidden)
                
                # Compute mean hidden state norm (h part of lstm hidden state)
                # actor_hidden is list of (h, c) for each agent. h is (batch_size, hidden_dim)
                h_norms = [torch.norm(h_state[0]).item() for h_state in actor_hidden]
                episode_h_norms.append(np.mean(h_norms))
                    
                episode_act_stds.append(np.std(actions))
                    
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                
                for i in range(current_num_agents):
                    curr_pos = env.agents[i].state[0:3]
                    trajectory_lengths[i] += np.linalg.norm(curr_pos - prev_positions[i])
                    prev_positions[i] = curr_pos.copy()
                
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
                
                agent.memory.push(obs, np.array(actions), rewards, next_obs, dones, done)
                
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
            
            agent_col_rate = info.get('agent_collision', np.zeros(current_num_agents, dtype=bool)).sum() / current_num_agents
            recent_collisions.append(agent_col_rate)
            recent_sat_rates.append(np.mean(episode_sat_rates))
            recent_act_stds.append(np.mean(episode_act_stds))
            
            agent_success = info.get('agent_success', np.zeros(current_num_agents, dtype=bool))
            agent_collision = info.get('agent_collision', np.zeros(current_num_agents, dtype=bool))
            agent_trapped = ~(agent_success | agent_collision)

            path_eff = np.sum(straight_line_dists) / max(np.sum(trajectory_lengths), 1e-5)
            trapped_rate = np.sum(agent_trapped) / current_num_agents
            
            recent_trapped.append(trapped_rate)
            recent_path_eff.append(path_eff)
            
            reward_history.append(episode_reward)
            success_history.append(info.get('individual_success_rate', 0.0))
            length_history.append(episode_length)
            collision_history.append(agent_col_rate)
            
            hidden_state_norm_history.append(np.mean(episode_h_norms) if hasattr(agent, 'actor') else np.nan)
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
            current_trapped = np.mean(recent_trapped) if len(recent_trapped) > 0 else 0.0
            current_path_eff = np.mean(recent_path_eff) if len(recent_path_eff) > 0 else 0.0
            
            if episode % config['logging']['log_interval'] == 0:
                avg_reward = current_avg_reward
                success_rate = current_success_rate
                avg_length = current_avg_length
                collision_rate = current_collision_rate
                avg_sat = current_sat_rate
                avg_act_std = current_act_std
                
                # FIX 5e: read current_sigma from AnnealedGaussianNoise for all agent types
                if hasattr(agent, 'noise'):
                    noise_obj = agent.noise[0] if isinstance(agent.noise, list) else agent.noise
                    if hasattr(noise_obj, 'current_sigma'):
                        current_sigma = noise_obj.current_sigma
                    elif hasattr(noise_obj, 'sigma'):
                        current_sigma = noise_obj.sigma
                    else:
                        current_sigma = 0.0
                else:
                    current_sigma = 0.0
                
                pbar.set_postfix({'Reward': f'{avg_reward:.2f}', 'Success': f'{success_rate:.2f}', 'PathEff': f'{current_path_eff:.2f}', 'Trapped': f'{current_trapped:.2f}', 'EpLen': f'{avg_length:.1f}'})
                
                if use_wandb:
                    log_data = {
                        'episode': episode,
                        'global_steps': global_step_count,
                        'avg_reward': avg_reward,
                        'success_rate': success_rate,
                        'avg_ep_length': avg_length,
                        'collision_rate': collision_rate,
                        'trapped_rate': current_trapped,
                        'path_efficiency': current_path_eff,
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
                # Print log to inform that the checkpoint was successfully saved
                print(f"Checkpoint saved: {save_path}")
                
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
    
    plots = [
        (reward_history, 'Reward', 'Reward', f'learning_curve_reward_{args.agent}.png', '#c0392b', [-700, 700]),
        (success_history, 'Success Rate', 'Success Rate', f'learning_curve_success_{args.agent}.png', '#2ecc71', [0, 1.1]),
        (length_history, 'Average Episode Length', 'Steps', f'learning_curve_length_{args.agent}.png', '#3498db', None),
        (collision_history, 'Average Collision Rate', 'Collision Rate', f'learning_curve_collision_{args.agent}.png', '#9b59b6', [0, 1.1]),
        (critic_loss_history, 'Critic Loss (TD Error)', 'MSE Loss', f'learning_curve_critic_loss_{args.agent}.png', '#e74c3c', None),
        (actor_loss_history, 'Actor Loss', 'Loss', f'learning_curve_actor_loss_{args.agent}.png', '#2ecc71', None),
        (q_value_mag_history, 'Mean Q-Value Magnitude', '|Q(s,a)|', f'learning_curve_q_mag_{args.agent}.png', '#f1c40f', None),
        (hidden_state_norm_history, 'Hidden State Norm ||h_t||_2', 'L2 Norm', f'learning_curve_h_norm_{args.agent}.png', '#8e44ad', None),
        (action_smoothness_history, 'Action Smoothness |a_t - a_{t-1}|', 'Mean Absolute Difference', f'learning_curve_action_smoothness_{args.agent}.png', '#d35400', None),
        (tracking_error_history, 'Velocity Tracking Error |v_{ref} - v|', 'Mean Absolute Error', f'learning_curve_tracking_error_{args.agent}.png', '#16a085', None),
        (min_agent_dist_history, 'Minimum Inter-Agent Distance', 'Distance (m)', f'learning_curve_min_agent_dist_{args.agent}.png', '#2980b9', None)
    ]
    
    for data, title, ylabel, fname, color, ylim in plots:
        save_learning_curve(data, f'{title} during Training ({args.agent.upper()})', 
                            ylabel, fname, color, ylim, config['logging']['log_dir'])

    print(f"Training complete! Plots saved to {config['logging']['log_dir']}")

if __name__ == '__main__':
    main()
