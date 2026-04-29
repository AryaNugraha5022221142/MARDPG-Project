# scripts/evaluate.py
import argparse
import yaml
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the project root to the Python path so it can find 'envs' and 'agents'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import QuadcopterEnv
from agents import MARDPG, MADDPG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--agent', type=str, default='mardpg', choices=['mardpg', 'maddpg', 'iddpg'], help='Agent type to evaluate')
    parser.add_argument('--scenario', type=str, default=None, help='Scenario name')
    parser.add_argument('--level', type=int, default=None, help='Curriculum level to evaluate on (0-4)')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--render', action='store_true', help='Enable 3D visualization')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config['training'].get('device') == 'cuda' else "cpu")
    
    env = QuadcopterEnv(
        num_agents=config['training']['num_agents'], 
        config=config['environment'],
        render_mode='human' if args.render else None,
        scenario=args.scenario
    )
    
    if args.level is not None and args.scenario is None:
        env.set_curriculum_level(args.level)
        print(f"Evaluating on Curriculum Level {args.level}")
        
    obs_dim = config['environment'].get('obs_dim', 34)
    
    if args.agent in ['mardpg', 'iddpg']:
        agent = MARDPG(
            obs_dim=obs_dim, 
            action_dim=4, 
            num_agents=config['training']['num_agents'], 
            config=config, 
            device=device,
            independent_critics=(args.agent == 'iddpg')
        )
    else:
        agent = MADDPG(
            obs_dim=obs_dim, 
            action_dim=4, 
            num_agents=config['training']['num_agents'], 
            config=config, 
            device=device
        )
    
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    successes = 0
    collisions = 0
    times_to_goal = []
    
    # Advanced EE Metrics
    all_jerks = []
    all_safety_frontiers = []
    agent_success_counts = np.zeros(env.num_agents)
    path_lengths = []
    ideal_lengths = []
    
    # Trajectory storage for plotting (first 5 episodes)
    all_trajectories = []
    
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            if args.agent == 'mardpg':
                actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
                critic_hidden = [agent.critics_1[i].init_hidden(1, device) for i in range(env.num_agents)]
            
            done = False
            steps = 0
            ep_trajectories = [[] for _ in range(env.num_agents)]
            
            while not done:
                if args.render:
                    env.render()
                
                for i in range(env.num_agents):
                    ep_trajectories[i].append(env.agents[i].state[:3].copy())
                    
                if args.agent in ['mardpg', 'iddpg']:
                    actions, actor_hidden, critic_hidden = agent.select_actions(obs, actor_hidden, critic_hidden, explore=False)
                else:
                    actions = agent.select_actions(obs, explore=False)
                    
                obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                steps += 1
                
            if ep < 5: # Save first 5 episodes for plotting
                all_trajectories.append({
                    'paths': ep_trajectories,
                    'goals': env.goals.copy(),
                    'obstacles': env.obstacles.copy()
                })
                
            if args.render:
                env.render()
                
            if info.get('success', False):
                successes += 1
                agent_success_counts += 1 # Assuming all agents reached goal for success
                times_to_goal.append(steps)
            elif info.get('collision', False):
                collisions += 1
                
            # Collect EE Metrics
            all_jerks.append(np.mean(env.total_jerk))
            all_safety_frontiers.append(np.mean(env.safety_frontier))
            
            # Path Efficiency
            for i in range(env.num_agents):
                path_lengths.append(np.sum(np.linalg.norm(np.diff(np.array(ep_trajectories[i]), axis=0), axis=1)))
                # Ideal distance from start to goal
                ideal_lengths.append(np.linalg.norm(env.goals[i] - ep_trajectories[i][0]))
                
            print(f"Episode {ep+1}/{args.episodes} - Steps: {steps}, Success: {info.get('success', False)}, Collision: {info.get('collision', False)}")
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user! Generating plots from collected data...")
        args.episodes = max(1, len(all_jerks)) # Avoid div by zero
        
    success_rate = successes / args.episodes * 100
    collision_rate = collisions / args.episodes * 100
    trapped_rate = 100 - success_rate - collision_rate
    avg_time = np.mean(times_to_goal) if times_to_goal else 0
    
    # Calculate Fairness Index (Jain's Fairness Index)
    agent_success_rates = agent_success_counts / args.episodes
    fairness_index = (np.sum(agent_success_rates)**2) / (env.num_agents * np.sum(agent_success_rates**2) + 1e-8)
    
    avg_jerk = np.mean(all_jerks)
    avg_safety = np.mean(all_safety_frontiers)
    
    # Path Efficiency
    valid_efficiencies = []
    for i in range(len(path_lengths)):
        if path_lengths[i] > 0.5:
            eff = (ideal_lengths[i] / path_lengths[i])
            valid_efficiencies.append(min(eff, 1.0))
        else:
            valid_efficiencies.append(0.0)
    avg_efficiency = np.mean(valid_efficiencies) if valid_efficiencies else 0.0
    
    print("=== Evaluation Results ===")
    print(f"Episodes: {args.episodes}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    print(f"Trapped Rate: {trapped_rate:.1f}%")
    print(f"Avg Time to Goal: {avg_time:.1f} steps")
    print(f"Path Efficiency: {avg_efficiency*100:.1f}%")
    print(f"Fairness Index: {fairness_index:.3f}")
    print(f"Avg Jerk (Smooth): {avg_jerk:.2f}")
    print(f"Safety Frontier: {avg_safety:.2f} m")
    
    # --- ACADEMIC PLOTTING (Fig 4 Style) ---
    print("\nGenerating Navigation Trajectory Plots...")
    
    if all_trajectories:
        plt.figure(figsize=(12, 12))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot the first successful episode's trajectory
        plot_ep = 0
        for i, ep_data in enumerate(all_trajectories):
            # Prefer plotting a success if available
            plot_ep = i
            break 

        ep_data = all_trajectories[plot_ep]
        
        # Plot Obstacles
        for obs_item in ep_data['obstacles']:
            if obs_item['type'] == 'box':
                rect = plt.Rectangle((obs_item['pos'][0]-obs_item['size'][0]/2, obs_item['pos'][1]-obs_item['size'][1]/2), 
                                     obs_item['size'][0], obs_item['size'][1], color='gray', alpha=0.4)
                plt.gca().add_patch(rect)
            else:
                circle = plt.Circle((obs_item['pos'][0], obs_item['pos'][1]), obs_item['radius'], color='gray', alpha=0.4)
                plt.gca().add_patch(circle)
                
        # Plot Paths
        colors = ['#c0392b', '#2980b9', '#27ae60', '#f39c12', '#8e44ad']
        for i in range(env.num_agents):
            path = np.array(ep_data['paths'][i])
            if len(path) > 0:
                plt.plot(path[:, 0], path[:, 1], color=colors[i % len(colors)], label=f'UAV {i+1}', linewidth=2.5)
                plt.scatter(path[0, 0], path[0, 1], color='red', marker='o', s=100, label='Start' if i==0 else "")
            
            plt.scatter(ep_data['goals'][i][0], ep_data['goals'][i][1], color='green', marker='*', s=250, label='Goal' if i==0 else "")
            # Draw the target radius around the goal
            goal_circle = plt.Circle((ep_data['goals'][i][0], ep_data['goals'][i][1]), env.goal_dist, color='green', alpha=0.2, linestyle='--')
            plt.gca().add_patch(goal_circle)

        plt.xlim(0, env.arena_size[0])
        plt.ylim(0, env.arena_size[1])
        plt.title(f'Navigation Trajectories ({args.agent.upper()}) - Evaluation Episode', fontsize=16, fontweight='bold')
        plt.xlabel('X (m)', fontsize=14)
        plt.ylabel('Y (m)', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        output_path = os.path.join(config['logging']['log_dir'], 'evaluation_trajectories.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Trajectory plot saved to {output_path}")
        plt.close() # Close trajectory plot
    else:
        print("No complete trajectories to plot. Skipping trajectory chart.")

    # Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics = ['Success (%)', 'Collision (%)', 'Path Efficiency (%)', 'Fairness Index (*100)']
    values = [success_rate, collision_rate, avg_efficiency * 100, fairness_index * 100]
    colors_bar = ['#27ae60', '#c0392b', '#2980b9', '#8e44ad']
    
    bars = plt.bar(metrics, values, color=colors_bar)
    plt.title('Evaluation Metrics Summary', fontsize=16, fontweight='bold')
    plt.ylim(0, 110)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')
    
    metrics_path = os.path.join(config['logging']['log_dir'], 'evaluation_metrics.png')
    plt.savefig(metrics_path, dpi=300)
    print(f"Metrics plot saved to {metrics_path}")
    
    plt.show()
    
    env.close()

if __name__ == '__main__':
    main()
