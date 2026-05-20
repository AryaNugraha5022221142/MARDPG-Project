import argparse
import yaml
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import BenchmarkSuite, QuadcopterKinematicEnv
from envs.benchmark_wrapped_env import BenchmarkWrappedEnv
from agents import MARDPG_Baseline

BENCHMARK_SCENES = tuple(BenchmarkSuite.REGISTRY.keys())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--agent', type=str, default='mardpg_baseline', choices=['mardpg_baseline'])
    parser.add_argument('--scenes', type=str, default='all', help='Comma separated scenes or "all" (for benchmark envs)')
    parser.add_argument('--scenario', type=str, default=None, help='Legacy: Specific scenario name (e.g. forest, pillars)')
    parser.add_argument('--legacy-random-env', action='store_true', help='Use legacy random environment')
    parser.add_argument('--num-agents', type=int, default=None)
    parser.add_argument('--level', type=str, default='1,2,3,4,5', help='Comma separated levels or single level (e.g., 1,2,3,4,5)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=250, help='Episodes per scene/level')
    parser.add_argument('--output-json', type=str, default='logs/multi_scene_evaluation_results.json')
    parser.add_argument('--output-csv', type=str, default='logs/multi_scene_evaluation_results.csv')
    parser.add_argument('--output-plot', type=str, default='logs/multi_scene_evaluation_metrics.png')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config['training'].get('device') == 'cuda' else "cpu")
    num_agents = args.num_agents if args.num_agents is not None else config['training'].get('num_agents', 5)
    
    obs_dim = 30
    agent = MARDPG_Baseline(
        obs_dim=obs_dim, 
        action_dim=2, 
        num_agents=num_agents,
        config=config, 
        device=device
    )
    
    agent.load(args.checkpoint)

    env_config = config['environment'].copy()
    env_config['seed'] = args.seed

    if args.scenario is not None:
        scenes = [args.scenario]
    elif args.legacy_random_env:
        scenes = ['legacy']
    elif args.scenes.lower() == 'all':
        scenes = list(BENCHMARK_SCENES)
    else:
        scenes = args.scenes.split(',')
        
    levels = [int(l.strip()) for l in str(args.level).split(',')]

    all_results = {}

    for scene in scenes:
        for level in levels:
            key = f"{scene}_L{level}"
            if scene == 'legacy':
                key = "legacy"
            print(f"\nEvaluating scene: {scene} (Level {level})")
            
            if scene == 'legacy':
                env = QuadcopterKinematicEnv(
                    num_agents=num_agents, 
                    config=env_config,
                    render_mode=None
                )
            else:
                env = BenchmarkWrappedEnv(
                    benchmark_name=scene,
                    level=level,
                    num_agents=num_agents,
                    config=env_config
                )
                
            scene_metrics = {
                'success': [],
                'collision': [],
                'trapped': [],
                'path_efficiency': [],
                'fairness': [],
                'smoothness': [],
                'jerk': [],
                'safety_clearance': [],
                'time_to_goal': [],
                'obstacle_count': 0,
                'dynamic_obstacle_count': 0
            }
            
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + ep * 100)
                actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(env.num_agents)]
                critic_hidden = [agent.critics[i].init_hidden(1, device) for i in range(env.num_agents)]
                
                if ep == 0:
                    scene_metrics['obstacle_count'] = len(env.obstacles)
                    scene_metrics['dynamic_obstacle_count'] = sum(1 for o in env.obstacles if o.get('is_dynamic', False))
            
                steps = 0
                ep_successes = 0
                ep_collisions = 0
                ep_smoothness = []
                ep_jerk = []
                ep_safety = []
                
                # For efficiency/fairness
                start_dist = np.linalg.norm(np.array([a.state[0:3] for a in env.agents]) - np.array(env.goals), axis=1)
                dist_traveled = np.zeros(num_agents)
                prev_positions = np.array([a.state[0:3] for a in env.agents])
                
                ep_positions = []
                if ep == 0:
                    ep_positions.append(prev_positions.copy())
    
                while steps < env.max_steps:
                    action, actor_hidden, critic_hidden = agent.select_actions(obs, actor_hidden, critic_hidden, explore=False)
                    
                    next_obs, rewards, terminated, truncated, info = env.step(action)
                    obs = next_obs
                    steps += 1
                    
                    current_positions = np.array([a.state[0:3] for a in env.agents])
                    dist_traveled += np.linalg.norm(current_positions - prev_positions, axis=1)
                    prev_positions = current_positions
                    if ep == 0:
                        ep_positions.append(current_positions.copy())
                    
                    ep_smoothness.append(info.get('action_smoothness', 0.0))
                    ep_jerk.append(info.get('avg_jerk', 0.0))
                    ep_safety.append(info.get('safety_frontier', 0.0))
                    
                    if terminated or truncated:
                        ep_successes = int(np.sum(info.get('agent_success', np.zeros(num_agents, dtype=bool))))
                        ep_collisions = int(np.sum(info.get('agent_collision', np.zeros(num_agents, dtype=bool))))
                        ep_any_collision = ep_collisions > 0
                        ep_success = (ep_successes == num_agents and not ep_any_collision)
                        
                        out_str = f"Ep {ep+1}/{args.episodes} | Steps: {steps} | Success: {ep_success} | Collision: {ep_any_collision}"
                        print(f"{out_str:<80}", end='\r')
                        break
                        
                ep_any_collision = ep_collisions > 0
                scene_metrics['success'].append(1.0 if (ep_successes == num_agents and not ep_any_collision) else 0.0)
                scene_metrics['collision'].append(1.0 if ep_any_collision else 0.0)
                scene_metrics['trapped'].append(1.0 if (ep_successes == 0 and not ep_any_collision) else 0.0)
                
                # Metrics computation
                eff = start_dist / np.maximum(dist_traveled, start_dist)
                scene_metrics['path_efficiency'].append(np.mean(eff))
                scene_metrics['fairness'].append(1.0 - np.std(eff))
                
                scene_metrics['smoothness'].append(np.mean(ep_smoothness))
                scene_metrics['jerk'].append(np.mean(ep_jerk))
                scene_metrics['safety_clearance'].append(np.mean(ep_safety))
                scene_metrics['time_to_goal'].append(steps)
                
                if ep == 0:
                    fig_traj = plt.figure(figsize=(10, 8))
                    ax_traj = fig_traj.add_subplot(111, projection='3d')
                    ep_positions_arr = np.array(ep_positions)  # shape (T, num_agents, 3)
                    for i in range(num_agents):
                        ax_traj.plot(ep_positions_arr[:, i, 0], ep_positions_arr[:, i, 1], ep_positions_arr[:, i, 2], label=f'Agent {i}')
                        ax_traj.scatter(ep_positions_arr[0, i, 0], ep_positions_arr[0, i, 1], ep_positions_arr[0, i, 2], color='green', marker='o')
                        ax_traj.scatter(ep_positions_arr[-1, i, 0], ep_positions_arr[-1, i, 1], ep_positions_arr[-1, i, 2], color='blue', marker='x')
                    
                    # Goal points
                    goals = np.array(env.goals)
                    for i in range(num_agents):
                        ax_traj.scatter(goals[i, 0], goals[i, 1], goals[i, 2], color='red', marker='*')
    
                    ax_traj.set_title(f'3D Trajectory - Benchmark: {key}')
                    ax_traj.set_xlabel('X')
                    ax_traj.set_ylabel('Y')
                    ax_traj.set_zlabel('Z')
                    plt.legend()
                    os.makedirs('logs', exist_ok=True)
                    plt.savefig(f'logs/{key}_trajectory_ep0.png')
                    plt.close(fig_traj)

        # Aggregate
        sr = float(np.mean(scene_metrics['success'])) * 100
        cr = float(np.mean(scene_metrics['collision'])) * 100
        at = float(np.mean(scene_metrics['time_to_goal']))
        
        all_results[key] = {
        'success_rate': float(np.mean(scene_metrics['success'])),
        'collision_rate': float(np.mean(scene_metrics['collision'])),
        'trapped_rate': float(np.mean(scene_metrics['trapped'])),
        'path_efficiency': float(np.mean(scene_metrics['path_efficiency'])),
        'fairness': float(np.mean(scene_metrics['fairness'])),
        'smoothness': float(np.mean(scene_metrics['smoothness'])),
        'jerk': float(np.mean(scene_metrics['jerk'])),
        'safety_clearance': float(np.mean(scene_metrics['safety_clearance'])),
        'average_time_to_goal': at,
        'obstacle_count': scene_metrics['obstacle_count'],
        'dynamic_obstacle_count': scene_metrics['dynamic_obstacle_count']
        }
        print(f"\n\nResults for {key}:\n" + "-"*30)
        print(f"Success Rate:    {all_results[key]['success_rate']*100:.1f}%")
        print(f"Collision Rate:  {all_results[key]['collision_rate']*100:.1f}%")
        print(f"Trapped Rate:    {all_results[key]['trapped_rate']*100:.1f}%")
        print(f"Path Efficiency: {all_results[key]['path_efficiency']:.3f}")
        print(f"Fairness Index:  {all_results[key]['fairness']:.3f}")
        print(f"Avg Time:        {all_results[key]['average_time_to_goal']:.1f} steps")
        print("-" * 30 + "\n")

    # Aggregate over scenes
    successes = [res['success_rate'] for res in all_results.values()]
    collisions = [res['collision_rate'] for res in all_results.values()]
    
    aggregate = {
        'mean_success': float(np.mean(successes)),
        'variance_success': float(np.var(successes)),
        'mean_collision': float(np.mean(collisions)),
        'worst_case_scene': min(all_results.keys(), key=lambda k: all_results[k]['success_rate']),
        'generalization_score': float(np.mean(successes) * (1.0 - np.std(successes)))
    }
    
    final_output = {
        'scenes': all_results,
        'aggregate': aggregate
    }
    
    # Save JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['scene', 'success_rate', 'collision_rate', 'trapped_rate', 'path_efficiency', 
                   'fairness', 'smoothness', 'jerk', 'safety_clearance', 'average_time_to_goal', 
                   'obstacle_count', 'dynamic_obstacle_count']
        writer.writerow(headers)
        for scene, res in all_results.items():
            writer.writerow([scene] + [res[h] for h in headers[1:]])
            
    # Save Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_results))
    width = 0.35
    
    scene_names = list(all_results.keys())
    ax.bar(x - width/2, [all_results[s]['success_rate']*100 for s in scene_names], width, label='Success Rate', color='#2ecc71')
    ax.bar(x + width/2, [all_results[s]['collision_rate']*100 for s in scene_names], width, label='Collision Rate', color='#e74c3c')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Multi-Scene Evaluation ({num_agents} UAVs)')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scene_names])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
    plt.savefig(args.output_plot, dpi=300)
    
    print("\n--- AGGREGATE METRICS ---")
    print(f"Mean Success: {aggregate['mean_success']*100:.1f}%")
    print(f"Mean Collision: {aggregate['mean_collision']*100:.1f}%")
    print(f"Generalization Score: {aggregate['generalization_score']:.3f}")
    print(f"Worst Case Scene: {aggregate['worst_case_scene'].capitalize()}")
    print(f"\nResults saved to logs/")

if __name__ == '__main__':
    main()
