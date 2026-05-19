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
    parser.add_argument('--scenes', type=str, default='all', help='Comma separated scenes or "all"')
    parser.add_argument('--scenario', type=str, default=None, help='Specific scenario name')
    parser.add_argument('--legacy-random-env', action='store_true')
    parser.add_argument('--num-agents', type=int, default=None)
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--output-json', type=str, default='logs/multi_scene_evaluation_results.json')
    parser.add_argument('--output-csv', type=str, default='logs/multi_scene_evaluation_results.csv')
    parser.add_argument('--output-plot', type=str, default='logs/multi_scene_evaluation_metrics.png')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config['training'].get('device') == 'cuda' else "cpu")
    num_agents = args.num_agents if args.num_agents is not None else config['training'].get('num_agents', 5)
    
    if args.legacy_random_env:
        scenes = ['legacy']
    else:
        if args.scenario:
            scenes = [args.scenario]
        else:
            if args.scenes.lower() == 'all':
                scenes = list(BENCHMARK_SCENES)
            else:
                scenes = args.scenes.split(',')

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
    
    all_results = {}
    
    for scene in scenes:
        print(f"\nEvaluating scene: {scene}")
        if scene == 'legacy':
            env = QuadcopterKinematicEnv(
                num_agents=num_agents, 
                config=env_config,
                render_mode=None
            )
        else:
            env = BenchmarkWrappedEnv(
                benchmark_name=scene,
                level=args.level,
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
            hidden = agent.init_hidden(batch_size=1)
            
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
            
            while steps < env.max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, hidden, _ = agent.select_action(obs_tensor, hidden, explore=False)
                action = action.squeeze(0).cpu().numpy()
                
                next_obs, rewards, terminated, truncated, info = env.step(action)
                obs = next_obs
                steps += 1
                
                current_positions = np.array([a.state[0:3] for a in env.agents])
                dist_traveled += np.linalg.norm(current_positions - prev_positions, axis=1)
                prev_positions = current_positions
                
                ep_smoothness.append(info.get('action_smoothness', 0.0))
                ep_jerk.append(info.get('avg_jerk', 0.0))
                ep_safety.append(info.get('safety_frontier', 0.0))
                
                if terminated or truncated:
                    ep_successes = sum(info.get('agent_success', np.zeros(num_agents)))
                    ep_collisions = sum(info.get('agent_collision', np.zeros(num_agents)))
                    break
                    
            scene_metrics['success'].append(1.0 if ep_successes == num_agents else 0.0)
            scene_metrics['collision'].append(1.0 if ep_collisions > 0 else 0.0)
            scene_metrics['trapped'].append(1.0 if (ep_successes < num_agents and ep_collisions == 0) else 0.0)
            
            # Metrics computation
            eff = start_dist / np.maximum(dist_traveled, start_dist)
            scene_metrics['path_efficiency'].append(np.mean(eff))
            scene_metrics['fairness'].append(1.0 - np.std(eff))
            
            scene_metrics['smoothness'].append(np.mean(ep_smoothness))
            scene_metrics['jerk'].append(np.mean(ep_jerk))
            scene_metrics['safety_clearance'].append(np.mean(ep_safety))
            scene_metrics['time_to_goal'].append(steps)
            
        # Aggregate
        all_results[scene] = {
            'success_rate': float(np.mean(scene_metrics['success'])),
            'collision_rate': float(np.mean(scene_metrics['collision'])),
            'trapped_rate': float(np.mean(scene_metrics['trapped'])),
            'path_efficiency': float(np.mean(scene_metrics['path_efficiency'])),
            'fairness': float(np.mean(scene_metrics['fairness'])),
            'smoothness': float(np.mean(scene_metrics['smoothness'])),
            'jerk': float(np.mean(scene_metrics['jerk'])),
            'safety_clearance': float(np.mean(scene_metrics['safety_clearance'])),
            'average_time_to_goal': float(np.mean(scene_metrics['time_to_goal'])),
            'obstacle_count': scene_metrics['obstacle_count'],
            'dynamic_obstacle_count': scene_metrics['dynamic_obstacle_count']
        }
        print(f"  > Success Rate: {all_results[scene]['success_rate']*100:.1f}%, Collision Rate: {all_results[scene]['collision_rate']*100:.1f}%")

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
