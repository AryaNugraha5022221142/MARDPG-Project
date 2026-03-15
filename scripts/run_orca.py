# scripts/run_orca.py
import numpy as np
import time
import os
import sys
import argparse
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadcopter_env import QuadcopterEnv
from agents.classical import ORCAAgent

def run_orca_baseline(scenario='city', num_agents=3, num_episodes=10, render=True, seed=42):
    if seed is not None:
        np.random.seed(seed)
        
    env = QuadcopterEnv(num_agents=num_agents, render_mode='human' if render else None, scenario=scenario)
    agent_logic = ORCAAgent(num_agents=num_agents)
    
    # Metrics storage
    successes = 0
    collisions = 0
    latencies = []
    path_lengths = []
    ideal_lengths = []
    velocity_consistencies = []
    all_jerks = []
    all_safety_frontiers = []
    agent_success_counts = np.zeros(num_agents)
    
    print(f"\n=== Evaluating Classical Baseline (ORCA/RVO) ===")
    print(f"Scenario: {scenario} | Episodes: {num_episodes} | Agents: {num_agents} | Seed: {seed}\n")

    best_trajectory = None
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_steps = 0
        ep_trajectories = [[] for _ in range(num_agents)]
        start_positions = [agent.state[:3].copy() for agent in env.agents]
        current_path_lengths = np.zeros(num_agents)
        last_positions = [p.copy() for p in start_positions]
        ep_velocities = []
        
        while not done:
            start_time = time.time()
            
            # Get internal states for the classical agent
            states = [a.state for a in env.agents]
            actions = agent_logic.select_actions(states, env.goals, env.obstacles)
            
            # Track velocities for consistency metric
            current_vels = [a.state[7:10] for a in env.agents]
            
            if num_agents > 1:
                avg_vel = np.mean(current_vels, axis=0)
                avg_vel_norm = np.linalg.norm(avg_vel)
                if avg_vel_norm > 1e-6:
                    consistencies = [np.dot(v, avg_vel) / (np.linalg.norm(v) * avg_vel_norm + 1e-6) for v in current_vels]
                    ep_velocities.append(np.mean(consistencies))

            latencies.append(time.time() - start_time)
            obs, rewards, terminated, truncated, info = env.step(actions)
            if render: env.render()
            
            for i in range(num_agents):
                curr_p = env.agents[i].state[:3]
                ep_trajectories[i].append(curr_p.copy())
                current_path_lengths[i] += np.linalg.norm(curr_p - last_positions[i])
                last_positions[i] = curr_p.copy()
            
            done = terminated or truncated
            ep_steps += 1

        if info.get('success'): 
            successes += 1
            agent_success_counts += 1
            if best_trajectory is None:
                best_trajectory = {
                    'paths': ep_trajectories,
                    'goals': [g.copy() for g in env.goals],
                    'obstacles': [o.copy() for o in env.obstacles]
                }
        if info.get('collision'): collisions += 1
        if ep_velocities: velocity_consistencies.append(np.mean(ep_velocities))
        all_jerks.append(np.mean(env.total_jerk))
        all_safety_frontiers.append(np.mean(env.safety_frontier))

        if ep == num_episodes - 1 and best_trajectory is None:
            best_trajectory = {
                'paths': ep_trajectories,
                'goals': [g.copy() for g in env.goals],
                'obstacles': [o.copy() for o in env.obstacles]
            }
        
        for i in range(num_agents):
            ideal = np.linalg.norm(env.goals[i] - start_positions[i])
            path_lengths.append(current_path_lengths[i])
            ideal_lengths.append(ideal)

        print(f"Episode {ep+1}/{num_episodes} | Steps: {ep_steps} | Result: {'SUCCESS' if info.get('success') else 'FAILED'}")

    # Report Generation (Same as APF)
    avg_success = (successes / num_episodes)
    avg_collision = (collisions / num_episodes)
    avg_latency = np.mean(latencies) * 1000
    avg_v_cons = np.mean(velocity_consistencies) if velocity_consistencies else 0.0
    valid_efficiencies = [(ideal_lengths[i] / path_lengths[i]) if path_lengths[i] > 0.5 else 0.0 for i in range(len(path_lengths))]
    efficiency = np.mean([min(e, 1.0) for e in valid_efficiencies])
    agent_success_rates = agent_success_counts / num_episodes
    fairness_index = (np.sum(agent_success_rates)**2) / (num_agents * np.sum(agent_success_rates**2) + 1e-8)
    avg_jerk = np.mean(all_jerks)
    avg_safety = np.mean(all_safety_frontiers)

    print("\n" + "="*40)
    print("FINAL PERFORMANCE REPORT (ORCA/RVO)")
    print("="*40)
    print(f"Success Rate:    {avg_success*100:.1f}%")
    print(f"Collision Rate:  {avg_collision*100:.1f}%")
    print(f"Path Efficiency: {efficiency*100:.1f}%")
    print(f"Fairness Index:  {fairness_index:.3f}")
    print(f"Avg Jerk (Smooth): {avg_jerk:.2f}")
    print(f"Safety Frontier: {avg_safety:.2f} m")
    print(f"Avg Latency:     {avg_latency:.3f} ms")
    print("="*40)

    # Plotting
    plt.figure(figsize=(10, 10))
    for obs_item in best_trajectory['obstacles']:
        if obs_item['type'] == 'box':
            rect = plt.Rectangle((obs_item['pos'][0]-obs_item['size'][0]/2, obs_item['pos'][1]-obs_item['size'][1]/2), 
                                 obs_item['size'][0], obs_item['size'][1], color='gray', alpha=0.5)
            plt.gca().add_patch(rect)
        else:
            circle = plt.Circle((obs_item['pos'][0], obs_item['pos'][1]), obs_item['radius'], color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
            
    colors_agents = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(num_agents):
        path = np.array(best_trajectory['paths'][i])
        if len(path) > 0:
            plt.plot(path[:, 0], path[:, 1], color=colors_agents[i % len(colors_agents)], label=f'UAV {i+1}', linewidth=2)
            plt.scatter(path[0, 0], path[0, 1], color='black', marker='o', s=50)
            plt.scatter(best_trajectory['goals'][i][0], best_trajectory['goals'][i][1], color='gold', marker='*', s=200)

    plt.xlim(0, env.arena_size[0])
    plt.ylim(0, env.arena_size[1])
    plt.title(f'ORCA Trajectory Tracking: {scenario}', fontweight='bold')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('orca_trajectory_tracking.png', dpi=300)
    print("\nAcademic report saved: 'orca_trajectory_tracking.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="city")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()
    
    run_orca_baseline(scenario=args.scenario, num_agents=args.agents, num_episodes=args.episodes, render=not args.no_render, seed=args.seed)
