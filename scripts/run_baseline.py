# scripts/run_baseline.py
import numpy as np
import time
import os
import sys
import argparse
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadcopter_env import QuadcopterEnv

def run_apf_baseline(scenario='city', num_agents=3, num_episodes=10, render=True, seed=42):
    """
    Runs the APF baseline and collects performance metrics.
    """
    if seed is not None:
        np.random.seed(seed)
        
    env = QuadcopterEnv(num_agents=num_agents, render_mode='human' if render else None, scenario=scenario)
    
    # Metrics storage
    successes = 0
    collisions = 0
    latencies = []
    path_lengths = []
    ideal_lengths = []
    velocity_consistencies = []
    
    # Advanced EE Metrics
    all_jerks = []
    all_safety_frontiers = []
    agent_success_counts = np.zeros(num_agents)
    
    print(f"\n=== Evaluating Classical Baseline (APF) ===")
    print(f"Scenario: {scenario} | Episodes: {num_episodes} | Agents: {num_agents} | Seed: {seed}\n")

    # Trajectory storage for plotting
    best_trajectory = None
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_steps = 0
        
        # Track trajectory for this episode
        ep_trajectories = [[] for _ in range(num_agents)]
        
        # Track path length
        start_positions = [agent.state[:3].copy() for agent in env.agents]
        current_path_lengths = np.zeros(num_agents)
        last_positions = [p.copy() for p in start_positions]
        
        # Velocity consistency for this episode
        ep_velocities = []
        
        # APF Hyperparameters
        k_att = 1.0
        k_rep = 50.0
        d_min = 5.0
        
        while not done:
            start_time = time.time()
            
            # APF Logic
            actions = []
            current_vels = []
            for i in range(num_agents):
                agent_pos = env.agents[i].state[:3]
                ep_trajectories[i].append(agent_pos.copy())
                goal_pos = env.goals[i]
                
                # 1. Attractive force
                f_att = (goal_pos - agent_pos)
                dist_to_goal = np.linalg.norm(f_att)
                if dist_to_goal > 0.1:
                    f_att = f_att / dist_to_goal * k_att
                
                # 2. Repulsive force
                f_rep = np.zeros(3)
                for obs_item in env.obstacles:
                    o_pos = obs_item['pos']
                    dist_vec = agent_pos - o_pos
                    dist = np.linalg.norm(dist_vec)
                    
                    if obs_item['type'] == 'box':
                        dist = max(0.1, dist - np.max(obs_item['size'])/2)
                    
                    if dist < d_min:
                        rep_mag = k_rep * (1.0/dist - 1.0/d_min) / (dist**2)
                        f_rep += (dist_vec / np.linalg.norm(dist_vec)) * rep_mag
                
                f_total = f_att + f_rep
                current_vels.append(f_total / (np.linalg.norm(f_total) + 1e-6))
                
                # Map force to discrete actions
                if f_total[2] > 0.5: actions.append(3) # Up
                elif f_total[2] < -0.5: actions.append(4) # Down
                else:
                    target_yaw = np.arctan2(f_total[1], f_total[0])
                    current_yaw = env.agents[i].state[5]
                    yaw_diff = (target_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
                    
                    if yaw_diff > 0.3: actions.append(1) # Right
                    elif yaw_diff < -0.3: actions.append(2) # Left
                    else: actions.append(0) # Forward
            
            # Calculate Velocity Consistency (Cosine Similarity between agents)
            if num_agents > 1:
                avg_vel = np.mean(current_vels, axis=0)
                avg_vel_norm = np.linalg.norm(avg_vel)
                if avg_vel_norm > 1e-6:
                    consistencies = [np.dot(v, avg_vel) / (np.linalg.norm(v) * avg_vel_norm + 1e-6) for v in current_vels]
                    ep_velocities.append(np.mean(consistencies))

            # Record Latency
            latencies.append(time.time() - start_time)
            
            # Step
            obs, rewards, terminated, truncated, info = env.step(actions)
            if render: env.render()
            
            # Update path length
            for i in range(num_agents):
                curr_p = env.agents[i].state[:3]
                ep_trajectories[i].append(curr_p.copy())
                current_path_lengths[i] += np.linalg.norm(curr_p - last_positions[i])
                last_positions[i] = curr_p.copy()
            
            done = terminated or truncated
            ep_steps += 1

        # Record Episode Results
        if info.get('success'): 
            successes += 1
            agent_success_counts += 1 # In APF, if success, all agents reached goal
            if best_trajectory is None:
                best_trajectory = {
                    'paths': ep_trajectories,
                    'goals': [g.copy() for g in env.goals],
                    'obstacles': [o.copy() for o in env.obstacles]
                }
        if info.get('collision'): collisions += 1
        if ep_velocities: velocity_consistencies.append(np.mean(ep_velocities))
        
        # Collect EE Metrics
        all_jerks.append(np.mean(env.total_jerk))
        all_safety_frontiers.append(np.mean(env.safety_frontier))

        # If last episode and still no success, take the last one
        if ep == num_episodes - 1 and best_trajectory is None:
            best_trajectory = {
                'paths': ep_trajectories,
                'goals': [g.copy() for g in env.goals],
                'obstacles': [o.copy() for o in env.obstacles]
            }
        
        # Path Efficiency (Ideal / Actual)
        for i in range(num_agents):
            ideal = np.linalg.norm(env.goals[i] - start_positions[i])
            path_lengths.append(current_path_lengths[i])
            ideal_lengths.append(ideal)

        print(f"Episode {ep+1}/{num_episodes} | Steps: {ep_steps} | Result: {'SUCCESS' if info.get('success') else 'FAILED'}")

    # --- Generate Final Report ---
    avg_success = (successes / num_episodes)
    avg_collision = (collisions / num_episodes)
    avg_latency = np.mean(latencies) * 1000 # ms
    avg_v_cons = np.mean(velocity_consistencies) if velocity_consistencies else 0.0
    
    # Avoid division by zero and handle very short paths (collisions)
    valid_efficiencies = []
    for i in range(len(path_lengths)):
        if path_lengths[i] > 0.5: # Only count if the UAV actually moved
            eff = (ideal_lengths[i] / path_lengths[i])
            valid_efficiencies.append(min(eff, 1.0)) # Cap at 100%
        else:
            valid_efficiencies.append(0.0)
            
    efficiency = np.mean(valid_efficiencies) if valid_efficiencies else 0.0
    
    # Calculate Fairness Index (Jain's Fairness Index on agent success rates)
    agent_success_rates = agent_success_counts / num_episodes
    fairness_index = (np.sum(agent_success_rates)**2) / (num_agents * np.sum(agent_success_rates**2) + 1e-8)
    
    avg_jerk = np.mean(all_jerks)
    avg_safety = np.mean(all_safety_frontiers)

    print("\n" + "="*40)
    print("FINAL PERFORMANCE REPORT (APF)")
    print("="*40)
    print(f"Success Rate:    {avg_success*100:.1f}%")
    print(f"Collision Rate:  {avg_collision*100:.1f}%")
    print(f"Path Efficiency: {efficiency*100:.1f}%")
    print(f"Fairness Index:  {fairness_index:.3f}")
    print(f"Avg Jerk (Smooth): {avg_jerk:.2f}")
    print(f"Safety Frontier: {avg_safety:.2f} m")
    print(f"Avg Latency:     {avg_latency:.3f} ms")
    print("="*40)

    # --- ACADEMIC PLOTTING ---
    # 1. Performance Bar Charts (Like Fig 3 in your image)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    metrics = ['Success Rate', 'Collision Rate', 'Velocity Consistency', 'Path Efficiency']
    values = [avg_success, avg_collision, avg_v_cons, efficiency]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f']
    
    for i in range(4):
        axs[i].bar([scenario], [values[i]], color=colors[i], width=0.4, edgecolor='black', hatch='//')
        axs[i].set_title(metrics[i], fontweight='bold')
        axs[i].set_ylim(0, 1.1)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300)
    
    # 2. Trajectory Tracking (Like Fig 4 in your image)
    plt.figure(figsize=(10, 10))
    # Plot Obstacles
    for obs_item in best_trajectory['obstacles']:
        if obs_item['type'] == 'box':
            rect = plt.Rectangle((obs_item['pos'][0]-obs_item['size'][0]/2, obs_item['pos'][1]-obs_item['size'][1]/2), 
                                 obs_item['size'][0], obs_item['size'][1], color='gray', alpha=0.5)
            plt.gca().add_patch(rect)
        else:
            circle = plt.Circle((obs_item['pos'][0], obs_item['pos'][1]), obs_item['radius'], color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
            
    # Plot Paths
    colors_agents = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(num_agents):
        path = np.array(best_trajectory['paths'][i])
        if len(path) > 0:
            plt.plot(path[:, 0], path[:, 1], color=colors_agents[i % len(colors_agents)], label=f'UAV {i+1}', linewidth=2)
            plt.scatter(path[0, 0], path[0, 1], color='black', marker='o', s=50) # Start
            plt.scatter(best_trajectory['goals'][i][0], best_trajectory['goals'][i][1], color='gold', marker='*', s=200) # Goal

    plt.xlim(0, env.arena_size[0])
    plt.ylim(0, env.arena_size[1])
    plt.title(f'Trajectory Tracking: {scenario}', fontweight='bold')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('trajectory_tracking.png', dpi=300)
    
    print("\nAcademic reports saved:")
    print("- 'performance_metrics.png' (Bar charts)")
    print("- 'trajectory_tracking.png' (Path tracking)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="city")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()
    
    run_apf_baseline(scenario=args.scenario, num_agents=args.agents, num_episodes=args.episodes, render=not args.no_render, seed=args.seed)
