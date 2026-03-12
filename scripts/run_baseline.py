# scripts/run_baseline.py
import numpy as np
import time
import os
import sys

# Add the project root to sys.path so it can find the 'envs' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadcopter_env import QuadcopterEnv

def run_apf_baseline(scenario='city', num_agents=3):
    """
    Runs a classical Artificial Potential Field (APF) baseline.
    """
    env = QuadcopterEnv(num_agents=num_agents, render_mode='human', scenario=scenario)
    obs, _ = env.reset()
    
    print(f"Running Classical Baseline (APF) on scenario: {scenario}")
    print("Close the window or press Ctrl+C to stop.")
    
    # APF Hyperparameters
    k_att = 1.0  # Attraction gain
    k_rep = 50.0 # Repulsion gain
    d_min = 5.0  # Distance at which repulsion starts
    
    try:
        while True:
            actions = []
            for i in range(num_agents):
                if env.dones[i]:
                    actions.append(5) # Hover if done
                    continue
                
                agent_pos = env.agents[i].state[:3]
                goal_pos = env.goals[i]
                
                # 1. Attractive Force (to goal)
                f_att = (goal_pos - agent_pos)
                f_att = f_att / np.linalg.norm(f_att) * k_att
                
                # 2. Repulsive Force (from obstacles)
                f_rep = np.zeros(3)
                for obs_item in env.obstacles:
                    o_pos = obs_item['pos']
                    dist_vec = agent_pos - o_pos
                    dist = np.linalg.norm(dist_vec)
                    
                    # Adjust distance for box size if necessary
                    if obs_item['type'] == 'box':
                        dist = max(0.1, dist - np.max(obs_item['size'])/2)
                    
                    if dist < d_min:
                        # Repulsive force is inversely proportional to distance squared
                        rep_mag = k_rep * (1.0/dist - 1.0/d_min) / (dist**2)
                        f_rep += (dist_vec / np.linalg.norm(dist_vec)) * rep_mag
                
                # Total Force
                f_total = f_att + f_rep
                
                # Map force to discrete actions
                # Action map: 0:Fwd, 1:Fwd+Right, 2:Fwd+Left, 3:Fwd+Up, 4:Fwd+Down, 5:Hover
                
                # Simple heuristic mapping for APF
                if f_total[2] > 0.5: actions.append(3) # Up
                elif f_total[2] < -0.5: actions.append(4) # Down
                else:
                    # Calculate angle to force
                    target_yaw = np.arctan2(f_total[1], f_total[0])
                    current_yaw = env.agents[i].state[3]
                    yaw_diff = (target_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
                    
                    if yaw_diff > 0.3: actions.append(1) # Right
                    elif yaw_diff < -0.3: actions.append(2) # Left
                    else: actions.append(0) # Forward
            
            obs, rewards, dones, truncated, info = env.step(actions)
            env.render()
            
            if all(dones):
                print("Episode finished. Resetting...")
                time.sleep(1.0)
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nBaseline stopped by user.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='city')
    parser.add_argument('--agents', type=int, default=3)
    args = parser.parse_args()
    
    run_apf_baseline(scenario=args.scenario, num_agents=args.agents)
