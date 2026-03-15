# agents/classical.py
import numpy as np
from typing import List

class PotentialFieldAgent:
    """
    A classical Artificial Potential Field agent for obstacle avoidance.
    """
    def __init__(self, num_agents: int, arena_size: np.ndarray):
        self.num_agents = num_agents
        self.arena_size = arena_size
        self.k_attr = 1.0  # Attraction gain
        self.k_rep = 5.0   # Repulsion gain
        self.d_min = 2.0   # Distance threshold for repulsion
        
    def select_actions(self, states: List[np.ndarray], goals: np.ndarray, obstacles: List[dict]) -> List[int]:
        actions = []
        for i in range(self.num_agents):
            pos = states[i][0:3]
            yaw = states[i][5]
            goal = goals[i]
            
            # Attractive force
            f_attr = self.k_attr * (goal - pos)
            
            # Repulsive force from obstacles
            f_rep = np.zeros(3)
            for obs in obstacles:
                obs_pos = obs['pos']
                dist = np.linalg.norm(pos - obs_pos)
                if obs['type'] == 'sphere':
                    dist -= obs['radius']
                
                if dist < self.d_min:
                    # Force points away from obstacle
                    direction = (pos - obs_pos) / (np.linalg.norm(pos - obs_pos) + 1e-6)
                    f_rep += self.k_rep * (1.0/dist - 1.0/self.d_min) * (1.0/dist**2) * direction
            
            # Repulsive force from other agents
            for j in range(self.num_agents):
                if i == j: continue
                other_pos = states[j][0:3]
                dist = np.linalg.norm(pos - other_pos)
                if dist < self.d_min:
                    direction = (pos - other_pos) / (dist + 1e-6)
                    f_rep += self.k_rep * (1.0/dist - 1.0/self.d_min) * (1.0/dist**2) * direction
            
            # Total force
            f_total = f_attr + f_rep
            
            # Map force to discrete actions
            # Simplified mapping:
            # 0: Forward, 1: Yaw Right, 2: Yaw Left, 3: Up, 4: Down, 5: Hover
            
            # Desired velocity direction in world frame
            desired_dir = f_total / (np.linalg.norm(f_total) + 1e-6)
            
            # Transform to local frame
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            local_dx = desired_dir[0] * cos_y + desired_dir[1] * sin_y
            local_dy = -desired_dir[0] * sin_y + desired_dir[1] * cos_y
            
            # Very simple mapping logic
            if desired_dir[2] > 0.3:
                action = 3 # Up
            elif desired_dir[2] < -0.3:
                action = 4 # Down
            elif local_dy > 0.3:
                action = 1 # Yaw Right (simplified)
            elif local_dy < -0.3:
                action = 2 # Yaw Left (simplified)
            else:
                action = 0 # Forward
                
            actions.append(action)
            
        return actions

class ORCAAgent:
    """
    A simplified Python implementation of Reciprocal Velocity Obstacles (RVO).
    Uses sampling-based optimization to find the best collision-free velocity.
    """
    def __init__(self, num_agents: int, max_speed: float = 5.0, neighbor_dist: float = 15.0, time_horizon: float = 2.0):
        self.num_agents = num_agents
        self.max_speed = max_speed
        self.neighbor_dist = neighbor_dist
        self.time_horizon = time_horizon
        self.radius = 1.0 # Safety radius for agents

    def select_actions(self, states: List[np.ndarray], goals: np.ndarray, obstacles: List[dict]) -> List[int]:
        actions = []
        for i in range(self.num_agents):
            pos = states[i][0:3]
            vel = states[i][7:10]
            yaw = states[i][5]
            goal = goals[i]
            
            # 1. Preferred Velocity
            pref_vel = (goal - pos)
            dist_to_goal = np.linalg.norm(pref_vel)
            if dist_to_goal > 0.1:
                pref_vel = (pref_vel / dist_to_goal) * self.max_speed
            else:
                pref_vel = np.zeros(3)

            # 2. Collect Obstacles
            vo_obstacles = []
            for j in range(self.num_agents):
                if i == j: continue
                other_pos = states[j][0:3]
                other_vel = states[j][7:10]
                dist = np.linalg.norm(pos - other_pos)
                if dist < self.neighbor_dist:
                    vo_obstacles.append({'pos': other_pos, 'vel': other_vel, 'radius': self.radius * 2, 'type': 'agent'})
            
            for obs in obstacles:
                dist = np.linalg.norm(pos - obs['pos'])
                if dist < self.neighbor_dist:
                    obs_radius = obs['radius'] if obs['type'] == 'sphere' else np.max(obs['size'])/2
                    vo_obstacles.append({'pos': obs['pos'], 'vel': obs.get('vel', np.zeros(3)), 'radius': self.radius + obs_radius, 'type': 'static'})

            # 3. Sampling-based Velocity Search
            best_v = pref_vel.copy()
            min_penalty = float('inf')
            
            # Sample directions and speeds
            samples = [pref_vel, vel, np.zeros(3)]
            for s in [self.max_speed * 0.5, self.max_speed]:
                for phi in np.linspace(0, 2*np.pi, 8):
                    for theta in np.linspace(0, np.pi, 4):
                        vx = s * np.sin(theta) * np.cos(phi)
                        vy = s * np.sin(theta) * np.sin(phi)
                        vz = s * np.cos(theta)
                        samples.append(np.array([vx, vy, vz]))

            for v_cand in samples:
                coll_penalty = 0
                for obs in vo_obstacles:
                    v_rel = v_cand - (vel + obs['vel'])/2 if obs['type'] == 'agent' else v_cand - obs['vel']
                    relative_pos = obs['pos'] - pos
                    
                    # Time to collision
                    a = np.dot(v_rel, v_rel)
                    b = -2 * np.dot(relative_pos, v_rel)
                    c = np.dot(relative_pos, relative_pos) - obs['radius']**2
                    
                    discriminant = b**2 - 4*a*c
                    if discriminant > 0:
                        t = (b - np.sqrt(discriminant)) / (2*a + 1e-6)
                        if 0 < t < self.time_horizon:
                            coll_penalty += (self.time_horizon - t) * 15.0
                
                penalty = np.linalg.norm(v_cand - pref_vel) + coll_penalty
                if penalty < min_penalty:
                    min_penalty = penalty
                    best_v = v_cand

            # Map to discrete actions
            if best_v[2] > 0.5: actions.append(3)
            elif best_v[2] < -0.5: actions.append(4)
            else:
                target_yaw = np.arctan2(best_v[1], best_v[0])
                yaw_diff = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
                if yaw_diff > 0.3: actions.append(1)
                elif yaw_diff < -0.3: actions.append(2)
                else: actions.append(0)
                
        return actions
