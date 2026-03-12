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
