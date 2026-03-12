# envs/quadcopter_env.py
import numpy as np
from typing import Tuple, Dict, Any, List
from .dynamics import QuadcopterDynamics

class QuadcopterEnv:
    """
    Multi-agent quadcopter environment with 3D obstacles.
    """
    def __init__(self, num_agents: int = 3, config: Dict[str, Any] = None, render_mode: str = None, scenario: str = None):
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.scenario = scenario
        self.fig = None
        self.ax = None
        
        # Load scenario config if provided and no explicit config
        if scenario is not None and config is None:
            from .scenarios import get_scenario_config
            config = get_scenario_config(scenario)
            
        if config is None:
            config = {
                'arena_size': [100.0, 100.0, 40.0],
                'num_obstacles': 25,
                'rangefinder_max_range': 30.0,
                'collision_distance': 0.8,
                'goal_distance': 2.0,
                'dt': 0.01,
                'dynamic_ratio': 0.3
            }
        self.arena_size = np.array(config['arena_size'], dtype=np.float32)
        self.num_obstacles = config['num_obstacles']
        self.max_range = config['rangefinder_max_range']
        self.collision_dist = config['collision_distance']
        self.goal_dist = config['goal_distance']
        self.dt = config['dt']
        self.dynamic_ratio = config.get('dynamic_ratio', 0.3)
        
        self.arena_diagonal = np.linalg.norm(self.arena_size)
        
        self.agents = [QuadcopterDynamics(dt=self.dt) for _ in range(self.num_agents)]
        
        # Dynamic goals based on arena size
        self.goals = np.zeros((self.num_agents, 3), dtype=np.float32)
        self._update_goals()
        
        self.obstacles = []
        self.step_count = 0
        self.max_steps = 1000 # Increased for larger arena
    
    def _update_goals(self):
        """Updates goals based on arena size."""
        # Place goals on the opposite side of the arena
        for i in range(self.num_agents):
            self.goals[i] = [
                self.arena_size[0] - np.random.uniform(5, 15),
                np.random.uniform(5, self.arena_size[1] - 5),
                np.random.uniform(5, self.arena_size[2] - 5)
            ]

    def _generate_obstacles(self):
        """Generates random spherical and box obstacles, some dynamic."""
        if self.scenario in ['narrow_passage', 'city', 'forest', 'warzone']:
            from .scenarios import apply_scenario_custom_logic
            apply_scenario_custom_logic(self, self.scenario)
            return

        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < self.num_obstacles and attempts < 1000:
            attempts += 1
            pos = np.array([
                np.random.uniform(5.0, self.arena_size[0] - 5.0),
                np.random.uniform(5.0, self.arena_size[1] - 5.0),
                np.random.uniform(1.0, self.arena_size[2] - 1.0)
            ])
            
            obs_type = np.random.choice(['sphere', 'box'])
            is_dynamic = np.random.random() < self.dynamic_ratio
            vel = np.random.uniform(-1.0, 1.0, size=3) if is_dynamic else np.zeros(3)
            
            if obs_type == 'sphere':
                radius = np.random.uniform(0.5, 1.5)
                # Check distance to other obstacles
                valid = True
                for obs in self.obstacles:
                    if obs['type'] == 'sphere':
                        if np.linalg.norm(pos - obs['pos']) < (radius + obs['radius'] + 1.0):
                            valid = False; break
                    else:
                        if np.linalg.norm(pos - obs['pos']) < (radius + np.max(obs['size']) + 1.0):
                            valid = False; break
                if valid:
                    self.obstacles.append({'type': 'sphere', 'pos': pos, 'radius': radius, 'vel': vel, 'origin': pos.copy()})
            else:
                size = np.random.uniform(0.5, 2.0, size=3)
                valid = True
                for obs in self.obstacles:
                    if obs['type'] == 'sphere':
                        if np.linalg.norm(pos - obs['pos']) < (np.max(size) + obs['radius'] + 1.0):
                            valid = False; break
                    else:
                        if np.linalg.norm(pos - obs['pos']) < (np.max(size) + np.max(obs['size']) + 1.0):
                            valid = False; break
                if valid:
                    self.obstacles.append({'type': 'box', 'pos': pos, 'size': size, 'vel': vel, 'origin': pos.copy()})

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and returns initial observations."""
        self.step_count = 0
        self._generate_obstacles()
        self._update_goals() # Refresh goals for variety
        
        # Random start positions on the "left" side
        for i in range(self.num_agents):
            start_pos = np.array([
                np.random.uniform(2.0, 8.0),
                np.random.uniform(5.0, self.arena_size[1] - 5.0),
                np.random.uniform(2.0, self.arena_size[2] - 2.0)
            ])
            start_yaw = np.random.uniform(-np.pi, np.pi)
            self.agents[i].reset(start_pos, start_yaw)
            
        obs = self._get_observations()
        return obs, {}

    def _get_observations(self) -> np.ndarray:
        """Computes observations for all agents."""
        obs_all = []
        for i in range(self.num_agents):
            state = self.agents[i].state
            pos = state[0:3]
            yaw = state[5]
            goal = self.goals[i]
            
            # Rangefinder (5x5 grid)
            h_angles = np.deg2rad(np.array([-60, -30, 0, 30, 60]))
            v_angles = np.deg2rad(np.array([-30, -15, 0, 15, 30]))
            ranges = np.ones(25, dtype=np.float32) * self.max_range
            
            idx = 0
            for ha in h_angles:
                for va in v_angles:
                    # Ray direction in world frame
                    ray_yaw = yaw + ha
                    ray_pitch = va # simplified
                    dir_vec = np.array([
                        np.cos(ray_pitch) * np.cos(ray_yaw),
                        np.cos(ray_pitch) * np.sin(ray_yaw),
                        np.sin(ray_pitch)
                    ])
                    
                    min_dist = self.max_range
                    
                    # Check obstacles
                    for obs in self.obstacles:
                        if obs['type'] == 'sphere':
                            obs_pos, obs_r = obs['pos'], obs['radius']
                            oc = obs_pos - pos
                            t = np.dot(oc, dir_vec)
                            if t > 0:
                                proj = pos + t * dir_vec
                                dist_to_ray = np.linalg.norm(obs_pos - proj)
                                if dist_to_ray < obs_r:
                                    d = t - np.sqrt(obs_r**2 - dist_to_ray**2)
                                    if d > 0 and d < min_dist: min_dist = d
                        else: # Box
                            # Ray-AABB intersection
                            b_min = obs['pos'] - obs['size']/2
                            b_max = obs['pos'] + obs['size']/2
                            t1 = (b_min - pos) / (dir_vec + 1e-8)
                            t2 = (b_max - pos) / (dir_vec + 1e-8)
                            t_min = np.max(np.minimum(t1, t2))
                            t_max = np.min(np.maximum(t1, t2))
                            if t_max >= t_min and t_max > 0:
                                if t_min > 0 and t_min < min_dist: min_dist = t_min
                                    
                    # Check other agents
                    for j in range(self.num_agents):
                        if i == j: continue
                        other_pos = self.agents[j].state[0:3]
                        oc = other_pos - pos
                        t = np.dot(oc, dir_vec)
                        if t > 0:
                            proj = pos + t * dir_vec
                            dist_to_ray = np.linalg.norm(other_pos - proj)
                            if dist_to_ray < 0.3: # agent radius approx
                                d = t - np.sqrt(0.3**2 - dist_to_ray**2)
                                if d > 0 and d < min_dist:
                                    min_dist = d
                                    
                    # Check walls
                    for dim in range(3):
                        if dir_vec[dim] > 1e-6:
                            d = (self.arena_size[dim] - pos[dim]) / dir_vec[dim]
                            if d > 0 and d < min_dist: min_dist = d
                        elif dir_vec[dim] < -1e-6:
                            d = (0.0 - pos[dim]) / dir_vec[dim]
                            if d > 0 and d < min_dist: min_dist = d
                            
                    ranges[idx] = min_dist
                    idx += 1
                    
            ranges_norm = ranges / self.max_range
            
            # Goal info
            rel_pos = goal - pos
            dist_to_goal = np.linalg.norm(rel_pos)
            dist_norm = dist_to_goal / self.arena_diagonal
            
            theta_goal = np.arctan2(rel_pos[1], rel_pos[0]) - yaw
            sin_theta = np.sin(theta_goal)
            cos_theta = np.cos(theta_goal)
            
            phi_goal = np.arctan2(rel_pos[2], np.sqrt(rel_pos[0]**2 + rel_pos[1]**2))
            sin_phi = np.sin(phi_goal)
            
            obs = np.concatenate([ranges_norm, [dist_norm, sin_theta, cos_theta, sin_phi]])[:28]
            obs_all.append(obs)
            
        return np.array(obs_all, dtype=np.float32)

    def _get_min_distance(self, agent_idx: int) -> float:
        """Calculates minimum distance to any obstacle, wall, or other agent."""
        pos = self.agents[agent_idx].state[0:3]
        min_dist = float('inf')
        
        # Walls
        min_dist = min(min_dist, np.min(pos))
        min_dist = min(min_dist, np.min(self.arena_size - pos))
        
        # Obstacles
        for obs in self.obstacles:
            if obs['type'] == 'sphere':
                d = np.linalg.norm(pos - obs['pos']) - obs['radius']
            else:
                # Distance to box (simplified)
                d = np.linalg.norm(np.maximum(0, np.abs(pos - obs['pos']) - obs['size']/2))
            if d < min_dist: min_dist = d
            
        # Other agents
        for j in range(self.num_agents):
            if agent_idx == j: continue
            d = np.linalg.norm(pos - self.agents[j].state[0:3]) - 0.3 # agent radius
            if d < min_dist: min_dist = d
            
        return min_dist

    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """Executes one environment step."""
        self.step_count += 1
        
        # Update dynamic obstacles
        for obs in self.obstacles:
            if np.any(obs['vel'] != 0):
                # Oscillate around origin
                obs['pos'] = obs['origin'] + obs['vel'] * np.sin(self.step_count * 0.05)
        
        # Action mapping (Increased speeds for real-life feel)
        action_map = {
            0: [10.0, 0.0, 0.0, 0.0], # Forward (Extreme)
            1: [5.0, 0.0, 0.0, 90.0],  # Forward + Yaw Right
            2: [5.0, 0.0, 0.0, -90.0], # Forward + Yaw Left
            3: [5.0, 0.0, 5.0, 0.0],   # Forward + Up
            4: [5.0, 0.0, -5.0, 0.0],  # Forward + Down
            5: [0.0, 0.0, 0.0, 0.0]    # Hover
        }
        
        # Apply actions
        for i in range(self.num_agents):
            cmd = np.array(action_map[actions[i]], dtype=np.float32)
            # Transform vx, vy to world frame based on yaw
            yaw = self.agents[i].state[5]
            vx_world = cmd[0] * np.cos(yaw) - cmd[1] * np.sin(yaw)
            vy_world = cmd[0] * np.sin(yaw) + cmd[1] * np.cos(yaw)
            world_cmd = np.array([vx_world, vy_world, cmd[2], cmd[3]])
            self.agents[i].step(world_cmd)
            
        obs = self._get_observations()
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        info = {'success': False, 'collision': False}
        
        success_count = 0
        
        for i in range(self.num_agents):
            pos = self.agents[i].state[0:3]
            goal = self.goals[i]
            dist_to_goal = np.linalg.norm(pos - goal)
            d_min = self._get_min_distance(i)
            
            # Rewards
            r_transfer = -dist_to_goal * 0.1
            r_collision = -100.0 * np.exp(-2.0 * max(d_min, 0.0))
            r_free_space = 0.5 if d_min > 2.0 else 0.0
            r_step = -0.01
            
            rewards[i] = r_transfer + r_collision + r_free_space + r_step
            
            if dist_to_goal < self.goal_dist:
                success_count += 1
                
            if d_min < self.collision_dist:
                terminated = True
                info['collision'] = True
                
        if success_count == self.num_agents:
            terminated = True
            info['success'] = True
            
        self.last_info = info
        return obs, rewards, terminated, truncated, info

    def render(self):
        """Renders the 3D environment using matplotlib."""
        if self.render_mode != 'human':
            return
            
        import matplotlib.pyplot as plt
        
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
        self.ax.clear()
        
        # Set limits
        self.ax.set_xlim(0, self.arena_size[0])
        self.ax.set_ylim(0, self.arena_size[1])
        self.ax.set_zlim(0, self.arena_size[2])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # Status indicators
        status_text = f'Step: {self.step_count}'
        if hasattr(self, 'last_info'):
            if self.last_info.get('collision'):
                status_text += " | COLLISION! 💥"
                self.ax.set_facecolor((1.0, 0.9, 0.9)) # Light red background
            elif self.last_info.get('success'):
                status_text += " | SUCCESS! 🎯"
                self.ax.set_facecolor((0.9, 1.0, 0.9)) # Light green background
            else:
                self.ax.set_facecolor('white')
        
        self.ax.set_title(status_text)
        
        # Draw obstacles (Better 3D representation)
        for obs in self.obstacles:
            p = obs['pos']
            alpha = obs.get('alpha', 0.6)
            if obs['type'] == 'sphere':
                self.ax.scatter(p[0], p[1], p[2], color=obs.get('color', 'gray'), s=(obs['radius']*10)**2, alpha=alpha)
            else:
                s = obs['size']
                # Use bar3d for real 3D boxes
                color = obs.get('color', 'gray')
                self.ax.bar3d(p[0]-s[0]/2, p[1]-s[1]/2, p[2]-s[2]/2, s[0], s[1], s[2], 
                             color=color, alpha=alpha)
            
        # Draw goals and agents
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
        for i in range(self.num_agents):
            c = colors[i % len(colors)]
            goal = self.goals[i]
            self.ax.scatter(goal[0], goal[1], goal[2], color=c, marker='x', s=100, label=f'Goal {i}')
            
            pos = self.agents[i].state[0:3]
            yaw = self.agents[i].state[5]
            
            # Draw UAV frame (cross)
            arm_len = 0.5
            dx = arm_len * np.cos(yaw)
            dy = arm_len * np.sin(yaw)
            self.ax.plot([pos[0]-dx, pos[0]+dx], [pos[1]-dy, pos[1]+dy], [pos[2], pos[2]], color=c, linewidth=2)
            self.ax.plot([pos[0]+dy, pos[0]-dy], [pos[1]-dx, pos[1]+dx], [pos[2], pos[2]], color=c, linewidth=2)
            self.ax.scatter(pos[0], pos[1], pos[2], color=c, marker='o', s=30)
            
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.draw()
        plt.pause(0.001)

    def close(self):
        """Closes the rendering window."""
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None
