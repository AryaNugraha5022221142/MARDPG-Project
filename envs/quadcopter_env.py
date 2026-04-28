# envs/quadcopter_env.py
import numpy as np
from typing import Tuple, Dict, Any, List
from .dynamics import QuadcopterDynamics
from .lqr_controller import PerAxisLQR

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
        
        # Load scenario config if provided
        if scenario is not None:
            from .scenarios import get_scenario_config
            scenario_config = get_scenario_config(scenario)
            if config is None:
                config = scenario_config
            else:
                # Merge scenario specific keys into config
                for k, v in scenario_config.items():
                    config[k] = v
            
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
        self.M = config.get('inner_steps', 10)
        self.cooperative = config.get('cooperative', False)
        self.rate_limit_per_step = config.get('rate_limit_per_step', 0.5)
        
        self.arena_diagonal = np.linalg.norm(self.arena_size)
        
        self.agents = [QuadcopterDynamics(dt=self.dt) for _ in range(self.num_agents)]
        self.lqr = PerAxisLQR(dt=self.dt)
        
        # Dynamic goals based on arena size
        self.goals = np.zeros((self.num_agents, 3), dtype=np.float32)
        self._update_goals()
        
        self.obstacles = []
        self.step_count = 0
        self.max_steps = config.get('max_steps', 2000)
        self.prev_dist_to_goal = np.zeros(self.num_agents, dtype=np.float32)
        self.agent_dones = np.zeros(self.num_agents, dtype=bool)
        self.prev_actions = np.zeros((self.num_agents, 4), dtype=np.float32)
        
        # Ablation options
        self.sensor_noise_std = config.get('sensor_noise_std', 0.02) # 2% default
        self.reward_type = config.get('reward_type', 'exponential') # 'linear' or 'exponential'
        
        # Reward weights from config
        self.reward_config = config.get('rewards', {
            'weights': {
                'transfer': 1.0, 
                'collision': 25.0, 
                'smoothness': 0.01, 
                'step_penalty': 0.1,
                'free_space': 0.05
            },
            'goal_bonus': 200.0,
            'collision_penalty': -50.0
        })
    
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
        if self.scenario in ['narrow_passage', 'city', 'forest', 'warzone', 'urban_canyon', 'search_and_rescue', 'dynamic_intercept']:
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
            phase = np.random.uniform(0, 2 * np.pi) if is_dynamic else 0.0
            freq = np.random.uniform(0.02, 0.08) if is_dynamic else 0.0
            
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
                    self.obstacles.append({
                        'type': 'sphere', 'pos': pos, 'radius': radius, 
                        'vel': vel, 'origin': pos.copy(), 'phase': phase, 'freq': freq
                    })
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
                    self.obstacles.append({
                        'type': 'box', 'pos': pos, 'size': size, 
                        'vel': vel, 'origin': pos.copy(), 'phase': phase, 'freq': freq
                    })

    def set_curriculum_level(self, level: int):
        """
        Updates environment difficulty based on curriculum level.
        Level 0: 30x30x15, 0 obstacles, 0% dynamic
        Level 1: 50x50x20, 5 obstacles, 0% dynamic
        Level 2: 75x75x30, 15 obstacles, 10% dynamic
        Level 3: 100x100x40, 25 obstacles, 20% dynamic
        """
        if level == 0:
            self.arena_size = np.array([30, 30, 15], dtype=np.float32)
            self.num_obstacles = 0
            self.dynamic_ratio = 0.0
        elif level == 1:
            self.arena_size = np.array([50, 50, 20], dtype=np.float32)
            self.num_obstacles = 5
            self.dynamic_ratio = 0.0
        elif level == 2:
            self.arena_size = np.array([75, 75, 30], dtype=np.float32)
            self.num_obstacles = 15
            self.dynamic_ratio = 0.1
        else:
            self.arena_size = np.array([100, 100, 40], dtype=np.float32)
            self.num_obstacles = 15 + (level - 2) * 10
            self.dynamic_ratio = min(0.1 + (level - 2) * 0.1, 0.5)
            
        self.arena_diagonal = float(np.linalg.norm(self.arena_size))
        print(f"[DEBUG] quadcopter_env initialized with arena_size: {self.arena_size}, scenario: {self.scenario}")
        self._generate_obstacles()
        print(f"Curriculum Level Updated: {level} (Arena: {self.arena_size}, Obstacles: {self.num_obstacles}, Dynamic: {self.dynamic_ratio:.1f})")

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and returns initial observations."""
        if seed is not None:
            np.random.seed(seed)
            
        self.step_count = 0
        self._episode_collision = False
        self.agent_dones = np.zeros(self.num_agents, dtype=bool)
        self._generate_obstacles()
        
        if self.scenario == 'search_and_rescue':
            self.targets_claimed = set()
            self.sar_targets = [
                np.array([
                    np.random.uniform(5, self.arena_size[0] - 5),
                    np.random.uniform(5, self.arena_size[1] - 5),
                    np.random.uniform(1, 5)   # low altitude — on the ground
                ])
                for _ in range(self.reward_config.get('num_targets', 6))
            ]
            # In SAR mode, goals are the nearest unclaimed target, dynamically updated
            self._update_sar_goals()
        else:
            self._update_goals() # Refresh goals for variety
        
        # Reset metrics
        self.total_jerk = np.zeros(self.num_agents, dtype=np.float32)
        self.safety_frontier = np.ones(self.num_agents, dtype=np.float32) * float('inf')
        self.prev_accel = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_vel = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_actions = np.zeros((self.num_agents, 4), dtype=np.float32)
        
        # Random start positions on the "left" side
        for i in range(self.num_agents):
            start_pos = np.array([
                np.random.uniform(4.0, 8.0),
                np.random.uniform(8.0, self.arena_size[1] - 8.0),
                np.random.uniform(4.0, self.arena_size[2] - 4.0)
            ])
            goal = self.goals[i]
            rough_dir = np.arctan2(goal[1] - start_pos[1], goal[0] - start_pos[0])
            start_yaw = rough_dir + np.random.uniform(-np.pi/4, np.pi/4)  # ±45° random
            self.agents[i].reset(start_pos, start_yaw)
            self.prev_dist_to_goal[i] = np.linalg.norm(start_pos - self.goals[i])
            
        obs = self._get_observations()
        return obs, {}

    def _update_sar_goals(self):
        """Assign each agent to its nearest unclaimed target."""
        for i in range(self.num_agents):
            pos = self.agents[i].state[0:3]
            unclaimed = [
                (t, np.linalg.norm(pos - t))
                for j, t in enumerate(self.sar_targets)
                if j not in self.targets_claimed
            ]
            if unclaimed:
                # Sort by distance and pick nearest
                self.goals[i] = min(unclaimed, key=lambda x: x[1])[0]
            else:
                # All targets claimed, stay at current position or go to a default
                self.goals[i] = pos.copy()

    def _get_observations(self) -> np.ndarray:
        """Computes observations for all agents using vectorized rangefinder."""
        obs_all = []
        
        # Precompute ray directions for all agents
        h_angles = np.deg2rad(np.array([-60, -30, 0, 30, 60]))
        v_angles = np.deg2rad(np.array([-30, -15, 0, 15, 30]))
        
        # Create a grid of angles: (25, 2)
        ha_grid, va_grid = np.meshgrid(h_angles, v_angles)
        ha_flat = ha_grid.flatten()
        va_flat = va_grid.flatten()
        
        # Obstacle data for vectorization
        sphere_pos = np.array([o['pos'] for o in self.obstacles if o['type'] == 'sphere'])
        sphere_rad = np.array([o['radius'] for o in self.obstacles if o['type'] == 'sphere'])
        
        box_min = np.array([o['pos'] - o['size']/2 for o in self.obstacles if o['type'] == 'box'])
        box_max = np.array([o['pos'] + o['size']/2 for o in self.obstacles if o['type'] == 'box'])
        
        for i in range(self.num_agents):
            state = self.agents[i].state
            pos = state[0:3]
            yaw = state[5]
            goal = self.goals[i]
            
            # 1. Vectorized Rangefinder
            # Ray directions in world frame: (25, 3)
            ray_yaws = yaw + ha_flat
            ray_pitches = va_flat
            
            dir_vecs = np.stack([
                np.cos(ray_pitches) * np.cos(ray_yaws),
                np.cos(ray_pitches) * np.sin(ray_yaws),
                np.sin(ray_pitches)
            ], axis=1) # (25, 3)
            
            ranges = np.ones(25, dtype=np.float32) * self.max_range
            
            # Sphere intersections (Vectorized)
            if len(sphere_pos) > 0:
                # oc: (N_spheres, 3)
                oc = pos - sphere_pos
                # a = 1.0 (unit vectors)
                # b: (25, N_spheres)
                b = 2.0 * np.dot(dir_vecs, oc.T)
                # c: (N_spheres,)
                c = np.sum(oc**2, axis=1) - sphere_rad**2
                
                # discriminant: (25, N_spheres)
                discriminant = b**2 - 4.0 * c # a=1
                
                mask = discriminant >= 0
                sqrt_disc = np.sqrt(np.maximum(0, discriminant))
                t1 = (-b - sqrt_disc) / 2.0
                t2 = (-b + sqrt_disc) / 2.0
                
                # We want the smallest positive t
                t_hits = np.where((t1 > 1e-4) & mask, t1, np.where((t2 > 1e-4) & mask, t2, self.max_range))
                ranges = np.minimum(ranges, np.min(t_hits, axis=1))
                
            # Box intersections (AABB) - Vectorized
            if len(box_min) > 0:
                # dir_vecs: (25, 3), box_min: (N_boxes, 3)
                # We need to broadcast to (25, N_boxes, 3)
                inv_dir = 1.0 / (dir_vecs[:, np.newaxis, :] + 1e-8)
                t1 = (box_min[np.newaxis, :, :] - pos) * inv_dir
                t2 = (box_max[np.newaxis, :, :] - pos) * inv_dir
                
                t_min = np.max(np.minimum(t1, t2), axis=2) # (25, N_boxes)
                t_max = np.min(np.maximum(t1, t2), axis=2) # (25, N_boxes)
                
                mask = (t_max >= t_min) & (t_max > 0)
                t_hits = np.where(mask & (t_min > 0), t_min, self.max_range)
                ranges = np.minimum(ranges, np.min(t_hits, axis=1))
                
            # Other agents (Vectorized)
            other_indices = [j for j in range(self.num_agents) if i != j and not self.agent_dones[j]]
            if other_indices:
                other_pos = np.array([self.agents[j].state[0:3] for j in other_indices])
                oc = other_pos - pos # (N_others, 3)
                t = np.dot(dir_vecs, oc.T) # (25, N_others)
                
                # Projection: pos + t * dir_vec
                # dist_to_ray: (25, N_others)
                # This part is a bit tricky to vectorize fully without large memory
                # But N_others is small (usually < 10)
                for idx, j_idx in enumerate(other_indices):
                    oc_j = oc[idx]
                    t_j = t[:, idx]
                    # dist_to_ray_sq = |oc_j|^2 - t_j^2
                    dist_sq = np.sum(oc_j**2) - t_j**2
                    mask = (t_j > 0) & (dist_sq < 0.3**2)
                    d = t_j - np.sqrt(np.maximum(0, 0.3**2 - dist_sq))
                    ranges = np.where(mask & (d < ranges), d, ranges)
                    
            # Walls (Vectorized)
            # dir_vecs: (25, 3), pos: (3,)
            for dim in range(3):
                # Positive direction
                mask_p = dir_vecs[:, dim] > 1e-6
                tp = (self.arena_size[dim] - pos[dim]) / (dir_vecs[:, dim] + 1e-8)
                ranges = np.where(mask_p & (tp < ranges), tp, ranges)
                
                # Negative direction
                mask_n = dir_vecs[:, dim] < -1e-6
                tn = -pos[dim] / (dir_vecs[:, dim] - 1e-8)
                ranges = np.where(mask_n & (tn < ranges), tn, ranges)
            
            # Apply noise and clip
            sensor_noise = np.random.normal(0, self.sensor_noise_std * self.max_range, 25)
            ranges = np.clip(ranges + sensor_noise, 0, self.max_range)
            ranges_norm = ranges / self.max_range
            
            # Goal info
            rel_pos = goal - pos
            dist_to_goal = np.linalg.norm(rel_pos)
            dist_norm = dist_to_goal / self.arena_diagonal
            theta_goal = np.arctan2(rel_pos[1], rel_pos[0]) - yaw
            phi_goal = np.arctan2(rel_pos[2], np.sqrt(rel_pos[0]**2 + rel_pos[1]**2))
            
            vel_norm = np.clip(self.agents[i].state[6:9] / 5.0, -1.0, 1.0)
            
            # Saturation indicator (Bug 14)
            is_saturated = float(getattr(self.agents[i], 'is_saturated', False))
            
            obs = np.concatenate([
                ranges_norm, 
                [dist_norm, np.sin(theta_goal), np.cos(theta_goal), np.sin(phi_goal), np.cos(phi_goal)],
                vel_norm,
                [is_saturated]
            ])
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

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """
        Executes one environment step with per-agent termination.
        actions: (num_agents, 4) in range [-1, 1]
        """
        self.step_count += 1
        
        # Update dynamic obstacles
        for obs in self.obstacles:
            if obs.get('is_interceptor', False):
                # Find nearest active agent
                nearest_dist = float('inf')
                nearest_pos = None
                for j in range(self.num_agents):
                    if self.agent_dones[j]: continue
                    d = np.linalg.norm(self.agents[j].state[0:3] - obs['pos'])
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_pos = self.agents[j].state[0:3].copy()
                if nearest_pos is not None and nearest_dist > 0.1:
                    direction = (nearest_pos - obs['pos']) / nearest_dist
                    obs['pos'] = obs['pos'] + direction * 1.5 * (self.dt * self.M)
            elif np.any(obs['vel'] != 0):
                phase = obs.get('phase', 0.0)
                freq = obs.get('freq', 0.05)
                obs['pos'] = obs['origin'] + obs['vel'] * np.sin(self.step_count * freq + phase)
        
        # Action scaling
        jerk_arr = np.zeros(self.num_agents, dtype=np.float32)
        r_smooth_arr = np.zeros(self.num_agents, dtype=np.float32)
        
        # Apply actions only to active agents
        for i in range(self.num_agents):
            if self.agent_dones[i]: continue
            
            action = actions[i]
            
            # Rate-of-change limiting (Bug 13)
            delta = np.abs(action - self.prev_actions[i])
            if np.any(delta > self.rate_limit_per_step):
                action = self.prev_actions[i] + np.clip(action - self.prev_actions[i], -self.rate_limit_per_step, self.rate_limit_per_step)
            
            r_smooth_arr[i] = -self.reward_config.get('weights', {}).get('smoothness', 0.01) * np.sum((action - self.prev_actions[i])**2)
            self.prev_actions[i] = action.copy()
            
            # Network already outputs in [-3.5, 3.5]; do NOT rescale again.
            vx_cmd, vy_cmd, vz_cmd = action[0], action[1], action[2]
            # Yaw: network outputs [-3.5, 3.5]; scale so max = pi/4 rad/s ~ 45 deg/s:
            yaw_rate_cmd = action[3] * (45.0 / 3.5)     # stays under thesis bound
            
            yaw = self.agents[i].state[5]
            vx_world = vx_cmd * np.cos(yaw) - vy_cmd * np.sin(yaw)
            vy_world = vx_cmd * np.sin(yaw) + vy_cmd * np.cos(yaw)
            world_cmd = np.array([vx_world, vy_world, vz_cmd, yaw_rate_cmd])
            
            old_vel = self.agents[i].state[6:9].copy()
            self.agents[i].rl_step(world_cmd, self.lqr, M=self.M)
            new_vel = self.agents[i].state[6:9]
            
            accel = (new_vel - old_vel) / (self.dt * self.M)
            jerk_val = np.linalg.norm(accel - self.prev_accel[i]) / (self.dt * self.M)
            self.total_jerk[i] += jerk_val
            self.prev_accel[i] = accel.copy()
            jerk_arr[i] = jerk_val
            
        obs = self._get_observations()
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        info = {'success': False, 'collision': getattr(self, '_episode_collision', False)}
        info['agent_terminated_now'] = np.zeros(self.num_agents, dtype=bool)
        
        sat_rates = [1.0 if getattr(self.agents[i], 'is_saturated', False) else 0.0 for i in range(self.num_agents)]
        info['sat_rate'] = np.mean(sat_rates)
        
        for i in range(self.num_agents):
            if self.agent_dones[i]: continue
            
            pos = self.agents[i].state[0:3]
            goal = self.goals[i]
            dist_to_goal = np.linalg.norm(pos - goal)
            d_min = self._get_min_distance(i)
            
            # Action penalty
            a_t = actions[i]
            a_prev = self.prev_actions[i]
            action_penalty = -0.01 * np.sum((a_t - a_prev)**2)

            r_transfer = 3.0 * (self.prev_dist_to_goal[i] - dist_to_goal)
            self.prev_dist_to_goal[i] = dist_to_goal
            
            collision_penalty = -5.0 * max(0.0, 1.5 - d_min)**2
            step_penalty = -0.02
            
            dense_r = r_transfer + action_penalty + collision_penalty + step_penalty
            
            terminal_bonus = 0.0
            
            if self.scenario == 'search_and_rescue':
                target_radius = self.reward_config.get('target_radius', 0.5)
                for j, t_pos in enumerate(self.sar_targets):
                    if j not in self.targets_claimed and np.linalg.norm(pos - t_pos) < target_radius:
                        self.targets_claimed.add(j)
                        terminal_bonus += 50.0  # SAR bonus
                        self._update_sar_goals()
                        break
            
            # Termination checks
            if dist_to_goal < self.goal_dist:
                if self.scenario != 'search_and_rescue':
                    self.agent_dones[i] = True
                    info['agent_terminated_now'][i] = True
                    terminal_bonus += self.reward_config.get('goal_bonus', 50.0)
            
            # Collision check
            if not self.agent_dones[i] and d_min < self.collision_dist:
                self.agent_dones[i] = True
                self._episode_collision = True
                info['agent_terminated_now'][i] = True
                info['collision'] = True
                penalty = self.reward_config.get('collision_penalty', -20.0)
                penalty_scale = 2.0
                penalty *= penalty_scale
                terminal_bonus += penalty
            
            rewards[i] = dense_r + terminal_bonus
            self.safety_frontier[i] = min(self.safety_frontier[i], d_min)

        info['agent_dones'] = self.agent_dones.copy()

        if self.cooperative:
            team_r = np.sum(rewards) / self.num_agents
            rewards[:] = team_r

        # Global termination
        terminated = np.all(self.agent_dones)
        if self.scenario == 'search_and_rescue' and len(self.targets_claimed) == len(self.sar_targets):
            terminated = True
            info['success'] = True
            info['individual_success_rate'] = len(self.targets_claimed) / max(1, len(self.sar_targets))
        else:
            individual_successes = sum(1 for i in range(self.num_agents) 
                                      if self.agent_dones[i] and 
                                      np.linalg.norm(self.agents[i].state[0:3] - self.goals[i]) < self.goal_dist)
            info['individual_success_rate'] = individual_successes / self.num_agents
            info['success'] = (individual_successes >= 1)
            
        truncated = self.step_count >= self.max_steps
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
        self.ax.set_box_aspect((self.arena_size[0], self.arena_size[1], self.arena_size[2]))
        
        # Plot ground plane to anchor visual
        xx, yy = np.meshgrid([0, self.arena_size[0]], [0, self.arena_size[1]])
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, color='green', alpha=0.1)

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
