import numpy as np
from typing import Tuple, Dict, Any, List
import math

from .quadcopter_env import QuadcopterEnv
from .kinematic_dynamics import KinematicDynamics

class QuadcopterKinematicEnv(QuadcopterEnv):
    """
    QuadcopterKinematicEnv implements the proper kinematic dynamics
    with the MARDPG paper's observation construction and rewards.
    """
    def __init__(self, num_agents: int = 3, config: Dict[str, Any] = None, render_mode: str = None, scenario: str = None):
        super().__init__(num_agents, config, render_mode, scenario)
        
        # Override agents to use KinematicDynamics
        self.agents = [KinematicDynamics(dt=self.dt, v=2.0) for _ in range(self.num_agents)]
        
        # Observation dimension: 25 rangefinders, 2 angles ([ϑ, ϕ]), 3 dist vector -> 30
        self.obs_dim = 30
        
        # Ensure the collision parameters match the paper's default
        if 'rewards' not in self.config:
            self.config['rewards'] = {}
        self.lambda_ = self.config['rewards'].get('collision_lambda', 5.0)
        self.sigma_ = self.config['rewards'].get('collision_sigma', 15.0)

    def set_scene_type(self, scene: str):
        """Randomly selects from the four obstacle types per episode."""
        self.scenario = scene

    def _generate_obstacles(self):
        """Generates random spherical, box, cylinder and irregular obstacles."""
        if self.scenario is None:
            # randomly select from the four obstacle types per episode
            obs_type_choice = np.random.choice(['sphere', 'box', 'cylinder', 'irregular'])
            self.scenario_obs_type = obs_type_choice
        else:
            scene_map = {'pillars': 'box', 'cylinders': 'cylinder', 'forest': 'cylinder', 'rings': 'irregular'}
            self.scenario_obs_type = scene_map.get(self.scenario, 'sphere')

        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < self.num_obstacles and attempts < 1000:
            attempts += 1
            pos = np.array([
                np.random.uniform(5.0, self.arena_size[0] - 5.0),
                np.random.uniform(5.0, self.arena_size[1] - 5.0),
                np.random.uniform(1.0, self.arena_size[2] - 1.0)
            ])
            
            vel = np.zeros(3)
            phase = 0.0
            freq = 0.0
            
            if self.scenario_obs_type == 'sphere':
                radius = np.random.uniform(0.5, 1.5)
                valid = True
                for obs in self.obstacles:
                    o_r = np.max(obs.get('size', [obs.get('radius', 1.0)]))
                    if np.linalg.norm(pos - obs['pos']) < (radius + o_r + 1.0):
                        valid = False; break
                if valid:
                    self.obstacles.append({
                        'type': 'sphere', 'pos': pos, 'radius': radius, 
                        'vel': vel, 'origin': pos.copy(), 'phase': phase, 'freq': freq
                    })
            elif self.scenario_obs_type == 'box':
                size = np.random.uniform(0.5, 2.0, size=3)
                if self.scenario == 'pillars':
                    h = np.random.uniform(10.0, self.arena_size[2])
                    size[2] = h
                    pos[2] = h / 2.0
                valid = True
                for obs in self.obstacles:
                    o_r = np.max(obs.get('size', [obs.get('radius', 1.0)]))
                    if np.linalg.norm(pos - obs['pos']) < (np.max(size) + o_r + 1.0):
                        valid = False; break
                if valid:
                    self.obstacles.append({
                        'type': 'box', 'pos': pos, 'size': size, 
                        'vel': vel, 'origin': pos.copy(), 'phase': phase, 'freq': freq
                    })
            elif self.scenario_obs_type == 'cylinder':
                radius = np.random.uniform(0.5, 1.5)
                h = np.random.uniform(3.0, 10.0)
                pos[2] = h / 2.0
                size = np.array([radius*2, radius*2, h])
                valid = True
                for obs in self.obstacles:
                    o_r = np.max(obs.get('size', [obs.get('radius', 1.0)]))
                    if np.linalg.norm(pos - obs['pos']) < (np.max(size) + o_r + 1.0):
                        valid = False; break
                if valid:
                    self.obstacles.append({
                        'type': 'cylinder', 'pos': pos, 'size': size, # Treat as box for physics simplification
                        'vel': vel, 'origin': pos.copy(), 'phase': phase, 'freq': freq
                    })
            elif self.scenario_obs_type == 'irregular':
                size = np.random.uniform(0.8, 2.5, size=3)
                valid = True
                for obs in self.obstacles:
                    o_r = np.max(obs.get('size', [obs.get('radius', 1.0)]))
                    if np.linalg.norm(pos - obs['pos']) < (np.max(size) + o_r + 1.0):
                        valid = False; break
                if valid:
                    self.obstacles.append({
                        'type': 'sphere', 'pos': pos, 'radius': np.max(size)*0.8, 
                        'vel': vel, 'origin': pos.copy(), 'phase': phase, 'freq': freq
                    })

    def _update_dynamic_obstacles(self):
        for obs in self.obstacles:
            vel = np.asarray(obs.get('vel', np.zeros(3)), dtype=float)
            if not np.any(vel != 0):
                continue

            origin = np.asarray(obs.get('origin', obs['pos']), dtype=float)
            phase = float(obs.get('phase', 0.0))
            freq = float(obs.get('freq', 0.05))
            obs['pos'] = origin + vel * np.sin(self.step_count * freq + phase)

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        self._episode_collision = False
        self.agent_dones = np.zeros(self.num_agents, dtype=bool)
        
        self.targets_claimed = set()
        self.total_jerk = np.zeros(self.num_agents, dtype=np.float32)
        self.safety_frontier = np.ones(self.num_agents, dtype=np.float32) * float('inf')
        self.prev_accel = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.prev_vel = np.zeros((self.num_agents, 3), dtype=np.float32)
        
        self._generate_obstacles()
        self._update_goals()
        
        for i in range(self.num_agents):
            start_pos = np.array([
                np.random.uniform(4.0, 8.0),
                np.random.uniform(8.0, self.arena_size[1] - 8.0),
                np.random.uniform(4.0, self.arena_size[2] - 4.0)
            ])
            goal = self.goals[i]
            rough_dir = np.arctan2(goal[1] - start_pos[1], goal[0] - start_pos[0])
            start_yaw = rough_dir + np.random.uniform(-np.pi/4, np.pi/4)
            self.agents[i].reset(start_pos, start_yaw)
            self.prev_dist_to_goal[i] = np.linalg.norm(start_pos - self.goals[i])
            self.prev_vel[i] = self.agents[i].state[6:9].copy()
            
        obs = self._get_observations()
        return obs, {}

    def _get_observations(self) -> np.ndarray:
        obs_all = []
        
        h_angles = np.deg2rad(np.array([-60, -30, 0, 30, 60]))
        v_angles = np.deg2rad(np.array([-30, -15, 0, 15, 30]))
        ha_grid, va_grid = np.meshgrid(h_angles, v_angles)
        ha_flat = ha_grid.flatten()
        va_flat = va_grid.flatten()
        
        sphere_pos = np.array([o['pos'] for o in self.obstacles if o['type'] == 'sphere'])
        sphere_rad = np.array([o['radius'] for o in self.obstacles if o['type'] == 'sphere'])
        
        box_min = np.array([o['pos'] - o['size']/2 for o in self.obstacles if o['type'] in ('box', 'cylinder')])
        box_max = np.array([o['pos'] + o['size']/2 for o in self.obstacles if o['type'] in ('box', 'cylinder')])
        
        for i in range(self.num_agents):
            state = self.agents[i].state
            pos = state[0:3]
            pitch = state[4]
            yaw = state[5] + np.random.normal(0, getattr(self, 'yaw_noise_std', 0.0))
            
            goal = self.goals[i]
            
            ray_yaws = yaw + ha_flat
            ray_pitches = pitch + va_flat
            
            dir_vecs = np.stack([
                np.cos(ray_pitches) * np.cos(ray_yaws),
                np.cos(ray_pitches) * np.sin(ray_yaws),
                np.sin(ray_pitches)
            ], axis=1)
            
            ranges = np.ones(25, dtype=np.float32) * self.max_range
            
            if len(sphere_pos) > 0:
                oc = pos - sphere_pos
                b = 2.0 * np.dot(dir_vecs, oc.T)
                c = np.sum(oc**2, axis=1) - sphere_rad**2
                discriminant = b**2 - 4.0 * c
                mask = discriminant >= 0
                sqrt_disc = np.sqrt(np.maximum(0, discriminant))
                t1 = (-b - sqrt_disc) / 2.0
                t2 = (-b + sqrt_disc) / 2.0
                t_hits = np.where((t1 > 1e-4) & mask, t1, np.where((t2 > 1e-4) & mask, t2, self.max_range))
                ranges = np.minimum(ranges, np.min(t_hits, axis=1))
                
            if len(box_min) > 0:
                inv_dir = 1.0 / (dir_vecs[:, np.newaxis, :] + 1e-8)
                t1 = (box_min[np.newaxis, :, :] - pos) * inv_dir
                t2 = (box_max[np.newaxis, :, :] - pos) * inv_dir
                t_min = np.max(np.minimum(t1, t2), axis=2)
                t_max = np.min(np.maximum(t1, t2), axis=2)
                mask = (t_max >= t_min) & (t_max > 0)
                t_hits = np.where(mask & (t_min > 0), t_min, self.max_range)
                ranges = np.minimum(ranges, np.min(t_hits, axis=1))
            
            ranges_norm = ranges / self.max_range
            ranges_norm = ranges_norm + np.random.normal(0, getattr(self, 'sensor_noise_std', 0.0), 25)
            ranges_norm = np.clip(ranges_norm, 0.0, 1.0)
            
            # Using dynamic diagonal normalization to prevent spatial observation drift
            arena_diagonal = np.linalg.norm(self.arena_size)
            goal_dist = np.linalg.norm(goal - pos) / arena_diagonal
            dx, dy, dz = goal - pos
            goal_h_angle = (np.arctan2(dy, dx) - yaw + np.pi) % (2*np.pi) - np.pi
            goal_v_angle = (np.arctan2(dz, np.sqrt(dx**2+dy**2)) - pitch + np.pi) % (2*np.pi) - np.pi
            
            # [ [ϕ, ϑ], 25 rangefinders, target ξ ]
            obs = np.concatenate([
                [pitch / (np.pi/2), yaw / np.pi],
                ranges_norm,
                [goal_dist, goal_h_angle / np.pi, goal_v_angle / np.pi]
            ])
            obs_all.append(obs)
            
        return np.array(obs_all, dtype=np.float32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        self._update_dynamic_obstacles()
        
        info = {'success': False, 'collision': getattr(self, '_episode_collision', False)}
        info['agent_terminated_now'] = np.zeros(self.num_agents, dtype=bool)
        info['agent_success'] = np.zeros(self.num_agents, dtype=bool)
        info['agent_collision'] = np.zeros(self.num_agents, dtype=bool)
        
        action_smoothness = []
        min_distances = []
        
        # Step 1: Apply actions and physics
        for i in range(self.num_agents):
            if self.agent_dones[i]: continue
            
            action = np.asarray(actions[i], dtype=np.float32).copy()
            delta = action - self.prev_actions[i]
            
            rate_limited_action = self.prev_actions[i] + np.clip(
                delta, 
                -self.rate_limit_per_step, 
                +self.rate_limit_per_step
            )
            action = np.clip(rate_limited_action, -self.action_bound, self.action_bound)

            action_smoothness.append(np.mean(np.abs(action - self.prev_actions[i])))
            self.prev_actions[i] = action.copy()
            
            old_vel = self.prev_vel[i].copy()
            self.agents[i].rl_step(action, M=self.M)
            new_vel = self.agents[i].state[6:9].copy()
            
            accel = (new_vel - old_vel) / self.dt
            jerk = (accel - self.prev_accel[i]) / self.dt
            self.total_jerk[i] += np.linalg.norm(jerk) * self.dt
            self.prev_accel[i] = accel.copy()
            self.prev_vel[i] = new_vel.copy()
            
            # Bound inside arena
            self.agents[i].state[0] = np.clip(self.agents[i].state[0], 0.0, self.arena_size[0])
            self.agents[i].state[1] = np.clip(self.agents[i].state[1], 0.0, self.arena_size[1])
            self.agents[i].state[2] = np.clip(self.agents[i].state[2], 0.0, self.arena_size[2])
            
        # Step 2: Get observations from the new state
        obs = self._get_observations()
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        # Step 3: Compute rewards
        for i in range(self.num_agents):
            if self.agent_dones[i]: continue
            
            pos = self.agents[i].state[0:3]
            goal = self.goals[i]
            dist_to_goal = np.linalg.norm(pos - goal)
            d_min = self._get_min_distance(i)
            min_distances.append(d_min)
            
            if d_min < self.safety_frontier[i]:
                self.safety_frontier[i] = d_min
                
            old_dist_to_goal = self.prev_dist_to_goal[i]
            self.prev_dist_to_goal[i] = dist_to_goal
            
            idx_ranges = obs[i, 2:27]
            all_clear = np.min(idx_ranges) >= 0.95

            alpha, lam, sigma = 3.0, 5.0, 15.0
            delta = [0.45, 0.30, 0.15, 0.10]
            
            d_min_agent = float('inf')
            for j in range(self.num_agents):
                if i == j or self.agent_dones[j]:
                    continue
                d = np.linalg.norm(pos - self.agents[j].state[0:3]) - 0.6  # 0.3 radius each
                if d < d_min_agent:
                    d_min_agent = d
                    
            lambda_sep = 2.0
            sigma_sep = 5.0
            r_sep = 0.0
            if d_min_agent != float('inf'):
                r_sep = -lambda_sep * np.exp(-sigma_sep * d_min_agent)
            
            r_trans = alpha * (old_dist_to_goal - dist_to_goal)
            r_col   = -lam * math.exp(-sigma * d_min)
            r_free  = 0.1 if all_clear else 0.0
            r_step  = -0.6
            
            delta_sep = 0.15
            rewards[i] = delta[0]*r_trans + delta[1]*r_col + delta[2]*r_free + delta[3]*r_step + delta_sep * r_sep
            
            if dist_to_goal < self.goal_dist:
                self.agent_dones[i] = True
                info['agent_terminated_now'][i] = True
                rewards[i] += self.reward_config.get('goal_bonus', 0.0)
            elif not self.agent_dones[i] and d_min < self.collision_dist:
                self.agent_dones[i] = True
                self._episode_collision = True
                info['agent_terminated_now'][i] = True
                info['collision'] = True
                rewards[i] += self.reward_config.get('collision_penalty', 0.0)
                
        info['agent_dones'] = self.agent_dones.copy()
        
        individual_successes = sum(1 for i in range(self.num_agents) 
                                  if self.agent_dones[i] and 
                                  np.linalg.norm(self.agents[i].state[0:3] - self.goals[i]) < self.goal_dist)
        info['individual_success_rate'] = individual_successes / self.num_agents
        info['success'] = (individual_successes == self.num_agents) and not getattr(self, '_episode_collision', False)
        
        info['agent_success'] = np.array([self.agent_dones[i] and np.linalg.norm(self.agents[i].state[0:3] - self.goals[i]) < self.goal_dist for i in range(self.num_agents)], dtype=bool)
        info['agent_collision'] = self.agent_dones & ~info['agent_success']
        
        info['action_smoothness'] = float(np.mean(action_smoothness)) if action_smoothness else 0.0
        info['avg_jerk'] = float(np.mean(self.total_jerk))
        info['safety_frontier'] = float(np.mean(self.safety_frontier))
        info['min_agent_dist'] = float(np.mean(min_distances)) if min_distances else 0.0
        
        terminated = bool(np.all(self.agent_dones))
        truncated = bool(self.step_count >= self.max_steps)
        self.last_info = info
        
        return obs, rewards, terminated, truncated, info
