import numpy as np
from envs.base_env import EnvironmentConfig, DifficultyLevel
from envs.benchmark_suite import BenchmarkSuite
from envs.quadcopter_kinematic_env import QuadcopterKinematicEnv

class BenchmarkWrappedEnv(QuadcopterKinematicEnv):
    def __init__(self, benchmark_name, level, num_agents, config):
        config_override = config.copy()
        super().__init__(num_agents=num_agents, config=config_override, render_mode=None, scenario=None)
        
        self.benchmark_name = benchmark_name
        self.base_seed = int(config.get('seed', 42))
        self.randomize_layouts = bool(config.get('randomize_layouts', True))
        self._reset_index = 0
        
        if isinstance(level, DifficultyLevel):
            self.difficulty = level
        else:
            self.difficulty = DifficultyLevel(level)
        
        # Override bypass so _generate_obstacles doesn't crash on init
        self._init_done = False
        self.set_curriculum_level(self.difficulty.value)
        self.b_cfg = self._make_benchmark_config(self.benchmark_name, self.base_seed)
        self._init_done = True
        self._generate_obstacles()
        
    def _make_benchmark_config(self, benchmark_name, seed):
        train_w = float(self.arena_size[0])
        train_d = float(self.arena_size[1])
        train_h = float(self.arena_size[2])
        return EnvironmentConfig(
            map_width=train_w,
            map_depth=train_d,
            map_height=train_h,
            difficulty=self.difficulty,
            name=f"{benchmark_name}_{self.difficulty.name}",
            n_agents=self.num_agents,
            seed=int(seed),
        )

    def _next_layout_seed(self, seed):
        if seed is not None:
            return int(seed)
        if not self.randomize_layouts:
            return self.base_seed
        layout_seed = self.base_seed + self._reset_index
        self._reset_index += 1
        return layout_seed

    @staticmethod
    def _obstacle_motion_fields(obstacle, pos):
        velocity = obstacle.velocity if obstacle.velocity is not None else np.zeros(3)
        return {
            'vel': np.asarray(velocity, dtype=float).copy(),
            'origin': np.asarray(pos, dtype=float).copy(),
            'phase': float(obstacle.metadata.get('phase', 0.0)),
            'freq': float(obstacle.metadata.get('freq', 0.05)),
            'is_dynamic': bool(obstacle.is_dynamic),
        }
        
    def set_curriculum_level(self, level: int):
        super().set_curriculum_level(level)
        if getattr(self, '_init_done', False):
            self.b_cfg = self._make_benchmark_config(self.benchmark_name, self.base_seed)

    def _generate_obstacles(self):
        if not getattr(self, '_init_done', False): return
        self.b_env = BenchmarkSuite.make(self.benchmark_name, self.b_cfg)
        
        self.obstacles = []
        for o in self.b_env.obstacles:
            rtype = o.obstacle_type.value
            if rtype == 'sphere':
                self.obstacles.append({
                    'type': 'sphere', 
                    'pos': o.position.copy(), 
                    'radius': np.max(o.dimensions) / 2.0,
                    'color': o.color,
                    **self._obstacle_motion_fields(o, o.position),
                })
            elif rtype in ('cylinder', 'dynamic'):
                radius, _, height = o.dimensions
                # For physics, treat as box, but store original cylinder size
                size = np.array([radius*2, radius*2, height])
                pos = o.position.copy()
                pos[2] += height / 2.0 # center Z for box collision logic
                self.obstacles.append({
                    'type': 'cylinder', 
                    'pos': pos, 
                    'size': size,
                    'orig_pos': o.position.copy(),
                    'radius': radius,
                    'height': height,
                    'color': o.color,
                    **self._obstacle_motion_fields(o, pos),
                })
            else:
                lo, hi = o.aabb
                size = hi - lo
                pos = (lo + hi) / 2.0
                self.obstacles.append({
                    'type': 'box', 
                    'pos': pos, 
                    'size': size,
                    'color': o.color,
                    **self._obstacle_motion_fields(o, pos),
                })
        
        self.agents_init_positions = [p.copy() for p in self.b_env.agents]
        self.goals_init_positions = [p.copy() for p in self.b_env.goals]

    def reset(self, seed=None):
        self.step_count = 0
        self._episode_collision = False
        self.agent_dones = np.zeros(self.num_agents, dtype=bool)
        
        self.targets_claimed = set()
        self.total_jerk = np.zeros(self.num_agents, dtype=np.float32)
        self.safety_frontier = np.ones(self.num_agents, dtype=np.float32) * float('inf')
        self.prev_accel = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.prev_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.prev_vel = np.zeros((self.num_agents, 3), dtype=np.float32)
        
        layout_seed = self._next_layout_seed(seed)
        self.b_cfg = self._make_benchmark_config(self.benchmark_name, layout_seed)
        
        self._generate_obstacles()
        
        self.goals = [g.copy() for g in self.goals_init_positions]
        
        for i in range(self.num_agents):
            start_pos = self.agents_init_positions[i]
            goal = self.goals[i]
            
            dx = goal[0] - start_pos[0]
            dy = goal[1] - start_pos[1]
            rough_dir = np.arctan2(dy, dx)
            start_yaw = rough_dir + np.random.uniform(-np.pi/4, np.pi/4)
            
            self.agents[i].reset(start_pos, start_yaw)
            self.prev_dist_to_goal[i] = np.linalg.norm(start_pos - self.goals[i])
            self.prev_vel[i] = self.agents[i].state[6:9].copy()
            
        return self._get_observations(), {}
        
    def set_scene_type(self, scene: str):
        self.benchmark_name = scene
        self.b_cfg = EnvironmentConfig(
            map_width=60.0,
            map_depth=60.0,
            map_height=20.0,
            difficulty=self.difficulty,
            name=f"{scene}_{self.difficulty.name}",
            n_agents=self.num_agents
        )
        self.scenario = scene
        # We don't need to generate obstacles here, it happens in reset()
