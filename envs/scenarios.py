# envs/scenarios.py
import numpy as np
from typing import Dict, Any, List

def get_scenario_config(scenario_name: str) -> Dict[str, Any]:
    """
    Returns configuration for different environment scenarios.
    """
    base_config = {
        'arena_size': [20.0, 20.0, 10.0],
        'num_obstacles': 15,
        'rangefinder_max_range': 15.0,
        'collision_distance': 0.5,
        'goal_distance': 1.0,
        'dt': 0.01,
        'dynamic_ratio': 0.3
    }
    
    if scenario_name == 'empty':
        config = base_config.copy()
        config['num_obstacles'] = 0
        return config
        
    elif scenario_name == 'static_dense':
        config = base_config.copy()
        config['num_obstacles'] = 40
        config['dynamic_ratio'] = 0.0
        return config
        
    elif scenario_name == 'dynamic_chaos':
        config = base_config.copy()
        config['num_obstacles'] = 20
        config['dynamic_ratio'] = 0.8
        return config
        
    elif scenario_name == 'city':
        config = base_config.copy()
        config['num_obstacles'] = 60
        config['dynamic_ratio'] = 0.1
        config['arena_size'] = [40.0, 40.0, 20.0]
        return config
        
    elif scenario_name == 'forest':
        config = base_config.copy()
        config['num_obstacles'] = 100
        config['dynamic_ratio'] = 0.2
        config['arena_size'] = [40.0, 40.0, 15.0]
        return config
        
    elif scenario_name == 'narrow_passage':
        config = base_config.copy()
        config['num_obstacles'] = 0 # Custom logic handles it
        config['arena_size'] = [30.0, 20.0, 10.0]
        return config
        
    return base_config

def apply_scenario_custom_logic(env, scenario_name: str):
    """
    Applies custom obstacle placement for specific scenarios.
    """
    env.obstacles = []
    if scenario_name == 'narrow_passage':
        # Create two walls with a gap in the middle
        wall_x = env.arena_size[0] / 2
        # Left wall
        env.obstacles.append({
            'type': 'box',
            'pos': np.array([wall_x, env.arena_size[1]/4 - 1.0, env.arena_size[2]/2]),
            'size': np.array([1.0, env.arena_size[1]/2 - 2.0, env.arena_size[2]]),
            'vel': np.zeros(3),
            'origin': np.array([wall_x, env.arena_size[1]/4 - 1.0, env.arena_size[2]/2])
        })
        # Right wall
        env.obstacles.append({
            'type': 'box',
            'pos': np.array([wall_x, 3*env.arena_size[1]/4 + 1.0, env.arena_size[2]/2]),
            'size': np.array([1.0, env.arena_size[1]/2 - 2.0, env.arena_size[2]]),
            'vel': np.zeros(3),
            'origin': np.array([wall_x, 3*env.arena_size[1]/4 + 1.0, env.arena_size[2]/2])
        })
        # Top wall (optional)
        env.obstacles.append({
            'type': 'box',
            'pos': np.array([wall_x, env.arena_size[1]/2, env.arena_size[2] - 1.0]),
            'size': np.array([1.0, 4.0, 2.0]),
            'vel': np.zeros(3),
            'origin': np.array([wall_x, env.arena_size[1]/2, env.arena_size[2] - 1.0])
        })
        
    elif scenario_name == 'city':
        # Generate tall buildings
        for _ in range(env.num_obstacles):
            pos = np.array([
                np.random.uniform(5.0, env.arena_size[0] - 5.0),
                np.random.uniform(5.0, env.arena_size[1] - 5.0),
                0.0 # Start from ground
            ])
            size = np.array([
                np.random.uniform(2.0, 4.0),
                np.random.uniform(2.0, 4.0),
                np.random.uniform(5.0, env.arena_size[2] * 0.8)
            ])
            pos[2] = size[2] / 2 # Center z
            env.obstacles.append({
                'type': 'box',
                'pos': pos,
                'size': size,
                'vel': np.zeros(3),
                'origin': pos.copy()
            })
            
    elif scenario_name == 'forest':
        # Generate many thin cylinders (trees)
        for _ in range(env.num_obstacles):
            pos = np.array([
                np.random.uniform(5.0, env.arena_size[0] - 5.0),
                np.random.uniform(5.0, env.arena_size[1] - 5.0),
                0.0
            ])
            radius = np.random.uniform(0.3, 0.8)
            height = np.random.uniform(4.0, env.arena_size[2])
            pos[2] = height / 2
            # Represent tree as a tall box for collision simplicity
            env.obstacles.append({
                'type': 'box',
                'pos': pos,
                'size': np.array([radius*2, radius*2, height]),
                'vel': np.zeros(3),
                'origin': pos.copy(),
                'is_tree': True
            })
