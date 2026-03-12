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
        
    elif scenario_name == 'narrow_passage':
        # This will require custom obstacle placement logic in the env
        config = base_config.copy()
        config['num_obstacles'] = 10
        config['is_narrow_passage'] = True
        return config
        
    return base_config

def apply_scenario_custom_logic(env, scenario_name: str):
    """
    Applies custom obstacle placement for specific scenarios.
    """
    if scenario_name == 'narrow_passage':
        env.obstacles = []
        # Create a wall with a hole in the middle (x=10)
        for y in np.linspace(0, 20, 10):
            for z in np.linspace(0, 10, 5):
                # Leave a hole at y=10, z=5
                if np.abs(y - 10) > 2.0 or np.abs(z - 5) > 2.0:
                    env.obstacles.append({
                        'type': 'box',
                        'pos': np.array([10.0, y, z]),
                        'size': np.array([1.0, 2.0, 2.0]),
                        'vel': np.zeros(3),
                        'origin': np.array([10.0, y, z])
                    })
