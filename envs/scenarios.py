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
        config['num_obstacles'] = 40 # Fewer but more complex
        config['dynamic_ratio'] = 0.05
        config['arena_size'] = [50.0, 50.0, 25.0]
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
        # Generate varied buildings with colors
        building_colors = ['#4682B4', '#708090', '#2F4F4F', '#A9A9A9', '#808080', '#B0C4DE']
        
        # Grid-like but randomized distribution for a city feel
        grid_size = 10.0
        for x in np.arange(5.0, env.arena_size[0] - 5.0, grid_size):
            for y in np.arange(5.0, env.arena_size[1] - 5.0, grid_size):
                if np.random.random() < 0.7: # 70% chance of a building in this grid cell
                    # Randomize position within cell
                    pos_x = x + np.random.uniform(-2.0, 2.0)
                    pos_y = y + np.random.uniform(-2.0, 2.0)
                    
                    # Randomize building type
                    b_type = np.random.choice(['tower', 'complex', 'low'])
                    color = np.random.choice(building_colors)
                    
                    if b_type == 'tower':
                        size = np.array([np.random.uniform(3, 5), np.random.uniform(3, 5), np.random.uniform(15, 22)])
                        pos = np.array([pos_x, pos_y, size[2]/2])
                        env.obstacles.append({'type': 'box', 'pos': pos, 'size': size, 'vel': np.zeros(3), 'origin': pos.copy(), 'color': color})
                    elif b_type == 'complex':
                        # Two boxes combined
                        size1 = np.array([np.random.uniform(4, 6), np.random.uniform(4, 6), np.random.uniform(8, 12)])
                        pos1 = np.array([pos_x, pos_y, size1[2]/2])
                        env.obstacles.append({'type': 'box', 'pos': pos1, 'size': size1, 'vel': np.zeros(3), 'origin': pos1.copy(), 'color': color})
                        
                        size2 = np.array([size1[0]*0.6, size1[1]*1.2, size1[2]*0.5])
                        pos2 = pos1 + np.array([0, 0, size1[2]/2 + size2[2]/2])
                        env.obstacles.append({'type': 'box', 'pos': pos2, 'size': size2, 'vel': np.zeros(3), 'origin': pos2.copy(), 'color': color})
                    else: # low
                        size = np.array([np.random.uniform(6, 10), np.random.uniform(6, 10), np.random.uniform(3, 6)])
                        pos = np.array([pos_x, pos_y, size[2]/2])
                        env.obstacles.append({'type': 'box', 'pos': pos, 'size': size, 'vel': np.zeros(3), 'origin': pos.copy(), 'color': color})
            
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
