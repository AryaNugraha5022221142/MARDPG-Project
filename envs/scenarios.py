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
        config['num_obstacles'] = 80 
        config['dynamic_ratio'] = 0.0
        config['arena_size'] = [100.0, 100.0, 40.0]
        config['altitude_limit'] = [5.0, 35.0]
        return config
        
    elif scenario_name == 'warzone':
        config = base_config.copy()
        config['num_obstacles'] = 30
        config['dynamic_ratio'] = 0.2
        config['arena_size'] = [100.0, 100.0, 50.0]
        return config
        
    elif scenario_name == 'forest':
        config = base_config.copy()
        config['num_obstacles'] = 150
        config['dynamic_ratio'] = 0.0
        config['arena_size'] = [60.0, 60.0, 20.0]
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
        # Urban Canyon: High-rise buildings in a grid
        building_colors = ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7']
        nfz_color = '#e74c3c' # Red for No-Fly Zones
        
        spacing = 12.0
        for x in np.arange(10.0, env.arena_size[0] - 10.0, spacing):
            for y in np.arange(10.0, env.arena_size[1] - 10.0, spacing):
                # Randomize building footprint and height
                w = np.random.uniform(6, 10)
                d = np.random.uniform(6, 10)
                h = np.random.uniform(15, 38)
                
                pos = np.array([x, y, h/2])
                env.obstacles.append({
                    'type': 'box',
                    'pos': pos,
                    'size': np.array([w, d, h]),
                    'vel': np.zeros(3),
                    'origin': pos.copy(),
                    'color': np.random.choice(building_colors),
                    'is_building': True
                })
                
                # Occasionally add a No-Fly Zone between buildings
                if np.random.random() < 0.1:
                    nfz_h = env.arena_size[2]
                    nfz_pos = np.array([x + spacing/2, y + spacing/2, nfz_h/2])
                    env.obstacles.append({
                        'type': 'box',
                        'pos': nfz_pos,
                        'size': np.array([4, 4, nfz_h]),
                        'vel': np.zeros(3),
                        'origin': nfz_pos.copy(),
                        'color': nfz_color,
                        'is_nfz': True,
                        'alpha': 0.3
                    })

    elif scenario_name == 'warzone':
        # Warzone: Terrain, Radars, and Missile Envelopes
        radar_color = '#f1c40f' # Yellow
        missile_color = '#c0392b' # Dark Red
        terrain_color = '#34495e'
        
        # 1. Terrain (Hills/Valleys)
        for x in np.arange(0, env.arena_size[0], 10):
            for y in np.arange(0, env.arena_size[1], 10):
                h = np.random.uniform(2, 12)
                pos = np.array([x+5, y+5, h/2])
                env.obstacles.append({
                    'type': 'box',
                    'pos': pos,
                    'size': np.array([10, 10, h]),
                    'vel': np.zeros(3),
                    'origin': pos.copy(),
                    'color': terrain_color,
                    'is_terrain': True
                })
        
        # 2. Radars (Large detection zones)
        for _ in range(5):
            pos = np.array([np.random.uniform(20, 80), np.random.uniform(20, 80), np.random.uniform(10, 30)])
            radius = np.random.uniform(15, 25)
            env.obstacles.append({
                'type': 'sphere',
                'pos': pos,
                'radius': radius,
                'vel': np.random.uniform(-1, 1, 3) * 0.5,
                'origin': pos.copy(),
                'color': radar_color,
                'is_radar': True,
                'alpha': 0.1
            })
            
        # 3. Missile Batteries (Lethal zones)
        for _ in range(8):
            pos = np.array([np.random.uniform(10, 90), np.random.uniform(10, 90), np.random.uniform(5, 40)])
            radius = np.random.uniform(5, 8)
            env.obstacles.append({
                'type': 'sphere',
                'pos': pos,
                'radius': radius,
                'vel': np.zeros(3),
                'origin': pos.copy(),
                'color': missile_color,
                'is_missile': True,
                'alpha': 0.4
            })

    elif scenario_name == 'forest':
        # Forest: Trunks and Branches
        trunk_color = '#5d4037' # Brown
        leaf_color = '#2e7d32' # Green
        
        for _ in range(env.num_obstacles):
            # Trunk
            tx, ty = np.random.uniform(5, env.arena_size[0]-5), np.random.uniform(5, env.arena_size[1]-5)
            th = np.random.uniform(8, 18)
            tpos = np.array([tx, ty, th/2])
            env.obstacles.append({
                'type': 'box',
                'pos': tpos,
                'size': np.array([0.8, 0.8, th]),
                'vel': np.zeros(3),
                'origin': tpos.copy(),
                'color': trunk_color,
                'is_trunk': True
            })
            
            # Branches (2-3 per tree)
            for _ in range(np.random.randint(1, 4)):
                bh_z = np.random.uniform(th*0.4, th)
                bl = np.random.uniform(2, 5)
                angle = np.random.uniform(0, 2*np.pi)
                bpos = np.array([tx + np.cos(angle)*bl/2, ty + np.sin(angle)*bl/2, bh_z])
                env.obstacles.append({
                    'type': 'box',
                    'pos': bpos,
                    'size': np.array([bl, 0.3, 0.3]) if np.random.random() > 0.5 else np.array([0.3, bl, 0.3]),
                    'vel': np.zeros(3),
                    'origin': bpos.copy(),
                    'color': leaf_color,
                    'is_branch': True,
                    'alpha': 0.7
                })
