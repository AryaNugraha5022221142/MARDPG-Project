#!/usr/bin/env python3
"""Script to visualize a snapshot of the quadcopter environment."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import matplotlib

# Set backend to Agg so it works without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from envs.quadcopter_kinematic_env import QuadcopterKinematicEnv

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def generate_scene():
    config = load_config()
    
    env_config = config.get('environment', config)
    scenes = config.get('evaluation', {}).get('scenes', ["urban", "forest", "terrain", "structured", "dynamic"])
    
    print(f"Generating snapshots for {len(scenes)} scenarios: {scenes}")
    
    # Wait, the render() method in envs uses interactive plotting which might conflict with Agg
    # But usually plt.pause is a no-op if Agg or fails. Let's patch plt.pause temporarily.
    _plt_pause = plt.pause
    plt.pause = lambda x: None
    
    try:
        for scenario in scenes:
            print(f"Initializing scene: {scenario}...")
            env = QuadcopterKinematicEnv(num_agents=3, config=env_config, render_mode="human", scenario=scenario)
            
            obs, info = env.reset()
            env.render()
            
            save_path = os.path.join(project_root, f"scene_{scenario}.png")
            env.fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✅ Saved: {save_path}")
            
            env.close()
    except Exception as e:
        print(f"Error while rendering: {e}")
    finally:
        plt.pause = _plt_pause

if __name__ == '__main__':
    generate_scene()
