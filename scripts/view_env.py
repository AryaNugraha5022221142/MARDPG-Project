import argparse
import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadcopter_kinematic_env import QuadcopterKinematicEnv

def main():
    parser = argparse.ArgumentParser(description="View the 3D environments with random agent actions.")
    parser.add_argument('--scene', type=str, default='pillars', choices=['pillars', 'cylinders', 'forest', 'rings', 'random'], help="Type of scene to load.")
    parser.add_argument('--steps', type=int, default=1000, help="Number of simulation steps to run.")
    parser.add_argument('--dt', type=float, default=0.05, help="Time step delta for the visualizer.")
    args = parser.parse_args()

    config = {
        'arena_size': [100.0, 100.0, 40.0],
        'num_obstacles': 25,
        'rangefinder_max_range': 30.0,
        'collision_distance': 0.8,
        'goal_distance': 2.0,
        'dt': args.dt,
        'dynamic_ratio': 0.0,
        'action_bound': np.pi / 6.0
    }

    env = QuadcopterKinematicEnv(num_agents=3, config=config, render_mode='human')
    
    if args.scene != 'random':
        env.set_scene_type(args.scene)
        
    env.reset()
    
    print(f"Viewing scene: {args.scene}")
    print("Close the matplotlib window or press Ctrl+C in the terminal to exit.")

    plt.ion() # Interactive mode on
    
    try:
        for _ in range(args.steps):
            # Generate random steering commands
            actions = np.random.uniform(-config['action_bound'], config['action_bound'], size=(3, 2))
            
            # Step the environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Render the environment
            env.render()
            
            if terminated or truncated:
                if args.scene == 'random':
                    # Pick a new random scene
                    import random
                    env.set_scene_type(random.choice(['pillars', 'cylinders', 'forest', 'rings']))
                env.reset()
                
    except KeyboardInterrupt:
        print("Visualization stopped by user.")
        
    print("Done. Keeping window open.")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
