# scripts/visualize_env.py
import sys
import os
import numpy as np
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import QuadcopterEnv
from envs.scenarios import get_scenario_config

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize the Quadcopter Environment')
    parser.add_argument('--scenario', type=str, default='city', 
                        choices=['empty', 'static_dense', 'dynamic_chaos', 'narrow_passage', 'city', 'forest', 'warzone'],
                        help='Scenario to visualize')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    args = parser.parse_args()

    print(f"Loading scenario: {args.scenario}")
    config = get_scenario_config(args.scenario)
    
    env = QuadcopterEnv(
        num_agents=args.agents, 
        config=config, 
        render_mode='human',
        scenario=args.scenario
    )
    
    obs, _ = env.reset()
    print("Environment reset. Starting visualization...")
    print("Close the window or press Ctrl+C to stop.")
    
    try:
        for step in range(1000):
            # Take random actions to see movement
            actions = [np.random.randint(6) for _ in range(env.num_agents)]
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            env.render()
            
            if terminated or truncated:
                print(f"Episode ended at step {step}. Resetting...")
                obs, _ = env.reset()
                
            # Slow down for visualization
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        env.close()

if __name__ == '__main__':
    main()
