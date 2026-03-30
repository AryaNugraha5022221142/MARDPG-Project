# scripts/generate_preview.py
import sys
import os
import numpy as np
import matplotlib
# Use Agg backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import QuadcopterEnv
from envs.scenarios import get_scenario_config

def main():
    scenario_name = 'city'
    print(f"Generating preview for scenario: {scenario_name}")
    config = get_scenario_config(scenario_name)
    
    env = QuadcopterEnv(
        num_agents=3, 
        config=config, 
        render_mode='rgb_array', # We'll handle rendering manually
        scenario=scenario_name
    )
    
    obs, _ = env.reset()
    
    # Create a directory for previews
    os.makedirs('previews', exist_ok=True)
    
    for step in range(101):
        # Take random actions
        actions = [np.random.uniform(-1, 1, size=4) for _ in range(env.num_agents)]
        env.step(actions)
        
        # Save a frame every 20 steps
        if step % 20 == 0:
            print(f"Capturing frame at step {step}...")
            env.render() # This calls the matplotlib code in quadcopter_env.py
            plt.savefig(f'previews/step_{step:03d}.png')
            plt.close()
            # Reset fig/ax so next render creates new ones
            env.fig = None
            env.ax = None
            
    env.close()
    print("Preview generation complete. Check the 'previews' folder.")

if __name__ == '__main__':
    main()
