import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import numpy as np
from envs import QuadcopterEnv
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="View Curriculum Levels for Screenshots")
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--level', type=int, default=None, help='Curriculum level (0-4)')
    parser.add_argument('--scenario', type=str, default=None, help='Specific scenario to render')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Initialize the environment in human render mode
    num_agents = config.get('training', {}).get('num_agents', 3)
    env = QuadcopterEnv(num_agents=num_agents, config=config['environment'], render_mode='human', scenario=args.scenario)
    
    # Force curriculum level
    if args.level is not None:
        env.set_curriculum_level(args.level)
        
    obs, _ = env.reset()
    env.render()
    
    print(f"\n---")
    print(f"🎯 Rendering Curriculum Level {args.level}")
    print(f"📐 Arena Size: {env.arena_size}")
    print(f"🧱 Obstacles: {env.num_obstacles}")
    print(f"---")
    print("Close the plot window to exit.")
    
    # Keep the plot open
    import os
    os.makedirs('logs', exist_ok=True)
    if args.scenario:
        save_path = f"logs/scenario_{args.scenario}_render.png"
    else:
        save_path = f"logs/level_{args.level}_render.png"
    plt.savefig(save_path, dpi=300)
    print(f"Render saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
