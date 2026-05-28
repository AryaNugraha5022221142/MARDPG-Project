import yaml
import time
import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.benchmark_wrapped_env import BenchmarkWrappedEnv

def main():
    parser = argparse.ArgumentParser(description="Visualize different environment scenes")
    parser.add_argument('--scene', type=str, default='all', help='Scene to visualize (urban, forest, terrain, structured, dynamic) or "all"')
    parser.add_argument('--num-agents', type=int, default=5, help='Number of agents')
    parser.add_argument('--level', type=int, default=3, help='Difficulty level (1-5)')
    args = parser.parse_args()

    with open("config/config_static.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    scenes = ['urban', 'forest', 'terrain', 'structured', 'dynamic'] if args.scene == 'all' else [args.scene]
    
    for scene in scenes:
        print(f"\n--- Visualizing {scene.upper()} Scene ---")
        
        # Determine arena size based on level (same logic as QuadcopterEnv set_curriculum_level)
        if args.level == 0:
            arena_size = [30.0, 30.0, 15.0]
        elif args.level == 1:
            arena_size = [50.0, 50.0, 20.0]
        elif args.level == 2:
            arena_size = [75.0, 75.0, 30.0]
        elif args.level == 3:
            arena_size = [100.0, 100.0, 40.0]
        else:
            arena_size = [130.0, 130.0, 50.0]
            
        env_config = config['environment'].copy()
        env_config['arena_size'] = arena_size
        
        env = BenchmarkWrappedEnv(
            benchmark_name=scene,
            level=args.level,
            num_agents=args.num_agents,
            config=env_config
        )
        env.render_mode = 'human'

        
        obs, _ = env.reset()
        env.render()
        
        # Take a few random steps just to show some movement
        import numpy as np
        print(f"Rendering {scene} for 5 seconds...")
        for _ in range(100):
            action_bound = getattr(env, 'action_bound', 0.5)
            actions = np.random.uniform(-action_bound, action_bound, size=(args.num_agents, 2))
            obs, rewards, terminated, truncated, info = env.step(actions)
            env.render()
            time.sleep(0.05)
            if terminated or truncated:
                break
                
        env.close()

if __name__ == "__main__":
    main()
