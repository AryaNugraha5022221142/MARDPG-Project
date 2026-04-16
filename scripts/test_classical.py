# scripts/test_classical.py
import sys
import os
import numpy as np
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import QuadcopterEnv
from envs.scenarios import get_scenario_config
from agents.classical import PotentialFieldAgent

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Classical Potential Field Algorithm')
    parser.add_argument('--scenario', type=str, default='static_dense', 
                        help='Scenario to test (e.g., empty, static_dense, basic_obstacles)')
    parser.add_argument('--render', action='store_true', help='Enable visualization')
    args = parser.parse_args()

    config = get_scenario_config(args.scenario)
    env = QuadcopterEnv(num_agents=3, config=config, render_mode='human' if args.render else None, scenario=args.scenario)
    
    agent = PotentialFieldAgent(num_agents=3, arena_size=np.array(config['arena_size']))
    
    obs, _ = env.reset()
    print(f"Starting test on scenario: {args.scenario}")
    
    successes = 0
    total_episodes = 10
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            # Get internal states for the classical agent
            states = [a.state for a in env.agents]
            actions = agent.select_actions(states, env.goals, env.obstacles)
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            steps += 1
            
            if args.render:
                env.render()
                time.sleep(0.01)
                
        if info.get('success', False):
            successes += 1
            print(f"Episode {ep+1}: Success in {steps} steps")
        else:
            print(f"Episode {ep+1}: Failed ({'Collision' if info.get('collision') else 'Timeout'})")
            
    print(f"\nFinal Success Rate: {successes/total_episodes*100:.1f}%")
    env.close()

if __name__ == '__main__':
    main()
