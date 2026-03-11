import numpy as np
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadcopter_env import QuadcopterEnv

def run_tests():
    print("Starting Environment Tests...")
    
    # Test 1: Reset
    print("Test 1: Resetting environment...")
    env = QuadcopterEnv(num_agents=3)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    assert obs.shape == (3, 28), f"Expected (3, 28), got {obs.shape}"
    print("Reset test passed!")
    
    # Test 2: Step
    print("Test 2: Taking a step...")
    actions = [0, 1, 2]
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"Rewards: {rewards}")
    assert rewards.shape == (3,), f"Expected rewards shape (3,), got {rewards.shape}"
    print("Step test passed!")
    
    # Test 3: Multiple steps with dynamic obstacles
    print("Test 3: Running 50 steps with dynamic obstacles...")
    for i in range(50):
        actions = [np.random.randint(6) for _ in range(3)]
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            print(f"Episode ended at step {i} due to {'collision' if terminated else 'timeout'}")
            break
    print("Dynamic obstacle test passed!")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"\nTests failed with error: {e}")
        import traceback
        traceback.print_exc()
