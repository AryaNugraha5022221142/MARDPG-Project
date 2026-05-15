import time
import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.base_env import EnvironmentConfig, DifficultyLevel
from envs.benchmark_suite import BenchmarkSuite

def main():
    parser = argparse.ArgumentParser(description="Check all environments and difficulty levels.")
    args = parser.parse_args()

    scenes = list(BenchmarkSuite.REGISTRY.keys())
    difficulties = list(DifficultyLevel)

    print(f"{'Scene':<12} | {'Difficulty':<10} | {'Obstacles':<9} | {'Gen Time (s)':<12} | {'Status'}")
    print("-" * 65)

    total_warnings = 0

    for scene in scenes:
        for diff in difficulties:
            # Configure environment
            cfg = EnvironmentConfig(
                map_width=60.0,
                map_depth=60.0,
                map_height=20.0,
                difficulty=diff,
                name=f"{scene}_{diff.name}"
            )
            
            # Start timer
            t0 = time.time()
            try:
                env = BenchmarkSuite.make(scene, cfg)
                generation_time = time.time() - t0
                
                # Check metrics & warnings
                metrics = env.validate()
                warnings = metrics.get("warnings", [])
                
                status = "OK"
                if warnings:
                    status = f"{len(warnings)} Warning(s)"
                    total_warnings += len(warnings)
                
                print(f"{scene:<12} | {diff.name:<10} | {len(env.obstacles):<9} | {generation_time:<12.4f} | {status}")
                
                for w in warnings:
                    print(f"  -> {w}")
                    
            except Exception as e:
                generation_time = time.time() - t0
                print(f"{scene:<12} | {diff.name:<10} | {'ERROR':<9} | {generation_time:<12.4f} | FAILED: {e}")

    print("-" * 65)
    print("Done checking all scenes and difficulties.")
    if total_warnings > 0:
         print(f"Note: There were {total_warnings} warnings in total. Some overlap or placement issues may occur in dense settings.")

if __name__ == '__main__':
    main()
