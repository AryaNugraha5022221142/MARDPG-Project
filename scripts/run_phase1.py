# scripts/run_phase1.py
import subprocess
import os
import json
import numpy as np
import pandas as pd

SEEDS = [42, 123, 456]
EPISODES = 5000
SCENARIO = 'basic_obstacles'
RESULTS_DIR = 'results/phase1'

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_phase1():
    print(f"=== Phase 1: Baseline Verification (MARDPG on {SCENARIO}) ===")
    results = []
    
    for seed in SEEDS:
        output_file = f"{RESULTS_DIR}/mardpg_seed{seed}.json"
        print(f"\n--- Training Seed {seed} ---")
        
        cmd = [
            'python', 'scripts/train.py',
            '--agent', 'mardpg',
            '--scenario', SCENARIO,
            '--num-episodes', str(EPISODES),
            '--seed', str(seed),
            '--output-json', output_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            with open(output_file, 'r') as f:
                results.append(json.load(f))
        except subprocess.CalledProcessError as e:
            print(f"Error during training seed {seed}: {e}")

    # Aggregate Results
    if results:
        success_rates = [r['success_rate'] for r in results]
        avg_rewards = [r['avg_reward'] for r in results]
        
        summary = {
            'Metric': ['Success Rate (%)', 'Average Reward'],
            'Mean': [np.mean(success_rates) * 100, np.mean(avg_rewards)],
            'Std Dev': [np.std(success_rates) * 100, np.std(avg_rewards)]
        }
        
        df = pd.DataFrame(summary)
        print("\n=== Phase 1 Summary Results ===")
        print(df.to_string(index=False))
        
        summary_path = f"{RESULTS_DIR}/summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

if __name__ == '__main__':
    run_phase1()
