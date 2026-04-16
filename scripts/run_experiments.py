import subprocess
import scipy.stats
import numpy as np
import os
import json

def run_experiments():
    seeds = [0, 1, 2, 3, 4]
    agents = ['mardpg', 'maddpg']
    results = {agent: [] for agent in agents}
    
    os.makedirs('results', exist_ok=True)
    
    for agent in agents:
        for seed in seeds:
            print(f"Running {agent} with seed {seed}")
            output_json = f"results/{agent}_seed_{seed}.json"
            subprocess.run([
                'python3', 'scripts/train.py', 
                '--agent', agent,
                '--seed', str(seed),
                '--num-episodes', '100',  # Short run for demonstration
                '--output-json', output_json
            ])
            
            if os.path.exists(output_json):
                with open(output_json, 'r') as f:
                    data = json.load(f)
                    results[agent].append(data['avg_reward'])
            else:
                print(f"Warning: {output_json} not found.")
                
    # Statistical evaluation
    if len(results['mardpg']) > 0 and len(results['maddpg']) > 0:
        a_rewards = results['mardpg']
        b_rewards = results['maddpg']
        
        t_stat, p_val = scipy.stats.ttest_ind(a_rewards, b_rewards, equal_var=False)
        
        mu_a = np.mean(a_rewards)
        mu_b = np.mean(b_rewards)
        var_a = np.var(a_rewards, ddof=1) if len(a_rewards) > 1 else 0
        var_b = np.var(b_rewards, ddof=1) if len(b_rewards) > 1 else 0
        
        pooled_std = np.sqrt((var_a + var_b) / 2) if (var_a + var_b) > 0 else 1e-8
        cohen_d = (mu_a - mu_b) / pooled_std
        
        print("\n=== Statistical Evaluation ===")
        print(f"MARDPG Mean Reward: {mu_a:.2f} ± {np.sqrt(var_a):.2f}")
        print(f"MADDPG Mean Reward: {mu_b:.2f} ± {np.sqrt(var_b):.2f}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_val:.4f}")
        print(f"Cohen's d: {cohen_d:.4f}")
    else:
        print("Not enough data to perform statistical evaluation.")

if __name__ == '__main__':
    run_experiments()
