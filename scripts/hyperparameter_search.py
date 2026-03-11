# scripts/hyperparameter_search.py
import yaml
import os
import pandas as pd

def main():
    """
    Generates configuration files for hyperparameter search.
    In a real scenario, this would launch multiple training jobs.
    """
    hidden_dims = [64, 128, 256]
    learning_rates = [1e-4, 5e-4, 1e-3]
    seq_lengths = [8, 16, 32]
    seeds = [42, 123, 456]
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Mocking the search by creating a CSV of combinations
    combinations = []
    run_id = 0
    
    for hd in hidden_dims:
        for lr in learning_rates:
            for sl in seq_lengths:
                for seed in seeds:
                    combinations.append({
                        'run_id': run_id,
                        'hidden_dim': hd,
                        'actor_lr': lr,
                        'seq_length': sl,
                        'seed': seed,
                        'status': 'pending'
                    })
                    run_id += 1
                    
    df = pd.DataFrame(combinations)
    df.to_csv(os.path.join(results_dir, 'hyperparameter_search.csv'), index=False)
    print(f"Generated {len(combinations)} hyperparameter combinations.")
    print(f"Saved to {os.path.join(results_dir, 'hyperparameter_search.csv')}")

if __name__ == '__main__':
    main()
