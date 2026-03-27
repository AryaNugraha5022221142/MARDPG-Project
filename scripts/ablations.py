"""
ablations.py — Runs actual training experiments for thesis ablations.

Ablation 1: MARDPG vs MADDPG (LSTM vs MLP)
Ablation 2: Centralized critic vs independent critics (CTDE vs IDDPG)
Ablation 3: With vs without sensor noise
Ablation 4: Reward function — linear proximity vs exponential collision penalty

Each experiment trains for N episodes across K seeds and reports mean ± std.
"""
import subprocess
import os
import json
import numpy as np
import pandas as pd
from scipy import stats


SEEDS = [42, 123, 456]          # 3 seeds per ablation (5 for camera-ready)
EPISODES = 5000                  # use same as main training
SCENARIO = 'basic_obstacles'    # controlled scenario for ablation
RESULTS_DIR = 'results/ablations'
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment(agent_type: str, seed: int, extra_flags: list = None) -> dict:
    """Run a single training experiment and return final metrics."""
    cmd = [
        'python', 'scripts/train.py',
        '--agent', agent_type,
        '--seed', str(seed),
        '--scenario', SCENARIO,
        '--num-episodes', str(EPISODES),
        '--output-json', f'{RESULTS_DIR}/{agent_type}_seed{seed}.json'
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    subprocess.run(cmd, check=True)
    with open(f'{RESULTS_DIR}/{agent_type}_seed{seed}.json') as f:
        return json.load(f)


def statistical_comparison(group_a: list, group_b: list, label_a: str, label_b: str):
    t_stat, p_val = stats.ttest_ind(group_a, group_b)
    effect_size = (np.mean(group_a) - np.mean(group_b)) / (
        np.sqrt((np.std(group_a)**2 + np.std(group_b)**2) / 2) + 1e-9
    )
    print(f"\n{label_a} vs {label_b}")
    print(f"  {label_a}: {np.mean(group_a):.2f} ± {np.std(group_a):.2f}")
    print(f"  {label_b}: {np.mean(group_b):.2f} ± {np.std(group_b):.2f}")
    print(f"  t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={effect_size:.3f}")
    return {'t': t_stat, 'p': p_val, 'd': effect_size}


def run_ablation_1():
    """MARDPG (LSTM) vs MADDPG (MLP) — tests recurrent memory hypothesis."""
    print("\n=== Ablation 1: LSTM vs MLP ===")
    mardpg_results, maddpg_results = [], []
    for seed in SEEDS:
        mardpg_results.append(run_experiment('mardpg', seed)['success_rate'])
        maddpg_results.append(run_experiment('maddpg', seed)['success_rate'])
    return statistical_comparison(mardpg_results, maddpg_results, 'MARDPG', 'MADDPG')


def run_ablation_2():
    """MARDPG vs IDDPG (independent critics) — tests CTDE hypothesis."""
    print("\n=== Ablation 2: Centralized vs Independent Critics ===")
    ctde_results, iddpg_results = [], []
    for seed in SEEDS:
        ctde_results.append(run_experiment('mardpg', seed)['success_rate'])
        iddpg_results.append(run_experiment('iddpg', seed)['success_rate'])
    return statistical_comparison(ctde_results, iddpg_results, 'MARDPG (CTDE)', 'IDDPG')


if __name__ == '__main__':
    abl1 = run_ablation_1()
    abl2 = run_ablation_2()
    summary = pd.DataFrame([
        {'ablation': 'LSTM vs MLP', **abl1},
        {'ablation': 'CTDE vs IDDPG', **abl2}
    ])
    summary.to_csv(f'{RESULTS_DIR}/ablation_summary.csv', index=False)
    print("\n=== Summary saved ===")
    print(summary.to_string())
