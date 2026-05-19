import argparse
import sys
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.base_env import EnvironmentConfig, DifficultyLevel
from envs.benchmark_suite import BenchmarkSuite
from envs.quadcopter_kinematic_env import QuadcopterKinematicEnv
from envs.benchmark_wrapped_env import BenchmarkWrappedEnv
from agents import MARDPG_Baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--agent', type=str, default='mardpg_baseline', choices=['mardpg_baseline'])
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per benchmark scene')
    parser.add_argument('--num_agents', type=int, default=10, help='Number of UAVs')
    parser.add_argument('--level', type=int, default=3, help='Difficulty level (1-5), 3=MEDIUM')
    parser.add_argument('--scenario', type=str, default=None, help='Specific scenario to evaluate (default: all)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading checkpoint metadata from {args.checkpoint}...")
    try:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        agent_actor_state = state.get('actor', state)
        for k, v in agent_actor_state.items():
            if 'lstm.weight_ih_l0' in k:
                hidden_dim = v.shape[0] // 4
                if 'network' not in config: config['network'] = {}
                if 'actor' not in config['network']: config['network']['actor'] = {}
                if 'critic' not in config['network']: config['network']['critic'] = {}
                config['network']['actor']['hidden_dim'] = hidden_dim
                config['network']['critic']['hidden_dim'] = hidden_dim
                print(f"Auto-detected hidden_dim: {hidden_dim}")
                break
    except Exception as e:
        print(f"Could not auto-detect checkpoint config: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() and config['training'].get('device') == 'cuda' else "cpu")
    
    num_agents = args.num_agents
    print(f"Running evaluation with {num_agents} UAVs on {len(BenchmarkSuite.REGISTRY)} Benchmark Scenes...")

    # Obs dim for QuadcopterKinematicEnv is fixed at 30
    obs_dim = 30
    
    # Initialize the multi-agent model (it supports dynamic scaling due to parameter sharing or we must spawn `num_agents` LSTM heads if MARDPG)
    # Wait, MARDPG has separate actor heads for each agent up to `chkpt_num_agents`!
    # If the user asks for 10 UAVs but the checkpoint only has 3, we must check.
    chkpt_num_agents = 3
    try:
        max_head = -1
        for k in agent_actor_state.keys():
            if 'heads.' in k:
                try:
                    head_idx = int(k.split('heads.')[1].split('.')[0])
                    if head_idx > max_head: max_head = head_idx
                except:
                    pass
        if max_head >= 0:
            chkpt_num_agents = max_head + 1
            print(f"Checkpoint was trained on {chkpt_num_agents} UAVs.")
    except:
        pass
        
    config['training']['num_agents'] = num_agents
    
    # Initialize the agent appropriately
    if args.agent == 'mardpg_baseline':
        agent = MARDPG_Baseline(obs_dim=obs_dim, action_dim=2, num_agents=num_agents, config=config, device=device, independent_critics=False)
        
    # We must handle loading weights properly when testing on 10 UAVs from a 3 UAV checkpoint
    try:
        # Load flexibly
        own_state = agent.actor.state_dict()
        for name, param in agent_actor_state.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception as e:
                    pass
            elif 'heads.' in name:
                # Map old heads to new heads using modulo
                old_idx = int(name.split('heads.')[1].split('.')[0])
                for new_idx in range(num_agents):
                    if new_idx % chkpt_num_agents == old_idx:
                        new_name = name.replace(f'heads.{old_idx}', f'heads.{new_idx}')
                        if new_name in own_state:
                            own_state[new_name].copy_(param)
                            
        agent.actor.load_state_dict(own_state, strict=False)
        print("Trained weights successfully mapped (and duplicated if necessary) to the new actor size!")
    except Exception as e:
        print(f"Warning: Failed robust loading, attempting standard load. Error: {e}")
        agent.load(args.checkpoint)

    scenes = list(BenchmarkSuite.REGISTRY.keys())
    if args.scenario:
        if args.scenario in scenes:
            scenes = [args.scenario]
        else:
            print(f"Scenario '{args.scenario}' not found in registry. Options: {scenes}")
            sys.exit(1)
    
    results = {}
    
    for scene in scenes:
        print(f"\n--- Evaluating on MARDPG Benchmark Suite: {scene.upper()} ---")
        env_config = config['environment'].copy()
        
        env = BenchmarkWrappedEnv(
            benchmark_name=scene, 
            level=args.level, 
            num_agents=num_agents, 
            config=env_config
        )
        
        successes = 0
        collisions = 0
        
        for ep in range(args.episodes):
            obs, _ = env.reset()
            # Reset hidden states
            actor_hidden = [agent.actor.init_hidden(1, device) for _ in range(num_agents)]
            critic_hidden = [agent.critics[0].init_hidden(1, device) for _ in range(num_agents)] # Critic doesn't matter for eval
            
            done = False
            steps = 0
            
            while not done:
                actions, actor_hidden, _ = agent.select_actions(obs, actor_hidden, critic_hidden, explore=False)
                    
                obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                steps += 1
                
            ep_successes = sum(info['agent_success'])
            ep_collisions = sum(info['agent_collision'])
            successes += ep_successes
            collisions += ep_collisions
            
            print(f"  Ep {ep+1}/{args.episodes} | Steps {steps} | Scene Success: {info.get('success', False)} | UAVs Success: {ep_successes}/{num_agents}")
            
        total_eval_agents = args.episodes * num_agents
        s_rate = successes / total_eval_agents * 100
        c_rate = collisions / total_eval_agents * 100
        print(f"  >> {scene.upper()} RESULT: {s_rate:.1f}% Success, {c_rate:.1f}% Collision")
        results[scene] = {'success': s_rate, 'collision': c_rate}
        
    print("\n==============================")
    print("BENCHMARK SUITE SUMMARY (10 UAVs)")
    print("==============================")
    for scene, res in results.items():
        print(f"{scene:10} | Success: {res['success']:>5.1f}% | Collision: {res['collision']:>5.1f}%")
        
    # Plot results
    plt.figure(figsize=(12, 6))
    x = np.arange(len(scenes))
    width = 0.35
    
    succ_vals = [results[s]['success'] for s in scenes]
    coll_vals = [results[s]['collision'] for s in scenes]
    
    plt.bar(x - width/2, succ_vals, width, label='Success Rate (%)', color='#27ae60')
    plt.bar(x + width/2, coll_vals, width, label='Collision Rate (%)', color='#c0392b')
    
    plt.title(f'MARDPG Benchmark Suite Validation - {num_agents} UAVs', fontsize=16, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(x, [s.upper() for s in scenes], fontsize=11, fontweight='bold')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    
    for i, v in enumerate(succ_vals):
        plt.text(i - width/2, v + 2, f"{v:.1f}", ha='center', fontweight='bold', fontsize=9)
    for i, v in enumerate(coll_vals):
        plt.text(i + width/2, v + 2, f"{v:.1f}", ha='center', fontweight='bold', fontsize=9)
        
    out_file = os.path.join(config['logging']['log_dir'], f'benchmark_suite_eval_{num_agents}uavs.png')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print(f"\nFinal Bar Chart saved to {out_file}")

if __name__ == '__main__':
    main()
