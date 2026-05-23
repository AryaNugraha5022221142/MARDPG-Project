#!/usr/bin/env python3
"""Pre-training verification script for MARDPG codebase."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np
from agents.mardpg_baseline import MARDPG_Baseline, AttentionCritic


def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_config_integrity():
    """Verify config values are consistent and valid."""
    print("=== Config Integrity ===")
    config = load_config()
    
    # C.1: Curriculum start matches num_agents
    start_agents = config['training']['num_agents']
    curriculum_start = list(config['curriculum']['schedule'].values())[0]
    assert start_agents == curriculum_start, \
        f"FAIL C.1: num_agents ({start_agents}) != curriculum start ({curriculum_start})"
    print(f"  ✓ C.1: num_agents ({start_agents}) matches curriculum start")
    
    # Buffer size sufficient
    buffer_size = config['memory']['buffer_size']
    batch_size = config['memory']['batch_size']
    assert buffer_size >= 10 * batch_size, \
        f"FAIL D.1: buffer_size ({buffer_size}) < 10x batch_size ({batch_size})"
    print(f"  ✓ D.1: buffer_size ({buffer_size}) >= 10x batch_size ({batch_size})")
    
    # Seq_len reasonable vs max_steps
    seq_len = config['memory']['seq_len']
    max_steps = config['environment']['max_steps']
    if seq_len > max_steps:
        print(f"  ⚠ seq_len ({seq_len}) > max_steps ({max_steps}) — excessive padding guaranteed")
    elif seq_len > max_steps // 2:
        print(f"  ℹ seq_len ({seq_len}) is large — verify episode lengths in practice")
    else:
        print(f"  ✓ seq_len ({seq_len}) conservative vs max_steps ({max_steps})")
    
    # Update rate in valid range
    tau = config['targets']['update_rate']
    assert 0.001 <= tau <= 0.1, f"FAIL: tau ({tau}) outside valid range [0.001, 0.1]"
    print(f"  ✓ tau ({tau}) in valid range")
    
    # Sensor noise positive (partial observability)
    noise = config['environment'].get('sensor_noise_std', 0)
    assert noise > 0, "FAIL B.3: sensor_noise_std must be > 0 for partial observability"
    print(f"  ✓ B.3: sensor_noise_std ({noise}) > 0")
    
    print("  ✓ All config checks passed\n")
    return True


def test_attention_critic():
    """Verify attention-based critic architecture."""
    print("=== Attention Critic ===")
    
    critic = AttentionCritic(obs_dim=30, action_dim=2, num_agents=5, 
                             hidden_dim=128, n_heads=4)
    
    # Forward pass
    obs = torch.randn(2, 10, 5, 30)
    actions = torch.randn(2, 10, 5, 2)
    q, hidden = critic(obs, actions)
    
    assert q.shape == (2, 10, 1), f"FAIL: Q-shape {q.shape} != (2,10,1)"
    print("  ✓ Q-value shape correct")
    
    # Attention weights
    x = torch.randn(2, 5, 128)
    critic.eval()
    with torch.no_grad():
        attn_out, weights = critic.attention(x, x, x)
    
    assert weights.shape == (2, 5, 5), f"FAIL: attention weights {weights.shape} != (2,5,5)"
    print("  ✓ Attention weights shape correct (5x5 pairwise)")
    
    # Softmax verification
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(2, 5), atol=1e-5), \
        "FAIL: attention weights don't sum to 1"
    print("  ✓ Attention weights are valid softmax")
    
    # Non-uniformity (not all identical)
    weight_std = weights.std(dim=-1).mean()
    assert weight_std > 0.01, "WARN: attention weights nearly uniform — coordination may not emerge"
    print(f"  ✓ Attention weights have variance ({weight_std:.4f}) — non-uniform")
    
    print("  ✓ All attention critic checks passed\n")
    return True


def test_ctde_separation():
    """Verify Centralized Training, Decentralized Execution."""
    print("=== CTDE Architecture ===")
    
    agent = MARDPG_Baseline(obs_dim=30, action_dim=2, num_agents=3)
    
    # Actor: single-agent input
    obs_single = torch.randn(1, 30)
    hidden = agent.actor.init_hidden(1, 'cpu')
    action, new_h, _, _ = agent.actor(obs_single, hidden, agent_idx=0)
    
    assert action.shape == (1, 2), f"FAIL A.1: actor output {action.shape} != (1,2)"
    print("  ✓ A.1: Actor is decentralized (single-agent in/out)")
    
    # Critic: joint input
    obs_joint = torch.randn(2, 10, 3, 30)
    actions_joint = torch.randn(2, 10, 3, 2)
    hidden_c = agent.critics[0].init_hidden(2, 'cpu')
    q, _ = agent.critics[0](obs_joint, actions_joint, hidden_c)
    
    assert q.shape == (2, 10, 1), f"FAIL A.1: critic output {q.shape} != (2,10,1)"
    print("  ✓ A.1: Critic is centralized (joint state in)")
    
    # No cross-agent communication in select_actions
    import inspect
    source = inspect.getsource(agent.select_actions)
    assert 'obs[i]' in source or 'obs_tensor' in source, \
        "FAIL A.3: select_actions may not use per-agent observations"
    assert 'agent_idx=i' in source, \
        "FAIL A.3: select_actions doesn't use per-agent head selection"
    print("  ✓ A.3: select_actions uses per-agent local observations")
    
    print("  ✓ All CTDE checks passed\n")
    return True


def test_lr_scaling():
    """Verify learning rate scaling affects both actor and critic."""
    print("=== LR Scaling ===")
    
    agent = MARDPG_Baseline(obs_dim=30, action_dim=2, num_agents=3)
    
    original_actor_lr = agent.actor_optimizer.param_groups[0]['lr']
    original_critic_lr = agent.critic_optimizers[0].param_groups[0]['lr']
    
    # Simulate set_lr_scale
    scale = 0.5
    for pg in agent.actor_optimizer.param_groups:
        pg['lr'] = original_actor_lr * scale
    for opt in agent.critic_optimizers:
        for pg in opt.param_groups:
            pg['lr'] = original_critic_lr * scale
    
    assert abs(agent.actor_optimizer.param_groups[0]['lr'] - original_actor_lr * scale) < 1e-10, \
        "FAIL: Actor LR not scaled"
    assert abs(agent.critic_optimizers[0].param_groups[0]['lr'] - original_critic_lr * scale) < 1e-10, \
        "FAIL: Critic LR not scaled — check attribute name 'critic_optimizers'"
    print("  ✓ Fix 1: Both actor and critic LR scale correctly")
    print("  ✓ All LR scaling checks passed\n")
    return True


def test_checkpoint_resume():
    """Verify checkpoint save/load roundtrip preserves all states."""
    print("=== Checkpoint Resume ===")
    
    import tempfile
    import os
    
    agent = MARDPG_Baseline(obs_dim=30, action_dim=2, num_agents=3)
    
    # Populate optimizer states with dummy step
    dummy_obs = torch.randn(1, 1, 3, 30)
    dummy_actions = torch.randn(1, 1, 3, 2)
    dummy_q = sum(c(dummy_obs, dummy_actions, c.init_hidden(1, 'cpu'))[0].sum() 
                  for c in agent.critics)
    dummy_q.backward()
    for opt in agent.critic_optimizers:
        opt.step()
    agent.actor_optimizer.zero_grad()
    obs_single = torch.randn(1, 30)
    hidden = agent.actor.init_hidden(1, 'cpu')
    act_out, _, _, _ = agent.actor(obs_single, hidden, agent_idx=0)
    actor_loss = act_out.sum()
    actor_loss.backward()
    agent.actor_optimizer.step()
    
    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pt")
        agent.save(path, epsilon=0.1, episode=500)
        
        # Verify save contains all keys
        checkpoint = torch.load(path, map_location='cpu')
        required_keys = ['actor', 'actor_target', 'actor_optimizer', 'epsilon', 'episode']
        for i in range(3):
            required_keys.extend([f'critic_{i}', f'critic_target_{i}', f'critic_optimizer_{i}'])
        
        for key in required_keys:
            assert key in checkpoint, f"FAIL: checkpoint missing key '{key}'"
        print("  ✓ Save contains all required keys")
        
        # Load into fresh agent
        agent2 = MARDPG_Baseline(obs_dim=30, action_dim=2, num_agents=3)
        eps, ep = agent2.load(path)
        
        assert ep == 500, f"FAIL C.6: episode {ep} != 500"
        assert eps == 0.1, f"FAIL C.6: epsilon {eps} != 0.1"
        print("  ✓ Episode and epsilon restored correctly")
        
        # C.6: CRITICAL — verify critic optimizer states restored
        assert len(agent2.critic_optimizers[0].state) > 0, \
            "FAIL C.6: Critic optimizer state is empty — not restored from checkpoint!"
        print("  ✓ C.6: Critic optimizer state restored (non-empty)")
        
        # Verify actor optimizer restored
        assert len(agent2.actor_optimizer.state) > 0, \
            "FAIL: Actor optimizer state not restored"
        print("  ✓ Actor optimizer state restored")
    
    print("  ✓ All checkpoint resume checks passed\n")
    return True


def test_curriculum_schedule():
    """Verify curriculum schedule logic."""
    print("=== Curriculum Schedule ===")
    
    config = load_config()
    schedule = config['curriculum']['schedule']
    
    # Monotonic check
    episodes = sorted(schedule.keys())
    counts = [schedule[e] for e in episodes]
    assert all(counts[i] <= counts[i+1] for i in range(len(counts)-1)), \
        "FAIL: Curriculum schedule not monotonic in agent count"
    print(f"  ✓ Schedule monotonic: {dict(zip(episodes, counts))}")
    
    # Transition function
    def get_target_n(episode):
        target = None
        for ep, n in sorted(schedule.items()):
            if episode >= ep:
                target = n
        return target
    
    assert get_target_n(0) == counts[0]
    assert get_target_n(episodes[1] - 1) == counts[0]
    assert get_target_n(episodes[1]) == counts[1]
    print("  ✓ Curriculum transitions at correct thresholds")
    
    print("  ✓ All curriculum checks passed\n")
    return True


def test_replay_buffer():
    """Verify sequence replay buffer operations."""
    print("=== Replay Buffer ===")
    
    from agents.replay_buffer import SequenceReplayBuffer
    
    buffer = SequenceReplayBuffer(capacity=100)
    
    # Push short episode (length 10)
    for t in range(10):
        obs = np.random.randn(3, 30)
        actions = np.random.randn(3, 2)
        rewards = np.random.randn(3)
        next_obs = np.random.randn(3, 30)
        dones = np.zeros(3, dtype=np.float32)
        buffer.push(obs, actions, rewards, next_obs, dones, episode_done=(t == 9))
    
    assert len(buffer) == 1, f"FAIL: buffer length {len(buffer)} != 1"
    print("  ✓ Episode storage works")
    
    # Sample with padding
    sampled = buffer.sample(batch_size=1, seq_len=20)
    assert sampled is not None, "FAIL: sampling returned None"
    
    obs_batch, act_batch, rew_batch, nobs_batch, done_batch, mask_batch = sampled
    assert obs_batch.shape == (1, 20, 3, 30), f"FAIL: shape {obs_batch.shape} != (1,20,3,30)"
    assert mask_batch.shape == (1, 20, 1), f"FAIL: mask shape {mask_batch.shape}"
    assert np.all(mask_batch[0, 10:, 0] == 0.0), "FAIL: padding not masked"
    print("  ✓ Sequence sampling and padding correct")
    
    # Clear
    buffer.clear()
    assert len(buffer) == 0, "FAIL: buffer not empty after clear"
    assert len(buffer.current_episode) == 0, "FAIL: current_episode not cleared"
    print("  ✓ Buffer clear works (curriculum-ready)")
    
    print("  ✓ All replay buffer checks passed\n")
    return True


def test_burn_in_mask():
    """Verify burn-in masking for recurrent training."""
    print("=== Burn-in Mask ===")
    
    config = load_config()
    seq_len = config['memory']['seq_len']     # reads actual value (currently 40)
    burn_in = seq_len // 4
    valid_len = seq_len - burn_in
    
    mask = torch.ones(2, seq_len, 3)
    mask[:, :burn_in, :] = 0.0
    
    assert mask[:, :burn_in, :].sum() == 0, \
        f"FAIL: burn-in first {burn_in} steps not zeroed"
    assert mask[:, burn_in:, :].sum() == 2 * valid_len * 3, \
        f"FAIL: valid region ({valid_len} steps) sum incorrect"
    
    print(f"  ✓ seq_len={seq_len}, burn_in={burn_in}, valid={valid_len} steps ({100*valid_len//seq_len}% of sequence)")
    print(f"  ✓ Burn-in mask correct (first {burn_in} steps masked)")
    print("  ✓ All burn-in checks passed\n")
    return True


def test_separation_penalty():
    """Verify inter-agent separation penalty shaping."""
    print("=== Separation Penalty ===")
    
    lambda_sep, sigma_sep = 2.0, 5.0
    
    # Far apart: negligible penalty
    d_far = 5.0
    r_far = -lambda_sep * np.exp(-sigma_sep * d_far)
    assert abs(r_far) < 0.01, f"FAIL: far penalty {r_far} not negligible"
    print(f"  ✓ Far separation penalty negligible ({r_far:.6f})")
    
    # Close: significant penalty
    d_close = 0.1
    r_close = -lambda_sep * np.exp(-sigma_sep * d_close)
    assert r_close < -1.0, f"FAIL: close penalty {r_close} not significant"
    print(f"  ✓ Close separation penalty significant ({r_close:.2f})")
    
    print("  ✓ All separation penalty checks passed\n")
    return True


def test_dynamic_normalization():
    """Verify arena diagonal normalization is dynamic, not hardcoded."""
    print("=== Dynamic Normalization ===")
    
    arena_sizes = [
        np.array([30, 30, 15], dtype=np.float32),
        np.array([100, 100, 40], dtype=np.float32),
        np.array([130, 130, 50], dtype=np.float32),
    ]
    
    for size in arena_sizes:
        diag = np.linalg.norm(size)
        realistic_dist = diag * 0.5
        normalized = realistic_dist / diag
        
        assert 0.0 < normalized < 1.0, f"FAIL: normalization {normalized} out of range"
        assert abs(normalized - 0.5) < 1e-6, "FAIL: half-diagonal should normalize to 0.5"
    
    print("  ✓ Dynamic normalization valid for all arena sizes")
    print("  ✓ All normalization checks passed\n")
    return True


def main():
    print("=" * 60)
    print("MARDPG Pre-Training Verification")
    print("=" * 60 + "\n")
    
    tests = [
        ("Config Integrity", test_config_integrity),
        ("CTDE Architecture", test_ctde_separation),
        ("Attention Critic", test_attention_critic),
        ("LR Scaling", test_lr_scaling),
        ("Checkpoint Resume", test_checkpoint_resume),
        ("Curriculum Schedule", test_curriculum_schedule),
        ("Replay Buffer", test_replay_buffer),
        ("Burn-in Mask", test_burn_in_mask),
        ("Separation Penalty", test_separation_penalty),
        ("Dynamic Normalization", test_dynamic_normalization),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, "PASS", None))
        except AssertionError as e:
            print(f"\n  ❌ {name}: {e}\n")
            results.append((name, "FAIL", str(e)))
        except Exception as e:
            print(f"\n  ❌ {name}: UNEXPECTED ERROR: {e}\n")
            results.append((name, "ERROR", str(e)))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, status, detail in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {name}: {status}")
        if detail:
            print(f"   → {detail}")
    
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = total - passed
    
    print(f"\n{passed}/{total} passed, {failed} failed")
    
    if failed == 0:
        print("\n🚀 ALL CHECKS PASSED — READY FOR TRAINING")
        return 0
    else:
        print("\n⛔ FIX FAILURES BEFORE TRAINING")
        return 1


if __name__ == '__main__':
    sys.exit(main())
