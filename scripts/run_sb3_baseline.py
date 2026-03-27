"""
Runs stable-baselines3 baselines against the gymnasium env.
Provides comparison points for the thesis: PPO, SAC, TD3 on 'urban_canyon'.

These single-agent baselines (treating all UAVs as one policy) establish
a lower bound that MARDPG with CTDE should exceed on cooperative metrics.
"""
import gymnasium as gym
import envs.gymnasium_env    # triggers registration
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import argparse
import os

ALGOS = {'ppo': PPO, 'sac': SAC, 'td3': TD3}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=list(ALGOS.keys()), default='ppo')
    parser.add_argument('--scenario', default='urban_canyon')
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    log_dir = f'logs/sb3_{args.algo}_{args.scenario}'
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env(
        'MultiUAVNav-v0',
        n_envs=4,
        seed=args.seed,
        env_kwargs={'scenario': args.scenario}
    )
    eval_env = gym.make('MultiUAVNav-v0', scenario=args.scenario)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    AlgoClass = ALGOS[args.algo]
    model = AlgoClass(
        'MlpPolicy',
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=log_dir,
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    model.save(f'{log_dir}/{args.algo}_final')
    print(f"Training complete. Model saved to {log_dir}/")


if __name__ == '__main__':
    main()
