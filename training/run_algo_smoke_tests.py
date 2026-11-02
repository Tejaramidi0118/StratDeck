"""
Phase 3 Smoke Tests for DQN and A2C.
Verifies MaskedDQN and MaskedA2C initialization, action masking, replay buffer/policy updates,
checkpoint saving/loading, self-play opponent interaction, and inference.

Run: python -m training.run_algo_smoke_tests
"""

import os
import shutil
import numpy as np

from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from training.masked_algos import MaskedDQN, MaskedA2C
from training.self_play_pool import SelfPlayPool
from training.evaluate_ppo import evaluate_policy, random_opponent, heuristic_opponent

DQN_SMOKE_DIR = "checkpoints/dqn_smoke"
A2C_SMOKE_DIR = "checkpoints/a2c_smoke"


def run_dqn_smoke_test():
    print("==================================================")
    print("STEP 4 — DQN SMOKE TEST")
    print("==================================================")

    if os.path.exists(DQN_SMOKE_DIR):
        shutil.rmtree(DQN_SMOKE_DIR)
    os.makedirs(DQN_SMOKE_DIR, exist_ok=True)

    pool = SelfPlayPool(DQN_SMOKE_DIR)
    env = AgentZeroEnv(num_players=2, seed=42)

    model = MaskedDQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        buffer_size=10_000,
        learning_starts=500,
        batch_size=32,
        exploration_fraction=0.3,
        seed=42,
    )

    smoke_timesteps = 2500
    model.learn(total_timesteps=smoke_timesteps)

    save_path = os.path.join(DQN_SMOKE_DIR, "dqn_smoke.zip")
    model.save(save_path)
    print(f"[DQN Smoke] Model trained for {smoke_timesteps} timesteps and saved to: {save_path}")

    # Verify reload
    reloaded = MaskedDQN.load(save_path)
    print("[DQN Smoke] Reloaded MaskedDQN model successfully.")

    # Evaluate
    res = evaluate_policy(reloaded, random_opponent, n_games=50, base_seed=8000)
    print(f"[DQN Smoke] Eval vs Random (50 games): Win Rate = {res['win_rate']:.1%}, Avg Reward = {res['avg_reward']:.2f}, Illegal Attempts = {res['illegal_attempts']}")
    assert res["illegal_attempts"] == 0, "Illegal action attempts in DQN!"

    # Self-play test
    pool.add_checkpoint(reloaded, step=smoke_timesteps, algo_name="dqn")
    opp_fn = pool.sample_opponent_policy()
    sp_env = AgentZeroEnv(num_players=2, opponent_policy=opp_fn, seed=888)
    sp_learner = MaskedDQN("MlpPolicy", sp_env, verbose=0, seed=43)
    sp_learner.learn(total_timesteps=1000)
    print("[DQN Smoke] Self-play step complete with 0 errors.")

    print("Step 4 DQN Smoke Test: PASSED!\n")


def run_a2c_smoke_test():
    print("==================================================")
    print("STEP 5 — A2C SMOKE TEST")
    print("==================================================")

    if os.path.exists(A2C_SMOKE_DIR):
        shutil.rmtree(A2C_SMOKE_DIR)
    os.makedirs(A2C_SMOKE_DIR, exist_ok=True)

    pool = SelfPlayPool(A2C_SMOKE_DIR)
    env = AgentZeroEnv(num_players=2, seed=42)

    model = MaskedA2C(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=7e-4,
        n_steps=64,
        seed=42,
    )

    smoke_timesteps = 2500
    model.learn(total_timesteps=smoke_timesteps)

    save_path = os.path.join(A2C_SMOKE_DIR, "a2c_smoke.zip")
    model.save(save_path)
    print(f"[A2C Smoke] Model trained for {smoke_timesteps} timesteps and saved to: {save_path}")

    # Verify reload
    reloaded = MaskedA2C.load(save_path)
    print("[A2C Smoke] Reloaded MaskedA2C model successfully.")

    # Evaluate
    res = evaluate_policy(reloaded, random_opponent, n_games=50, base_seed=8500)
    print(f"[A2C Smoke] Eval vs Random (50 games): Win Rate = {res['win_rate']:.1%}, Avg Reward = {res['avg_reward']:.2f}, Illegal Attempts = {res['illegal_attempts']}")
    assert res["illegal_attempts"] == 0, "Illegal action attempts in A2C!"

    # Self-play test
    pool.add_checkpoint(reloaded, step=smoke_timesteps, algo_name="a2c")
    opp_fn = pool.sample_opponent_policy()
    sp_env = AgentZeroEnv(num_players=2, opponent_policy=opp_fn, seed=888)
    sp_learner = MaskedA2C("MlpPolicy", sp_env, verbose=0, seed=43)
    sp_learner.learn(total_timesteps=1000)
    print("[A2C Smoke] Self-play step complete with 0 errors.")

    print("Step 5 A2C Smoke Test: PASSED!\n")


if __name__ == "__main__":
    run_dqn_smoke_test()
    run_a2c_smoke_test()
