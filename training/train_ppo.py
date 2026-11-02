"""
Phase 2: Train the base self-play MaskablePPO agent.
Uses sb3-contrib's MaskablePPO with ActionMasker to strictly enforce legal action masking.
Run: python -m training.train_ppo
"""

import json
import os
import sys
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from env.gym_wrapper import AgentZeroEnv
from training.self_play_pool import SelfPlayPool
from training.evaluate_ppo import evaluate_policy, random_opponent, heuristic_opponent, run_full_evaluation

CHECKPOINT_DIR = "checkpoints/ppo"
RESULTS_PATH = "data/results.json"
TOTAL_TIMESTEPS = 500_000
SNAPSHOT_EVERY = 50_000
SEED = 42


def mask_fn(env: AgentZeroEnv) -> np.ndarray:
    return env.action_masks()


def make_env(pool: SelfPlayPool, seed: int = SEED):
    def _init():
        opponent_policy = pool.sample_opponent_policy()
        env = AgentZeroEnv(num_players=2, opponent_policy=opponent_policy, seed=seed)
        return ActionMasker(env, mask_fn)
    return _init


def train_self_play_ppo(total_timesteps: int = TOTAL_TIMESTEPS, snapshot_every: int = SNAPSHOT_EVERY):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    pool = SelfPlayPool(CHECKPOINT_DIR)
    env = make_env(pool, seed=SEED)()

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./tb_logs/ppo",
        seed=SEED,
    )

    steps_done = 0
    progression_logs = []

    print("==================================================")
    print(f"STARTING PPO SELF-PLAY TRAINING ({total_timesteps:,} TIMESTEPS)")
    print("==================================================")

    while steps_done < total_timesteps:
        chunk = min(snapshot_every, total_timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk

        # Add snapshot checkpoint to self-play pool
        pool.add_checkpoint(model, steps_done, algo_name="ppo")

        # Evaluate against baselines
        eval_rnd = evaluate_policy(model, random_opponent, n_games=100, base_seed=10000 + steps_done)
        eval_h = evaluate_policy(model, heuristic_opponent, n_games=100, base_seed=20000 + steps_done)

        print(f"[Step {steps_done:6d}/{total_timesteps}] Win-rate vs Random: {eval_rnd['win_rate']:.1%} | vs Heuristic: {eval_h['win_rate']:.1%} | Illegal Attempts: {eval_rnd['illegal_attempts']}")

        progression_logs.append({
            "step": steps_done,
            "win_rate_random": eval_rnd["win_rate"],
            "avg_reward_random": eval_rnd["avg_reward"],
            "avg_len_random": eval_rnd["avg_length"],
            "win_rate_heuristic": eval_h["win_rate"],
            "avg_reward_heuristic": eval_h["avg_reward"],
            "avg_len_heuristic": eval_h["avg_length"],
        })

        # Update environment with fresh opponent sampled from self-play pool
        env = make_env(pool, seed=SEED + steps_done)()
        model.set_env(env)

    final_model_path = os.path.join(CHECKPOINT_DIR, "ppo_final")
    model.save(final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}.zip")

    # Run full final evaluation
    first_ckpt = os.path.join(CHECKPOINT_DIR, f"ppo_step{snapshot_every}.zip")
    final_eval = run_full_evaluation(final_model_path + ".zip", checkpoint_opp_path=first_ckpt, n_games=500)

    # Save to data/results.json
    results_data = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results_data = json.load(f)
        except Exception:
            results_data = {}

    results_data["ppo_training_progression"] = progression_logs
    results_data["ppo_final_evaluation"] = final_eval
    if "algorithm_comparison" not in results_data:
        results_data["algorithm_comparison"] = {}
    results_data["algorithm_comparison"]["PPO"] = {
        "win_rate": final_eval["random"]["win_rate"],
        "avg_reward": final_eval["random"]["avg_reward"],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Saved evaluation results to {RESULTS_PATH}")
    return model, final_eval, progression_logs


if __name__ == "__main__":
    train_self_play_ppo()
