"""
Phase 2 Step 3 & Step 6 Smoke Tests.
Verifies PPO initialization, MaskablePPO action masking, rollout collection,
checkpoint saving/loading, self-play opponent interaction, and inference.

Run: python -m training.run_smoke_tests
"""

import os
import shutil
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from stable_baselines3.common.env_util import make_vec_env


from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from training.self_play_pool import SelfPlayPool
from training.evaluate_ppo import evaluate_policy, random_opponent

SMOKE_DIR = "checkpoints/ppo_smoke"


def mask_fn(env: AgentZeroEnv) -> np.ndarray:
    return env.action_masks()


def make_smoke_env(pool: SelfPlayPool):
    def _init():
        opp_policy = pool.sample_opponent_policy()
        env = AgentZeroEnv(num_players=2, opponent_policy=opp_policy, seed=42)
        return ActionMasker(env, mask_fn)
    return _init


def run_ppo_smoke_test():
    print("==================================================")
    print("STEP 3 — PPO SMOKE TEST")
    print("==================================================")

    if os.path.exists(SMOKE_DIR):
        shutil.rmtree(SMOKE_DIR)
    os.makedirs(SMOKE_DIR, exist_ok=True)

    pool = SelfPlayPool(SMOKE_DIR)
    env = make_smoke_env(pool)()

    # Initialize MaskablePPO
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=256,
        batch_size=64,
        n_epochs=3,
        learning_rate=3e-4,
        seed=123,
    )

    smoke_timesteps = 1024
    model.learn(total_timesteps=smoke_timesteps)

    save_path = os.path.join(SMOKE_DIR, "smoke_model.zip")
    model.save(save_path)
    print(f"[Smoke Test] Model trained for {smoke_timesteps} timesteps and saved to: {save_path}")

    # Verify model reloading
    reloaded_model = MaskablePPO.load(save_path)
    print("[Smoke Test] Reloaded MaskablePPO model successfully.")

    # Verify inference & action masking
    test_env = AgentZeroEnv(num_players=2, seed=999)
    obs, _ = test_env.reset()
    mask = test_env.action_masks()
    action, _ = reloaded_model.predict(obs, action_masks=mask, deterministic=True)
    decoded = AgentZeroEnv._decode_action(int(action))
    legal = test_env.engine.legal_actions(0)
    
    print(f"[Smoke Test] Predicted action: {action} ({decoded.name if decoded != -1 else 'DRAW'})")
    print(f"[Smoke Test] Mask at action 0 (HITMAN): {mask[0]} (Must be False)")
    print(f"[Smoke Test] Action is legal: {decoded in legal}")
    assert not mask[0], "HITMAN action mask was True!"
    assert decoded in legal, "Predicted action was not legal!"

    eval_res = evaluate_policy(reloaded_model, random_opponent, n_games=50, base_seed=7000)
    print(f"[Smoke Test] Eval vs Random (50 games): Win Rate = {eval_res['win_rate']:.1%}, Avg Reward = {eval_res['avg_reward']:.2f}, Illegal Attempts = {eval_res['illegal_attempts']}")
    assert eval_res["illegal_attempts"] == 0, "Illegal action attempts detected during evaluation!"

    print("Step 3 Smoke Test: PASSED!\n")
    return reloaded_model, save_path


def run_self_play_smoke_test(base_model, initial_ckpt_path):
    print("==================================================")
    print("STEP 6 — SELF-PLAY SMOKE TEST")
    print("==================================================")

    pool = SelfPlayPool(SMOKE_DIR)
    # Add initial checkpoint to pool
    pool.add_checkpoint(base_model, step=1000)
    print(f"[Self-Play Smoke] Added initial checkpoint to pool. Pool size: {len(pool._models)}")

    # Sample opponent from pool
    opp_fn = pool.sample_opponent_policy()
    assert opp_fn is not None, "Opponent policy sampling failed!"

    # Create self-play env with sampled checkpoint opponent
    raw_env = AgentZeroEnv(num_players=2, opponent_policy=opp_fn, seed=888)
    env = ActionMasker(raw_env, mask_fn)

    # Train current learner against checkpoint opponent for 1024 timesteps
    learner = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=256,
        batch_size=64,
        n_epochs=3,
        learning_rate=3e-4,
        seed=456,
    )
    learner.learn(total_timesteps=1024)

    learner_ckpt_path = os.path.join(SMOKE_DIR, "self_play_learner.zip")
    learner.save(learner_ckpt_path)
    pool.add_checkpoint(learner, step=2000)
    print(f"[Self-Play Smoke] Trained learner for 1024 timesteps vs checkpoint opponent.")
    print(f"[Self-Play Smoke] Updated pool size: {len(pool._models)}")

    # Evaluate learner against historical checkpoint opponent
    res_vs_ckpt = evaluate_policy(learner, opp_fn, n_games=50, base_seed=9000)
    print(f"[Self-Play Smoke] Learner vs Historical Checkpoint (50 games):")
    print(f"   Win Rate: {res_vs_ckpt['win_rate']:.1%}")
    print(f"   Avg Reward: {res_vs_ckpt['avg_reward']:.2f}")
    print(f"   Avg Episode Length: {res_vs_ckpt['avg_length']:.1f}")
    print(f"   Illegal Action Attempts: {res_vs_ckpt['illegal_attempts']}")

    assert res_vs_ckpt["illegal_attempts"] == 0, "Illegal action attempts in self-play!"
    print("Step 6 Self-Play Smoke Test: PASSED!\n")


if __name__ == "__main__":
    base_model, ckpt_path = run_ppo_smoke_test()
    run_self_play_smoke_test(base_model, ckpt_path)
