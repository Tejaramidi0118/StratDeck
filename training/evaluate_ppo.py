"""
Phase 2 Evaluation Pipeline for Trained PPO Checkpoints.
Evaluates a MaskablePPO model against:
1. Random legal-action opponent
2. Rule-based heuristic opponent
3. Historical checkpoint opponent (if provided)

Run: python -m training.evaluate_ppo --model checkpoints/ppo/ppo_final.zip
"""

import argparse
import os
import random
import numpy as np
from sb3_contrib import MaskablePPO
from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from env.cards import Card


def random_opponent(raw_obs: dict, legal_actions: list):
    return random.choice(legal_actions)


def heuristic_opponent(raw_obs: dict, legal_actions: list):
    action_cards = [a for a in legal_actions if a != -1]
    if action_cards and random.random() < 0.8:
        return random.choice(action_cards)
    return -1


def evaluate_policy(model, opponent_fn, n_games=500, base_seed=5000):
    wins = 0
    total_reward = 0.0
    lengths = []
    illegal_attempts = 0

    for i in range(n_games):
        seed = base_seed + i
        env = AgentZeroEnv(num_players=2, opponent_policy=opponent_fn, seed=seed)
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        while not done and steps < 300:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            
            # Verify action is legal
            decoded = AgentZeroEnv._decode_action(int(action))
            legal = env.engine.legal_actions(env.agent_player_id)
            if decoded not in legal:
                illegal_attempts += 1

            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += reward
            steps += 1

        total_reward += ep_reward
        lengths.append(steps)
        if env.engine.winner == 0:
            wins += 1


    win_rate = wins / n_games
    avg_reward = total_reward / n_games
    avg_len = sum(lengths) / len(lengths)
    return {
        "games": n_games,
        "wins": wins,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_len,
        "illegal_attempts": illegal_attempts,
    }


def load_any_model(model_path: str):
    from training.masked_algos import MaskedDQN, MaskedA2C
    base = os.path.basename(model_path).lower()
    if "dqn" in base:
        return MaskedDQN.load(model_path)
    elif "a2c" in base:
        return MaskedA2C.load(model_path)
    else:
        return MaskablePPO.load(model_path)


def run_full_evaluation(model_path: str, checkpoint_opp_path: str = None, n_games: int = 500):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = load_any_model(model_path)
    print(f"Loaded model from: {model_path}")

    # 1. Vs Random Opponent
    res_rnd = evaluate_policy(model, random_opponent, n_games=n_games, base_seed=10000)
    print("\n--- Evaluation vs RANDOM Baseline ---")
    print(f"Games: {res_rnd['games']}, Win Rate: {res_rnd['win_rate']:.1%}, Avg Reward: {res_rnd['avg_reward']:.2f}, Avg Length: {res_rnd['avg_length']:.1f}, Illegal Attempts: {res_rnd['illegal_attempts']}")

    # 2. Vs Heuristic Opponent
    res_h = evaluate_policy(model, heuristic_opponent, n_games=n_games, base_seed=20000)
    print("\n--- Evaluation vs HEURISTIC Baseline ---")
    print(f"Games: {res_h['games']}, Win Rate: {res_h['win_rate']:.1%}, Avg Reward: {res_h['avg_reward']:.2f}, Avg Length: {res_h['avg_length']:.1f}, Illegal Attempts: {res_h['illegal_attempts']}")

    # 3. Vs Historical Checkpoint Opponent (if provided)
    res_ckpt = None
    if checkpoint_opp_path and os.path.exists(checkpoint_opp_path):
        opp_model = load_any_model(checkpoint_opp_path)
        def ckpt_opponent(raw_obs: dict, legal_actions: list):
            obs_vec = AgentZeroEnv.encode_raw_obs(raw_obs)
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            for a in legal_actions:
                mask[AgentZeroEnv._encode_action(a)] = True
            act_id, _ = opp_model.predict(obs_vec, action_masks=mask, deterministic=False)
            return AgentZeroEnv._decode_action(int(act_id))

        res_ckpt = evaluate_policy(model, ckpt_opponent, n_games=n_games, base_seed=30000)
        print(f"\n--- Evaluation vs CHECKPOINT Opponent ({os.path.basename(checkpoint_opp_path)}) ---")
        print(f"Games: {res_ckpt['games']}, Win Rate: {res_ckpt['win_rate']:.1%}, Avg Reward: {res_ckpt['avg_reward']:.2f}, Avg Length: {res_ckpt['avg_length']:.1f}, Illegal Attempts: {res_ckpt['illegal_attempts']}")

    return {"random": res_rnd, "heuristic": res_h, "checkpoint": res_ckpt}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/ppo/ppo_final.zip")
    parser.add_argument("--opponent_checkpoint", type=str, default=None)
    parser.add_argument("--games", type=int, default=500)
    args = parser.parse_args()

    run_full_evaluation(args.model, args.opponent_checkpoint, args.games)
