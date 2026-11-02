"""
Phase 4: Trajectory Data Collection Pipeline.
Collects 100 complete game move histories per policy source across 8 agent policies:
1. Random
2. Heuristic
3. Early PPO (step 50k)
4. Final PPO (step 500k)
5. Early DQN (step 50k)
6. Final DQN (step 500k)
7. Early A2C (step 50k)
8. Final A2C (step 500k)

Total samples: 800 (8 sources * 100 games).
One sample = one player's behavior over one complete game.
"""

import sys
import os
import random
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from env.cards import Card
from training.evaluate_ppo import load_any_model, random_opponent, heuristic_opponent

POLICY_SOURCES = {
    "Random": None,
    "Heuristic": None,
    "PPO_early": "checkpoints/ppo/ppo_step50000.zip",
    "PPO_final": "checkpoints/ppo/ppo_final.zip",
    "DQN_early": "checkpoints/dqn/dqn_step50000.zip",
    "DQN_final": "checkpoints/dqn/dqn_final.zip",
    "A2C_early": "checkpoints/a2c/a2c_step50000.zip",
    "A2C_final": "checkpoints/a2c/a2c_final.zip",
}


def make_agent_policy(source_name: str, ckpt_path: str):
    if source_name == "Random":
        return lambda obs, legal: random.choice(legal)
    elif source_name == "Heuristic":
        return lambda obs, legal: (
            random.choice([a for a in legal if a != -1])
            if [a for a in legal if a != -1] and random.random() < 0.8
            else -1
        )
    else:
        model = load_any_model(ckpt_path)
        def model_policy(obs_raw, legal_actions):
            obs_vec = AgentZeroEnv.encode_raw_obs(obs_raw)
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            for a in legal_actions:
                mask[AgentZeroEnv._encode_action(a)] = True
            act_id, _ = model.predict(obs_vec, action_masks=mask, deterministic=False)
            return AgentZeroEnv._decode_action(int(act_id))
        return model_policy


def collect_dataset(games_per_source: int = 100, base_seed: int = 42) -> list:
    dataset = []
    sample_counter = 0

    print("=== Phase 4: Trajectory Collection Started ===")

    for source_name, ckpt_path in POLICY_SOURCES.items():
        print(f"Collecting {games_per_source} games for policy source: {source_name}...")
        agent_fn = make_agent_policy(source_name, ckpt_path)

        for g in range(games_per_source):
            seed = base_seed + g
            # Player 0 uses agent_fn, Player 1 uses Heuristic opponent as standardized baseline
            env = AgentZeroEnv(num_players=2, opponent_policy=heuristic_opponent, seed=seed)
            obs, _ = env.reset()
            done = False
            steps = 0

            # Collect move history specifically for Player 0
            while not done and steps < 300:
                mask = env.action_masks()
                legal = env.engine.legal_actions(0)
                raw_obs = env.engine._get_obs(0)
                
                # Get action from target policy
                card_act = agent_fn(raw_obs, legal)
                act_idx = AgentZeroEnv._encode_action(card_act)

                obs, reward, done, truncated, info = env.step(act_idx)
                steps += 1

            p0_history = list(env.engine.players[0].move_history)
            # Format history as list of [turn_idx, action_val]
            formatted_history = [[t, int(act) if act != -1 else -1] for t, act in p0_history]

            sample_data = {
                "sample_id": sample_counter,
                "source": source_name,
                "game_id": g,
                "player_id": 0,
                "move_history": formatted_history,
                "total_turns": len(formatted_history),
                "won": (env.engine.winner == 0),
            }
            dataset.append(sample_data)
            sample_counter += 1

    print(f"Collection complete: {len(dataset)} total player trajectory samples collected.")
    return dataset


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = collect_dataset(games_per_source=100, base_seed=42)
    output_path = "data/trajectories_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved dataset to {output_path}")
