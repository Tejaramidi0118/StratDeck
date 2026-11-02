"""
Phase 6: Explanation Generation Script.

Samples decision points from real MaskablePPO evaluation games, extracts observable evidence,
generates grounded LLM explanations, runs automatic grounding validation, and serializes
the dataset to data/explanations/explanation_dataset.json.
"""

import sys
import os
import json
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from training.evaluate_ppo import heuristic_opponent
from modules.explanation_evidence import extract_explanation_evidence
from modules.explainer import ExplainerModule, MockBackend
from modules.profiler import OpponentProfiler
from modules.dda import DDAController


def generate_explanation_dataset(n_games: int = 20, base_seed: int = 42) -> list:
    os.makedirs("data/explanations", exist_ok=True)
    os.makedirs("results/explanations", exist_ok=True)

    print("=== Phase 6: Generating Explanation Dataset ===")
    print(f"Loading frozen MaskablePPO model from checkpoints/ppo/ppo_final.zip...")
    model = MaskablePPO.load("checkpoints/ppo/ppo_final.zip")

    # Load fitted Phase 4 profiler and DDA controller
    profiler = OpponentProfiler(n_clusters=2, seed=42)
    dataset_path = "data/trajectories_dataset.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        histories = [d["move_history"] for d in dataset]
        profiler.fit(histories)

    dda = DDAController(initial_difficulty=2, profiler=profiler)
    explainer = ExplainerModule()  # Auto-detects local Ollama qwen3.5:4b backend
    backend_type = type(explainer.backend).__name__
    model_name = getattr(explainer.backend, "model", "qwen3.5:4b")

    print(f"Explainer initialized with Backend: {backend_type}, Model: {model_name}")

    records = []
    record_id = 0

    for g in range(n_games):
        seed = base_seed + g
        env = AgentZeroEnv(num_players=2, opponent_policy=heuristic_opponent, seed=seed)
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 300:
            mask = env.action_masks()
            action_idx, _ = model.predict(obs, action_masks=mask, deterministic=True)
            chosen_act = int(action_idx)

            p0_history = list(env.engine.players[0].move_history)
            formatted_history = [[t, int(act) if act != -1 else -1] for t, act in p0_history]

            # Extract evidence
            evidence = extract_explanation_evidence(
                model=model,
                env=env,
                obs=obs,
                chosen_action_idx=chosen_act,
                player_id=0,
                dda_controller=dda,
                profiler=profiler,
                move_history=formatted_history
            )

            # Generate explanation via real Ollama
            exp_result = explainer.explain_move(evidence)

            record = {
                "record_id": record_id,
                "episode_id": g,
                "step_id": steps,
                "seed": seed,
                "backend": backend_type,
                "model_name": model_name,
                "evidence": evidence,
                "explanation_output": exp_result,
            }
            records.append(record)
            record_id += 1

            # Step environment
            obs, reward, done, truncated, info = env.step(chosen_act)
            steps += 1

            p0_won = (env.engine.winner == 0) if done else False
            if done:
                dda.update(player_won=p0_won, move_history=formatted_history, game_length=steps)

    print(f"Generated {len(records)} decision explanation records across {n_games} games.")

    dataset_file = "data/explanations/explanation_dataset.json"
    with open(dataset_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved explanation dataset to {dataset_file}")

    return records


if __name__ == "__main__":
    generate_explanation_dataset(n_games=20, base_seed=42)
