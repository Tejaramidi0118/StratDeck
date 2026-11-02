"""
Phase 3: Algorithm Comparison Runner (MaskablePPO vs MaskedDQN vs MaskedA2C).
Evaluates all three trained models using identical evaluation protocol and fixed seeds.
Outputs a machine-generated factual comparison table and saves to data/results.json.

Run: python -m training.compare_algorithms
"""

import json
import os
import sys

from training.evaluate_ppo import run_full_evaluation

RESULTS_PATH = "data/results.json"
MODELS = {
    "MaskablePPO": ("checkpoints/ppo/ppo_final.zip", "checkpoints/ppo/ppo_step50000.zip"),
    "MaskedDQN": ("checkpoints/dqn/dqn_final.zip", "checkpoints/dqn/dqn_step50000.zip"),
    "MaskedA2C": ("checkpoints/a2c/a2c_final.zip", "checkpoints/a2c/a2c_step50000.zip"),
}


def run_comparison():
    print("==================================================")
    print("PHASE 3 — ALGORITHM COMPARISON BENCHMARK (500 GAMES PER EVALUATION)")
    print("==================================================")

    comparison_results = {}

    for algo_name, (model_path, ckpt_opp_path) in MODELS.items():
        if not os.path.exists(model_path):
            print(f"Skipping {algo_name}: checkpoint missing at {model_path}")
            continue

        print(f"\n[Evaluating {algo_name}]")
        eval_res = run_full_evaluation(model_path, checkpoint_opp_path=ckpt_opp_path, n_games=500)

        comparison_results[algo_name] = {
            "model_path": model_path,
            "training_timesteps": 500000,
            "random_win_rate": eval_res["random"]["win_rate"],
            "random_avg_reward": eval_res["random"]["avg_reward"],
            "random_avg_length": eval_res["random"]["avg_length"],
            "heuristic_win_rate": eval_res["heuristic"]["win_rate"],
            "heuristic_avg_reward": eval_res["heuristic"]["avg_reward"],

            "heuristic_avg_length": eval_res["heuristic"]["avg_length"],
            "checkpoint_win_rate": eval_res["checkpoint"]["win_rate"] if eval_res["checkpoint"] else None,
            "checkpoint_avg_reward": eval_res["checkpoint"]["avg_reward"] if eval_res["checkpoint"] else None,
            "illegal_attempts": eval_res["random"]["illegal_attempts"] + eval_res["heuristic"]["illegal_attempts"],
        }

    # Update data/results.json
    results_data = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results_data = json.load(f)
        except Exception:
            results_data = {}

    results_data["algorithm_comparison"] = comparison_results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_data, f, indent=2)

    # Print Factual Comparison Table
    print("\n" + "=" * 95)
    print("PHASE 3 FACTUAL ALGORITHM COMPARISON TABLE")
    print("=" * 95)
    header = f"{'Algorithm':<14} | {'Timesteps':<10} | {'Random WR':<10} | {'Heuristic WR':<12} | {'Ckpt WR':<10} | {'Avg Reward':<11} | {'Avg Len':<8} | {'Illegal'}"
    print(header)
    print("-" * 95)

    for algo_name, data in comparison_results.items():
        ckpt_str = f"{data['checkpoint_win_rate']:.1%}" if data['checkpoint_win_rate'] is not None else "N/A"
        row = f"{algo_name:<14} | {data['training_timesteps']:<10,d} | {data['random_win_rate']:<10.1%} | {data['heuristic_win_rate']:<12.1%} | {ckpt_str:<10} | {data['random_avg_reward']:<11.2f} | {data['random_avg_length']:<8.1f} | {data['illegal_attempts']}"
        print(row)
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_comparison()
