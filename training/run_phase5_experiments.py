"""
Phase 5: Experimental Evaluation & Visualization Pipeline for DDA.

Compares:
1. Fixed Easy Opponent (Level 1: a2c_final.zip)
2. Fixed Hard Opponent (Level 5: ppo_step350000.zip)
3. Adaptive DDA Controller (Dynamic Checkpoint Selection across Levels 0..5)

Across 100 sequential games each against a standardized baseline player (Heuristic Player),
tracking rolling win rate, target-band occupancy (45-55%), MAD from 0.50, difficulty transitions,
oscillation counts, and cluster-wise profiling breakdown.
"""

import sys
import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from env.cards import Card
from training.evaluate_ppo import load_any_model, heuristic_opponent
from modules.profiler import OpponentProfiler
from modules.dda import DDAController, DEFAULT_DIFFICULTY_MAPPING


def load_model_cache():
    print("Loading model checkpoints for DDA execution...")
    cache = {}
    for lvl, info in DEFAULT_DIFFICULTY_MAPPING.items():
        if info["type"] == "model" and info["path"] and os.path.exists(info["path"]):
            cache[lvl] = load_any_model(info["path"])
            print(f"  Level {lvl} ({info['name']}): Loaded successfully.")
        else:
            cache[lvl] = None
            print(f"  Level {lvl} ({info['name']}): Rule-based / Random policy.")
    return cache


def get_opponent_action(model, raw_obs, legal_actions):
    if model is None:
        return random.choice(legal_actions)
    obs_vec = AgentZeroEnv.encode_raw_obs(raw_obs)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in legal_actions:
        mask[AgentZeroEnv._encode_action(a)] = True
    act_id, _ = model.predict(obs_vec, action_masks=mask, deterministic=False)
    return AgentZeroEnv._decode_action(int(act_id))


def run_sequential_match_suite(mode="adaptive", fixed_level=2, n_games=100, base_seed=9000, model_cache=None, profiler=None):
    controller = DDAController(
        target_band=(0.45, 0.55),
        window_size=20,
        min_games_between_adjustments=5,
        initial_difficulty=fixed_level if mode == "fixed" else 2,
        profiler=profiler
    )

    telemetry = []
    player_wins = 0

    print(f"\n--- Running Match Suite: Mode={mode.upper()} (Fixed Level={fixed_level if mode=='fixed' else 'N/A'}, Games={n_games}) ---")

    for g in range(n_games):
        seed = base_seed + g
        curr_lvl = controller.current_difficulty if mode == "adaptive" else fixed_level
        opp_model = model_cache.get(curr_lvl)

        # Policy function for Player 1 (Opponent)
        def opp_policy(raw_obs, legal):
            return get_opponent_action(opp_model, raw_obs, legal)

        env = AgentZeroEnv(num_players=2, opponent_policy=opp_policy, seed=seed)
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 300:
            legal = env.engine.legal_actions(0)
            raw_obs = env.engine._get_obs(0)
            p0_act = heuristic_opponent(raw_obs, legal)
            act_idx = AgentZeroEnv._encode_action(p0_act)
            obs, reward, done, truncated, info = env.step(act_idx)
            steps += 1

        p0_won = (env.engine.winner == 0)
        p0_history = list(env.engine.players[0].move_history)
        formatted_history = [[t, int(act) if act != -1 else -1] for t, act in p0_history]

        if p0_won:
            player_wins += 1

        if mode == "adaptive":
            status = controller.update(player_won=p0_won, move_history=formatted_history, game_length=steps)
        else:
            controller.total_games_played += 1
            controller.results_window.append(1 if p0_won else 0)
            wr = controller.current_win_rate()
            c_id = profiler.predict_cluster(formatted_history) if profiler else None
            status = {
                "game_idx": controller.total_games_played,
                "player_won": p0_won,
                "rolling_win_rate": wr,
                "in_target_band": controller.is_in_target_band(),
                "current_difficulty": fixed_level,
                "checkpoint_name": DEFAULT_DIFFICULTY_MAPPING[fixed_level]["name"],
                "checkpoint_path": DEFAULT_DIFFICULTY_MAPPING[fixed_level]["path"],
                "transition_occurred": False,
                "transition_reason": "FIXED",
                "cluster_id": c_id,
                "game_length": steps,
            }

        telemetry.append(status)

    overall_wr = player_wins / n_games
    rolling_wrs = [t["rolling_win_rate"] for t in telemetry]
    target_occupancy = sum(1 for wr in rolling_wrs if 0.45 <= wr <= 0.55) / n_games
    mad = float(np.mean([abs(wr - 0.50) for wr in rolling_wrs]))

    print(f"Suite Finished ({mode}): Player WR: {overall_wr:.1%}, Target Occupancy: {target_occupancy:.1%}, MAD: {mad:.4f}, Transitions: {len(controller.transition_history)}, Oscillations: {controller.oscillation_count}")

    return {
        "mode": mode,
        "fixed_level": fixed_level if mode == "fixed" else None,
        "n_games": n_games,
        "overall_win_rate": overall_wr,
        "target_occupancy": target_occupancy,
        "mad_from_target": mad,
        "total_transitions": len(controller.transition_history),
        "oscillation_count": controller.oscillation_count,
        "telemetry": telemetry,
        "transition_history": controller.transition_history,
    }


def run_phase5_experiment_pipeline():
    os.makedirs("data/dda", exist_ok=True)
    os.makedirs("results/dda", exist_ok=True)

    # 1. Load fitted Phase 4 profiler
    print("Loading Phase 4 Opponent Profiler...")
    dataset_path = "data/trajectories_dataset.json"
    profiler = OpponentProfiler(n_clusters=2, seed=42)
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        histories = [d["move_history"] for d in dataset]
        profiler.fit(histories)
        print(f"Profiler fitted on {len(histories)} historical trajectories.")

    model_cache = load_model_cache()

    # 2. Run Match Suites: Fixed Easy (L1), Fixed Hard (L5), Adaptive DDA
    fixed_easy_res = run_sequential_match_suite("fixed", fixed_level=1, n_games=100, base_seed=10000, model_cache=model_cache, profiler=profiler)
    fixed_hard_res = run_sequential_match_suite("fixed", fixed_level=5, n_games=100, base_seed=11000, model_cache=model_cache, profiler=profiler)
    adaptive_dda_res = run_sequential_match_suite("adaptive", n_games=100, base_seed=12000, model_cache=model_cache, profiler=profiler)

    # 3. Cluster-Specific Analysis for Adaptive DDA
    cluster_metrics = {}
    for cid in [0, 1]:
        c_telemetry = [t for t in adaptive_dda_res["telemetry"] if t["cluster_id"] == cid]
        if c_telemetry:
            c_wins = sum(1 for t in c_telemetry if t["player_won"])
            c_wrs = [t["rolling_win_rate"] for t in c_telemetry]
            cluster_metrics[cid] = {
                "sample_count": len(c_telemetry),
                "win_rate": c_wins / len(c_telemetry),
                "target_occupancy": sum(1 for wr in c_wrs if 0.45 <= wr <= 0.55) / len(c_telemetry),
                "mad": float(np.mean([abs(wr - 0.50) for wr in c_wrs])),
                "avg_difficulty": float(np.mean([t["current_difficulty"] for t in c_telemetry])),
                "avg_length": float(np.mean([t["game_length"] for t in c_telemetry])),
            }

    print("\n--- Cluster-Wise Adaptive DDA Performance ---")
    for cid, m in cluster_metrics.items():
        print(f"Cluster {cid} ({m['sample_count']} games): WR={m['win_rate']:.1%}, Occupancy={m['target_occupancy']:.1%}, MAD={m['mad']:.4f}, Avg Difficulty={m['avg_difficulty']:.2f}")

    # 4. Visualizations
    print("\nGenerating Analytical Visualizations...")

    # Plot 1: Rolling Player Win Rate Comparison
    plt.figure(figsize=(10, 5))
    games_x = range(1, 101)
    plt.axhspan(0.45, 0.55, color="green", alpha=0.15, label="Target Band (45%-55%)")
    plt.axhline(0.50, color="gray", linestyle="--", alpha=0.6)

    plt.plot(games_x, [t["rolling_win_rate"] for t in fixed_easy_res["telemetry"]], label="Fixed Easy (Level 1)", color="blue", alpha=0.7)
    plt.plot(games_x, [t["rolling_win_rate"] for t in fixed_hard_res["telemetry"]], label="Fixed Hard (Level 5)", color="red", alpha=0.7)
    plt.plot(games_x, [t["rolling_win_rate"] for t in adaptive_dda_res["telemetry"]], label="Adaptive DDA", color="purple", linewidth=2)

    plt.xlabel("Game Index")
    plt.ylabel("Rolling Player Win Rate")
    plt.title("Phase 5: Rolling Player Win Rate Trajectory Across Systems")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("data/dda/rolling_win_rate.png", dpi=300)
    plt.savefig("results/dda/rolling_win_rate.png", dpi=300)
    plt.close()

    # Plot 2: Difficulty Level Progression in Adaptive DDA
    plt.figure(figsize=(10, 5))
    diffs = [t["current_difficulty"] for t in adaptive_dda_res["telemetry"]]
    plt.plot(games_x, diffs, color="teal", linewidth=2, marker="o", markersize=3, label="Opponent Difficulty Level")

    # Mark transition points
    for trans in adaptive_dda_res["transition_history"]:
        g_idx = trans["game_idx"]
        new_lvl = trans["new_difficulty"]
        plt.scatter(g_idx, new_lvl, color="orange", s=70, zorder=5)

    plt.yticks(range(6), [f"L{i}: {DEFAULT_DIFFICULTY_MAPPING[i]['name']}" for i in range(6)])
    plt.xlabel("Game Index")
    plt.ylabel("Difficulty Level")
    plt.title("Phase 5: Dynamic Difficulty Adjustment Level Progression")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("data/dda/difficulty_progression.png", dpi=300)
    plt.savefig("results/dda/difficulty_progression.png", dpi=300)
    plt.close()

    # Plot 3: Stability & Performance Metrics Bar Chart
    plt.figure(figsize=(9, 5))
    labels = ["Fixed Easy", "Fixed Hard", "Adaptive DDA"]
    win_rates = [fixed_easy_res["overall_win_rate"], fixed_hard_res["overall_win_rate"], adaptive_dda_res["overall_win_rate"]]
    occupancies = [fixed_easy_res["target_occupancy"], fixed_hard_res["target_occupancy"], adaptive_dda_res["target_occupancy"]]
    mads = [fixed_easy_res["mad_from_target"], fixed_hard_res["mad_from_target"], adaptive_dda_res["mad_from_target"]]

    x = np.arange(len(labels))
    width = 0.25

    plt.bar(x - width, win_rates, width, label="Player Win Rate", color="royalblue")
    plt.bar(x, occupancies, width, label="Target Occupancy (45-55%)", color="mediumseagreen")
    plt.bar(x + width, mads, width, label="MAD from 0.50", color="salmon")

    plt.ylabel("Metric Value / Ratio")
    plt.title("Phase 5: Baseline vs Adaptive DDA Performance Comparison")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/dda/baseline_comparison.png", dpi=300)
    plt.savefig("results/dda/baseline_comparison.png", dpi=300)
    plt.close()

    # 5. Serialize Output to data/dda_results.json
    output_data = {
        "metadata": {
            "phase": 5,
            "target_band": [0.45, 0.55],
            "window_size": 20,
            "cooldown_min_games": 5,
            "difficulty_mapping": DEFAULT_DIFFICULTY_MAPPING,
        },
        "calibration": {
            str(lvl): {
                "name": info["name"],
                "path": info["path"],
                "empirically_measured_player_win_rate": info["target_wr"]
            } for lvl, info in DEFAULT_DIFFICULTY_MAPPING.items()
        },
        "baselines": {
            "fixed_easy": fixed_easy_res,
            "fixed_hard": fixed_hard_res,
        },
        "adaptive_dda": adaptive_dda_res,
        "cluster_wise_analysis": cluster_metrics,
    }

    with open("data/dda_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("Saved data/dda_results.json successfully!")


if __name__ == "__main__":
    run_phase5_experiment_pipeline()
