"""
Phase 7: Controlled Ablation Studies and Interaction Analysis.

Evaluates four experimental conditions:
- Condition A: Base RL (Fixed level 2, no profiling, no DDA adaptation)
- Condition B: Profiling Only (Fixed level 2, observational profiling, no DDA adaptation)
- Condition C: DDA Only (Rolling win-rate DDA without profile-aware selection)
- Condition D: Profiling + DDA (Full Phase 5 profile-aware DDA controller)

Evaluated across multiple seeds (42, 43, 44, 45, 46) with 100 games per seed.
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
from training.evaluate_ppo import load_any_model, heuristic_opponent
from modules.profiler import OpponentProfiler
from modules.dda import DDAController, DEFAULT_DIFFICULTY_MAPPING


def load_model_cache():
    print("Loading model checkpoints for Phase 7 Ablation Study...")
    cache = {}
    for lvl, info in DEFAULT_DIFFICULTY_MAPPING.items():
        if info["type"] == "model" and info["path"] and os.path.exists(info["path"]):
            cache[lvl] = load_any_model(info["path"])
            print(f"  Level {lvl} ({info['name']}): Loaded successfully.")
        else:
            cache[lvl] = None
            print(f"  Level {lvl} ({info['name']}): Rule-based / Random policy.")
    return cache


def load_fitted_profiler():
    dataset_path = "data/trajectories_dataset.json"
    profiler = OpponentProfiler(n_clusters=2, seed=42)
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        histories = [d["move_history"] for d in dataset]
        profiler.fit(histories)
        print(f"Profiler fitted on {len(histories)} historical trajectories.")
    return profiler


def get_opponent_action(model, raw_obs, legal_actions):
    if model is None:
        return random.choice(legal_actions)
    obs_vec = AgentZeroEnv.encode_raw_obs(raw_obs)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in legal_actions:
        mask[AgentZeroEnv._encode_action(a)] = True
    act_id, _ = model.predict(obs_vec, action_masks=mask, deterministic=False)
    return AgentZeroEnv._decode_action(int(act_id))


def run_single_suite(condition_key, seed, n_games=100, model_cache=None, profiler=None):
    """
    Executes 100 games under a specific condition and seed.
    Condition A: Base RL (Fixed level 2, no profiling, no DDA)
    Condition B: Profiling Only (Fixed level 2, profiler active observationally, no DDA)
    Condition C: DDA Only (Adaptive WR-only DDA, profiler=None)
    Condition D: Profiling + DDA (Adaptive Profile-Aware DDA, profiler=profiler)
    """
    is_adaptive = condition_key in ["C", "D"]
    is_profiled = condition_key in ["B", "D"]
    
    dda_profiler = profiler if condition_key == "D" else None
    
    controller = DDAController(
        target_band=(0.45, 0.55),
        window_size=20,
        min_games_between_adjustments=5,
        initial_difficulty=2,
        profiler=dda_profiler
    )

    telemetry = []
    player_wins = 0

    for g in range(n_games):
        game_seed = seed * 1000 + g
        curr_lvl = controller.current_difficulty if is_adaptive else 2
        opp_model = model_cache.get(curr_lvl)

        def opp_policy(raw_obs, legal):
            return get_opponent_action(opp_model, raw_obs, legal)

        env = AgentZeroEnv(num_players=2, opponent_policy=opp_policy, seed=game_seed)
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

        c_id = None
        if profiler is not None and len(formatted_history) > 0:
            try:
                c_id = int(profiler.predict_cluster(formatted_history))
            except Exception:
                c_id = None

        if is_adaptive:
            status = controller.update(player_won=p0_won, move_history=formatted_history, game_length=steps)
            if not is_profiled:
                status["cluster_id"] = None
            else:
                status["cluster_id"] = c_id
        else:
            controller.total_games_played += 1
            controller.results_window.append(1 if p0_won else 0)
            wr = controller.current_win_rate()
            status = {
                "game_idx": controller.total_games_played,
                "player_won": p0_won,
                "rolling_win_rate": wr,
                "in_target_band": controller.is_in_target_band(),
                "current_difficulty": 2,
                "checkpoint_name": DEFAULT_DIFFICULTY_MAPPING[2]["name"],
                "checkpoint_path": DEFAULT_DIFFICULTY_MAPPING[2]["path"],
                "transition_occurred": False,
                "transition_reason": "FIXED",
                "cluster_id": c_id if is_profiled else None,
                "game_length": steps,
            }

        telemetry.append(status)

    overall_wr = player_wins / n_games
    rolling_wrs = [t["rolling_win_rate"] for t in telemetry]
    target_occupancy = sum(1 for wr in rolling_wrs if 0.45 <= wr <= 0.55) / n_games
    mad = float(np.mean([abs(wr - 0.50) for wr in rolling_wrs]))

    diff_counts = {lvl: 0 for lvl in range(6)}
    for t in telemetry:
        diff_counts[t["current_difficulty"]] += 1

    return {
        "condition": condition_key,
        "seed": seed,
        "n_games": n_games,
        "overall_win_rate": overall_wr,
        "target_occupancy": target_occupancy,
        "mad_from_target": mad,
        "total_transitions": len(controller.transition_history),
        "oscillation_count": controller.oscillation_count,
        "difficulty_counts": diff_counts,
        "telemetry": telemetry,
        "transition_history": controller.transition_history,
    }


def compute_aggregate_metrics(seed_results):
    wrs = [r["overall_win_rate"] for r in seed_results]
    occs = [r["target_occupancy"] for r in seed_results]
    mads = [r["mad_from_target"] for r in seed_results]
    trans = [r["total_transitions"] for r in seed_results]
    oscs = [r["oscillation_count"] for r in seed_results]

    return {
        "win_rate": {
            "mean": float(np.mean(wrs)),
            "std": float(np.std(wrs)),
            "min": float(np.min(wrs)),
            "max": float(np.max(wrs)),
        },
        "target_occupancy": {
            "mean": float(np.mean(occs)),
            "std": float(np.std(occs)),
            "min": float(np.min(occs)),
            "max": float(np.max(occs)),
        },
        "mad": {
            "mean": float(np.mean(mads)),
            "std": float(np.std(mads)),
            "min": float(np.min(mads)),
            "max": float(np.max(mads)),
        },
        "transitions": {
            "mean": float(np.mean(trans)),
            "std": float(np.std(trans)),
            "min": int(np.min(trans)),
            "max": int(np.max(trans)),
        },
        "oscillations": {
            "mean": float(np.mean(oscs)),
            "std": float(np.std(oscs)),
            "min": int(np.min(oscs)),
            "max": int(np.max(oscs)),
        },
    }


def generate_ablation_plots(results_by_cond, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/ablation", exist_ok=True)

    conditions = ["A", "B", "C", "D"]
    cond_labels = {
        "A": "Cond A: Base RL",
        "B": "Cond B: Profiling Only",
        "C": "Cond C: DDA Only",
        "D": "Cond D: Profiling + DDA",
    }
    colors = {"A": "blue", "B": "green", "C": "orange", "D": "purple"}

    # 1. Plot 1: Win Rate Comparison
    plt.figure(figsize=(8, 5))
    x = np.arange(len(conditions))
    means = [results_by_cond[c]["aggregate"]["win_rate"]["mean"] for c in conditions]
    stds = [results_by_cond[c]["aggregate"]["win_rate"]["std"] for c in conditions]
    plt.bar(x, means, yerr=stds, capsize=5, color=[colors[c] for c in conditions], alpha=0.85)
    plt.xticks(x, [cond_labels[c] for c in conditions], rotation=15)
    plt.ylabel("Win Rate")
    plt.title("Phase 7: Condition Comparison — Player Win Rate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "condition_comparison_win_rate.png"), dpi=300)
    plt.savefig("results/ablation/condition_comparison_win_rate.png", dpi=300)
    plt.close()

    # 2. Plot 2: Target-Band Occupancy Comparison
    plt.figure(figsize=(8, 5))
    means = [results_by_cond[c]["aggregate"]["target_occupancy"]["mean"] for c in conditions]
    stds = [results_by_cond[c]["aggregate"]["target_occupancy"]["std"] for c in conditions]
    plt.bar(x, means, yerr=stds, capsize=5, color=[colors[c] for c in conditions], alpha=0.85)
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    plt.xticks(x, [cond_labels[c] for c in conditions], rotation=15)
    plt.ylabel("Target-Band Occupancy (45-55%)")
    plt.title("Phase 7: Condition Comparison — Target-Band Occupancy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "condition_comparison_target_occupancy.png"), dpi=300)
    plt.savefig("results/ablation/condition_comparison_target_occupancy.png", dpi=300)
    plt.close()

    # 3. Plot 3: MAD Comparison
    plt.figure(figsize=(8, 5))
    means = [results_by_cond[c]["aggregate"]["mad"]["mean"] for c in conditions]
    stds = [results_by_cond[c]["aggregate"]["mad"]["std"] for c in conditions]
    plt.bar(x, means, yerr=stds, capsize=5, color=[colors[c] for c in conditions], alpha=0.85)
    plt.xticks(x, [cond_labels[c] for c in conditions], rotation=15)
    plt.ylabel("Mean Absolute Deviation (MAD from 0.50)")
    plt.title("Phase 7: Condition Comparison — MAD from Target Center")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "condition_comparison_mad.png"), dpi=300)
    plt.savefig("results/ablation/condition_comparison_mad.png", dpi=300)
    plt.close()

    # 4. Plot 4: DDA Rolling Win Rate Trajectories (Mean over seeds)
    plt.figure(figsize=(10, 5))
    games_x = range(1, 101)
    plt.axhspan(0.45, 0.55, color="green", alpha=0.15, label="Target Band (45%-55%)")
    plt.axhline(0.50, color="gray", linestyle="--", alpha=0.6)

    for c in conditions:
        all_tele = [s_res["telemetry"] for s_res in results_by_cond[c]["seeds"]]
        avg_wrs = np.mean([[t["rolling_win_rate"] for t in tele] for tele in all_tele], axis=0)
        plt.plot(games_x, avg_wrs, label=cond_labels[c], color=colors[c], linewidth=2)

    plt.xlabel("Game Index")
    plt.ylabel("Rolling Player Win Rate")
    plt.title("Phase 7: Rolling Win Rate Trajectory Across Conditions (5-Seed Avg)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dda_rolling_win_rate_trajectory.png"), dpi=300)
    plt.savefig("results/ablation/dda_rolling_win_rate_trajectory.png", dpi=300)
    plt.close()

    # 5. Plot 5: DDA Difficulty Level Trajectories for Cond C vs Cond D
    plt.figure(figsize=(10, 5))
    for c in ["C", "D"]:
        all_tele = [s_res["telemetry"] for s_res in results_by_cond[c]["seeds"]]
        avg_diffs = np.mean([[t["current_difficulty"] for t in tele] for tele in all_tele], axis=0)
        plt.plot(games_x, avg_diffs, label=cond_labels[c], color=colors[c], linewidth=2, marker="o", markersize=3)

    plt.yticks(range(6), [f"L{i}: {DEFAULT_DIFFICULTY_MAPPING[i]['name']}" for i in range(6)])
    plt.xlabel("Game Index")
    plt.ylabel("Difficulty Level")
    plt.title("Phase 7: DDA Difficulty Progression — DDA Only vs Profiling+DDA")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dda_difficulty_trajectory.png"), dpi=300)
    plt.savefig("results/ablation/dda_difficulty_trajectory.png", dpi=300)
    plt.close()

    # 6. Plot 6: Difficulty Level Occupancy
    plt.figure(figsize=(9, 5))
    x_levels = np.arange(6)
    width = 0.2
    for idx, c in enumerate(conditions):
        all_counts = [s_res["difficulty_counts"] for s_res in results_by_cond[c]["seeds"]]
        avg_counts = [np.mean([cnt[lvl] for cnt in all_counts]) for lvl in range(6)]
        plt.bar(x_levels + (idx - 1.5) * width, avg_counts, width, label=cond_labels[c], color=colors[c], alpha=0.8)

    plt.xlabel("Difficulty Level")
    plt.ylabel("Average Games Spent")
    plt.title("Phase 7: Difficulty Level Occupancy Across Conditions")
    plt.xticks(x_levels, [f"L{i}" for i in range(6)])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "difficulty_occupancy.png"), dpi=300)
    plt.savefig("results/ablation/difficulty_occupancy.png", dpi=300)
    plt.close()

    # 7. Plot 7 & 8: Cluster x Difficulty and Performance Breakdown for Cond B and D
    plt.figure(figsize=(9, 5))
    cluster_stats = {}
    for c in ["B", "D"]:
        all_tele = []
        for s_res in results_by_cond[c]["seeds"]:
            all_tele.extend(s_res["telemetry"])
        
        c0 = [t for t in all_tele if t.get("cluster_id") == 0]
        c1 = [t for t in all_tele if t.get("cluster_id") == 1]
        
        cluster_stats[c] = {
            0: {
                "count": len(c0),
                "wr": sum(1 for t in c0 if t["player_won"]) / len(c0) if c0 else 0,
                "avg_diff": np.mean([t["current_difficulty"] for t in c0]) if c0 else 0,
                "target_occ": sum(1 for t in c0 if 0.45 <= t["rolling_win_rate"] <= 0.55) / len(c0) if c0 else 0,
            },
            1: {
                "count": len(c1),
                "wr": sum(1 for t in c1 if t["player_won"]) / len(c1) if c1 else 0,
                "avg_diff": np.mean([t["current_difficulty"] for t in c1]) if c1 else 0,
                "target_occ": sum(1 for t in c1 if 0.45 <= t["rolling_win_rate"] <= 0.55) / len(c1) if c1 else 0,
            }
        }

    # Plot Cluster Performance
    x_c = np.arange(2)
    width = 0.35
    c0_wrs = [cluster_stats["B"][0]["wr"], cluster_stats["D"][0]["wr"]]
    c1_wrs = [cluster_stats["B"][1]["wr"], cluster_stats["D"][1]["wr"]]

    plt.bar(x_c - width/2, c0_wrs, width, label="Cluster 0 (Draw-heavy)", color="deepskyblue")
    plt.bar(x_c + width/2, c1_wrs, width, label="Cluster 1 (High-defense)", color="coral")
    plt.xticks(x_c, ["Cond B: Profiling Only", "Cond D: Profiling + DDA"])
    plt.ylabel("Win Rate")
    plt.title("Phase 7: Cluster-Wise Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_performance_comparison.png"), dpi=300)
    plt.savefig("results/ablation/cluster_performance_comparison.png", dpi=300)
    plt.close()

    # Plot Cluster Difficulty Distribution
    plt.figure(figsize=(9, 5))
    c0_diffs = [cluster_stats["B"][0]["avg_diff"], cluster_stats["D"][0]["avg_diff"]]
    c1_diffs = [cluster_stats["B"][1]["avg_diff"], cluster_stats["D"][1]["avg_diff"]]

    plt.bar(x_c - width/2, c0_diffs, width, label="Cluster 0 (Draw-heavy)", color="deepskyblue")
    plt.bar(x_c + width/2, c1_diffs, width, label="Cluster 1 (High-defense)", color="coral")
    plt.xticks(x_c, ["Cond B: Profiling Only", "Cond D: Profiling + DDA"])
    plt.ylabel("Average Difficulty Level")
    plt.title("Phase 7: Cluster-Wise Average Assigned Difficulty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_difficulty_distribution.png"), dpi=300)
    plt.savefig("results/ablation/cluster_difficulty_distribution.png", dpi=300)
    plt.close()

    return cluster_stats


def run_phase7_ablation_pipeline(seeds=[42, 43, 44, 45, 46], n_games=100):
    output_dir = "data/ablation/plots"
    results_dir = "data/ablation/results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=== Phase 7: Running Controlled Ablation Pipeline ===")
    print(f"Seeds: {seeds}")
    print(f"Games per seed per condition: {n_games}")

    model_cache = load_model_cache()
    profiler = load_fitted_profiler()

    conditions = ["A", "B", "C", "D"]
    results_by_cond = {}

    for c in conditions:
        print(f"\nEvaluating Condition {c} across {len(seeds)} seeds...")
        seed_results = []
        for seed in seeds:
            res = run_single_suite(c, seed, n_games=n_games, model_cache=model_cache, profiler=profiler)
            seed_results.append(res)
            print(f"  Condition {c}, Seed {seed}: WR={res['overall_win_rate']:.1%}, Occupancy={res['target_occupancy']:.1%}, MAD={res['mad_from_target']:.4f}, Trans={res['total_transitions']}, Osc={res['oscillation_count']}")

        agg = compute_aggregate_metrics(seed_results)
        results_by_cond[c] = {
            "aggregate": agg,
            "seeds": seed_results,
        }

    cluster_stats = generate_ablation_plots(results_by_cond, output_dir)

    # Save summary JSON
    summary_data = {
        "metadata": {
            "phase": 7,
            "seeds": seeds,
            "n_games_per_seed": n_games,
            "conditions": {
                "A": "Base RL (Fixed L2, no profiling, no DDA)",
                "B": "Profiling Only (Fixed L2, observational profiling, no DDA)",
                "C": "DDA Only (WR-only DDA, no profile-aware selection)",
                "D": "Profiling + DDA (Full Phase 5 profile-aware DDA)",
            }
        },
        "results_by_condition": {
            c: {
                "aggregate": results_by_cond[c]["aggregate"],
                "per_seed": [
                    {
                        "seed": r["seed"],
                        "win_rate": r["overall_win_rate"],
                        "target_occupancy": r["target_occupancy"],
                        "mad": r["mad_from_target"],
                        "transitions": r["total_transitions"],
                        "oscillations": r["oscillation_count"],
                        "difficulty_counts": r["difficulty_counts"],
                    } for r in results_by_cond[c]["seeds"]
                ]
            } for c in conditions
        },
        "cluster_stats": cluster_stats,
    }

    summary_file = os.path.join(results_dir, "ablation_summary_metrics.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved aggregated metrics to {summary_file}")

    print("\nPhase 7 Ablation Experiment Pipeline Completed Successfully!")
    return summary_data


if __name__ == "__main__":
    run_phase7_ablation_pipeline()
