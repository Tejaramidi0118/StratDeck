"""
Phase 8: Static dashboard (matplotlib) summarizing results from all phases.
Run: python -m ui.dashboard
Reads results that you should log to data/results.json as you complete each
phase (see the RESULTS TEMPLATE below) rather than hardcoding numbers here.
"""

import json
import os
import matplotlib.pyplot as plt

RESULTS_PATH = "data/results.json"

RESULTS_TEMPLATE = {
    "algorithm_comparison": {
        "PPO": {"win_rate": 0.0, "avg_reward": 0.0},
        "DQN": {"win_rate": 0.0, "avg_reward": 0.0},
        "A2C": {"win_rate": 0.0, "avg_reward": 0.0},
    },
    "dda_win_rate_over_time": [],   # list of floats, one per evaluation checkpoint
    "faithfulness": {"without_dda": 0.0, "with_dda": 0.0},
    "profiling_dda_conflict_rate": 0.0,
}


def load_results():
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(RESULTS_TEMPLATE, f, indent=2)
        print(f"No results found - wrote a template to {RESULTS_PATH}. "
              f"Fill in real numbers from each phase's scripts, then re-run this dashboard.")
        return RESULTS_TEMPLATE
    with open(RESULTS_PATH) as f:
        return json.load(f)


def plot_algorithm_comparison(results):
    algos = list(results["algorithm_comparison"].keys())
    win_rates = [results["algorithm_comparison"][a]["win_rate"] for a in algos]
    plt.figure()
    plt.bar(algos, win_rates, color=["#4C72B0", "#DD8452", "#55A868"])
    plt.ylabel("Win rate vs random baseline")
    plt.title("Phase 3: Algorithm Comparison")
    plt.savefig("data/plot_algorithm_comparison.png")
    plt.close()


def plot_dda_win_rate(results):
    series = results["dda_win_rate_over_time"]
    if not series:
        return
    plt.figure()
    plt.plot(series, marker="o")
    plt.axhspan(0.45, 0.55, color="green", alpha=0.15, label="target band")
    plt.ylabel("Rolling win rate")
    plt.xlabel("Evaluation checkpoint")
    plt.title("Phase 5: DDA Win-Rate Stability")
    plt.legend()
    plt.savefig("data/plot_dda_stability.png")
    plt.close()


def plot_faithfulness(results):
    labels = ["Without DDA", "With DDA"]
    values = [results["faithfulness"]["without_dda"], results["faithfulness"]["with_dda"]]
    plt.figure()
    plt.bar(labels, values, color=["#4C72B0", "#C44E52"])
    plt.ylabel("Explanation faithfulness")
    plt.title("Phase 7a: Faithfulness Under DDA (your core finding)")
    plt.savefig("data/plot_faithfulness.png")
    plt.close()


if __name__ == "__main__":
    results = load_results()
    plot_algorithm_comparison(results)
    plot_dda_win_rate(results)
    plot_faithfulness(results)
    print("Plots saved to data/plot_*.png")
    print(f"Profiling-vs-DDA conflict rate: {results['profiling_dda_conflict_rate']:.1%}")
