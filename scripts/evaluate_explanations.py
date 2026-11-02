"""
Phase 6: Quantitative Faithfulness & Grounding Evaluation Pipeline.

Evaluates generated explanations against strict factual evidence standards across 8 quantitative metrics:
1. Grounding Rate (% fully grounded)
2. Selected Action Consistency Rate
3. Hidden Information Violation Rate (target: 0.0%)
4. Q-Value Violation Rate (target: 0.0%)
5. Action Substitution Violation Rate (target: 0.0%)
6. Probability Consistency Rate
7. Explanation Coverage Rate
8. Alternative Awareness Rate
9. Explanation Length Statistics (Word counts)
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_explanation_evaluation():
    dataset_file = "data/explanations/explanation_dataset.json"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Explanation dataset missing at {dataset_file}. Run generate_explanations.py first.")

    with open(dataset_file, "r") as f:
        records = json.load(f)

    n_samples = len(records)
    print(f"=== Phase 6: Quantitative Explanation Evaluation ({n_samples} samples) ===")

    grounded_count = 0
    action_consistent_count = 0
    prob_consistent_count = 0
    legal_consistent_count = 0
    state_consistent_count = 0
    hidden_violation_count = 0
    q_value_violation_count = 0
    substitution_violation_count = 0
    coverage_count = 0
    alternative_awareness_count = 0

    word_counts = []
    chosen_probs = []
    value_estimates = []

    for r in records:
        ev = r["evidence"]
        exp_out = r["explanation_output"]
        val = exp_out.get("validation", {})

        if val.get("grounded", False):
            grounded_count += 1
        if val.get("selected_action_consistent", False):
            action_consistent_count += 1
        if val.get("probabilities_consistent", False):
            prob_consistent_count += 1
        if val.get("legal_actions_consistent", False):
            legal_consistent_count += 1
        if val.get("state_facts_consistent", False):
            state_consistent_count += 1
        if val.get("hidden_information_violation", False):
            hidden_violation_count += 1
        if val.get("q_value_violation", False):
            q_value_violation_count += 1
        if val.get("action_substitution", False):
            substitution_violation_count += 1

        # Coverage check (cites at least 1 evidence item)
        if exp_out.get("evidence") and len(exp_out["evidence"]) > 0:
            coverage_count += 1

        # Alternative awareness check (references legal alternatives if >1 legal actions exist)
        if len(ev["state_facts"]["legal_actions"]) > 1:
            if exp_out.get("legal_alternatives") or ("alternative" in exp_out.get("explanation", "").lower()):
                alternative_awareness_count += 1

        text = exp_out.get("explanation", "")
        word_counts.append(len(text.split()))

        chosen_probs.append(ev["policy_evidence"]["chosen_action_probability"])
        value_estimates.append(ev["policy_evidence"]["value_estimate_V_s"])

    # Calculate metrics
    grounding_rate = grounded_count / n_samples
    action_consistency_rate = action_consistent_count / n_samples
    prob_consistency_rate = prob_consistent_count / n_samples
    legal_consistency_rate = legal_consistent_count / n_samples
    state_consistency_rate = state_consistent_count / n_samples
    hidden_violation_rate = hidden_violation_count / n_samples
    q_value_violation_rate = q_value_violation_count / n_samples
    substitution_violation_rate = substitution_violation_count / n_samples
    coverage_rate = coverage_count / n_samples
    alternative_awareness_rate = alternative_awareness_count / n_samples

    mean_words = float(np.mean(word_counts))
    min_words = int(np.min(word_counts))
    max_words = int(np.max(word_counts))

    metrics = {
        "sample_size": n_samples,
        "seed_protocol": "fixed (base_seed=42)",
        "checkpoint": "checkpoints/ppo/ppo_final.zip",
        "grounding_rate": grounding_rate,
        "selected_action_consistency_rate": action_consistency_rate,
        "probability_consistency_rate": prob_consistency_rate,
        "legal_actions_consistency_rate": legal_consistency_rate,
        "state_facts_consistency_rate": state_consistency_rate,
        "hidden_information_violation_rate": hidden_violation_rate,
        "q_value_violation_rate": q_value_violation_rate,
        "action_substitution_violation_rate": substitution_violation_rate,
        "coverage_rate": coverage_rate,
        "alternative_awareness_rate": alternative_awareness_rate,
        "word_length_statistics": {
            "mean": mean_words,
            "min": min_words,
            "max": max_words
        }
    }

    print("\n--- Quantitative Explanation Evaluation Results ---")
    print(f"Sample Size: {n_samples}")
    print(f"Grounding Rate: {grounding_rate:.1%}")
    print(f"Selected Action Consistency: {action_consistency_rate:.1%}")
    print(f"Probability Consistency (+/- 5%): {prob_consistency_rate:.1%}")
    print(f"Hidden Information Violation Rate: {hidden_violation_rate:.1%}")
    print(f"Q-Value Violation Rate: {q_value_violation_rate:.1%}")
    print(f"Action Substitution Rate: {substitution_violation_rate:.1%}")
    print(f"Explanation Coverage Rate: {coverage_rate:.1%}")
    print(f"Alternative Awareness Rate: {alternative_awareness_rate:.1%}")
    print(f"Word Lengths: Mean={mean_words:.1f}, Min={min_words}, Max={max_words}")

    # Save metrics JSON
    metrics_file = "data/explanations/explanation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")

    # Visualizations
    print("Generating Analytical Visualizations...")

    # Plot 1: Grounding & Faithfulness Metrics Bar Chart
    plt.figure(figsize=(9, 5))
    metric_names = ["Grounding", "Action Match", "Prob Match", "Coverage", "Alt Awareness"]
    metric_vals = [grounding_rate, action_consistency_rate, prob_consistency_rate, coverage_rate, alternative_awareness_rate]

    plt.bar(metric_names, [v * 100 for v in metric_vals], color="teal", alpha=0.85)
    plt.axhline(100, color="gray", linestyle="--", alpha=0.5)
    plt.ylabel("Percentage (%)")
    plt.title("Phase 6: Grounded LLM Explanation Faithfulness Metrics")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig("data/explanations/grounding_metrics.png", dpi=300)
    plt.savefig("results/explanations/grounding_metrics.png", dpi=300)
    plt.close()

    # Plot 2: Action Probability vs Value Estimate Scatter Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(chosen_probs, value_estimates, color="purple", alpha=0.6, edgecolors="none")
    plt.xlabel("Selected Action Probability pi(a|s)")
    plt.ylabel("Value Estimate V(s)")
    plt.title("Phase 6: PPO Policy Inference Evidence Distribution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("data/explanations/action_probabilities.png", dpi=300)
    plt.savefig("results/explanations/action_probabilities.png", dpi=300)
    plt.close()

    # Plot 3: Explanation Word Length Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(word_counts, bins=10, color="indigo", alpha=0.75, edgecolor="black")
    plt.xlabel("Explanation Word Count")
    plt.ylabel("Frequency")
    plt.title("Phase 6: Explanation Word Length Distribution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("data/explanations/explanation_length.png", dpi=300)
    plt.savefig("results/explanations/explanation_length.png", dpi=300)
    plt.close()

    print("Visualizations saved successfully!")


if __name__ == "__main__":
    run_explanation_evaluation()
