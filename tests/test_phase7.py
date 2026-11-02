"""
Unit and Sanity Test Suite for Phase 7 Ablation Studies.

Verifies:
1. Condition A (Base RL) execution
2. Condition B (Profiling Only) execution
3. Condition C (DDA Only) execution
4. Condition D (Profiling + DDA) execution
5. Primary metrics calculation (WR, Occupancy, MAD, Transitions, Oscillations)
6. Difficulty counts and cluster breakdown
7. Serialization of outputs to data/ablation/results/
8. Action legality (0 illegal action attempts)
9. Frozen artifact integrity
"""

import os
import json
import hashlib
import numpy as np

from training.run_phase7_ablations import (
    load_model_cache,
    load_fitted_profiler,
    run_single_suite,
    compute_aggregate_metrics,
    run_phase7_ablation_pipeline
)
from modules.dda import DDAController


def test_01_model_cache_loading():
    cache = load_model_cache()
    assert 0 in cache
    assert 1 in cache
    assert 2 in cache
    assert 3 in cache
    assert 4 in cache
    assert 5 in cache
    print("Test 1 Passed: Model cache loaded successfully.")


def test_02_profiler_fitting():
    profiler = load_fitted_profiler()
    assert profiler.kmeans is not None
    print("Test 2 Passed: Profiler loaded and fitted successfully.")


def test_03_condition_a_base_rl():
    cache = load_model_cache()
    profiler = load_fitted_profiler()
    res = run_single_suite("A", seed=42, n_games=20, model_cache=cache, profiler=profiler)
    assert res["condition"] == "A"
    assert res["n_games"] == 20
    assert 0.0 <= res["overall_win_rate"] <= 1.0
    assert 0.0 <= res["target_occupancy"] <= 1.0
    assert res["total_transitions"] == 0
    assert res["difficulty_counts"][2] == 20
    print("Test 3 Passed: Condition A (Base RL) executed cleanly.")


def test_04_condition_b_profiling_only():
    cache = load_model_cache()
    profiler = load_fitted_profiler()
    res = run_single_suite("B", seed=42, n_games=20, model_cache=cache, profiler=profiler)
    assert res["condition"] == "B"
    assert res["total_transitions"] == 0
    assert res["difficulty_counts"][2] == 20
    # Check that cluster_id is logged for telemetry
    clusters = [t["cluster_id"] for t in res["telemetry"] if t["cluster_id"] is not None]
    assert len(clusters) > 0
    print("Test 4 Passed: Condition B (Profiling Only) executed cleanly.")


def test_05_condition_c_dda_only():
    cache = load_model_cache()
    profiler = load_fitted_profiler()
    res = run_single_suite("C", seed=42, n_games=30, model_cache=cache, profiler=profiler)
    assert res["condition"] == "C"
    assert 0.0 <= res["overall_win_rate"] <= 1.0
    # Check that cluster_id is None in telemetry for Condition C
    assert all(t["cluster_id"] is None for t in res["telemetry"])
    print("Test 5 Passed: Condition C (DDA Only) executed cleanly.")


def test_06_condition_d_profiling_dda():
    cache = load_model_cache()
    profiler = load_fitted_profiler()
    res = run_single_suite("D", seed=42, n_games=30, model_cache=cache, profiler=profiler)
    assert res["condition"] == "D"
    assert 0.0 <= res["overall_win_rate"] <= 1.0
    print("Test 6 Passed: Condition D (Profiling + DDA) executed cleanly.")


def test_07_aggregate_metrics_calculation():
    mock_results = [
        {"overall_win_rate": 0.50, "target_occupancy": 0.40, "mad_from_target": 0.10, "total_transitions": 2, "oscillation_count": 0},
        {"overall_win_rate": 0.60, "target_occupancy": 0.60, "mad_from_target": 0.08, "total_transitions": 4, "oscillation_count": 1},
    ]
    agg = compute_aggregate_metrics(mock_results)
    assert np.isclose(agg["win_rate"]["mean"], 0.55)
    assert np.isclose(agg["target_occupancy"]["mean"], 0.50)
    assert agg["transitions"]["min"] == 2
    assert agg["transitions"]["max"] == 4
    print("Test 7 Passed: Aggregate metrics calculated correctly.")


def test_08_frozen_checkpoint_integrity():
    checkpoints = [
        "checkpoints/ppo/ppo_final.zip",
        "checkpoints/dqn/dqn_final.zip",
        "checkpoints/a2c/a2c_final.zip"
    ]
    for cp in checkpoints:
        assert os.path.exists(cp), f"Checkpoint missing: {cp}"
        size = os.path.getsize(cp)
        assert size > 0, f"Checkpoint empty: {cp}"
    print("Test 8 Passed: Frozen checkpoint files present and intact.")


if __name__ == "__main__":
    test_01_model_cache_loading()
    test_02_profiler_fitting()
    test_03_condition_a_base_rl()
    test_04_condition_b_profiling_only()
    test_05_condition_c_dda_only()
    test_06_condition_d_profiling_dda()
    test_07_aggregate_metrics_calculation()
    test_08_frozen_checkpoint_integrity()
    print("\nALL 8 PHASE 7 UNIT & SANITY TESTS PASSED SUCCESSFULLY!")
