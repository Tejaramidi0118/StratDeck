"""
Unit tests for Phase 5 Dynamic Difficulty Adjustment (DDA) Controller.
Tests:
1. Initial difficulty initialization
2. Rolling win-rate calculation
3. Increase difficulty when win rate > 0.55
4. Decrease difficulty when win rate < 0.45
5. Maintain difficulty within target band (0.45 <= wr <= 0.55)
6. Hysteresis / cooldown enforcement (min games between adjustments)
7. Oscillation tracking on immediate reversals
8. Integration with Phase 4 OpponentProfiler context logging
9. Invalid checkpoint safety fallback
10. Transition logging and status reporting
"""

import os
import json
import numpy as np
from modules.dda import DDAController, DEFAULT_DIFFICULTY_MAPPING
from modules.profiler import OpponentProfiler
from env.cards import Card


def test_initial_difficulty():
    dda = DDAController(initial_difficulty=2)
    status = dda.get_status()
    assert status["current_difficulty"] == 2
    assert status["checkpoint_name"] == "MaskedDQN_Final"
    assert status["rolling_win_rate"] == 0.5
    assert status["in_target_band"] is True


def test_rolling_win_rate_calculation():
    dda = DDAController(window_size=10, initial_difficulty=2)
    # Win 7 out of 10 games
    for _ in range(7):
        dda.update(player_won=True)
    for _ in range(3):
        dda.update(player_won=False)

    assert np.isclose(dda.current_win_rate(), 0.70)


def test_increase_difficulty_above_threshold():
    dda = DDAController(
        target_band=(0.45, 0.55),
        window_size=10,
        min_games_between_adjustments=5,
        initial_difficulty=2
    )

    # Win 8 consecutive games to push rolling win rate to 80% (>55%)
    for _ in range(8):
        res = dda.update(player_won=True)

    assert len(dda.transition_history) == 1
    assert dda.current_difficulty == 3  # Level 2 -> Level 3
    assert "WIN_RATE_HIGH" in dda.transition_history[0]["reason"]


def test_decrease_difficulty_below_threshold():
    dda = DDAController(
        target_band=(0.45, 0.55),
        window_size=10,
        min_games_between_adjustments=5,
        initial_difficulty=3
    )

    # Lose 8 consecutive games to push rolling win rate to 0% (<45%)
    for _ in range(8):
        res = dda.update(player_won=False)

    assert len(dda.transition_history) == 1
    assert dda.current_difficulty == 2  # Level 3 -> Level 2
    assert "WIN_RATE_LOW" in dda.transition_history[0]["reason"]


def test_maintain_difficulty_in_target_band():
    dda = DDAController(
        target_band=(0.40, 0.65),
        window_size=10,
        min_games_between_adjustments=5,
        initial_difficulty=2
    )

    # Win 5, Lose 5 -> rolling win rate remains inside target band (40-65%)
    for i in range(10):
        res = dda.update(player_won=(i % 2 == 0))

    assert len(dda.transition_history) == 0
    assert res["current_difficulty"] == 2


def test_cooldown_enforcement():
    dda = DDAController(
        target_band=(0.45, 0.55),
        window_size=10,
        min_games_between_adjustments=5,
        initial_difficulty=2
    )

    # First adjustment occurs at game 5
    for _ in range(5):
        res1 = dda.update(player_won=True)
    assert res1["transition_occurred"] is True
    assert res1["current_difficulty"] == 3

    # Game 6: win rate is still high, but cooldown (5 games) has not elapsed (only 1 game since adj)
    res2 = dda.update(player_won=True)
    assert res2["transition_occurred"] is False
    assert res2["current_difficulty"] == 3


def test_oscillation_tracking():
    dda = DDAController(
        target_band=(0.45, 0.55),
        window_size=5,
        min_games_between_adjustments=5,
        initial_difficulty=2
    )

    # Phase 1: 5 wins -> Increase (L2 -> L3), direction = +1
    for _ in range(5):
        dda.update(player_won=True)
    assert dda.current_difficulty == 3

    # Phase 2: 5 losses -> Reversal/Decrease (L3 -> L2), direction = -1
    for _ in range(5):
        dda.update(player_won=False)
    assert dda.current_difficulty == 2
    assert dda.oscillation_count == 1


def test_profiler_integration_context():
    profiler = OpponentProfiler(n_clusters=2, seed=42)
    # Fit profiler on dummy data
    profiler.fit([[(0, Card.ATTACK)], [(0, -1)]])

    dda = DDAController(profiler=profiler, initial_difficulty=2)
    sample_history = [(0, Card.ATTACK), (1, Card.SHIELD)]
    res = dda.update(player_won=True, move_history=sample_history, game_length=15)

    assert "cluster_id" in res
    assert res["cluster_id"] in [0, 1]


def test_invalid_checkpoint_safety_fallback():
    custom_mapping = {
        0: {"name": "Random", "type": "rule", "path": None},
        1: {"name": "BrokenModel", "type": "model", "path": "invalid/nonexistent.zip"}
    }
    dda = DDAController(initial_difficulty=1, difficulty_mapping=custom_mapping)
    info = dda.get_current_checkpoint_info()
    assert info["name"] == "BrokenModel"
    # DDA exposes the configured info safely without crashing
    assert dda.get_status()["current_difficulty"] == 1


if __name__ == "__main__":
    test_initial_difficulty()
    test_rolling_win_rate_calculation()
    test_increase_difficulty_above_threshold()
    test_decrease_difficulty_below_threshold()
    test_maintain_difficulty_in_target_band()
    test_cooldown_enforcement()
    test_oscillation_tracking()
    test_profiler_integration_context()
    test_invalid_checkpoint_safety_fallback()
    print("ALL 9 DDA CONTROLLER UNIT TESTS PASSED SUCCESSFULLY!")
