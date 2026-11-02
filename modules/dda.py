"""
Phase 5: Dynamic Difficulty Adjustment (DDA) Controller.

Adaptively selects opponent policy checkpoints based on rolling player win rate
and contextual opponent behavioral profiles from Phase 4.

Key Design Principles:
1. Frozen underlying RL models (no online gradient updates or weight modifications).
2. Checkpoint Selection based on empirical calibration.
3. Rolling player win-rate signal over a sliding window (default: 20 games).
4. Target win-rate band: 45% - 55% (0.45 <= rolling_win_rate <= 0.55).
5. Cooldown & Hysteresis to prevent high-frequency oscillation (min 5 games between adjustments).
6. Integration with Phase 4 OpponentProfiler for context logging and cluster-wise analysis.
"""

from collections import deque
import os
import numpy as np

DEFAULT_DIFFICULTY_MAPPING = {
    0: {"name": "Random", "type": "rule", "path": None, "target_wr": 0.50},
    1: {"name": "MaskedA2C_Final", "type": "model", "path": "checkpoints/a2c/a2c_final.zip", "target_wr": 0.37},
    2: {"name": "MaskedDQN_Final", "type": "model", "path": "checkpoints/dqn/dqn_final.zip", "target_wr": 0.20},
    3: {"name": "MaskablePPO_Step50k", "type": "model", "path": "checkpoints/ppo/ppo_step50000.zip", "target_wr": 0.11},
    4: {"name": "MaskablePPO_Final", "type": "model", "path": "checkpoints/ppo/ppo_final.zip", "target_wr": 0.09},
    5: {"name": "MaskablePPO_Step350k", "type": "model", "path": "checkpoints/ppo/ppo_step350000.zip", "target_wr": 0.04},
}


class DDAController:
    def __init__(
        self,
        target_band: tuple = (0.45, 0.55),
        window_size: int = 20,
        min_games_between_adjustments: int = 5,
        initial_difficulty: int = 2,
        difficulty_mapping: dict = None,
        profiler = None,
    ):
        self.target_band = target_band
        self.window_size = window_size
        self.min_games_between_adjustments = min_games_between_adjustments
        self.current_difficulty = initial_difficulty
        self.difficulty_mapping = difficulty_mapping if difficulty_mapping else DEFAULT_DIFFICULTY_MAPPING
        self.profiler = profiler

        self.min_level = min(self.difficulty_mapping.keys())
        self.max_level = max(self.difficulty_mapping.keys())
        self.is_fixed = False

        self.results_window = deque(maxlen=window_size)
        self.games_since_adjustment = 0
        self.total_games_played = 0

        self.transition_history = []
        self.last_adjustment_direction = 0  # +1 for increase, -1 for decrease
        self.oscillation_count = 0

    def set_difficulty(self, level: int, fixed: bool = True):
        """Allows direct user selection of difficulty level."""
        self.current_difficulty = max(self.min_level, min(self.max_level, level))
        self.is_fixed = fixed

    def current_win_rate(self) -> float:
        if not self.results_window:
            return 0.5
        return sum(self.results_window) / len(self.results_window)

    def is_in_target_band(self) -> bool:
        wr = self.current_win_rate()
        return self.target_band[0] <= wr <= self.target_band[1]

    def get_current_checkpoint_info(self) -> dict:
        return self.difficulty_mapping.get(self.current_difficulty, self.difficulty_mapping[self.min_level])

    def update(self, player_won: bool, move_history: list = None, game_length: int = 0) -> dict:
        self.total_games_played += 1
        self.games_since_adjustment += 1
        self.results_window.append(1 if player_won else 0)

        # Contextual profiling via Phase 4 profiler if provided
        cluster_id = None
        if self.profiler is not None and move_history is not None and len(move_history) > 0:
            try:
                cluster_id = self.profiler.predict_cluster(move_history)
            except Exception:
                cluster_id = None

        wr = self.current_win_rate()
        low, high = self.target_band
        transition_occurred = False
        transition_reason = "MAINTAINED"
        old_difficulty = self.current_difficulty

        # If user explicitly chose a fixed difficulty, do not auto-adjust
        if getattr(self, "is_fixed", False):
            transition_reason = f"FIXED_BY_USER (Level {self.current_difficulty})"
            checkpoint_info = self.get_current_checkpoint_info()
            return {
                "player_won": player_won,
                "rolling_win_rate": wr,
                "in_target_band": self.is_in_target_band(),
                "current_difficulty": self.current_difficulty,
                "is_fixed": True,
                "checkpoint_name": checkpoint_info["name"],
                "checkpoint_path": checkpoint_info["path"],
                "transition_occurred": False,
                "transition_reason": transition_reason,
                "cluster_id": cluster_id,
                "game_length": game_length,
            }

        # Evaluate adaptation condition: window must have at least 5 games and cooldown passed
        if len(self.results_window) >= 5 and self.games_since_adjustment >= self.min_games_between_adjustments:
            direction = 0
            if wr > high and self.current_difficulty < self.max_level:
                # Primary signal: Player winning too much -> Increase opponent difficulty
                direction = +1
                transition_reason = f"WIN_RATE_HIGH ({wr:.1%} > {high:.1%})"
            elif wr < low and self.current_difficulty > self.min_level:
                # Primary signal: Player losing too much -> Decrease opponent difficulty
                direction = -1
                transition_reason = f"WIN_RATE_LOW ({wr:.1%} < {low:.1%})"

            if direction != 0:
                target_level = self.current_difficulty + direction

                # Secondary Profile-Aware Contextual Selector:
                # If target_level is adjacent to alternative checkpoint types,
                # use behavioral cluster to select between model families:
                # Cluster 0 (Draw-heavy) prefers PPO model family (Levels 3, 4, 5).
                # Cluster 1 (High-defense/High-diversity) prefers A2C/DQN model family (Levels 1, 2).
                if cluster_id == 0 and target_level in [1, 2] and direction < 0:
                    # For Cluster 0 players needing a step down, Level 3 (PPO-50k) is selected as profile-matched checkpoint
                    target_level = max(self.min_level, 3)
                elif cluster_id == 1 and target_level == 3 and direction < 0:
                    # For Cluster 1 players needing a step down, Level 2 (DQN-Final) is selected as profile-matched checkpoint
                    target_level = 2

                if target_level != self.current_difficulty:
                    self.current_difficulty = target_level
                    transition_occurred = True

            if transition_occurred:
                # Check for oscillation (reversal within cooldown)
                if self.last_adjustment_direction != 0 and direction == -self.last_adjustment_direction:
                    self.oscillation_count += 1
                self.last_adjustment_direction = direction
                self.games_since_adjustment = 0

                self.transition_history.append({
                    "game_idx": self.total_games_played,
                    "old_difficulty": old_difficulty,
                    "new_difficulty": self.current_difficulty,
                    "win_rate": wr,
                    "cluster_id": cluster_id,
                    "reason": transition_reason,
                })

        checkpoint_info = self.get_current_checkpoint_info()

        return {
            "game_idx": self.total_games_played,
            "player_won": player_won,
            "rolling_win_rate": wr,
            "in_target_band": self.is_in_target_band(),
            "current_difficulty": self.current_difficulty,
            "checkpoint_name": checkpoint_info["name"],
            "checkpoint_path": checkpoint_info["path"],
            "transition_occurred": transition_occurred,
            "transition_reason": transition_reason,
            "cluster_id": cluster_id,
            "game_length": game_length,
        }

    def get_status(self) -> dict:
        info = self.get_current_checkpoint_info()
        is_fixed = getattr(self, "is_fixed", False)
        return {
            "total_games": self.total_games_played,
            "window_size": len(self.results_window),
            "rolling_win_rate": self.current_win_rate(),
            "in_target_band": self.is_in_target_band(),
            "current_difficulty": self.current_difficulty,
            "checkpoint": info["name"],
            "checkpoint_name": info["name"],
            "checkpoint_path": info["path"],
            "total_transitions": len(self.transition_history),
            "oscillation_count": self.oscillation_count,
            "is_fixed": is_fixed,
            "mode": f"Fixed (Level {self.current_difficulty})" if is_fixed else "Auto (DDA Adaptive)",
        }
