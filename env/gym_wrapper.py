"""
Gymnasium wrapper around GameEngine. Provides a fixed-size observation vector
and a discrete action space with legal-action masking (required for PPO/DQN/A2C
via Stable-Baselines3 + sb3-contrib's MaskablePPO, or manual masking otherwise).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .game_engine import GameEngine
from .cards import Card, NUM_CARD_TYPES

MAX_HAND_SIZE = 10
ACTION_DRAW = NUM_CARD_TYPES  # extra action id for "draw"
NUM_ACTIONS = NUM_CARD_TYPES + 1  # one action per card type + draw


class AgentZeroEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_players: int = 2, opponent_policy=None, seed: int = None, max_episode_steps: int = 300):
        super().__init__()
        self.num_players = num_players
        self.opponent_policy = opponent_policy  # callable(obs, legal_actions) -> action, for self-play
        self.engine = GameEngine(num_players=num_players, seed=seed)
        self.agent_player_id = 0
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.last_opponent_infos = []

        # Observation: hand counts (9) + discard counts (9) + deck_size (1) + num_alive (1) + pending_draws (1) = 21
        obs_len = NUM_CARD_TYPES * 2 + 3
        self.observation_space = spaces.Box(low=0, high=99, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.last_opponent_infos = []
        raw_obs = self.engine.reset()
        self._play_opponents_until_agent_turn()
        return self._encode_obs(), {}

    def step(self, action):
        self.step_count += 1
        self.last_opponent_infos = []
        card_action = self._decode_action(action)
        legal = self.engine.legal_actions(self.engine.current_player)
        if card_action not in legal:
            # illegal action: penalize but don't crash - substitute a legal fallback
            card_action = self.engine.DRAW_ACTION

        _, reward, done, info = self.engine.step(card_action)

        if not done:
            done = self._play_opponents_until_agent_turn()

        info["opponent_infos"] = self.last_opponent_infos

        if done:
            info["winner"] = self.engine.winner

        obs = self._encode_obs()
        truncated = (self.step_count >= self.max_episode_steps) and not done
        return obs, reward, done, truncated, info



    def action_masks(self) -> np.ndarray:
        """For MaskablePPO (sb3-contrib). Returns a boolean mask over NUM_ACTIONS."""
        legal = self.engine.legal_actions(self.agent_player_id)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for a in legal:
            mask[self._encode_action(a)] = True
        return mask

    # ------------------------------------------------------------------ #

    def _play_opponents_until_agent_turn(self) -> bool:
        """Advances the game using self.opponent_policy until it's the agent's turn
        or the game ends. Returns True if the game ended."""
        self.last_opponent_infos = []
        while (
            not self.engine.game_over
            and self.engine.current_player != self.agent_player_id
        ):
            opp_id = self.engine.current_player
            legal = self.engine.legal_actions(opp_id)
            if self.opponent_policy is not None:
                action = self.opponent_policy(self.engine._get_obs(opp_id), legal)
            else:
                import random
                action = random.choice(legal)
            _, _, done, opp_info = self.engine.step(action)
            self.last_opponent_infos.append(opp_info)
            if done:
                return True
        return self.engine.game_over

    def _encode_obs(self) -> np.ndarray:
        raw = self.engine._get_obs(self.agent_player_id)
        return self.encode_raw_obs(raw)

    @staticmethod
    def encode_raw_obs(raw_obs: dict) -> np.ndarray:
        hand_counts = np.zeros(NUM_CARD_TYPES, dtype=np.float32)
        for c in raw_obs["hand"]:
            hand_counts[int(c)] += 1

        discard_counts = np.zeros(NUM_CARD_TYPES, dtype=np.float32)
        for c in raw_obs.get("discard", []):
            discard_counts[int(c)] += 1

        extra = np.array(
            [raw_obs["deck_size"], len(raw_obs["alive_players"]), raw_obs["pending_draw_count"]],
            dtype=np.float32,
        )
        return np.concatenate([hand_counts, discard_counts, extra])

    @staticmethod
    def _decode_action(action_id: int):
        if action_id == ACTION_DRAW:
            return GameEngine.DRAW_ACTION
        return Card(action_id)

    @staticmethod
    def _encode_action(card_action) -> int:
        if card_action == GameEngine.DRAW_ACTION:
            return ACTION_DRAW
        return int(card_action)

