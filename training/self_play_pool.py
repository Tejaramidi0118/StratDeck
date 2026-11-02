"""
Manages a pool of past model checkpoints to use as opponents during self-play,
so the agent doesn't overfit to only ever playing against its current self.
Supports MaskablePPO, MaskedDQN, and MaskedA2C.
"""

import os
import random
import numpy as np
from sb3_contrib import MaskablePPO
from training.masked_algos import MaskedDQN, MaskedA2C


class SelfPlayPool:
    def __init__(self, checkpoint_dir: str, max_pool_size: int = 8):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.max_pool_size = max_pool_size
        self._models = []  # list of checkpoint file paths, oldest first
        self._cache = {}   # path -> loaded model instance

    def add_checkpoint(self, model, step: int, algo_name: str = "ppo"):
        path = os.path.join(self.checkpoint_dir, f"{algo_name}_step{step}.zip")
        model.save(path)
        self._models.append(path)
        self._cache[path] = model
        if len(self._models) > self.max_pool_size:
            oldest = self._models.pop(0)  # drop oldest to keep pool bounded
            if oldest in self._cache:
                del self._cache[oldest]

    def _load_model(self, path: str):
        if "dqn" in os.path.basename(path).lower():
            return MaskedDQN.load(path)
        elif "a2c" in os.path.basename(path).lower():
            return MaskedA2C.load(path)
        else:
            return MaskablePPO.load(path)

    def sample_opponent_policy(self):
        """Returns a callable(raw_obs, legal_actions) -> action, sampled from the pool.
        Mixes: 70% most recent checkpoint, 30% a random older one."""
        if not self._models:
            return None  # caller should fall back to random policy

        if len(self._models) == 1 or random.random() < 0.7:
            path = self._models[-1]
        else:
            path = random.choice(self._models[:-1])

        if path not in self._cache:
            self._cache[path] = self._load_model(path)
        model = self._cache[path]

        def policy_fn(raw_obs: dict, legal_actions: list):
            from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
            obs_vec = AgentZeroEnv.encode_raw_obs(raw_obs)

            # Construct boolean action mask for legal actions
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            for a in legal_actions:
                mask[AgentZeroEnv._encode_action(a)] = True

            action_id, _ = model.predict(obs_vec, action_masks=mask, deterministic=False)
            decoded = AgentZeroEnv._decode_action(int(action_id))
            return decoded

        return policy_fn



