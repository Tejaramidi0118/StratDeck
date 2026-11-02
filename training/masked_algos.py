"""
MaskedDQN and MaskedA2C wrappers for StratDeck.
Provides PyTorch-level action-logit/Q-value masking for DQN and A2C so they
never select illegal action indices or the unplayable HITMAN hazard (action 0).
"""

import numpy as np
import torch
from stable_baselines3 import DQN, A2C
from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS


class MaskedDQN(DQN):
    """DQN subclass with Q-value action masking for training and prediction."""

    def predict(self, observation, state=None, episode_start=None, deterministic=True, action_masks=None):
        if action_masks is not None:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                q_values = self.q_net(obs_tensor).cpu().numpy().copy()

            mask = np.array(action_masks, dtype=bool)
            if len(mask.shape) == 1:
                q_values[0][~mask] = -1e9
                action = int(np.argmax(q_values[0]))
                return np.array(action), state
            else:
                q_values[~mask] = -1e9
                actions = np.argmax(q_values, axis=1)
                return actions, state

        return super().predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        """Override action sampling during rollout collection to respect action masks."""
        # Get current action mask from env
        try:
            mask = self.env.envs[0].unwrapped.action_masks()
        except Exception:
            mask = np.ones(NUM_ACTIONS, dtype=bool)

        legal_indices = np.where(mask)[0]

        if self.num_timesteps < learning_starts:
            act = np.random.choice(legal_indices)
            return np.array([act]), np.array([act])

        # Epsilon-greedy with masking
        if np.random.rand() < self.exploration_rate:
            act = np.random.choice(legal_indices)
            return np.array([act]), np.array([act])
        else:
            with torch.no_grad():
                obs_tensor = self._last_obs
                if not isinstance(obs_tensor, torch.Tensor):
                    obs_tensor = torch.as_tensor(obs_tensor, dtype=torch.float32, device=self.device)
                q_values = self.q_net(obs_tensor).cpu().numpy().copy()
            q_values[0][~mask] = -1e9
            act = int(np.argmax(q_values[0]))
            return np.array([act]), np.array([act])


class MaskedA2C(A2C):
    """A2C subclass with PyTorch action-logit masking during both training rollouts and prediction."""

    def _setup_model(self) -> None:
        super()._setup_model()
        self._apply_action_masking_to_policy()

    def set_env(self, env, force_reset: bool = True) -> None:
        super().set_env(env, force_reset=force_reset)
        self._apply_action_masking_to_policy()

    def _apply_action_masking_to_policy(self) -> None:
        if hasattr(self, "policy") and self.policy is not None:
            model_self = self
            def get_masked_action_dist(latent_pi):
                action_logits = model_self.policy.action_net(latent_pi)
                try:
                    m = model_self.env.envs[0].unwrapped.action_masks()
                except Exception:
                    m = None
                if m is not None:
                    mask_tensor = torch.as_tensor(m, dtype=torch.bool, device=action_logits.device)
                    if len(mask_tensor.shape) == 1:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    action_logits = torch.where(mask_tensor, action_logits, torch.tensor(-1e9, device=action_logits.device))
                return model_self.policy.action_dist.proba_distribution(action_logits=action_logits)

            self.policy._get_action_dist_from_latent = get_masked_action_dist

    def predict(self, observation, state=None, episode_start=None, deterministic=True, action_masks=None):
        if action_masks is not None:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                distribution = self.policy.get_distribution(obs_tensor)
                logits = distribution.distribution.logits.cpu().numpy().copy()

            mask = np.array(action_masks, dtype=bool)
            if len(mask.shape) == 1:
                logits[0][~mask] = -1e9
                if deterministic:
                    act = int(np.argmax(logits[0]))
                else:
                    exp_l = np.exp(logits[0] - np.max(logits[0]))
                    probs = exp_l / exp_l.sum()
                    act = int(np.random.choice(len(probs), p=probs))
                return np.array(act), state
            else:
                logits[~mask] = -1e9
                if deterministic:
                    actions = np.argmax(logits, axis=1)
                else:
                    actions = []
                    for row in logits:
                        exp_l = np.exp(row - np.max(row))
                        probs = exp_l / exp_l.sum()
                        actions.append(np.random.choice(len(probs), p=probs))
                    actions = np.array(actions)
                return actions, state

        return super().predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)
