"""
Test PyTorch-level Action Masking for DQN and A2C.
"""
import torch
import numpy as np
from stable_baselines3 import DQN, A2C
from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from env.cards import Card


def masked_dqn_predict(model: DQN, obs: np.ndarray, mask: np.ndarray, deterministic: bool = True, epsilon: float = 0.0) -> int:
    """Masked Q-value action selection for DQN."""
    if not deterministic and np.random.rand() < epsilon:
        legal_indices = np.where(mask)[0]
        return np.random.choice(legal_indices)
    
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        q_values = model.q_net(obs_tensor).squeeze(0).cpu().numpy().copy()
        
    q_values[~mask] = -1e9
    return int(np.argmax(q_values))


def masked_a2c_predict(model: A2C, obs: np.ndarray, mask: np.ndarray, deterministic: bool = True) -> int:
    """Masked logits action selection for A2C."""
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        distribution = model.policy.get_distribution(obs_tensor)
        logits = distribution.distribution.logits.squeeze(0).cpu().numpy().copy()
        
    logits[~mask] = -1e9
    if deterministic:
        return int(np.argmax(logits))
    else:
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        return int(np.random.choice(len(probs), p=probs))


def test_action_masking():
    env = AgentZeroEnv(num_players=2, seed=42)
    obs, _ = env.reset()
    mask = env.action_masks()

    # Instantiate models
    dqn_model = DQN("MlpPolicy", env, verbose=0, seed=42)
    a2c_model = A2C("MlpPolicy", env, verbose=0, seed=42)

    dqn_action = masked_dqn_predict(dqn_model, obs, mask, deterministic=True)
    a2c_action = masked_a2c_predict(a2c_model, obs, mask, deterministic=True)

    print(f"Mask at action 0 (HITMAN): {mask[0]}")
    print(f"DQN selected action: {dqn_action} ({AgentZeroEnv._decode_action(dqn_action)})")
    print(f"A2C selected action: {a2c_action} ({AgentZeroEnv._decode_action(a2c_action)})")

    assert mask[0] == False, "HITMAN mask must be False!"
    assert mask[dqn_action] == True, "DQN selected an illegal action!"
    assert mask[a2c_action] == True, "A2C selected an illegal action!"
    print("Action masking test: PASSED!")


if __name__ == "__main__":
    test_action_masking()
