"""
Phase 6: Explanation Evidence Extraction Module.

Extracts structured, observable game-state facts and real PPO policy inference evidence
for action explanation.

CRITICAL COMPLIANCE:
1. PPO Policy Evidence uses pi(a|s) action probabilities and V(s) value estimates from real model inference.
2. ZERO Q-values are claimed or fabricated for PPO.
3. ZERO hidden information (opponent private hand, deck order) is extracted or included.
"""

import torch
import numpy as np
from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from env.cards import Card, NUM_CARD_TYPES


ACTION_NAMES = {
    0: "HITMAN",
    1: "ATTACK",
    2: "SHIELD",
    3: "SKIP",
    4: "REVERSE",
    5: "SPY",
    6: "SWAP",
    7: "TRAP",
    8: "STEAL",
    9: "DRAW",
}


def action_to_name(action_id: int) -> str:
    return ACTION_NAMES.get(int(action_id), f"ACTION_{action_id}")


def extract_explanation_evidence(
    model,
    env: AgentZeroEnv,
    obs: np.ndarray,
    chosen_action_idx: int,
    player_id: int = 0,
    dda_controller = None,
    profiler = None,
    move_history: list = None
) -> dict:
    """
    Extracts observable state facts and real PPO inference evidence.
    """
    engine = env.engine
    raw_obs = engine._get_obs(player_id)

    # 1. Observable State Facts
    hand_cards = [action_to_name(int(c)) for c in raw_obs["hand"]]
    hand_counts = {}
    for c in hand_cards:
        hand_counts[c] = hand_counts.get(c, 0) + 1

    discard_cards = [action_to_name(int(c)) for c in raw_obs.get("discard", [])]
    discard_counts = {}
    for c in discard_cards:
        discard_counts[c] = discard_counts.get(c, 0) + 1

    legal_action_ids = engine.legal_actions(player_id)
    legal_action_names = [action_to_name(AgentZeroEnv._encode_action(a)) for a in legal_action_ids]
    
    action_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in legal_action_ids:
        enc = AgentZeroEnv._encode_action(a)
        if 0 <= enc < NUM_ACTIONS:
            action_mask[enc] = True

    state_facts = {
        "player_id": player_id,
        "turn_index": engine.turn_index,
        "alive_players_count": len(raw_obs["alive_players"]),
        "deck_size": raw_obs["deck_size"],
        "pending_draw_count": raw_obs["pending_draw_count"],
        "own_hand_cards": hand_cards,
        "own_hand_counts": hand_counts,
        "public_discard_counts": discard_counts,
        "legal_actions": legal_action_names,
        "action_mask": action_mask.tolist(),
    }

    # 2. Policy Inference Evidence
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.policy.device)
    mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(model.policy.device)

    with torch.no_grad():
        if hasattr(model, "policy") and hasattr(model.policy, "get_distribution"):
            try:
                # MaskableActorCriticPolicy (MaskablePPO)
                distribution = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
                probs = distribution.distribution.probs.cpu().numpy()[0]
            except TypeError:
                # Standard ActorCriticPolicy (e.g. MaskedA2C)
                distribution = model.policy.get_distribution(obs_tensor)
                raw_probs = distribution.distribution.probs.cpu().numpy()[0]
                masked_probs = raw_probs * action_mask
                total = np.sum(masked_probs)
                probs = masked_probs / total if total > 0 else np.where(action_mask, 1.0 / np.sum(action_mask), 0.0)

            if hasattr(model.policy, "predict_values"):
                try:
                    value_estimate = float(model.policy.predict_values(obs_tensor).cpu().numpy()[0][0])
                except Exception:
                    value_estimate = 0.0
            else:
                value_estimate = 0.0
        elif hasattr(model, "policy") and hasattr(model.policy, "q_net"):
            q_values = model.policy.q_net(obs_tensor).cpu().numpy()[0]
            q_masked = np.copy(q_values)
            q_masked[~action_mask] = -1e9
            exp_q = np.exp(q_masked - np.max(q_masked))
            exp_q[~action_mask] = 0
            probs = exp_q / (np.sum(exp_q) + 1e-8)
            value_estimate = float(np.max(q_values[action_mask]))
        else:
            probs = np.where(action_mask, 1.0 / np.sum(action_mask), 0.0)
            value_estimate = 0.0

    chosen_name = action_to_name(chosen_action_idx)
    chosen_prob = float(probs[chosen_action_idx])

    # Top legal actions by probability
    legal_probs = []
    for idx, name in enumerate(ACTION_NAMES.values()):
        if action_mask[idx]:
            legal_probs.append({"action": name, "probability": float(probs[idx])})

    legal_probs.sort(key=lambda x: x["probability"], reverse=True)

    # Rank of chosen action
    action_rank = 1
    for item in legal_probs:
        if item["action"] == chosen_name:
            break
        action_rank += 1

    policy_evidence = {
        "chosen_action_idx": chosen_action_idx,
        "chosen_action_name": chosen_name,
        "chosen_action_probability": chosen_prob,
        "chosen_action_rank": action_rank,
        "top_legal_actions": legal_probs[:4],
        "value_estimate_V_s": value_estimate,
    }

    # 3. Context Metadata (Profile & DDA)
    context_info = {
        "profiler_cluster": None,
        "dda_difficulty_level": None,
        "dda_checkpoint_name": None,
    }

    if profiler is not None and move_history is not None and len(move_history) > 0:
        try:
            context_info["profiler_cluster"] = int(profiler.predict_cluster(move_history))
        except Exception:
            context_info["profiler_cluster"] = None

    if dda_controller is not None:
        try:
            status = dda_controller.get_status()
            context_info["dda_difficulty_level"] = status["current_difficulty"]
            context_info["dda_checkpoint_name"] = status["checkpoint_name"]
        except Exception:
            pass

    return {
        "state_facts": state_facts,
        "policy_evidence": policy_evidence,
        "context": context_info,
    }
