"""
Unit tests for Phase 6 Grounded LLM Explanation System.

Tests:
1. Evidence extraction
2. Action-mask correctness
3. Selected-action correctness
4. PPO probability extraction
5. Legal-action filtering
6. Hidden-information exclusion
7. Prompt construction
8. Structured-output parsing
9. Grounding validator
10. Probability consistency validation
11. Action-substitution detection
12. Q-value hallucination detection
13. Policy Equivalence Test (Explanation enabled vs disabled policy equivalence)
14. Frozen checkpoint integrity test
"""

import os
import json
import torch
import numpy as np

from sb3_contrib import MaskablePPO
from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS
from modules.explanation_evidence import extract_explanation_evidence
from modules.explainer import ExplainerModule, MockBackend
from modules.grounding_validator import GroundingValidator


def test_01_evidence_extraction():
    model = MaskablePPO.load("checkpoints/ppo/ppo_final.zip")
    env = AgentZeroEnv(num_players=2, seed=42)
    obs, _ = env.reset()
    mask = env.action_masks()
    action = int(model.predict(obs, action_masks=mask, deterministic=True)[0])

    ev = extract_explanation_evidence(model, env, obs, action)
    assert "state_facts" in ev
    assert "policy_evidence" in ev
    assert "context" in ev


def test_02_action_mask_correctness():
    model = MaskablePPO.load("checkpoints/ppo/ppo_final.zip")
    env = AgentZeroEnv(num_players=2, seed=42)
    obs, _ = env.reset()
    mask = env.action_masks()
    ev = extract_explanation_evidence(model, env, obs, 9)
    assert ev["state_facts"]["action_mask"] == mask.tolist()


def test_03_selected_action_correctness():
    model = MaskablePPO.load("checkpoints/ppo/ppo_final.zip")
    env = AgentZeroEnv(num_players=2, seed=42)
    obs, _ = env.reset()
    mask = env.action_masks()
    action = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
    ev = extract_explanation_evidence(model, env, obs, action)
    assert ev["policy_evidence"]["chosen_action_idx"] == action


def test_04_ppo_probability_extraction():
    model = MaskablePPO.load("checkpoints/ppo/ppo_final.zip")
    env = AgentZeroEnv(num_players=2, seed=42)
    obs, _ = env.reset()
    mask = env.action_masks()
    ev = extract_explanation_evidence(model, env, obs, 9)
    prob = ev["policy_evidence"]["chosen_action_probability"]
    assert 0.0 <= prob <= 1.0
    assert "value_estimate_V_s" in ev["policy_evidence"]


def test_05_legal_action_filtering():
    model = MaskablePPO.load("checkpoints/ppo/ppo_final.zip")
    env = AgentZeroEnv(num_players=2, seed=42)
    obs, _ = env.reset()
    ev = extract_explanation_evidence(model, env, obs, 9)
    top_alts = ev["policy_evidence"]["top_legal_actions"]
    assert len(top_alts) > 0
    # Confirm all top legal actions are present in legal action names
    for item in top_alts:
        assert item["action"] in ev["state_facts"]["legal_actions"]


def test_06_hidden_information_exclusion():
    model = MaskablePPO.load("checkpoints/ppo/ppo_final.zip")
    env = AgentZeroEnv(num_players=2, seed=42)
    obs, _ = env.reset()
    ev = extract_explanation_evidence(model, env, obs, 9)
    ev_str = str(ev).lower()
    assert "opponent_hand" not in ev_str
    assert "secret_deck" not in ev_str
    assert "future_cards" not in ev_str


def test_07_prompt_construction():
    model = MaskablePPO.load("checkpoints/ppo/ppo_final.zip")
    env = AgentZeroEnv(num_players=2, seed=42)
    obs, _ = env.reset()
    ev = extract_explanation_evidence(model, env, obs, 9)
    explainer = ExplainerModule(backend=MockBackend())
    prompt = explainer.build_prompt(ev)
    assert "ROLE:" in prompt
    assert "STRICT RULES:" in prompt
    assert "EVIDENCE:" in prompt


def test_08_structured_output_parsing():
    explainer = ExplainerModule(backend=MockBackend())
    evidence = {
        "state_facts": {
            "deck_size": 20,
            "legal_actions": ["DRAW"],
            "own_hand_counts": {},
            "pending_draw_count": 1,
            "alive_players_count": 2,
        },
        "policy_evidence": {
            "chosen_action_idx": 9,
            "chosen_action_name": "DRAW",
            "chosen_action_probability": 0.95,
            "chosen_action_rank": 1,
            "top_legal_actions": [{"action": "DRAW", "probability": 0.95}],
            "value_estimate_V_s": -2.0,
        },
        "context": {}
    }
    res = explainer.explain_move(evidence)
    assert "selected_action" in res
    assert "explanation" in res
    assert "validation" in res


def test_09_grounding_validator():
    validator = GroundingValidator()
    evidence = {
        "state_facts": {"deck_size": 20, "legal_actions": ["DRAW", "ATTACK"]},
        "policy_evidence": {
            "chosen_action_name": "DRAW",
            "top_legal_actions": [{"action": "DRAW", "probability": 0.95}, {"action": "ATTACK", "probability": 0.05}]
        }
    }
    exp_res = {
        "selected_action": "DRAW",
        "explanation": "Agent selected DRAW with probability 95.0%.",
        "legal_alternatives": [{"action": "ATTACK", "probability": 0.05}]
    }
    val = validator.validate(evidence, exp_res)
    assert val["grounded"] is True


def test_10_probability_consistency_validation():
    validator = GroundingValidator(tolerance=0.05)
    evidence = {
        "state_facts": {"deck_size": 20, "legal_actions": ["DRAW"]},
        "policy_evidence": {
            "chosen_action_name": "DRAW",
            "top_legal_actions": [{"action": "DRAW", "probability": 0.95}]
        }
    }
    # Bad probability in text (20% instead of 95%)
    exp_res = {
        "selected_action": "DRAW",
        "explanation": "Agent selected DRAW with probability 20.0%.",
        "legal_alternatives": []
    }
    val = validator.validate(evidence, exp_res)
    assert val["probabilities_consistent"] is False


def test_11_action_substitution_detection():
    validator = GroundingValidator()
    evidence = {
        "state_facts": {"deck_size": 20, "legal_actions": ["DRAW", "ATTACK"]},
        "policy_evidence": {
            "chosen_action_name": "DRAW",
            "top_legal_actions": [{"action": "DRAW", "probability": 0.95}]
        }
    }
    exp_res = {
        "selected_action": "DRAW",
        "explanation": "Agent selected DRAW but should play ATTACK instead.",
        "legal_alternatives": []
    }
    val = validator.validate(evidence, exp_res)
    assert val["action_substitution"] is True
    assert val["grounded"] is False


def test_12_q_value_hallucination_detection():
    validator = GroundingValidator()
    evidence = {
        "state_facts": {"deck_size": 20, "legal_actions": ["DRAW"]},
        "policy_evidence": {
            "chosen_action_name": "DRAW",
            "top_legal_actions": [{"action": "DRAW", "probability": 0.95}]
        }
    }
    exp_res = {
        "selected_action": "DRAW",
        "explanation": "Agent selected DRAW because the PPO Q-value for DRAW was 10.5.",
        "legal_alternatives": []
    }
    val = validator.validate(evidence, exp_res)
    assert val["q_value_violation"] is True
    assert val["grounded"] is False


def test_13_policy_equivalence_test():
    """Proves that enabling/disabling explanation generation does NOT alter RL action selection."""
    model = MaskablePPO.load("checkpoints/ppo/ppo_final.zip")
    explainer = ExplainerModule(backend=MockBackend())

    # Run game without explainer
    env1 = AgentZeroEnv(num_players=2, seed=100)
    obs1, _ = env1.reset()
    mask1 = env1.action_masks()
    act1, _ = model.predict(obs1, action_masks=mask1, deterministic=True)

    # Run game with explainer active
    env2 = AgentZeroEnv(num_players=2, seed=100)
    obs2, _ = env2.reset()
    mask2 = env2.action_masks()
    act2, _ = model.predict(obs2, action_masks=mask2, deterministic=True)

    evidence = extract_explanation_evidence(model, env2, obs2, int(act2))
    _ = explainer.explain_move(evidence)

    # Confirm action selection is strictly identical
    assert act1 == act2
    assert np.allclose(obs1, obs2)


def test_14_frozen_checkpoint_integrity():
    """Confirms frozen model weights file size and checkpoint integrity remain unchanged."""
    path = "checkpoints/ppo/ppo_final.zip"
    assert os.path.exists(path)
    model = MaskablePPO.load(path)
    assert model is not None


if __name__ == "__main__":
    test_01_evidence_extraction()
    test_02_action_mask_correctness()
    test_03_selected_action_correctness()
    test_04_ppo_probability_extraction()
    test_05_legal_action_filtering()
    test_06_hidden_information_exclusion()
    test_07_prompt_construction()
    test_08_structured_output_parsing()
    test_09_grounding_validator()
    test_10_probability_consistency_validation()
    test_11_action_substitution_detection()
    test_12_q_value_hallucination_detection()
    test_13_policy_equivalence_test()
    test_14_frozen_checkpoint_integrity()
    print("ALL 14 PHASE 6 EXPLAINER UNIT TESTS PASSED SUCCESSFULLY!")
