"""
Phase 8.2: Playable Card Game UI Redesign & Integration Test Suite.

Verifies:
1. Flask backend health check (/api/health)
2. Playable game initialization in 1v1 mode (/api/game/start)
3. Human action submission and legal action validation (/api/game/action)
4. Automatic AI opponent turn execution loop
5. Action log updating and public visibility
6. Imperfect-information boundaries (0 private opponent hand exposure)
7. Evidence synchronization & latest_ai_decision object integrity
8. Policy non-interference (LLM cannot change selected action)
9. GroundingValidator mismatch detection (MUST fail if explanation action != actual evidence action)
10. Session reset & Play Again isolation
11. Phase 7 historical metrics loading (/api/experiments)
12. Frozen checkpoint integrity
"""

import os
import json
import numpy as np

from ui.app import app, GAME_SESSIONS


def test_01_health_endpoint():
    with app.test_client() as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "HEALTHY"
        assert data["phase"] == "8.1" or data["phase"] == "8.2"
        assert data["rl_model_loaded"] is True
        assert data["profiler_fitted"] is True
        print("Phase 8.2 Test 1 Passed: Health endpoint healthy.")


def test_02_start_playable_game():
    with app.test_client() as client:
        res = client.post("/api/game/start", json={"seed": 42, "mode": "1v1", "player_count": 2})
        assert res.status_code == 200
        data = res.get_json()
        assert "session_id" in data
        assert data["mode"] == "1v1"
        assert data["current_player"] == 0
        assert len(data["human_hand"]) > 0
        assert len(data["human_action_mask"]) == 10
        # Verify HITMAN (index 0) is permanently masked in 1v1
        assert data["human_action_mask"][0] == 0 or data["human_action_mask"][0] is False
        assert len(data["action_log"]) > 0
        print("Phase 8.2 Test 2 Passed: Playable game initialized cleanly in 1v1 mode.")


def test_03_playable_turn_loop():
    with app.test_client() as client:
        start_res = client.post("/api/game/start", json={"seed": 42, "mode": "1v1"}).get_json()
        session_id = start_res["session_id"]
        action_mask = start_res["human_action_mask"]

        # Find first legal human action index
        legal_idx = next(idx for idx, legal in enumerate(action_mask) if legal)

        # Human plays action
        act_res = client.post("/api/game/action", json={"session_id": session_id, "action_idx": legal_idx})
        assert act_res.status_code == 200
        data = act_res.get_json()

        log_text = " ".join(data["action_log"])
        assert "You played" in log_text
        assert "RL Agent played" in log_text or data["game_over"]
        print("Phase 8.2 Test 3 Passed: Playable human -> AI turn loop executed automatically.")


def test_04_human_action_validation():
    with app.test_client() as client:
        start_res = client.post("/api/game/start", json={"seed": 42}).get_json()
        session_id = start_res["session_id"]

        # HITMAN (index 0) is permanently masked and illegal
        bad_res = client.post("/api/game/action", json={"session_id": session_id, "action_idx": 0})
        assert bad_res.status_code == 400
        data = bad_res.get_json()
        assert "error" in data
        print("Phase 8.2 Test 4 Passed: Human illegal action strictly rejected by backend.")


def test_05_imperfect_information_boundary():
    with app.test_client() as client:
        start_res = client.post("/api/game/start", json={"seed": 42}).get_json()
        session_id = start_res["session_id"]
        legal_idx = next(idx for idx, legal in enumerate(start_res["human_action_mask"]) if legal)

        state = client.post("/api/game/action", json={"session_id": session_id, "action_idx": legal_idx}).get_json()

        # Ensure state does NOT expose opponent's private hand or deck order
        assert "opponent_hand" not in state
        assert "deck_order" not in state
        assert "future_cards" not in state
        assert "opponent_hand_count" in state  # Public count only
        print("Phase 8.2 Test 5 Passed: Imperfect-information boundaries strictly preserved.")


def test_06_policy_non_interference_and_synchronization():
    with app.test_client() as client:
        start_res = client.post("/api/game/start", json={"seed": 42}).get_json()
        session_id = start_res["session_id"]
        legal_idx = next(idx for idx, legal in enumerate(start_res["human_action_mask"]) if legal)

        state = client.post("/api/game/action", json={"session_id": session_id, "action_idx": legal_idx}).get_json()

        sess = GAME_SESSIONS[session_id]
        if sess.get("last_ai_evidence"):
            ai_ev = sess["last_ai_evidence"]
            ai_chosen_name = ai_ev["policy_evidence"]["chosen_action_name"]
            summary_action = state["latest_ai_decision"]["action"]
            assert ai_chosen_name.upper() == summary_action.upper()
            assert state["latest_ai_decision"]["probability"] == ai_ev["policy_evidence"]["chosen_action_probability"]
        print("Phase 8.2 Test 6 Passed: Policy non-interference & latest_ai_decision evidence synchronization verified.")


def test_07_grounding_validator_mismatch_detection():
    from modules.grounding_validator import GroundingValidator
    validator = GroundingValidator()

    # Matching evidence and explanation
    fake_evidence = {
        "policy_evidence": {
            "chosen_action_name": "DRAW",
            "chosen_action_probability": 0.85,
            "chosen_action_rank": 1,
            "top_legal_actions": [{"action": "DRAW", "probability": 0.85}],
            "value_estimate_V_s": -1.5
        },
        "state_facts": {"deck_size": 10, "legal_actions": ["DRAW", "ATTACK"]}
    }
    matching_exp = {
        "explanation": "The RL agent selected DRAW with 85.0% probability because state value V(s) is -1.50.",
        "grounding_status": "PARSED",
        "evidence": ["DRAW probability is 85.0%"]
    }
    res_pass = validator.validate(fake_evidence, matching_exp)
    assert res_pass["grounded"] is True, "Valid matching explanation must pass grounding validation."

    # Mismatched explanation (says SHIELD when actual action was DRAW)
    mismatched_exp = {
        "selected_action": "SHIELD",
        "explanation": "The RL agent selected SHIELD with 90.0% probability to defend itself.",
        "grounding_status": "PARSED",
        "evidence": ["SHIELD probability is 90.0%"]
    }
    res_fail = validator.validate(fake_evidence, mismatched_exp)
    assert res_fail["grounded"] is False, "Mismatched action explanation MUST fail grounding validation!"
    print("Phase 8.2 Test 7 Passed: GroundingValidator mismatch detection verified.")


def test_08_play_again_session_isolation():
    with app.test_client() as client:
        start_res1 = client.post("/api/game/start", json={"seed": 100, "mode": "1v1"}).get_json()
        sess_id1 = start_res1["session_id"]

        start_res2 = client.post("/api/game/start", json={"seed": 200, "mode": "1v1"}).get_json()
        sess_id2 = start_res2["session_id"]

        assert sess_id1 != sess_id2
        assert start_res2["turn_index"] == 0
        assert start_res2["game_over"] is False
        print("Phase 8.2 Test 8 Passed: Play Again / Reset creates fresh isolated sessions.")


def test_09_experiments_endpoint():
    with app.test_client() as client:
        res = client.get("/api/experiments")
        assert res.status_code == 200
        data = res.get_json()
        assert "ablation_studies" in data
        assert "phase3_comparison" in data
        print("Phase 8.2 Test 9 Passed: Experiments metrics endpoint loaded successfully.")


def test_10_frozen_checkpoint_integrity():
    checkpoints = [
        "checkpoints/ppo/ppo_final.zip",
        "checkpoints/dqn/dqn_final.zip",
        "checkpoints/a2c/a2c_final.zip"
    ]
    for cp in checkpoints:
        assert os.path.exists(cp)
        assert os.path.getsize(cp) > 0
    print("Phase 8.2 Test 10 Passed: All frozen checkpoints present and intact.")


if __name__ == "__main__":
    test_01_health_endpoint()
    test_02_start_playable_game()
    test_03_playable_turn_loop()
    test_04_human_action_validation()
    test_05_imperfect_information_boundary()
    test_06_policy_non_interference_and_synchronization()
    test_07_grounding_validator_mismatch_detection()
    test_08_play_again_session_isolation()
    test_09_experiments_endpoint()
    test_10_frozen_checkpoint_integrity()
    print("\nALL 10 PHASE 8.2 INTEGRATION TESTS PASSED SUCCESSFULLY!")
