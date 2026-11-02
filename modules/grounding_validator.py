"""
Phase 6: Grounding Validator Module.

Performs automatic machine-readable grounding and faithfulness verification on generated
LLM action explanations against actual PPO policy evidence and observable state facts.

Validation Checks:
1. Selected Action Consistency
2. Probability Consistency (+/- 5% tolerance)
3. Legal-Action Consistency
4. State-Fact Consistency
5. Hidden Information Exclusion (no opponent hand / deck secrets)
6. Q-Value Violation Check (no PPO Q-value claims)
7. Action Substitution Check (no recommending different actions)
"""

import re
import numpy as np


class GroundingValidator:
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance

    def validate(self, evidence: dict, explanation_result: dict) -> dict:
        state_facts = evidence["state_facts"]
        policy_evidence = evidence["policy_evidence"]

        chosen_name = policy_evidence["chosen_action_name"]
        legal_actions = state_facts["legal_actions"]
        actual_probs = {item["action"]: item["probability"] for item in policy_evidence["top_legal_actions"]}

        exp_text = explanation_result.get("explanation", "")
        exp_action = explanation_result.get("selected_action", "")
        exp_alternatives = explanation_result.get("legal_alternatives", [])

        # 1. Selected Action Consistency
        selected_action_consistent = (exp_action.upper() == chosen_name.upper()) or (chosen_name.upper() in exp_text.upper())

        # 2. Probability Consistency Check
        prob_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", exp_text)
        probabilities_consistent = True
        for p_str in prob_matches:
            val = float(p_str) / 100.0
            # Check if val is close to any actual legal probability
            matched = any(abs(val - actual_p) <= self.tolerance for actual_p in actual_probs.values())
            if not matched:
                probabilities_consistent = False
                break

        # Also check legal_alternatives list in JSON
        for alt in exp_alternatives:
            alt_name = alt.get("action", "").upper()
            alt_prob = alt.get("probability", 0.0)
            if alt_name in actual_probs:
                if abs(alt_prob - actual_probs[alt_name]) > self.tolerance:
                    probabilities_consistent = False

        # 3. Legal-Action Consistency
        legal_actions_consistent = True
        for alt in exp_alternatives:
            alt_name = alt.get("action", "").upper()
            if alt_name not in [a.upper() for a in legal_actions]:
                legal_actions_consistent = False
                break

        # 4. State-Fact Consistency
        state_facts_consistent = True
        deck_matches = re.findall(r"deck(?:\s+size)?\s*(?:of|=|:)?\s*(\d+)", exp_text, re.IGNORECASE)
        for d_str in deck_matches:
            if int(d_str) != state_facts["deck_size"]:
                state_facts_consistent = False

        # 5. Hidden Information Violation Check
        hidden_keywords = [
            "opponent hand", "opponent's hand", "enemy hand", "opponent holds",
            "next card is hitman", "hitman probability", "deck secret", "future card",
            "deck contains hitman at position"
        ]
        hidden_information_violation = any(kw in exp_text.lower() for kw in hidden_keywords)

        # 6. Q-Value Violation Check (PPO uses V(s) and pi(a|s), NOT Q-values)
        q_value_keywords = ["q-value", "q value", "q(s,a)", "q-score", "q score"]
        q_value_violation = any(kw in exp_text.lower() for kw in q_value_keywords)

        # 7. Action Substitution Check
        sub_keywords = [
            f"should play", f"better to play", f"instead of {chosen_name.lower()}",
            f"recommend playing", f"ought to choose"
        ]
        action_substitution = any(kw in exp_text.lower() for kw in sub_keywords)

        # 8. Unsupported Causal Overstatement Check
        causal_keywords = [
            "because v(s)", "caused the action", "the agent knew", "the agent determined",
            "the opponent probably", "sole viable strategy", "only viable strategy", "action was optimal"
        ]
        causal_overstatement_violation = any(kw in exp_text.lower() for kw in causal_keywords)

        # Overall Grounded Status
        is_grounded = (
            selected_action_consistent
            and probabilities_consistent
            and legal_actions_consistent
            and state_facts_consistent
            and not hidden_information_violation
            and not q_value_violation
            and not action_substitution
            and not causal_overstatement_violation
        )

        return {
            "grounded": is_grounded,
            "selected_action_consistent": selected_action_consistent,
            "probabilities_consistent": probabilities_consistent,
            "legal_actions_consistent": legal_actions_consistent,
            "state_facts_consistent": state_facts_consistent,
            "hidden_information_violation": hidden_information_violation,
            "q_value_violation": q_value_violation,
            "action_substitution": action_substitution,
            "causal_overstatement_violation": causal_overstatement_violation,
        }
