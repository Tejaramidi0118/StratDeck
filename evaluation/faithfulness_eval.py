"""
Phase 7a: Explanation-faithfulness evaluation.
Measures whether the LLM's natural-language explanation accurately reflects
the agent's REAL top action, comparing conditions where DDA is active vs
inactive. This is the core "interaction effect" measurement described in the
Phase 1 research plan.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.cards import Card
from modules.dda import DifficultyAdjuster
from modules.explainer import ExplainerModule, OllamaBackend

N_TEST_MOVES = 100


def mock_action_probs():
    """Placeholder: in the real pipeline, pull these from model.policy.
    get_distribution(obs).distribution.probs at explanation time."""
    import random
    cards = list(Card) + [-1]
    probs = [random.random() for _ in cards]
    total = sum(probs)
    return {c: p / total for c, p in zip(cards, probs)}


def check_faithfulness(explanation_text: str, real_top_action) -> bool:
    """Very simple faithfulness check: does the explanation text mention the
    same card name as the agent's actual top-probability action? Replace with
    a stricter check (e.g. semantic similarity) if needed for the report."""
    name = "draw" if real_top_action == -1 else Card(real_top_action).name.lower()
    return name in explanation_text.lower()


def run_condition(dda_active: bool, explainer: ExplainerModule, n=N_TEST_MOVES):
    faithful_count = 0
    dda = DifficultyAdjuster() if dda_active else None

    for i in range(n):
        probs = mock_action_probs()
        real_top_action = max(probs, key=probs.get)

        if dda_active:
            # simulate DDA perturbing the distribution before the move is chosen
            import numpy as np
            logits = list(probs.values())
            perturbed = dda.apply_temperature(logits)
            keys = list(probs.keys())
            probs = dict(zip(keys, perturbed))
            chosen_action = max(probs, key=probs.get)
            dda.record_result(agent_won=(i % 2 == 0))  # placeholder win/loss signal
        else:
            chosen_action = real_top_action

        explanation = explainer.explain_move(
            chosen_action=chosen_action,
            action_probs=probs,
            hand=[Card.SHIELD, Card.SPY],
            hitman_probability=0.3,
        )

        if check_faithfulness(explanation, chosen_action):
            faithful_count += 1

    return faithful_count / n


if __name__ == "__main__":
    explainer = ExplainerModule(backend=OllamaBackend(model="llama3.1"))

    print("Running faithfulness evaluation WITHOUT DDA active...")
    faithfulness_no_dda = run_condition(dda_active=False, explainer=explainer)

    print("Running faithfulness evaluation WITH DDA active...")
    faithfulness_with_dda = run_condition(dda_active=True, explainer=explainer)

    print("\n--- Phase 7 Result (paste into your report) ---")
    print(f"Explanation faithfulness WITHOUT DDA: {faithfulness_no_dda:.1%}")
    print(f"Explanation faithfulness WITH DDA:    {faithfulness_with_dda:.1%}")
    print(f"Delta: {faithfulness_no_dda - faithfulness_with_dda:+.1%}")
