"""
Phase 7b: Profiling-vs-DDA conflict evaluation.
Measures how often the strategy bias suggested by opponent-style profiling
disagrees with the direction DDA independently wants to push the agent
(strengthen vs weaken), and by how much.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from modules.profiler import OpponentProfiler, STYLE_LABELS
from modules.dda import DifficultyAdjuster
from env.cards import Card

N_TRIALS = 200


def synthetic_move_history(style: str, length: int = 30):
    """Generates a fake move history biased toward a given style, for offline
    calibration/testing without needing real game logs yet."""
    weighted = {
        "aggressive": [Card.ATTACK] * 5 + [Card.STEAL] * 3 + list(Card),
        "defensive": [Card.SHIELD] * 5 + [Card.SKIP] * 3 + list(Card),
        "risk_taker": [Card.TRAP] * 4 + [Card.SWAP] * 3 + list(Card),
        "conservative": [-1] * 6 + [Card.SPY] * 2 + list(Card),
    }[style]
    return [(i, random.choice(weighted)) for i in range(length)]


def run_conflict_eval():
    profiler = OpponentProfiler(n_clusters=4)
    histories = [synthetic_move_history(s) for s in STYLE_LABELS for _ in range(10)]
    profiler.fit(histories)

    conflicts = 0
    for _ in range(N_TRIALS):
        style = random.choice(STYLE_LABELS)
        history = synthetic_move_history(style)
        detected_style = profiler.predict_style(history)
        bias = profiler.strategy_bias_for_style(detected_style)

        dda = DifficultyAdjuster()
        # simulate a lopsided run of results to force DDA to want to weaken the agent
        for _ in range(10):
            dda.record_result(agent_won=True)
        dda_wants_weaker = dda.temperature > 1.0

        # profiling bias > 1.0 on average = wants agent to play MORE aggressively (stronger)
        avg_bias = sum(bias.values()) / len(bias) if bias else 1.0
        profiling_wants_stronger = avg_bias > 1.0

        if dda_wants_weaker and profiling_wants_stronger:
            conflicts += 1

    return conflicts / N_TRIALS


if __name__ == "__main__":
    conflict_rate = run_conflict_eval()
    print("\n--- Phase 7 Result (paste into your report) ---")
    print(f"Profiling-vs-DDA conflict rate: {conflict_rate:.1%}")
    print("(Fraction of trials where DDA wants to weaken the agent while")
    print(" opponent-profiling simultaneously suggests a stronger strategy.)")
