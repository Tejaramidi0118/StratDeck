"""
Phase 1 validation: run N games with random-action bots directly against the
GameEngine (no gym wrapper needed) and report win-rate / average game length.
Run: python test_phase1.py
"""

import random
from env.game_engine import GameEngine

N_GAMES = 2000


def run_random_game(seed):
    engine = GameEngine(num_players=2, seed=seed)
    engine.reset()
    steps = 0
    while not engine.game_over and steps < 500:
        legal = engine.legal_actions(engine.current_player)
        action = random.choice(legal)
        engine.step(action)
        steps += 1
    return engine.winner, steps


def test_phase1_random_game():
    winner, steps = run_random_game(seed=42)
    assert winner in (0, 1, None)
    assert steps > 0


if __name__ == "__main__":
    wins = {0: 0, 1: 0, None: 0}
    lengths = []
    for i in range(N_GAMES):
        winner, steps = run_random_game(seed=i)
        wins[winner] += 1
        lengths.append(steps)

    print(f"Ran {N_GAMES} games between two random bots.")
    print(f"Player 0 wins: {wins[0]} ({wins[0]/N_GAMES:.1%})")
    print(f"Player 1 wins: {wins[1]} ({wins[1]/N_GAMES:.1%})")
    print(f"Draws/timeouts: {wins[None]}")
    print(f"Average game length: {sum(lengths)/len(lengths):.1f} steps")
    print("\nIf both win rates are roughly balanced (~45-55%) and there are")
    print("few timeouts, the engine is working correctly. Proceed to Phase 2.")
