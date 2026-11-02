"""
Comprehensive Phase 1 Unit Tests & Verification Suite for GameEngine & AgentZeroEnv.
Tests all 24 requirements: card mechanics, edge cases, deterministic transitions,
Gymnasium wrapper integration, 21-dim observation vector shape, action masking,
reproducible seeded execution, HITMAN hazard mask verification, and large-scale
random/rule-based simulations across 2, 3, and 4 player matches.

Run: python test_engine_mechanics.py
"""

import random
import unittest
import numpy as np

from env.cards import Card, build_deck, NUM_CARD_TYPES, DEFAULT_ACTION_COUNTS
from env.game_engine import GameEngine, PlayerState
from env.gym_wrapper import AgentZeroEnv, NUM_ACTIONS


class TestGameEngineMechanics(unittest.TestCase):

    def test_01_deck_creation(self):
        rng = random.Random(42)
        hands, deck = build_deck(num_players=2, rng=rng)
        # 2 players: each starting hand has 5 cards (1 guaranteed SHIELD + 4 action cards)
        self.assertEqual(len(hands), 2)
        self.assertEqual(len(hands[0]), 5)
        self.assertEqual(len(hands[1]), 5)
        self.assertGreaterEqual(hands[0].count(Card.SHIELD), 1)
        self.assertGreaterEqual(hands[1].count(Card.SHIELD), 1)
        # HITMAN cards should only be in the draw deck, never in initial hands
        self.assertNotIn(Card.HITMAN, hands[0])
        self.assertNotIn(Card.HITMAN, hands[1])
        self.assertEqual(deck.count(Card.HITMAN), 1)  # num_players - 1 = 1

    def test_02_draw_normal_card(self):
        engine = GameEngine(num_players=2, seed=100)
        p0 = engine.players[0]
        initial_hand_len = len(p0.hand)

        # Force top card to be ATTACK (not Hitman)
        engine.deck.append(Card.ATTACK)
        obs, reward, done, info = engine.step(engine.DRAW_ACTION)

        self.assertEqual(len(p0.hand), initial_hand_len + 1)
        self.assertEqual(p0.hand[-1], Card.ATTACK)
        self.assertFalse(done)

    def test_03_hitman_elimination_without_shield(self):
        engine = GameEngine(num_players=2, seed=101)
        p0 = engine.players[0]
        p0.hand = [c for c in p0.hand if c != Card.SHIELD]  # ensure no shield
        engine.deck.append(Card.HITMAN)  # put Hitman on top

        obs, reward, done, info = engine.step(engine.DRAW_ACTION)

        self.assertFalse(p0.alive)
        self.assertTrue(done)
        self.assertEqual(engine.winner, 1)
        self.assertIn(0, info["eliminated"])

    def test_04_hitman_defuse_with_shield(self):
        engine = GameEngine(num_players=2, seed=102)
        p0 = engine.players[0]
        p0.hand = [Card.SHIELD]
        engine.deck = [Card.ATTACK, Card.HITMAN]  # top is HITMAN

        obs, reward, done, info = engine.step(engine.DRAW_ACTION)

        self.assertTrue(p0.alive)
        self.assertNotIn(Card.SHIELD, p0.hand)
        self.assertIn(Card.SHIELD, engine.discard)
        self.assertIn(Card.HITMAN, engine.deck)  # re-inserted into deck
        self.assertTrue(info.get("defused"))

    def test_05_attack_mechanic(self):
        engine = GameEngine(num_players=2, seed=103)
        p0 = engine.players[0]
        p0.hand.append(Card.ATTACK)

        # Player 0 plays ATTACK
        obs, reward, done, info = engine.step(Card.ATTACK)

        # Player 0's turn ends immediately, turn moves to Player 1, Player 1 has 2 turns
        self.assertEqual(engine.current_player, 1)
        self.assertEqual(engine.turns_remaining_for_current, 2)

    def test_06_skip_mechanic(self):
        engine = GameEngine(num_players=2, seed=104)
        p0 = engine.players[0]
        p0.hand.append(Card.SKIP)

        obs, reward, done, info = engine.step(Card.SKIP)

        # SKIP ends turn without drawing
        self.assertEqual(engine.current_player, 1)

    def test_07_reverse_mechanic(self):
        engine = GameEngine(num_players=3, seed=105)
        p0 = engine.players[0]
        p0.hand.append(Card.REVERSE)

        obs, reward, done, info = engine.step(Card.REVERSE)

        # Direction reverses from 1 to -1. Turn advances from 0 to (0 - 1) % 3 = 2
        self.assertEqual(engine.direction, -1)
        self.assertEqual(engine.current_player, 2)

    def test_08_spy_mechanic(self):
        engine = GameEngine(num_players=2, seed=106)
        p0 = engine.players[0]
        p0.hand.append(Card.SPY)

        obs, reward, done, info = engine.step(Card.SPY)

        self.assertIn("spy_peek", info)
        self.assertEqual(engine.current_player, 0)  # SPY does not end turn

    def test_09_swap_mechanic(self):
        engine = GameEngine(num_players=2, seed=107)
        p0 = engine.players[0]
        p1 = engine.players[1]
        p0.hand = [Card.SWAP, Card.ATTACK]
        p1.hand = [Card.STEAL]

        obs, reward, done, info = engine.step(Card.SWAP)

        # p0 played SWAP, gave ATTACK, got STEAL
        self.assertIn(Card.STEAL, p0.hand)
        self.assertIn(Card.ATTACK, p1.hand)
        self.assertEqual(engine.current_player, 0)  # SWAP does not end turn

    def test_10_trap_mechanic(self):
        engine = GameEngine(num_players=2, seed=108)
        p0 = engine.players[0]
        p0.hand.append(Card.TRAP)

        obs, reward, done, info = engine.step(Card.TRAP)

        # Player 0 plays TRAP: turn advances to Player 1, pending_draw_count becomes 2
        self.assertEqual(engine.current_player, 1)
        self.assertEqual(engine.pending_draw_count, 2)

    def test_11_steal_mechanic(self):
        engine = GameEngine(num_players=2, seed=109)
        p0 = engine.players[0]
        p1 = engine.players[1]
        p0.hand = [Card.STEAL]
        p1.hand = [Card.SKIP]

        obs, reward, done, info = engine.step(Card.STEAL)

        self.assertIn(Card.SKIP, p0.hand)
        self.assertEqual(len(p1.hand), 0)
        self.assertEqual(engine.current_player, 0)  # STEAL does not end turn

    def test_12_gym_wrapper_obs_and_actions(self):
        env = AgentZeroEnv(num_players=2, seed=200)
        obs, info = env.reset()

        # Observation shape must be 21 (9 hand + 9 discard + 3 extras)
        self.assertEqual(obs.shape, (21,))
        self.assertTrue(np.all(obs >= 0))

        # Action space Discrete(10)
        self.assertEqual(env.action_space.n, NUM_ACTIONS)

        # Action mask shape
        mask = env.action_masks()
        self.assertEqual(len(mask), NUM_ACTIONS)
        self.assertTrue(mask[-1])  # DRAW is always legal

    def test_13_invalid_action_fallback(self):
        env = AgentZeroEnv(num_players=2, seed=201)
        obs, info = env.reset()

        # Execute an illegal card action (e.g. Card 0 HITMAN when not in hand)
        # Should gracefully fall back to DRAW without crashing
        obs, reward, done, truncated, info = env.step(0)
        self.assertIsInstance(done, bool)

    def test_14_seeded_reproducibility(self):
        """Verify stochastic environment with reproducible seeded execution."""
        seed = 777
        
        # Trajectory 1
        e1 = GameEngine(num_players=3, seed=seed)
        r1 = random.Random(seed)
        e1.reset()
        t1 = []
        while not e1.game_over and len(t1) < 50:
            legal = e1.legal_actions(e1.current_player)
            act = r1.choice(legal)
            obs, reward, done, info = e1.step(act)
            t1.append((e1.current_player, act, done, e1.winner))

        # Trajectory 2
        e2 = GameEngine(num_players=3, seed=seed)
        r2 = random.Random(seed)
        e2.reset()
        t2 = []
        while not e2.game_over and len(t2) < 50:
            legal = e2.legal_actions(e2.current_player)
            act = r2.choice(legal)
            obs, reward, done, info = e2.step(act)
            t2.append((e2.current_player, act, done, e2.winner))

        self.assertEqual(t1, t2)
        self.assertEqual(e1.winner, e2.winner)

    def test_15_observation_specification(self):
        """Verify 21-dim observation vector bounds, structure, and privacy."""
        env = AgentZeroEnv(num_players=2, seed=300)
        obs, _ = env.reset()

        self.assertEqual(obs.shape, (21,))
        self.assertTrue(np.all(obs >= 0))
        self.assertTrue(np.all(obs <= 99))

        # Check raw obs privacy (no opponent hand or deck order leak)
        raw = env.engine._get_obs(0)
        self.assertNotIn("opponents_hand", raw)
        self.assertNotIn("deck_order", raw)

    def test_16_hitman_action_mask_always_false(self):
        """Verify that Action Index 0 (HITMAN hazard) action mask is ALWAYS False."""
        env = AgentZeroEnv(num_players=4, seed=400)
        obs, _ = env.reset()

        # Step through multiple turns and verify mask[0] is strictly False
        for step_idx in range(50):
            if env.engine.game_over:
                break
            mask = env.action_masks()
            self.assertFalse(mask[0], f"Action mask 0 (HITMAN) was True at step {step_idx}!")
            legal = env.engine.legal_actions(env.engine.current_player)
            self.assertNotIn(Card.HITMAN, legal)
            obs, reward, done, truncated, info = env.step(env.action_space.sample())


def run_simulations(n_games=5000, num_players=2, mode="heuristic"):
    """Runs N games with random/rule-based choices to check stability."""
    wins = {i: 0 for i in range(num_players)}
    wins[None] = 0
    lengths = []
    timeouts = 0

    for i in range(n_games):
        engine = GameEngine(num_players=num_players, seed=i)
        agent_rng = random.Random(i)
        engine.reset()
        steps = 0
        while not engine.game_over and steps < 500:
            p_id = engine.current_player
            legal = engine.legal_actions(p_id)

            if mode == "heuristic":
                # Rule-based heuristic: 80% play action card if available, 20% draw
                action_cards = [a for a in legal if a != engine.DRAW_ACTION]
                if action_cards and agent_rng.random() < 0.8:
                    action = agent_rng.choice(action_cards)
                else:
                    action = engine.DRAW_ACTION
            else:
                action = agent_rng.choice(legal)

            obs, reward, done, info = engine.step(action)
            steps += 1

        if not engine.game_over:
            timeouts += 1
            wins[None] += 1
        else:
            wins[engine.winner] += 1
        lengths.append(steps)

    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    return wins, avg_len, min_len, max_len, timeouts


if __name__ == "__main__":
    print("Running Phase 1 Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGameEngineMechanics)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nAll 16 Unit Tests Passed Successfully!")
        print("\nRunning Regression Simulations...")
        
        # 1. 2,000 two-player random games
        w_2p_rnd, avg_2p_rnd, min_2p_rnd, max_2p_rnd, t_2p_rnd = run_simulations(n_games=2000, num_players=2, mode="random")
        print(f"2-Player Random Games (2,000):")
        print(f"   P0 Wins: {w_2p_rnd[0]} ({w_2p_rnd[0]/2000:.1%}), P1 Wins: {w_2p_rnd[1]} ({w_2p_rnd[1]/2000:.1%}), Timeouts: {t_2p_rnd}")
        print(f"   Game Lengths: avg={avg_2p_rnd:.1f}, min={min_2p_rnd}, max={max_2p_rnd}")

        # 2. 5,000 two-player heuristic games
        w_2p_h, avg_2p_h, min_2p_h, max_2p_h, t_2p_h = run_simulations(n_games=5000, num_players=2, mode="heuristic")
        print(f"\n2-Player Heuristic Games (5,000):")
        print(f"   P0 Wins: {w_2p_h[0]} ({w_2p_h[0]/5000:.1%}), P1 Wins: {w_2p_h[1]} ({w_2p_h[1]/5000:.1%}), Timeouts: {t_2p_h}")
        print(f"   Game Lengths: avg={avg_2p_h:.1f}, min={min_2p_h}, max={max_2p_h}")

        # 3. 2,000 three-player heuristic games
        w_3p_h, avg_3p_h, min_3p_h, max_3p_h, t_3p_h = run_simulations(n_games=2000, num_players=3, mode="heuristic")
        print(f"\n3-Player Heuristic Games (2,000):")
        print(f"   P0: {w_3p_h[0]} ({w_3p_h[0]/2000:.1%}), P1: {w_3p_h[1]} ({w_3p_h[1]/2000:.1%}), P2: {w_3p_h[2]} ({w_3p_h[2]/2000:.1%}), Timeouts: {t_3p_h}")
        print(f"   Game Lengths: avg={avg_3p_h:.1f}, min={min_3p_h}, max={max_3p_h}")

        # 4. 2,000 four-player heuristic games
        w_4p_h, avg_4p_h, min_4p_h, max_4p_h, t_4p_h = run_simulations(n_games=2000, num_players=4, mode="heuristic")
        print(f"\n4-Player Heuristic Games (2,000):")
        print(f"   P0: {w_4p_h[0]} ({w_4p_h[0]/2000:.1%}), P1: {w_4p_h[1]} ({w_4p_h[1]/2000:.1%}), P2: {w_4p_h[2]} ({w_4p_h[2]/2000:.1%}), P3: {w_4p_h[3]} ({w_4p_h[3]/2000:.1%}), Timeouts: {t_4p_h}")
        print(f"   Game Lengths: avg={avg_4p_h:.1f}, min={min_4p_h}, max={max_4p_h}")
    else:
        print("\nUnit Tests Failed!")
