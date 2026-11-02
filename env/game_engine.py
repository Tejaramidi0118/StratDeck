"""
Core game engine for Agent Zero. Pure game logic, no RL/gym dependency here
so it can be unit-tested and played by random/rule-based bots independently.
"""

import random
from dataclasses import dataclass, field
from .cards import Card, build_deck, DEFAULT_ACTION_COUNTS

STARTING_HAND_SIZE = 5


@dataclass
class PlayerState:
    player_id: int
    hand: list = field(default_factory=list)
    alive: bool = True
    move_history: list = field(default_factory=list)  # list of (turn_idx, Card) played


class GameEngine:
    """
    A single game of Agent Zero. Call reset() to start, then step(action) with
    the current player's chosen action. Legal actions are always: any card in
    hand, or DRAW (represented as None / a sentinel action id).
    """

    DRAW_ACTION = -1  # sentinel: draw a card from the deck (ends turn unless attacked)

    def __init__(self, num_players: int = 2, seed: int = None):
        self.num_players = num_players
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        hands, self.deck = build_deck(self.num_players, self.rng)
        self.discard = []
        self.players = [PlayerState(i, hand=hands[i]) for i in range(self.num_players)]
        self.current_player = 0
        self.direction = 1
        self.turns_remaining_for_current = 1  # >1 after being ATTACKed
        self.pending_draw_count = 1            # >1 after being TRAPped
        self.turn_index = 0
        self.game_over = False
        self.winner = None
        return self._get_obs(self.current_player)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def legal_actions(self, player_id: int) -> list:
        """Returns a list of legal action ids for the given player's turn."""
        p = self.players[player_id]
        actions = list(set(p.hand))  # unique card types in hand
        actions.append(self.DRAW_ACTION)
        return actions

    def step(self, action):
        """
        Executes one action for self.current_player.
        Returns (obs, reward, done, info).
        """
        if self.game_over:
            raise RuntimeError("Game is already over; call reset().")

        p = self.players[self.current_player]
        info = {"eliminated": [], "action": action}
        reward = 0.0

        if action == self.DRAW_ACTION:
            reward = self._handle_draw(p, info)
        else:
            reward = self._handle_play_card(p, action, info)

        p.move_history.append((self.turn_index, action))
        self.turn_index += 1

        done = self._check_game_over()
        if done:
            reward += 100.0 if self.winner == p.player_id else 0.0
            info["winner"] = self.winner

        self.last_action_info = info
        obs = self._get_obs(self.current_player)
        return obs, reward, done, info

    # ------------------------------------------------------------------ #
    # Internal logic
    # ------------------------------------------------------------------ #

    def _handle_draw(self, p: PlayerState, info: dict) -> float:
        reward = 0.0
        for _ in range(self.pending_draw_count):
            if not self.deck:
                if self.discard:
                    self._reshuffle_discard_into_deck()
                else:
                    break  # no cards left anywhere
            card = self.deck.pop()
            if card == Card.HITMAN:
                if Card.SHIELD in p.hand:
                    p.hand.remove(Card.SHIELD)
                    self.discard.append(Card.SHIELD)
                    # Re-insert HITMAN into deck at a random position
                    insert_idx = self.rng.randint(0, len(self.deck)) if self.deck else 0
                    self.deck.insert(insert_idx, Card.HITMAN)
                    reward -= 10.0  # used a defensive resource
                    info["defused"] = True
                else:
                    p.alive = False
                    # Discard eliminated player's hand to discard pile
                    self.discard.extend(p.hand)
                    p.hand.clear()
                    info["eliminated"].append(p.player_id)
                    reward -= 100.0
                    break
            else:
                p.hand.append(card)
        self.pending_draw_count = 1
        self._advance_turn()
        return reward


    def _handle_play_card(self, p: PlayerState, card: Card, info: dict) -> float:
        if card not in p.hand:
            # illegal move guard - treat as a no-op skip with penalty
            return -5.0
        p.hand.remove(card)
        self.discard.append(card)
        reward = 0.0

        if card == Card.ATTACK:
            # ATTACK ends current player's turn immediately without drawing,
            # and passes 2 turns (or current remaining + 2) to the next player.
            extra_turns = (self.turns_remaining_for_current if self.turns_remaining_for_current > 1 else 1) + 1
            self.current_player = self._next_alive_player_id(self.current_player)
            self.turns_remaining_for_current = extra_turns
        elif card == Card.SKIP:
            self._advance_turn()
        elif card == Card.REVERSE:
            self.direction *= -1
            self._advance_turn()
        elif card == Card.SPY:
            info["spy_peek"] = list(reversed(self.deck[-3:])) if self.deck else []
            reward += 5.0
            # does not end turn
        elif card == Card.SWAP:
            target = self._next_alive_player(self.current_player)
            if target.hand:
                idx = self.rng.randrange(len(target.hand))
                swapped = target.hand.pop(idx)
                given = None
                if p.hand:
                    given_idx = self.rng.randrange(len(p.hand))
                    given = p.hand.pop(given_idx)
                    target.hand.append(given)
                p.hand.append(swapped)
                info["swap_given"] = given
                info["swap_taken"] = swapped
            reward += 5.0
        elif card == Card.TRAP:
            self.pending_draw_count = 2
            reward += 10.0
            self._advance_turn()
        elif card == Card.STEAL:
            target = self._next_alive_player(self.current_player)
            if target.hand:
                idx = self.rng.randrange(len(target.hand))
                stolen = target.hand.pop(idx)
                p.hand.append(stolen)
                info["stolen_card"] = stolen
            reward += 20.0  # eliminating opponent options
        else:
            self._advance_turn()

        return reward

    def _advance_turn(self):
        if not self.players[self.current_player].alive or self.turns_remaining_for_current <= 1:
            self.current_player = self._next_alive_player_id(self.current_player)
            self.turns_remaining_for_current = 1
        else:
            self.turns_remaining_for_current -= 1


    def _next_alive_player_id(self, from_id: int) -> int:
        n = self.num_players
        idx = from_id
        for _ in range(n):
            idx = (idx + self.direction) % n
            if self.players[idx].alive:
                return idx
        return from_id

    def _next_alive_player(self, from_id: int) -> PlayerState:
        return self.players[self._next_alive_player_id(from_id)]

    def _reshuffle_discard_into_deck(self):
        self.deck = self.discard[:]
        self.discard = []
        self.rng.shuffle(self.deck)

    def _check_game_over(self) -> bool:
        alive = [p for p in self.players if p.alive]
        if len(alive) <= 1:
            self.game_over = True
            self.winner = alive[0].player_id if alive else None
            return True
        return False

    def _get_obs(self, player_id: int) -> dict:
        p = self.players[player_id]
        return {
            "hand": list(p.hand),
            "deck_size": len(self.deck),
            "discard": list(self.discard),
            "discard_top": self.discard[-1] if self.discard else None,
            "alive_players": [pl.player_id for pl in self.players if pl.alive],
            "current_player": self.current_player,
            "pending_draw_count": self.pending_draw_count,
        }

