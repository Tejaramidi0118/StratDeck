"""
Card definitions for Agent Zero.
Deadly card: HITMAN. Action cards: ATTACK, SHIELD, SKIP, REVERSE, SPY, SWAP, TRAP, STEAL.
"""

from enum import IntEnum
import random


class Card(IntEnum):
    HITMAN = 0      # deadly card - drawing this eliminates you unless you play SHIELD first
    ATTACK = 1      # end your turn without drawing, force next player to take 2 turns
    SHIELD = 2      # play to negate a HITMAN draw (single-use, consumed)
    SKIP = 3        # end your turn without drawing
    REVERSE = 4      # reverse turn order
    SPY = 5          # peek at the top N cards of the deck
    SWAP = 6         # swap a random card with the next player's hand
    TRAP = 7         # force the next player to draw 2 cards instead of 1
    STEAL = 8        # steal a random card from a target player's hand


CARD_NAMES = {c: c.name for c in Card}

# Default deck composition. HITMAN count = num_players - 1 (so exactly one player
# survives when all HITMANs have been drawn and not defused).
DEFAULT_ACTION_COUNTS = {
    Card.ATTACK: 4,
    Card.SHIELD: 6,
    Card.SKIP: 4,
    Card.REVERSE: 4,
    Card.SPY: 4,
    Card.SWAP: 3,
    Card.TRAP: 3,
    Card.STEAL: 3,
}

ACTION_CARDS = list(DEFAULT_ACTION_COUNTS.keys())
NUM_CARD_TYPES = len(Card)


def build_deck(num_players: int, rng: random.Random):
    """
    Builds starting hands and draw deck for given number of players.
    Each player receives 1 SHIELD and 4 random action cards to start.
    Then (num_players - 1) HITMAN cards are inserted into the remaining draw deck and shuffled.
    Returns (hands, draw_deck).
    """
    action_pool = []
    for card, count in DEFAULT_ACTION_COUNTS.items():
        if card == Card.SHIELD:
            # Reserve 1 SHIELD per player for starting hands
            rem_shields = max(0, count - num_players)
            action_pool.extend([Card.SHIELD] * rem_shields)
        else:
            action_pool.extend([card] * count)
    rng.shuffle(action_pool)

    # Deal 1 SHIELD + 4 random action cards to each player
    hands = []
    for _ in range(num_players):
        hand = [Card.SHIELD]
        for _ in range(4):
            if action_pool:
                hand.append(action_pool.pop())
        hands.append(hand)

    # Insert HITMAN cards into the remaining draw deck and shuffle
    draw_deck = action_pool[:]
    hitmen = [Card.HITMAN] * (num_players - 1)
    draw_deck.extend(hitmen)
    rng.shuffle(draw_deck)

    return hands, draw_deck

