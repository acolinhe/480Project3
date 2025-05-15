import random
import time
import math
from typing import List, Tuple
from collections import Counter

SUITS = ["h", "d", "c", "s"]
RANKS = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
RANK_VAL = {v: k for k, v in RANKS.items()}

class Card:
    def __init__(self, rank: int, suit: int):
        self.rank = rank  # 2-14
        self.suit = suit  # 0-3
    
    def __repr__(self) -> str:
        return f"{RANKS[self.rank]}{SUITS[self.suit]}"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

class Deck:
    def __init__(self):
        self.cards = []
        self.reset()
    
    def reset(self):
        """Resets the deck with all 52 cards."""
        self.cards = []
        for rank in range(2, 15):
            for suit in range(4):
                self.cards.append(Card(rank, suit))
    
    def shuffle(self):
        """Shuffles the deck."""
        random.shuffle(self.cards)
    
    def draw(self) -> Card:
        """Draws a card from the deck."""
        if not self.cards:
            raise ValueError("No cards left in the deck")
        return self.cards.pop()
    
    def draw_excluding(self, excluded_cards: List[Card]) -> Card:
        """Draws a card from the deck excluding specific cards."""
        available_cards = [card for card in self.cards if card not in excluded_cards]
        if not available_cards:
            raise ValueError("No valid cards left to draw")
        card = random.choice(available_cards)
        self.cards.remove(card)
        return card

    def __len__(self) -> int:
        return len(self.cards)

def evaluate_hand(hole_cards: List[Card], community_cards: List[Card]) -> Tuple[int, List[int]]:
    """
    Evaluates a poker hand and returns a tuple (hand_type, tiebreaker_values).
    hand_type is 8 for straight flush, 7 for four of a kind, and so on.
    tiebreaker_values are used to break ties within the same hand type.
    
    Returns (hand_type, tiebreaker_cards) where:
    - 8: Straight Flush
    - 7: Four of a Kind
    - 6: Full House
    - 5: Flush
    - 4: Straight
    - 3: Three of a Kind
    - 2: Two Pair
    - 1: One Pair
    - 0: High Card
    """
    all_cards = hole_cards + community_cards
    
    # Group by rank and suit
    ranks = [card.rank for card in all_cards]
    rank_count = Counter(ranks)
    suits = [card.suit for card in all_cards]
    suit_count = Counter(suits)
    
    # Check for flush (5 or more cards of the same suit)
    flush_suit = None
    for suit, count in suit_count.items():
        if count >= 5:
            flush_suit = suit
            break
    
    flush_cards = [card for card in all_cards if flush_suit is not None and card.suit == flush_suit]
    
    # Check for straight (5 or more consecutive ranks)
    unique_ranks = sorted(set(ranks), reverse=True)
    
    # Handle Ace as low card (A, 2, 3, 4, 5)
    if 14 in unique_ranks and 2 in unique_ranks and 3 in unique_ranks and 4 in unique_ranks and 5 in unique_ranks:
        straight_high = 5
        has_straight = True
    else:
        straight_high = None
        has_straight = False
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                straight_high = unique_ranks[i]
                has_straight = True
                break
    
    # Check for straight flush
    if flush_suit is not None and has_straight:
        flush_ranks = sorted([card.rank for card in flush_cards], reverse=True)
        
        # Handle Ace as low for straight flush
        if 14 in flush_ranks and 2 in flush_ranks and 3 in flush_ranks and 4 in flush_ranks and 5 in flush_ranks:
            return (8, [5, 4, 3, 2, 1])
        
        for i in range(len(flush_ranks) - 4):
            if all(flush_ranks[i+j] == flush_ranks[i] - j for j in range(5)):
                return (8, [flush_ranks[i]])
    
    # Four of a kind
    for rank, count in rank_count.items():
        if count == 4:
            kickers = [r for r in sorted(ranks, reverse=True) if r != rank]
            return (7, [rank, kickers[0]])
    
    # Full house
    three_of_a_kind = None
    pairs = []
    
    for rank, count in sorted(rank_count.items(), key=lambda x: (x[1], x[0]), reverse=True):
        if count >= 3 and three_of_a_kind is None:
            three_of_a_kind = rank
        elif count >= 2:
            pairs.append(rank)
    
    if three_of_a_kind is not None and pairs:
        return (6, [three_of_a_kind, pairs[0]])
    
    # Flush
    if flush_suit is not None:
        flush_ranks = sorted([card.rank for card in flush_cards], reverse=True)[:5]
        return (5, flush_ranks)
    
    # Straight
    if has_straight:
        if straight_high == 5 and 14 in ranks:
            return (4, [5, 4, 3, 2, 1])
        return (4, [straight_high])
    
    # Three of a kind
    if three_of_a_kind is not None:
        kickers = [r for r in sorted(ranks, reverse=True) if r != three_of_a_kind][:2]
        return (3, [three_of_a_kind] + kickers)
    
    # Two pair
    if len(pairs) >= 2:
        kicker = max([r for r in ranks if r != pairs[0] and r != pairs[1]])
        return (2, [pairs[0], pairs[1], kicker])
    
    # One pair
    if pairs:
        kickers = [r for r in sorted(ranks, reverse=True) if r != pairs[0]][:3]
        return (1, [pairs[0]] + kickers)
    
    # High card
    high_cards = sorted(ranks, reverse=True)[:5]
    return (0, high_cards)

def compare_hands(hand1: Tuple[List[Card], List[Card]], hand2: Tuple[List[Card], List[Card]]) -> int:
    """
    Compare two hands and return:
    1 if hand1 wins
    -1 if hand2 wins
    0 if tie
    Each hand is a tuple of (hole_cards, community_cards)
    """
    hand1_value = evaluate_hand(hand1[0], hand1[1])
    hand2_value = evaluate_hand(hand2[0], hand2[1])
    
    # Compare hand types
    if hand1_value[0] > hand2_value[0]:
        return 1
    elif hand1_value[0] < hand2_value[0]:
        return -1
    
    # Same hand type, compare tiebreakers
    for tb1, tb2 in zip(hand1_value[1], hand2_value[1]):
        if tb1 > tb2:
            return 1
        elif tb1 < tb2:
            return -1
    
    return 0  # Tie

class MCTSNode:
    def __init__(self, parent=None, remaining_cards: List[Card] = None):
        self.parent = parent
        self.visits = 0
        self.wins = 0
        self.remaining_cards = remaining_cards if remaining_cards else []
        
    def ucb1(self, exploration_weight=1.414):
        """UCB1 formula for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def win_probability(self):
        """Return the probability of winning based on simulations"""
        if self.visits == 0:
            return 0
        return self.wins / self.visits

class PokerBot:
    def __init__(self, time_limit=10):
        self.time_limit = time_limit
    
    def make_decision(self, hole_cards: List[Card], community_cards: List[Card], phase: str) -> bool:
        """
        Decide whether to fold or stay based on MCTS simulations.
        Returns True to stay, False to fold.
        """
        start_time = time.time()
        
        print(f"\nDeciding at {phase}...")
        print(f"Your hole cards: {hole_cards}")
        print(f"Community cards: {community_cards}")
        
        # Setup for MCTS
        root = MCTSNode()
        simulations = 0
        wins = 0
        
        # Get all visible cards to exclude them from simulations
        visible_cards = hole_cards + community_cards
        
        # Calculate how many community cards we need to generate based on phase
        remaining_community = 5 - len(community_cards)
        
        while time.time() - start_time < self.time_limit:
            # Create a copy of the deck excluding visible cards
            deck = Deck()
            for card in visible_cards:
                if card in deck.cards:
                    deck.cards.remove(card)
            
            # 1. Simulate opponent hole cards
            opponent_hole = []
            for _ in range(2):
                opponent_hole.append(deck.draw())
            
            # 2. Simulate remaining community cards
            simulated_community = community_cards.copy()
            for _ in range(remaining_community):
                simulated_community.append(deck.draw())
            
            # 3. Evaluate and compare hands
            hand1 = (hole_cards, simulated_community)
            hand2 = (opponent_hole, simulated_community)
            
            result = compare_hands(hand1, hand2)
            
            # Update statistics
            root.visits += 1
            if result >= 0:  # Win or tie
                root.wins += 1
                wins += 1
            
            simulations += 1
        
        win_probability = root.wins / root.visits if root.visits > 0 else 0
        
        print(f"Completed {simulations} simulations in {time.time() - start_time:.2f} seconds")
        print(f"Estimated win probability: {win_probability:.2%}")
        
        # Stay if win probability is at least 50%, otherwise fold
        decision = win_probability >= 0.5
        print(f"Decision: {'STAY' if decision else 'FOLD'}")
        
        return decision

def card_from_string(card_str: str) -> Card:
    """Convert a string representation to a Card object (e.g., 'As' -> Ace of spades)"""
    if len(card_str) != 2:
        raise ValueError("Card string must be 2 characters (rank + suit)")
    
    rank_char = card_str[0].upper()
    suit_char = card_str[1].lower()
    
    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    suit_map = {'h': 0, 'd': 1, 'c': 2, 's': 3}
    
    if rank_char not in rank_map:
        raise ValueError(f"Invalid rank: {rank_char}")
    if suit_char not in suit_map:
        raise ValueError(f"Invalid suit: {suit_char}")
    
    return Card(rank_map[rank_char], suit_map[suit_char])

def parse_cards(card_strings: List[str]) -> List[Card]:
    """Convert a list of card strings to Card objects"""
    return [card_from_string(s) for s in card_strings]

def demo():
    """Run a demonstration of the poker bot"""
    bot = PokerBot(time_limit=10)
    
    hole_cards = parse_cards(["Ah", "Ac"])
    community_cards = []
    
    print("=== Pre-Flop Decision ===")
    bot.make_decision(hole_cards, community_cards, "Pre-Flop")
    
    community_cards = parse_cards(["2h", "7d", "Kc"])
    print("\n=== Pre-Turn Decision ===")
    bot.make_decision(hole_cards, community_cards, "Pre-Turn")
    
    community_cards = parse_cards(["2h", "7d", "Kc", "Qs"])
    print("\n=== Pre-River Decision ===")
    bot.make_decision(hole_cards, community_cards, "Pre-River")
    
    print("\n=== Weaker Hand Example ===")
    hole_cards = parse_cards(["7h", "2c"])
    community_cards = parse_cards(["As", "Kd", "Qh"])
    bot.make_decision(hole_cards, community_cards, "Pre-Turn")

if __name__ == "__main__":
    demo()