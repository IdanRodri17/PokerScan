"""
Poker Hand Evaluator for PokerVision

This module evaluates poker hands from detected cards, identifying hand rankings
such as pairs, straights, flushes, and other poker combinations.
"""

import logging
from typing import List, Dict, Tuple, Set, Optional
from enum import Enum
from collections import Counter

logger = logging.getLogger(__name__)


class HandRank(Enum):
    """Poker hand rankings from lowest to highest"""
    HIGH_CARD = (1, "High Card")
    ONE_PAIR = (2, "One Pair")
    TWO_PAIR = (3, "Two Pair")
    THREE_OF_A_KIND = (4, "Three of a Kind")
    STRAIGHT = (5, "Straight")
    FLUSH = (6, "Flush")
    FULL_HOUSE = (7, "Full House")
    FOUR_OF_A_KIND = (8, "Four of a Kind")
    STRAIGHT_FLUSH = (9, "Straight Flush")
    ROYAL_FLUSH = (10, "Royal Flush")
    
    def __init__(self, rank_value: int, display_name: str):
        self.rank_value = rank_value
        self.display_name = display_name
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.rank_value < other.rank_value
        return NotImplemented


class Card:
    """Represents a single playing card"""
    
    # Map card names to numeric values for comparison
    RANK_VALUES = {
        'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
        '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
    }
    
    SUIT_NAMES = {
        's': 'Spades', 'h': 'Hearts', 'd': 'Diamonds', 'c': 'Clubs'
    }
    
    def __init__(self, card_name: str):
        """
        Initialize a card from its name (e.g., 'As', 'Kh', 'Td')
        
        Args:
            card_name: Card name in format 'RankSuit' (e.g., 'As', '10h')
        """
        self.card_name = card_name.strip()
        
        if len(self.card_name) < 2:
            raise ValueError(f"Invalid card name: {card_name}")
        
        # Handle 10 specially (represented as 'T' or '10')
        if self.card_name.startswith('10'):
            self.rank = 'T'
            self.suit = self.card_name[2].lower()
        else:
            self.rank = self.card_name[0].upper()
            self.suit = self.card_name[1].lower()
        
        # Validate rank and suit
        if self.rank not in self.RANK_VALUES:
            raise ValueError(f"Invalid rank: {self.rank}")
        if self.suit not in self.SUIT_NAMES:
            raise ValueError(f"Invalid suit: {self.suit}")
        
        self.rank_value = self.RANK_VALUES[self.rank]
        self.suit_name = self.SUIT_NAMES[self.suit]
    
    def __str__(self) -> str:
        return self.card_name
    
    def __repr__(self) -> str:
        return f"Card('{self.card_name}')"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
    
    def __lt__(self, other) -> bool:
        if isinstance(other, Card):
            return self.rank_value < other.rank_value
        return NotImplemented


class PokerHand:
    """Represents and evaluates a poker hand"""
    
    def __init__(self, cards: List[Card]):
        """
        Initialize poker hand
        
        Args:
            cards: List of Card objects (typically 5 cards for evaluation)
        """
        self.cards = sorted(cards, key=lambda c: c.rank_value, reverse=True)
        self.hand_rank = None
        self.rank_details = None
        self.hand_strength = 0
        
        if len(self.cards) >= 5:
            self._evaluate_hand()
    
    def _evaluate_hand(self) -> None:
        """Evaluate the poker hand and determine its ranking"""
        # Get the best 5 cards for evaluation
        best_5_cards = self.cards[:5]
        
        ranks = [card.rank_value for card in best_5_cards]
        suits = [card.suit for card in best_5_cards]
        
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        is_flush = len(suit_counts) == 1
        is_straight = self._is_straight(ranks)
        
        # Count pairs, trips, etc.
        count_values = sorted(rank_counts.values(), reverse=True)
        unique_ranks = len(rank_counts)
        
        # Determine hand rank
        if is_straight and is_flush:
            if min(ranks) == 10:  # A, K, Q, J, 10
                self.hand_rank = HandRank.ROYAL_FLUSH
                self.rank_details = {"high_card": max(ranks)}
            else:
                self.hand_rank = HandRank.STRAIGHT_FLUSH
                self.rank_details = {"high_card": max(ranks)}
        elif count_values[0] == 4:
            self.hand_rank = HandRank.FOUR_OF_A_KIND
            four_kind = [rank for rank, count in rank_counts.items() if count == 4][0]
            kicker = [rank for rank, count in rank_counts.items() if count == 1][0]
            self.rank_details = {"four_kind": four_kind, "kicker": kicker}
        elif count_values == [3, 2]:
            self.hand_rank = HandRank.FULL_HOUSE
            trips = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair = [rank for rank, count in rank_counts.items() if count == 2][0]
            self.rank_details = {"trips": trips, "pair": pair}
        elif is_flush:
            self.hand_rank = HandRank.FLUSH
            self.rank_details = {"high_cards": sorted(ranks, reverse=True)}
        elif is_straight:
            self.hand_rank = HandRank.STRAIGHT
            self.rank_details = {"high_card": max(ranks)}
        elif count_values[0] == 3:
            self.hand_rank = HandRank.THREE_OF_A_KIND
            trips = [rank for rank, count in rank_counts.items() if count == 3][0]
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            self.rank_details = {"trips": trips, "kickers": kickers}
        elif count_values == [2, 2, 1]:
            self.hand_rank = HandRank.TWO_PAIR
            pairs = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)
            kicker = [rank for rank, count in rank_counts.items() if count == 1][0]
            self.rank_details = {"high_pair": pairs[0], "low_pair": pairs[1], "kicker": kicker}
        elif count_values[0] == 2:
            self.hand_rank = HandRank.ONE_PAIR
            pair = [rank for rank, count in rank_counts.items() if count == 2][0]
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            self.rank_details = {"pair": pair, "kickers": kickers}
        else:
            self.hand_rank = HandRank.HIGH_CARD
            self.rank_details = {"high_cards": sorted(ranks, reverse=True)}
        
        # Calculate hand strength for comparison
        self.hand_strength = self._calculate_hand_strength()
    
    def _is_straight(self, ranks: List[int]) -> bool:
        """Check if ranks form a straight"""
        sorted_ranks = sorted(set(ranks))
        
        if len(sorted_ranks) != 5:
            return False
        
        # Check for normal straight
        if sorted_ranks[-1] - sorted_ranks[0] == 4:
            return True
        
        # Check for A-2-3-4-5 straight (wheel)
        if sorted_ranks == [2, 3, 4, 5, 14]:
            return True
        
        return False
    
    def _calculate_hand_strength(self) -> int:
        """Calculate numeric hand strength for comparison"""
        # Base strength from hand rank
        strength = self.hand_rank.rank_value * 1000000
        
        # Add details for tie-breaking
        if self.hand_rank == HandRank.ROYAL_FLUSH:
            strength += 0  # All royal flushes are equal
        elif self.hand_rank == HandRank.STRAIGHT_FLUSH:
            strength += self.rank_details["high_card"] * 1000
        elif self.hand_rank == HandRank.FOUR_OF_A_KIND:
            strength += self.rank_details["four_kind"] * 1000 + self.rank_details["kicker"]
        elif self.hand_rank == HandRank.FULL_HOUSE:
            strength += self.rank_details["trips"] * 1000 + self.rank_details["pair"]
        elif self.hand_rank == HandRank.FLUSH:
            for i, card_rank in enumerate(self.rank_details["high_cards"]):
                strength += card_rank * (100 ** (4 - i))
        elif self.hand_rank == HandRank.STRAIGHT:
            strength += self.rank_details["high_card"] * 1000
        elif self.hand_rank == HandRank.THREE_OF_A_KIND:
            strength += self.rank_details["trips"] * 10000
            for i, kicker in enumerate(self.rank_details["kickers"]):
                strength += kicker * (100 ** (1 - i))
        elif self.hand_rank == HandRank.TWO_PAIR:
            strength += (self.rank_details["high_pair"] * 10000 + 
                        self.rank_details["low_pair"] * 100 + 
                        self.rank_details["kicker"])
        elif self.hand_rank == HandRank.ONE_PAIR:
            strength += self.rank_details["pair"] * 10000
            for i, kicker in enumerate(self.rank_details["kickers"]):
                strength += kicker * (100 ** (2 - i))
        else:  # High card
            for i, card_rank in enumerate(self.rank_details["high_cards"]):
                strength += card_rank * (100 ** (4 - i))
        
        return strength
    
    def __str__(self) -> str:
        if self.hand_rank:
            return f"{self.hand_rank.display_name} ({', '.join(str(c) for c in self.cards[:5])})"
        return f"Hand ({', '.join(str(c) for c in self.cards)})"
    
    def __lt__(self, other) -> bool:
        if isinstance(other, PokerHand):
            return self.hand_strength < other.hand_strength
        return NotImplemented


class PokerHandEvaluator:
    """Main poker hand evaluation system"""
    
    def __init__(self):
        """Initialize the hand evaluator"""
        pass
    
    def parse_cards(self, card_names: List[str]) -> List[Card]:
        """
        Parse card names into Card objects
        
        Args:
            card_names: List of card names (e.g., ['As', 'Kh', 'Qd'])
            
        Returns:
            List of Card objects
        """
        cards = []
        for card_name in card_names:
            try:
                card = Card(card_name)
                cards.append(card)
            except ValueError as e:
                logger.warning(f"Invalid card name '{card_name}': {e}")
                continue
        
        return cards
    
    def evaluate_best_hand(self, card_names: List[str]) -> Optional[PokerHand]:
        """
        Evaluate the best 5-card poker hand from available cards
        
        Args:
            card_names: List of card names
            
        Returns:
            PokerHand object with the best hand, or None if insufficient cards
        """
        cards = self.parse_cards(card_names)
        
        if len(cards) < 5:
            logger.warning(f"Not enough cards for evaluation: {len(cards)} < 5")
            return None
        
        # If exactly 5 cards, evaluate directly
        if len(cards) == 5:
            return PokerHand(cards)
        
        # If more than 5 cards, find the best combination
        from itertools import combinations
        
        best_hand = None
        best_strength = -1
        
        # Try all possible 5-card combinations
        for card_combo in combinations(cards, 5):
            hand = PokerHand(list(card_combo))
            if hand.hand_strength > best_strength:
                best_hand = hand
                best_strength = hand.hand_strength
        
        return best_hand
    
    def evaluate_community_and_hole_cards(self, community_cards: List[str], 
                                        hole_cards: List[str]) -> Dict:
        """
        Evaluate poker hand with community and hole cards (Texas Hold'em style)
        
        Args:
            community_cards: List of community card names
            hole_cards: List of hole card names (typically 2)
            
        Returns:
            Dictionary with evaluation results
        """
        all_cards = community_cards + hole_cards
        best_hand = self.evaluate_best_hand(all_cards)
        
        if not best_hand:
            return {
                'valid': False,
                'error': 'Not enough cards for evaluation',
                'total_cards': len(all_cards)
            }
        
        return {
            'valid': True,
            'hand_rank': best_hand.hand_rank.display_name,
            'hand_rank_value': best_hand.hand_rank.rank_value,
            'hand_strength': best_hand.hand_strength,
            'best_5_cards': [str(card) for card in best_hand.cards[:5]],
            'rank_details': best_hand.rank_details,
            'community_cards': community_cards,
            'hole_cards': hole_cards,
            'total_cards': len(all_cards)
        }
    
    def compare_hands(self, hand1_cards: List[str], hand2_cards: List[str]) -> Dict:
        """
        Compare two poker hands
        
        Args:
            hand1_cards: Cards for first hand
            hand2_cards: Cards for second hand
            
        Returns:
            Comparison result dictionary
        """
        hand1 = self.evaluate_best_hand(hand1_cards)
        hand2 = self.evaluate_best_hand(hand2_cards)
        
        if not hand1 or not hand2:
            return {
                'valid': False,
                'error': 'One or both hands have insufficient cards'
            }
        
        if hand1.hand_strength > hand2.hand_strength:
            winner = 'hand1'
        elif hand2.hand_strength > hand1.hand_strength:
            winner = 'hand2'
        else:
            winner = 'tie'
        
        return {
            'valid': True,
            'winner': winner,
            'hand1': {
                'rank': hand1.hand_rank.display_name,
                'strength': hand1.hand_strength,
                'cards': [str(card) for card in hand1.cards[:5]]
            },
            'hand2': {
                'rank': hand2.hand_rank.display_name,
                'strength': hand2.hand_strength,
                'cards': [str(card) for card in hand2.cards[:5]]
            }
        }


def create_hand_evaluator() -> PokerHandEvaluator:
    """
    Factory function to create a PokerHandEvaluator instance
    
    Returns:
        Configured PokerHandEvaluator instance
    """
    return PokerHandEvaluator()