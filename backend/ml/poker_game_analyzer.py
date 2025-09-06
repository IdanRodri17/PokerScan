"""
Complete Poker Game Analyzer
Identifies players, community cards, and determines the winner
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from itertools import combinations

logger = logging.getLogger(__name__)

# Poker hand rankings
class HandRank(Enum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10

@dataclass
class Card:
    """Represents a playing card"""
    rank: str
    suit: str
    value: int  # Numeric value for comparison
    
    def __str__(self):
        return f"{self.rank}{self.suit}"

@dataclass
class PokerHand:
    """Represents a poker hand evaluation"""
    rank: HandRank
    rank_name: str
    high_cards: List[int]  # For tiebreakers
    cards: List[Card]
    description: str
    
@dataclass
class Player:
    """Represents a player with their cards"""
    id: int
    name: str
    hole_cards: List[Card]
    best_hand: Optional[PokerHand] = None
    position: str = ""  # "top", "bottom", etc.

@dataclass
class GameState:
    """Complete game state with all cards identified"""
    community_cards: List[Card]
    players: List[Player]
    winner: Optional[Player] = None
    tie: bool = False
    tied_players: List[Player] = None

class PokerGameAnalyzer:
    """
    Analyzes poker game state and determines winner
    """
    
    def __init__(self):
        # Card value mappings
        self.rank_values = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11, '10': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
            '4': 4, '3': 3, '2': 2
        }
        
        # For ace-low straights
        self.rank_values_ace_low = {
            'A': 1, 'K': 13, 'Q': 12, 'J': 11, '10': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
            '4': 4, '3': 3, '2': 2
        }
    
    def analyze_game(self, detections: List[Dict], image_shape: Tuple[int, int]) -> GameState:
        """
        Main method to analyze the complete game
        
        Args:
            detections: List of card detections with positions
            image_shape: (height, width) of image
            
        Returns:
            Complete GameState with winner determined
        """
        # Step 1: Identify which cards belong to whom
        grouped_cards = self._group_cards_by_position(detections, image_shape)
        
        # Step 2: Convert to Card objects
        game_state = self._create_game_state(grouped_cards)
        
        # Step 3: Evaluate hands for each player
        for player in game_state.players:
            player.best_hand = self._evaluate_best_hand(
                player.hole_cards, 
                game_state.community_cards
            )
        
        # Step 4: Determine winner
        game_state = self._determine_winner(game_state)
        
        # Step 5: Log results
        self._log_game_analysis(game_state)
        
        return game_state
    
    def _group_cards_by_position(self, detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """
        Group cards by their position on the table
        """
        height, width = image_shape
        
        # Sort detections by Y position
        sorted_by_y = sorted(detections, key=lambda x: x['center'][1])
        
        # Define regions based on typical poker layout
        groups = {
            'player1': [],  # Top of image
            'community': [],  # Middle of image
            'player2': []  # Bottom of image
        }
        
        for det in sorted_by_y:
            y = det['center'][1]
            y_ratio = y / height
            
            if y_ratio < 0.25:
                # Top quarter - Player 1
                groups['player1'].append(det)
            elif y_ratio > 0.85:
                # Very bottom - treat as community (likely misplaced)
                groups['community'].append(det)
            elif y_ratio > 0.65:
                # Bottom region - Player 2
                groups['player2'].append(det)
            else:
                # Middle region - Community cards
                groups['community'].append(det)
        
        # Sort each group by X position (left to right)
        for key in groups:
            groups[key] = sorted(groups[key], key=lambda x: x['center'][0])
        
        # Debug: Log card positions and groupings
        logger.info("🔍 Card grouping analysis:")
        for group_name, cards in groups.items():
            card_names = [c['card_name'] for c in cards]
            logger.info(f"  {group_name}: {card_names}")
            for card in cards:
                logger.info(f"    {card['card_name']} at y={card['center'][1]:.0f} (ratio: {card['center'][1]/image_shape[0]:.2f})")
        
        # Validate and fix card counts
        groups = self._fix_card_groups(groups)
        
        # Final validation
        self._validate_card_groups(groups)
        
        return groups
    
    def _fix_card_groups(self, groups: Dict) -> Dict:
        """Fix card groups to ensure proper poker layout"""
        logger.info("🔧 Fixing card group assignments...")
        
        # If any player has more than 2 cards, move extras to community
        for player in ['player1', 'player2']:
            if len(groups[player]) > 2:
                # Move excess cards to community (keep the 2 with best positions)
                excess_cards = groups[player][2:]  # Cards beyond first 2
                groups[player] = groups[player][:2]  # Keep only first 2
                groups['community'].extend(excess_cards)
                
                card_names = [c['card_name'] for c in excess_cards]
                logger.info(f"  Moved {card_names} from {player} to community")
        
        # If community has fewer than 5 cards and we have 9 total, try to balance
        total_cards = sum(len(cards) for cards in groups.values())
        if len(groups['community']) < 5 and total_cards == 9:
            # If a player has exactly 2 cards and community needs more, this is correct
            pass
        
        logger.info("🔧 After fixing:")
        for group_name, cards in groups.items():
            card_names = [c['card_name'] for c in cards]
            logger.info(f"  {group_name}: {card_names}")
            
        return groups
    
    def _validate_card_groups(self, groups: Dict):
        """Validate that card groups make sense for poker"""
        # Each player should have exactly 2 cards
        for player in ['player1', 'player2']:
            if len(groups[player]) != 2 and len(groups[player]) != 0:
                logger.warning(f"{player} has {len(groups[player])} cards, expected 2 or 0")
        
        # Community should have 3-5 cards
        if not (3 <= len(groups['community']) <= 5):
            logger.warning(f"Community has {len(groups['community'])} cards, expected 3-5")
    
    def _create_game_state(self, grouped_cards: Dict) -> GameState:
        """Create GameState from grouped cards"""
        # Parse community cards
        community_cards = []
        for det in grouped_cards['community']:
            card = self._parse_card(det['card_name'])
            if card:
                community_cards.append(card)
        
        # Create players
        players = []
        
        # Player 1 (top)
        if len(grouped_cards['player1']) >= 2:
            player1_cards = []
            for det in grouped_cards['player1'][:2]:
                card = self._parse_card(det['card_name'])
                if card:
                    player1_cards.append(card)
            
            players.append(Player(
                id=1,
                name="Player 1 (Top)",
                hole_cards=player1_cards,
                position="top"
            ))
        
        # Player 2 (bottom)
        if len(grouped_cards['player2']) >= 2:
            player2_cards = []
            for det in grouped_cards['player2'][:2]:
                card = self._parse_card(det['card_name'])
                if card:
                    player2_cards.append(card)
            
            players.append(Player(
                id=2,
                name="Player 2 (Bottom)",
                hole_cards=player2_cards,
                position="bottom"
            ))
        
        return GameState(
            community_cards=community_cards,
            players=players
        )
    
    def _parse_card(self, card_str: str) -> Optional[Card]:
        """Parse card string into Card object"""
        card_str = card_str.upper()
        
        # Handle different formats
        if len(card_str) >= 2:
            if card_str[:-1] in self.rank_values:
                rank = card_str[:-1]
                suit = card_str[-1]
            else:
                rank = card_str[0]
                suit = card_str[1]
            
            if rank in self.rank_values:
                return Card(
                    rank=rank,
                    suit=suit,
                    value=self.rank_values[rank]
                )
        
        logger.warning(f"Could not parse card: {card_str}")
        return None
    
    def _evaluate_best_hand(self, hole_cards: List[Card], community_cards: List[Card]) -> PokerHand:
        """
        Evaluate the best possible poker hand from hole cards and community cards
        """
        # Combine all available cards
        all_cards = hole_cards + community_cards
        
        if len(all_cards) < 5:
            return PokerHand(
                rank=HandRank.HIGH_CARD,
                rank_name="High Card",
                high_cards=[max(card.value for card in all_cards)],
                cards=all_cards,
                description="Not enough cards for a full hand"
            )
        
        # Generate all possible 5-card combinations
        best_hand = None
        
        for combo in combinations(all_cards, 5):
            hand = self._evaluate_five_cards(list(combo))
            if best_hand is None or self._compare_hands(hand, best_hand) > 0:
                best_hand = hand
        
        return best_hand
    
    def _evaluate_five_cards(self, cards: List[Card]) -> PokerHand:
        """Evaluate exactly 5 cards"""
        # Sort by value
        cards.sort(key=lambda x: x.value, reverse=True)
        
        # Check for flush
        is_flush = len(set(card.suit for card in cards)) == 1
        
        # Check for straight
        is_straight = self._is_straight(cards)
        is_straight_ace_low = self._is_straight_ace_low(cards)
        
        # Count ranks
        rank_counts = {}
        for card in cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        
        # Sort by count then value
        sorted_ranks = sorted(rank_counts.items(), 
                            key=lambda x: (x[1], self.rank_values[x[0]]), 
                            reverse=True)
        
        # Determine hand rank
        counts = [count for rank, count in sorted_ranks]
        
        # Royal Flush
        if is_flush and is_straight and cards[0].value == 14:  # Ace high
            return PokerHand(
                rank=HandRank.ROYAL_FLUSH,
                rank_name="Royal Flush",
                high_cards=[14, 13, 12, 11, 10],
                cards=cards,
                description=f"Royal Flush in {cards[0].suit}"
            )
        
        # Straight Flush
        if is_flush and (is_straight or is_straight_ace_low):
            high_card = cards[0].value if is_straight else 5  # Ace-low straight
            return PokerHand(
                rank=HandRank.STRAIGHT_FLUSH,
                rank_name="Straight Flush",
                high_cards=[high_card],
                cards=cards,
                description=f"Straight Flush, {cards[0].rank} high"
            )
        
        # Four of a Kind
        if counts == [4, 1]:
            quads_rank = sorted_ranks[0][0]
            kicker_rank = sorted_ranks[1][0]
            return PokerHand(
                rank=HandRank.FOUR_OF_A_KIND,
                rank_name="Four of a Kind",
                high_cards=[self.rank_values[quads_rank], self.rank_values[kicker_rank]],
                cards=cards,
                description=f"Four {quads_rank}s"
            )
        
        # Full House
        if counts == [3, 2]:
            trips_rank = sorted_ranks[0][0]
            pair_rank = sorted_ranks[1][0]
            return PokerHand(
                rank=HandRank.FULL_HOUSE,
                rank_name="Full House",
                high_cards=[self.rank_values[trips_rank], self.rank_values[pair_rank]],
                cards=cards,
                description=f"{trips_rank}s full of {pair_rank}s"
            )
        
        # Flush
        if is_flush:
            return PokerHand(
                rank=HandRank.FLUSH,
                rank_name="Flush",
                high_cards=[card.value for card in cards],
                cards=cards,
                description=f"Flush, {cards[0].rank} high"
            )
        
        # Straight
        if is_straight or is_straight_ace_low:
            high_card = cards[0].value if is_straight else 5
            return PokerHand(
                rank=HandRank.STRAIGHT,
                rank_name="Straight",
                high_cards=[high_card],
                cards=cards,
                description=f"Straight, {'5' if is_straight_ace_low else cards[0].rank} high"
            )
        
        # Three of a Kind
        if counts == [3, 1, 1]:
            trips_rank = sorted_ranks[0][0]
            kickers = [self.rank_values[sorted_ranks[i][0]] for i in range(1, 3)]
            return PokerHand(
                rank=HandRank.THREE_OF_A_KIND,
                rank_name="Three of a Kind",
                high_cards=[self.rank_values[trips_rank]] + kickers,
                cards=cards,
                description=f"Three {trips_rank}s"
            )
        
        # Two Pair
        if counts == [2, 2, 1]:
            pair1_rank = sorted_ranks[0][0]
            pair2_rank = sorted_ranks[1][0]
            kicker_rank = sorted_ranks[2][0]
            return PokerHand(
                rank=HandRank.TWO_PAIR,
                rank_name="Two Pair",
                high_cards=[self.rank_values[pair1_rank], 
                          self.rank_values[pair2_rank], 
                          self.rank_values[kicker_rank]],
                cards=cards,
                description=f"{pair1_rank}s and {pair2_rank}s"
            )
        
        # Pair
        if counts == [2, 1, 1, 1]:
            pair_rank = sorted_ranks[0][0]
            kickers = [self.rank_values[sorted_ranks[i][0]] for i in range(1, 4)]
            return PokerHand(
                rank=HandRank.PAIR,
                rank_name="Pair",
                high_cards=[self.rank_values[pair_rank]] + kickers,
                cards=cards,
                description=f"Pair of {pair_rank}s"
            )
        
        # High Card
        return PokerHand(
            rank=HandRank.HIGH_CARD,
            rank_name="High Card",
            high_cards=[card.value for card in cards],
            cards=cards,
            description=f"{cards[0].rank} high"
        )
    
    def _is_straight(self, cards: List[Card]) -> bool:
        """Check if 5 cards form a straight"""
        values = sorted([card.value for card in cards], reverse=True)
        for i in range(len(values) - 1):
            if values[i] - values[i + 1] != 1:
                return False
        return True
    
    def _is_straight_ace_low(self, cards: List[Card]) -> bool:
        """Check for A-2-3-4-5 straight"""
        ranks = [card.rank for card in cards]
        return set(ranks) == {'A', '2', '3', '4', '5'}
    
    def _compare_hands(self, hand1: PokerHand, hand2: PokerHand) -> int:
        """
        Compare two poker hands
        Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        # Compare hand ranks
        if hand1.rank.value > hand2.rank.value:
            return 1
        elif hand1.rank.value < hand2.rank.value:
            return -1
        
        # Same rank, compare high cards
        for h1, h2 in zip(hand1.high_cards, hand2.high_cards):
            if h1 > h2:
                return 1
            elif h1 < h2:
                return -1
        
        return 0  # Tie
    
    def _determine_winner(self, game_state: GameState) -> GameState:
        """Determine the winner of the game"""
        if len(game_state.players) < 2:
            logger.warning("Not enough players to determine winner")
            return game_state
        
        # Compare all players
        best_players = []
        best_hand = None
        
        for player in game_state.players:
            if player.best_hand:
                if best_hand is None:
                    best_players = [player]
                    best_hand = player.best_hand
                else:
                    comparison = self._compare_hands(player.best_hand, best_hand)
                    if comparison > 0:
                        best_players = [player]
                        best_hand = player.best_hand
                    elif comparison == 0:
                        best_players.append(player)
        
        # Set winner(s)
        if len(best_players) == 1:
            game_state.winner = best_players[0]
            game_state.tie = False
        elif len(best_players) > 1:
            game_state.tie = True
            game_state.tied_players = best_players
        
        return game_state
    
    def _log_game_analysis(self, game_state: GameState):
        """Log the complete game analysis"""
        logger.info("=" * 60)
        logger.info("🃏 POKER GAME ANALYSIS")
        logger.info("=" * 60)
        
        # Community cards
        community_str = ", ".join(str(card) for card in game_state.community_cards)
        logger.info(f"Community Cards: {community_str}")
        logger.info("-" * 40)
        
        # Each player
        for player in game_state.players:
            hole_cards_str = ", ".join(str(card) for card in player.hole_cards)
            logger.info(f"{player.name}:")
            logger.info(f"  Hole Cards: {hole_cards_str}")
            if player.best_hand:
                logger.info(f"  Best Hand: {player.best_hand.rank_name}")
                logger.info(f"  Description: {player.best_hand.description}")
        
        logger.info("-" * 40)
        
        # Winner
        if game_state.winner:
            logger.info(f"🏆 WINNER: {game_state.winner.name}")
            logger.info(f"   With: {game_state.winner.best_hand.description}")
        elif game_state.tie:
            tied_names = ", ".join(p.name for p in game_state.tied_players)
            logger.info(f"🤝 TIE between: {tied_names}")
            logger.info(f"   With: {game_state.tied_players[0].best_hand.description}")
        
        logger.info("=" * 60)

# Main function to use
def analyze_poker_game(detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
    """
    Main function to analyze poker game and determine winner
    
    Args:
        detections: List of card detections
        image_shape: (height, width) of image
        
    Returns:
        Dictionary with game analysis results
    """
    analyzer = PokerGameAnalyzer()
    game_state = analyzer.analyze_game(detections, image_shape)
    
    # Create result dictionary for easy use
    result = {
        'community_cards': [str(card) for card in game_state.community_cards],
        'players': [],
        'winner': None,
        'tie': game_state.tie
    }
    
    for player in game_state.players:
        player_info = {
            'id': player.id,
            'name': player.name,
            'position': player.position,
            'hole_cards': [str(card) for card in player.hole_cards],
            'best_hand': player.best_hand.rank_name if player.best_hand else None,
            'hand_description': player.best_hand.description if player.best_hand else None
        }
        result['players'].append(player_info)
    
    if game_state.winner:
        result['winner'] = {
            'id': game_state.winner.id,
            'name': game_state.winner.name,
            'winning_hand': game_state.winner.best_hand.description
        }
    elif game_state.tied_players:
        result['tied_players'] = [
            {'id': p.id, 'name': p.name} for p in game_state.tied_players
        ]
    
    return result