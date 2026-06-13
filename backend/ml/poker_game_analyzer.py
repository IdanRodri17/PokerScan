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
        Improved grouping logic based on strict spatial zones
        """
        height, width = image_shape
        
        # Sort detections by Y position
        sorted_by_y = sorted(detections, key=lambda x: x['center'][1])
        
        # Initialize groups
        groups = {
            'player1': [],
            'community': [],
            'player2': []
        }
        
        # IMPROVED ZONE DEFINITIONS
        # Based on your test results, we need stricter boundaries
        TOP_ZONE_LIMIT = 0.20      # Top 20% for Player 1
        MIDDLE_ZONE_START = 0.25   # Community starts at 25%
        MIDDLE_ZONE_END = 0.70     # Community ends at 70%
        BOTTOM_ZONE_START = 0.75   # Bottom 25% for Player 2
        
        # First pass: Rough grouping
        top_zone = []
        middle_zone = []
        bottom_zone = []
        
        for det in sorted_by_y:
            y = det['center'][1]
            y_ratio = y / height
            
            if y_ratio <= TOP_ZONE_LIMIT:
                top_zone.append(det)
            elif y_ratio >= BOTTOM_ZONE_START:
                bottom_zone.append(det)
            elif MIDDLE_ZONE_START <= y_ratio <= MIDDLE_ZONE_END:
                middle_zone.append(det)
            else:
                # Transition zones - assign based on nearest boundary
                if y_ratio < MIDDLE_ZONE_START:
                    # Between top and middle - check which is closer
                    if y_ratio - TOP_ZONE_LIMIT < MIDDLE_ZONE_START - y_ratio:
                        top_zone.append(det)
                    else:
                        middle_zone.append(det)
                else:
                    # Between middle and bottom
                    if y_ratio - MIDDLE_ZONE_END < BOTTOM_ZONE_START - y_ratio:
                        middle_zone.append(det)
                    else:
                        bottom_zone.append(det)
        
        # Debug logging
        logger.info("🔍 Initial zone assignment:")
        logger.info(f"  Top zone ({TOP_ZONE_LIMIT*100:.0f}%): {[d['card_name'] for d in top_zone]}")
        logger.info(f"  Middle zone ({MIDDLE_ZONE_START*100:.0f}-{MIDDLE_ZONE_END*100:.0f}%): {[d['card_name'] for d in middle_zone]}")
        logger.info(f"  Bottom zone ({BOTTOM_ZONE_START*100:.0f}%+): {[d['card_name'] for d in bottom_zone]}")

        # CRITICAL FIX: Handle duplicate ranks within PLAYER zones only
        # Players can have max 2 cards of different ranks
        # If a player zone has 2 cards with same rank, keep highest confidence and remove the other

        def remove_duplicate_ranks_in_zone(zone_cards, zone_name):
            """Remove duplicate ranks within a single zone, keeping highest confidence"""
            if not zone_cards:
                return zone_cards

            rank_map = {}  # rank -> list of cards with that rank
            for card in zone_cards:
                rank = card['card_name'][0]  # First char is rank (A, 2, K, etc.)
                if rank not in rank_map:
                    rank_map[rank] = []
                rank_map[rank].append(card)

            # Build cleaned list - for each rank, keep only highest confidence
            cleaned = []
            for rank, cards_with_rank in rank_map.items():
                if len(cards_with_rank) > 1:
                    # Sort by confidence and keep best
                    best_card = max(cards_with_rank, key=lambda x: x['confidence'])
                    cleaned.append(best_card)
                    logger.info(f"  🔧 {zone_name}: Found {len(cards_with_rank)} cards with rank '{rank}'")
                    logger.info(f"     Keeping: {best_card['card_name']} ({best_card['confidence']:.3f})")
                    for card in cards_with_rank:
                        if card != best_card:
                            logger.info(f"     Removing: {card['card_name']} ({card['confidence']:.3f}) - duplicate rank")
                else:
                    cleaned.append(cards_with_rank[0])

            return cleaned

        # Apply duplicate rank removal to player zones (not community - community can have duplicates legitimately)
        top_zone = remove_duplicate_ranks_in_zone(top_zone, "Top zone")
        bottom_zone = remove_duplicate_ranks_in_zone(bottom_zone, "Bottom zone")

        logger.info("🔍 After duplicate rank removal:")
        logger.info(f"  Top zone: {[d['card_name'] for d in top_zone]}")
        logger.info(f"  Middle zone: {[d['card_name'] for d in middle_zone]}")
        logger.info(f"  Bottom zone: {[d['card_name'] for d in bottom_zone]}")
        
        # Second pass: Apply poker rules
        # Player 1 - max 2 cards, highest confidence if more
        if len(top_zone) > 2:
            # Sort by confidence (descending) to keep the best detections
            top_zone_sorted = sorted(top_zone, key=lambda x: x['confidence'], reverse=True)
            kept_cards = top_zone_sorted[:2]
            # Sort kept cards by X position for display
            groups['player1'] = sorted(kept_cards, key=lambda x: x['center'][0])
            # Extra cards go to community
            for card in top_zone_sorted[2:]:
                middle_zone.append(card)
                logger.info(f"  Moved {card['card_name']} from player1 to community (excess, lower confidence)")
        else:
            groups['player1'] = sorted(top_zone, key=lambda x: x['center'][0])
        
        # Player 2 - max 2 cards, highest confidence if more
        if len(bottom_zone) > 2:
            # Sort by confidence (descending) to keep the best detections
            bottom_zone_sorted = sorted(bottom_zone, key=lambda x: x['confidence'], reverse=True)
            kept_cards = bottom_zone_sorted[:2]
            # Sort kept cards by X position for display
            groups['player2'] = sorted(kept_cards, key=lambda x: x['center'][0])
            # Extra cards go to community
            for card in bottom_zone_sorted[2:]:
                middle_zone.append(card)
                logger.info(f"  Moved {card['card_name']} from player2 to community (excess, lower confidence)")
        else:
            groups['player2'] = sorted(bottom_zone, key=lambda x: x['center'][0])
        
        # Community - max 5 cards, highest confidence if more
        if len(middle_zone) > 5:
            # FIXED: Keep 5 highest confidence cards instead of "most centered"
            # The "centered" logic was keeping wrong cards (Qd, 3s) instead of correct ones (3c, 6h)
            middle_zone_sorted = sorted(middle_zone, key=lambda x: x['confidence'], reverse=True)
            kept_community = middle_zone_sorted[:5]
            # Sort kept cards by X position for display
            groups['community'] = sorted(kept_community, key=lambda x: x['center'][0])
            logger.info(f"  Limited community to 5 highest confidence cards")
            logger.info(f"    Kept: {[c['card_name'] for c in kept_community]}")
            logger.info(f"    Removed: {[c['card_name'] for c in middle_zone_sorted[5:]]}")
        else:
            # Sort community cards by X position for display
            groups['community'] = sorted(middle_zone, key=lambda x: x['center'][0])
        
        # Final logging
        logger.info("🔧 Final card groups:")
        total = 0
        for group_name, cards in groups.items():
            card_names = [c['card_name'] for c in cards]
            logger.info(f"  {group_name}: {card_names} ({len(cards)} cards)")
            total += len(cards)
        logger.info(f"  Total: {total} cards")
        
        # Validate
        self._validate_card_groups(groups)
        
        return groups

    def _validate_card_groups(self, groups: Dict):
        """Validate that card groups make sense for poker"""
        warnings = []
        
        # Each player should have exactly 2 cards (or 0 if not playing)
        for player in ['player1', 'player2']:
            count = len(groups[player])
            if count != 0 and count != 2:
                warnings.append(f"{player} has {count} cards, expected 0 or 2")
        
        # Community should have 0 (preflop), 3 (flop), 4 (turn), or 5 (river) cards
        comm_count = len(groups['community'])
        if comm_count not in [0, 3, 4, 5]:
            warnings.append(f"Community has {comm_count} cards, expected 0, 3, 4, or 5")
        
        # Total should be 7 (one player) or 9 (two players)
        total = sum(len(cards) for cards in groups.values())
        if total not in [7, 9]:
            warnings.append(f"Total card count is {total}, expected 7 or 9")
        
        # Log warnings
        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️ {warning}")
    
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
            
            if len(player1_cards) == 2:  # Only add if we have exactly 2 cards
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
            
            if len(player2_cards) == 2:  # Only add if we have exactly 2 cards
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
    
    def _parse_card(self, card_str: str) -> Optional[Card]:
        """Parse card string into Card object - handles various formats"""
        if not card_str:
            return None
            
        card_str = card_str.strip().upper()
        
        # Handle different formats: "KS", "10D", "TC", etc.
        if len(card_str) >= 2:
            # Check for 10/T
            if card_str.startswith('10'):
                rank = '10'
                suit = card_str[2] if len(card_str) > 2 else card_str[-1]
            elif card_str[0] == 'T':
                rank = '10'  # Convert T to 10 internally
                suit = card_str[1]
            elif card_str[:-1] in self.rank_values:
                rank = card_str[:-1]
                suit = card_str[-1]
            else:
                rank = card_str[0]
                suit = card_str[1] if len(card_str) > 1 else 'S'  # Default to spades if missing
            
            # Normalize rank
            if rank == 'T':
                rank = '10'
            
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