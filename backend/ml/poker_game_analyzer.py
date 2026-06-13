"""
Complete Poker Game Analyzer
Identifies players, community cards, and determines the winner
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from ml.hand_evaluator import create_hand_evaluator, PokerHand as EvaluatedHand

logger = logging.getLogger(__name__)

# Adaptive row-clustering parameters
MAX_ROWS = 3
# A vertical gap must exceed this fraction of the image height to separate two
# rows. This is a scale (minimum row separation), NOT a fixed band position:
# rows are detected wherever the cards actually are.
MIN_ROW_GAP_FRACTION = 0.08


@dataclass
class Card:
    """Represents a playing card"""
    rank: str
    suit: str
    value: int  # Numeric value for comparison
    
    def __str__(self):
        return f"{self.rank}{self.suit}"

@dataclass
class Player:
    """Represents a player with their cards"""
    id: int
    name: str
    hole_cards: List[Card]
    best_hand: Optional[EvaluatedHand] = None  # ml.hand_evaluator.PokerHand
    position: str = ""  # "top", "bottom", etc.

@dataclass
class GameState:
    """Complete game state with all cards identified"""
    community_cards: List[Card]
    players: List[Player]
    winner: Optional[Player] = None
    tie: bool = False
    tied_players: List[Player] = None
    layout_confidence: float = 1.0  # 0-1: how cleanly the card rows separated

class PokerGameAnalyzer:
    """
    Analyzes poker game state and determines winner
    """
    
    def __init__(self):
        # Card value mappings (used by _parse_card)
        self.rank_values = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11, '10': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
            '4': 4, '3': 3, '2': 2
        }

        # Shared hand evaluator — the single source of truth for hand strength.
        # Both this analyzer (/upload) and /evaluate-winner go through it, so the
        # two paths can never disagree on hand ranking or the winner.
        self.hand_evaluator = create_hand_evaluator()
    
    def analyze_game(self, detections: List[Dict], image_shape: Tuple[int, int]) -> GameState:
        """
        Main method to analyze the complete game
        
        Args:
            detections: List of card detections with positions
            image_shape: (height, width) of image
            
        Returns:
            Complete GameState with winner determined
        """
        # Step 1: Identify which cards belong to whom (adaptive row detection)
        grouped_cards, layout_confidence = self._group_cards_by_position(detections, image_shape)

        # Step 2: Convert to Card objects
        game_state = self._create_game_state(grouped_cards)
        game_state.layout_confidence = layout_confidence

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
    
    def _group_cards_by_position(self, detections: List[Dict],
                                 image_shape: Tuple[int, int]) -> Tuple[Dict, float]:
        """
        Group detected cards into player1 / community / player2 using adaptive
        row detection instead of fixed vertical bands.

        Cards are clustered into up to MAX_ROWS horizontal rows by the gaps
        between their sorted y-centers, then identified by size and position: the
        largest row (3-5 cards) nearest the vertical middle is the community, the
        row above is one player and the row below the other. Handles 1, 2 and 3
        rows (e.g. preflop has no community).

        Returns the groups dict (same shape as before) and a layout_confidence in
        [0, 1] describing how cleanly the rows separated. A low value means the
        layout was ambiguous (a card drifting between rows) -- a good signal to
        later ask the user for a clearer photo.
        """
        height, width = image_shape
        groups = {'player1': [], 'community': [], 'player2': []}

        if not detections:
            return groups, 1.0

        # Step 1: cluster cards into rows by the vertical gaps between them
        rows, boundary_gaps, within_gaps = self._cluster_rows(detections, height)

        # Step 2: decide which row is community / player1 / player2
        raw_groups = self._assign_rows_to_groups(rows, height)

        # Step 3: apply poker per-row limits + duplicate-rank removal
        groups = self._filter_groups(raw_groups)

        # Step 4: score how cleanly the rows separated
        layout_confidence = self._compute_layout_confidence(boundary_gaps, within_gaps)

        logger.info("🔧 Adaptive grouping: %d row(s), layout_confidence=%.2f",
                    len(rows), layout_confidence)
        for name, cards in groups.items():
            logger.info("  %s: %s", name, [c['card_name'] for c in cards])

        # Validate (logs warnings on unusual counts)
        self._validate_card_groups(groups)

        return groups, layout_confidence

    def _cluster_rows(self, detections: List[Dict], image_height: int):
        """Cluster detections into up to MAX_ROWS rows by gaps between y-centers.

        Returns (rows, boundary_gaps, within_gaps):
          rows          -- rows ordered top to bottom, each a list of detections
          boundary_gaps -- the vertical gaps chosen as row separators
          within_gaps   -- the remaining (within-row) gaps
        """
        sorted_dets = sorted(detections, key=lambda d: d['center'][1])
        ys = [d['center'][1] for d in sorted_dets]
        n = len(ys)

        if n <= 1:
            return [sorted_dets], [], []

        gaps = [ys[i + 1] - ys[i] for i in range(n - 1)]

        # A gap separates two rows only if it is larger than a minimum fraction of
        # the image height. This avoids splitting a single, slightly tilted row,
        # while still adapting to wherever the rows actually sit.
        min_gap = MIN_ROW_GAP_FRACTION * image_height
        candidates = sorted(
            (i for i in range(len(gaps)) if gaps[i] > min_gap),
            key=lambda i: gaps[i], reverse=True,
        )
        boundary_idx = sorted(candidates[:MAX_ROWS - 1])  # at most 2 boundaries -> 3 rows

        rows = []
        start = 0
        for b in boundary_idx:
            rows.append(sorted_dets[start:b + 1])
            start = b + 1
        rows.append(sorted_dets[start:])

        boundary_gaps = [gaps[i] for i in boundary_idx]
        within_gaps = [gaps[i] for i in range(len(gaps)) if i not in boundary_idx]
        return rows, boundary_gaps, within_gaps

    def _assign_rows_to_groups(self, rows: List[List[Dict]], image_height: int) -> Dict:
        """Identify the community and player rows from clustered rows (top to bottom)."""
        groups = {'player1': [], 'community': [], 'player2': []}
        if not rows:
            return groups

        middle_y = image_height / 2.0

        def row_mean_y(row):
            return sum(d['center'][1] for d in row) / len(row)

        # The community is a row of 3+ cards (flop/turn/river) nearest the middle.
        community_like = [i for i, row in enumerate(rows) if len(row) >= 3]
        if community_like:
            community_idx = min(
                community_like, key=lambda i: abs(row_mean_y(rows[i]) - middle_y)
            )
            groups['community'] = rows[community_idx]
            above = [i for i in range(len(rows)) if i < community_idx]
            below = [i for i in range(len(rows)) if i > community_idx]
            if above:
                groups['player1'] = rows[above[-1]]   # nearest row above community
            if below:
                groups['player2'] = rows[below[0]]    # nearest row below community
        else:
            # No community row (e.g. preflop): the rows are players. Assign by
            # vertical position -- topmost is player1, bottommost is player2.
            groups['player1'] = rows[0]
            if len(rows) >= 2:
                groups['player2'] = rows[-1]

        return groups

    @staticmethod
    def _by_confidence(card: Dict):
        """Sort key: highest confidence first, ties broken deterministically by x
        position then card name so the result never depends on detection order."""
        return (-card['confidence'], card['center'][0], card['card_name'])

    def _filter_groups(self, raw_groups: Dict) -> Dict:
        """Apply poker per-row limits + duplicate-rank removal, x-sorted for display."""
        groups = {'player1': [], 'community': [], 'player2': []}

        # Players: drop duplicate ranks, then keep the 2 highest-confidence cards.
        for player in ('player1', 'player2'):
            cards = self._dedupe_player_ranks(raw_groups[player], player)
            cards = sorted(cards, key=self._by_confidence)[:2]
            groups[player] = sorted(cards, key=lambda c: c['center'][0])

        # Community: keep up to 5 highest-confidence cards (duplicate ranks allowed).
        community = sorted(raw_groups['community'], key=self._by_confidence)[:5]
        groups['community'] = sorted(community, key=lambda c: c['center'][0])

        return groups

    @staticmethod
    def _card_rank(card_name: str) -> str:
        """Rank of a card name with the trailing suit removed ('10c' -> '10', 'As' -> 'A')."""
        return card_name[:-1] if len(card_name) > 1 else card_name

    @staticmethod
    def _dedupe_player_ranks(cards: List[Dict], zone_name: str = "") -> List[Dict]:
        """Within a player row, keep only the highest-confidence card per rank.

        Ties are broken deterministically (see _by_confidence) so the kept card
        never depends on detection order.
        """
        if not cards:
            return list(cards)

        rank_map = {}
        for card in cards:
            rank_map.setdefault(PokerGameAnalyzer._card_rank(card['card_name']), []).append(card)

        cleaned = []
        for rank, same_rank in rank_map.items():
            best = sorted(same_rank, key=PokerGameAnalyzer._by_confidence)[0]
            cleaned.append(best)
            if len(same_rank) > 1:
                logger.info("  🔧 %s: kept %s for rank '%s', dropped %d duplicate(s)",
                            zone_name, best['card_name'], rank, len(same_rank) - 1)
        return cleaned

    @staticmethod
    def _compute_layout_confidence(boundary_gaps: List[float],
                                   within_gaps: List[float]) -> float:
        """How cleanly the rows separated, in [0, 1].

        1.0 when there is a single row, or when the between-row gaps dwarf the
        within-row spread. Lower as a within-row gap approaches the smallest
        between-row gap (e.g. a card drifting between rows in a tilted photo).
        """
        if not boundary_gaps:
            return 1.0
        min_between = min(boundary_gaps)
        if min_between <= 0:
            return 0.0
        max_within = max(within_gaps) if within_gaps else 0.0
        ambiguity = min(max_within / min_between, 1.0)
        return round(1.0 - ambiguity, 3)

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
    
    def _evaluate_best_hand(self, hole_cards: List[Card],
                            community_cards: List[Card]) -> Optional[EvaluatedHand]:
        """
        Evaluate the best possible 5-card hand using the shared PokerHandEvaluator.

        All hand-strength logic lives in ml.hand_evaluator (the single source of
        truth), so this /upload path and the /evaluate-winner path can never
        disagree on hand ranking. Returns None when fewer than 5 cards are
        available (e.g. an incomplete board), which callers already treat as
        "no hand".
        """
        card_names = [str(card) for card in hole_cards] + [str(card) for card in community_cards]
        return self.hand_evaluator.evaluate_best_hand(card_names)
    
    def _determine_winner(self, game_state: GameState) -> GameState:
        """Determine the winner by comparing each player's evaluated hand strength."""
        if len(game_state.players) < 2:
            logger.warning("Not enough players to determine winner")
            return game_state

        # Compare all players by the evaluator's single comparable hand strength
        best_players = []
        best_strength = None

        for player in game_state.players:
            if not player.best_hand:
                continue
            strength = player.best_hand.hand_strength
            if best_strength is None or strength > best_strength:
                best_players = [player]
                best_strength = strength
            elif strength == best_strength:
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
                logger.info(f"  Best Hand: {player.best_hand.hand_rank.display_name}")
        
        logger.info("-" * 40)
        
        # Winner
        if game_state.winner:
            logger.info(f"🏆 WINNER: {game_state.winner.name}")
            logger.info(f"   With: {game_state.winner.best_hand.hand_rank.display_name}")
        elif game_state.tie:
            tied_names = ", ".join(p.name for p in game_state.tied_players)
            logger.info(f"🤝 TIE between: {tied_names}")
            logger.info(f"   With: {game_state.tied_players[0].best_hand.hand_rank.display_name}")
        
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
        'tie': game_state.tie,
        'layout_confidence': game_state.layout_confidence
    }
    
    for player in game_state.players:
        player_info = {
            'id': player.id,
            'name': player.name,
            'position': player.position,
            'hole_cards': [str(card) for card in player.hole_cards],
            'best_hand': player.best_hand.hand_rank.display_name if player.best_hand else None,
            'hand_description': player.best_hand.hand_rank.display_name if player.best_hand else None
        }
        result['players'].append(player_info)
    
    if game_state.winner:
        result['winner'] = {
            'id': game_state.winner.id,
            'name': game_state.winner.name,
            'winning_hand': game_state.winner.best_hand.hand_rank.display_name if game_state.winner.best_hand else "Unknown"
        }
    elif game_state.tied_players:
        result['tied_players'] = [
            {'id': p.id, 'name': p.name} for p in game_state.tied_players
        ]
    
    return result