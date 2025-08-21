"""
Spatial Analysis for Poker Table Card Positioning

This module analyzes the spatial relationships between detected cards
to identify community cards, player hands, and game state.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import yaml

from .card_detector import CardDetection

logger = logging.getLogger(__name__)


class GameStage(Enum):
    """Poker game stages based on community cards"""
    PREFLOP = "preflop"      # 0 community cards
    FLOP = "flop"            # 3 community cards
    TURN = "turn"            # 4 community cards
    RIVER = "river"          # 5 community cards
    UNKNOWN = "unknown"      # Unexpected number


@dataclass
class PlayerHand:
    """Represents a player's hole cards"""
    player_id: int
    cards: List[CardDetection]
    position: Tuple[float, float]  # Center position of hand
    confidence: float  # Average confidence of cards in hand


@dataclass
class CommunityCards:
    """Represents the community cards (flop, turn, river)"""
    cards: List[CardDetection]
    stage: GameStage
    position: Tuple[float, float]  # Center position of community cards


@dataclass
class PokerTableAnalysis:
    """Complete analysis of a poker table"""
    community_cards: Optional[CommunityCards]
    player_hands: List[PlayerHand]
    unassigned_cards: List[CardDetection]
    total_cards: int
    confidence_score: float
    analysis_metadata: Dict


class PokerSpatialAnalyzer:
    """Analyzes spatial relationships of cards on a poker table"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the spatial analyzer
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        logger.info("Initialized PokerSpatialAnalyzer")
    
    def _load_config(self) -> Dict:
        """Load spatial analysis configuration"""
        if self.config_path:
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config.get('spatial', {})
            except Exception as e:
                logger.warning(f"Could not load config from {self.config_path}: {e}")
        
        # Default configuration
        return {
            'clustering_method': 'dbscan',
            'eps': 0.1,
            'min_samples': 1,
            'community_card_threshold': 0.3,
            'player_hand_size': 2,
            'max_players': 3
        }
    
    def analyze_table(self, detections: List[CardDetection], image_shape: Optional[Tuple[int, int]] = None) -> PokerTableAnalysis:
        """
        Analyze the spatial distribution of cards on a poker table
        
        Args:
            detections: List of card detections
            image_shape: (height, width) of the original image
            
        Returns:
            PokerTableAnalysis object with organized card groups
        """
        if not detections:
            return PokerTableAnalysis(
                community_cards=None,
                player_hands=[],
                unassigned_cards=[],
                total_cards=0,
                confidence_score=0.0,
                analysis_metadata={'error': 'No cards detected'}
            )
        
        logger.info(f"Analyzing {len(detections)} detected cards")
        
        # Normalize card positions if image shape is provided
        normalized_detections = self._normalize_positions(detections, image_shape)
        
        # Cluster cards by spatial proximity
        clusters = self._cluster_cards(normalized_detections)
        
        # Identify community cards and player hands
        community_cards, player_hands, unassigned = self._identify_card_groups(clusters)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(detections, community_cards, player_hands)
        
        # Create metadata
        metadata = {
            'clustering_method': self.config.get('clustering_method'),
            'num_clusters': len(clusters),
            'image_shape': image_shape,
            'normalization_applied': image_shape is not None
        }
        
        analysis = PokerTableAnalysis(
            community_cards=community_cards,
            player_hands=player_hands,
            unassigned_cards=unassigned,
            total_cards=len(detections),
            confidence_score=confidence_score,
            analysis_metadata=metadata
        )
        
        logger.info(f"Analysis complete: {len(player_hands)} players, "
                   f"{'community cards found' if community_cards else 'no community cards'}")
        
        return analysis
    
    def _normalize_positions(self, detections: List[CardDetection], image_shape: Optional[Tuple[int, int]]) -> List[CardDetection]:
        """Normalize card positions to [0,1] range if image shape is provided"""
        if image_shape is None:
            return detections
        
        height, width = image_shape
        normalized = []
        
        for detection in detections:
            # Normalize center coordinates
            norm_center = (
                detection.center[0] / width,
                detection.center[1] / height
            )
            
            # Normalize bounding box
            norm_bbox = [
                detection.bbox[0] / width,   # x1
                detection.bbox[1] / height,  # y1
                detection.bbox[2] / width,   # x2
                detection.bbox[3] / height   # y2
            ]
            
            # Create new detection with normalized coordinates
            norm_detection = CardDetection(
                card_name=detection.card_name,
                confidence=detection.confidence,
                bbox=norm_bbox,
                center=norm_center
            )
            normalized.append(norm_detection)
        
        return normalized
    
    def _cluster_cards(self, detections: List[CardDetection]) -> List[List[CardDetection]]:
        """Cluster cards by spatial proximity"""
        if len(detections) <= 1:
            return [detections]
        
        # Extract positions for clustering
        positions = np.array([det.center for det in detections])
        
        # Choose clustering method
        method = self.config.get('clustering_method', 'dbscan').lower()
        
        try:
            if method == 'dbscan':
                labels = self._cluster_dbscan(positions)
            elif method == 'kmeans':
                labels = self._cluster_kmeans(positions)
            elif method == 'agglomerative':
                labels = self._cluster_agglomerative(positions)
            else:
                logger.warning(f"Unknown clustering method: {method}, using DBSCAN")
                labels = self._cluster_dbscan(positions)
                
        except Exception as e:
            logger.error(f"Clustering failed: {e}, falling back to single cluster")
            labels = np.zeros(len(detections))
        
        # Group detections by cluster labels
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Noise point in DBSCAN
                label = f"noise_{i}"
            
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(detections[i])
        
        return list(clusters.values())
    
    def _cluster_dbscan(self, positions: np.ndarray) -> np.ndarray:
        """Cluster using DBSCAN"""
        eps = self.config.get('eps', 0.1)
        min_samples = self.config.get('min_samples', 1)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        return clustering.fit_predict(positions)
    
    def _cluster_kmeans(self, positions: np.ndarray) -> np.ndarray:
        """Cluster using K-Means"""
        # Estimate number of clusters (community cards + player hands)
        max_players = self.config.get('max_players', 3)
        n_clusters = min(max_players + 1, len(positions))  # +1 for community cards
        
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return clustering.fit_predict(positions)
    
    def _cluster_agglomerative(self, positions: np.ndarray) -> np.ndarray:
        """Cluster using Agglomerative clustering"""
        max_players = self.config.get('max_players', 3)
        n_clusters = min(max_players + 1, len(positions))
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return clustering.fit_predict(positions)
    
    def _identify_card_groups(self, clusters: List[List[CardDetection]]) -> Tuple[Optional[CommunityCards], List[PlayerHand], List[CardDetection]]:
        """Identify community cards and player hands from clusters"""
        community_cards = None
        player_hands = []
        unassigned = []
        
        # Sort clusters by size and position
        sorted_clusters = self._sort_clusters_by_characteristics(clusters)
        
        for i, cluster in enumerate(sorted_clusters):
            cluster_size = len(cluster)
            
            # Community cards are typically 3-5 cards in the center
            if cluster_size in [3, 4, 5] and community_cards is None:
                stage = self._determine_game_stage(cluster_size)
                center_pos = self._calculate_cluster_center(cluster)
                
                community_cards = CommunityCards(
                    cards=cluster,
                    stage=stage,
                    position=center_pos
                )
                logger.info(f"Identified community cards: {stage.value} with {cluster_size} cards")
            
            # Player hands are typically 2 cards
            elif cluster_size == 2:
                center_pos = self._calculate_cluster_center(cluster)
                avg_confidence = np.mean([card.confidence for card in cluster])
                
                player_hand = PlayerHand(
                    player_id=len(player_hands) + 1,
                    cards=cluster,
                    position=center_pos,
                    confidence=avg_confidence
                )
                player_hands.append(player_hand)
                logger.info(f"Identified player hand {player_hand.player_id}")
            
            else:
                # Cards that don't fit expected patterns
                unassigned.extend(cluster)
                logger.info(f"Cluster of {cluster_size} cards marked as unassigned")
        
        return community_cards, player_hands, unassigned
    
    def _sort_clusters_by_characteristics(self, clusters: List[List[CardDetection]]) -> List[List[CardDetection]]:
        """Sort clusters by size and centrality for better identification"""
        def cluster_key(cluster):
            size = len(cluster)
            center = self._calculate_cluster_center(cluster)
            # Prefer larger clusters and those closer to image center
            centrality = 1.0 - (abs(center[0] - 0.5) + abs(center[1] - 0.5))
            return (-size, -centrality)  # Sort descending
        
        return sorted(clusters, key=cluster_key)
    
    def _calculate_cluster_center(self, cluster: List[CardDetection]) -> Tuple[float, float]:
        """Calculate the center position of a cluster"""
        x_coords = [card.center[0] for card in cluster]
        y_coords = [card.center[1] for card in cluster]
        return (np.mean(x_coords), np.mean(y_coords))
    
    def _determine_game_stage(self, num_cards: int) -> GameStage:
        """Determine poker game stage from number of community cards"""
        stage_map = {
            0: GameStage.PREFLOP,
            3: GameStage.FLOP,
            4: GameStage.TURN,
            5: GameStage.RIVER
        }
        return stage_map.get(num_cards, GameStage.UNKNOWN)
    
    def _calculate_confidence_score(self, detections: List[CardDetection], 
                                  community_cards: Optional[CommunityCards],
                                  player_hands: List[PlayerHand]) -> float:
        """Calculate overall confidence score for the analysis"""
        if not detections:
            return 0.0
        
        # Base confidence from card detection accuracies
        detection_confidence = np.mean([det.confidence for det in detections])
        
        # Structural confidence based on expected poker layout
        structure_score = 0.0
        
        # Bonus for having community cards
        if community_cards:
            stage_scores = {
                GameStage.FLOP: 0.9,
                GameStage.TURN: 0.95,
                GameStage.RIVER: 1.0,
                GameStage.PREFLOP: 0.7,
                GameStage.UNKNOWN: 0.3
            }
            structure_score += stage_scores.get(community_cards.stage, 0.3) * 0.5
        
        # Bonus for having complete player hands
        if player_hands:
            complete_hands = len([hand for hand in player_hands if len(hand.cards) == 2])
            structure_score += (complete_hands / max(1, len(player_hands))) * 0.3
        
        # Penalty for too many unassigned cards
        assigned_cards = (len(community_cards.cards) if community_cards else 0) + \
                        sum(len(hand.cards) for hand in player_hands)
        unassigned_ratio = (len(detections) - assigned_cards) / len(detections)
        structure_penalty = min(unassigned_ratio * 0.2, 0.2)
        
        final_confidence = (detection_confidence * 0.6 + structure_score * 0.4) - structure_penalty
        return max(0.0, min(1.0, final_confidence))
    
    def get_analysis_summary(self, analysis: PokerTableAnalysis) -> Dict:
        """Get a summary of the poker table analysis"""
        summary = {
            'total_cards': analysis.total_cards,
            'confidence_score': analysis.confidence_score,
            'game_stage': analysis.community_cards.stage.value if analysis.community_cards else 'preflop',
            'community_cards_count': len(analysis.community_cards.cards) if analysis.community_cards else 0,
            'player_count': len(analysis.player_hands),
            'unassigned_cards': len(analysis.unassigned_cards)
        }
        
        if analysis.community_cards:
            summary['community_cards'] = [card.card_name for card in analysis.community_cards.cards]
        
        for i, hand in enumerate(analysis.player_hands):
            summary[f'player_{hand.player_id}_cards'] = [card.card_name for card in hand.cards]
            summary[f'player_{hand.player_id}_confidence'] = hand.confidence
        
        return summary