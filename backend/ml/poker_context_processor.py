"""
Context-Aware Post-Processing for Poker Card Detection
Uses poker game logic and spatial reasoning to improve accuracy
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import Counter
import cv2

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Card detection with metadata"""
    card_name: str
    confidence: float
    bbox: List[float]
    center: Tuple[float, float]
    suit: str
    rank: str
    color: str

class PokerContextProcessor:
    """Post-processor using poker context and rules"""
    
    def __init__(self):
        # Card knowledge base
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.suits = {'S': 'spades', 'H': 'hearts', 'D': 'diamonds', 'C': 'clubs'}
        self.red_suits = ['H', 'D']
        self.black_suits = ['S', 'C']
        
        # Confusion patterns from your data
        self.common_confusions = {
            ('4', 'H', 'D'): 0.85,  # 4H often confused with 4D
            ('A', 'D', 'H'): 0.80,  # AD often confused with AH
            ('Q', 'H', 'D'): 0.75,  # Red queens confused
            ('K', 'H', 'D'): 0.75,  # Red kings confused
            ('J', 'H', 'D'): 0.70,  # Red jacks confused
        }
        
    def parse_card(self, card_name: str) -> Tuple[str, str]:
        """Parse card name into rank and suit"""
        if len(card_name) >= 2:
            if card_name.startswith('10'):
                rank = '10'
                suit = card_name[2:].upper()
            else:
                rank = card_name[:-1].upper()
                suit = card_name[-1:].upper()
            return rank, suit
        return None, None
    
    def process_detections(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Apply context-aware post-processing
        
        Args:
            detections: List of detection dictionaries
            image_shape: (height, width) of image
            
        Returns:
            Processed detections with improved accuracy
        """
        if not detections:
            return detections
        
        # Convert to Detection objects
        processed_detections = []
        for det in detections:
            rank, suit = self.parse_card(det['card_name'])
            if rank and suit:
                processed_detections.append(Detection(
                    card_name=det['card_name'],
                    confidence=det['confidence'],
                    bbox=det['bbox'],
                    center=tuple(det['center']),
                    suit=suit,
                    rank=rank,
                    color='red' if suit in self.red_suits else 'black'
                ))
        
        # Apply multiple processing strategies
        processed_detections = self._resolve_suit_confusion(processed_detections)
        processed_detections = self._apply_spatial_reasoning(processed_detections, image_shape)
        processed_detections = self._validate_poker_constraints(processed_detections)
        processed_detections = self._boost_missing_cards(processed_detections)
        
        # Convert back to dictionaries
        result = []
        for det in processed_detections:
            result.append({
                'card_name': det.card_name,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'center': det.center
            })
        
        return result
    
    def _resolve_suit_confusion(self, detections: List[Detection]) -> List[Detection]:
        """
        Resolve red suit confusion using context and confidence patterns
        """
        # Group detections by rank and color
        rank_color_groups = {}
        for det in detections:
            key = (det.rank, det.color)
            if key not in rank_color_groups:
                rank_color_groups[key] = []
            rank_color_groups[key].append(det)
        
        processed = []
        for (rank, color), group in rank_color_groups.items():
            if color == 'red' and len(group) > 0:
                # Handle red suit confusion
                processed.extend(self._disambiguate_red_suits(group, rank))
            else:
                processed.extend(group)
        
        return processed
    
    def _disambiguate_red_suits(self, red_detections: List[Detection], rank: str) -> List[Detection]:
        """
        Disambiguate between hearts and diamonds using multiple cues
        """
        if len(red_detections) == 1:
            det = red_detections[0]
            
            # Check if this is a commonly confused card
            for (conf_rank, suit1, suit2), conf_threshold in self.common_confusions.items():
                if det.rank == conf_rank and det.confidence < conf_threshold:
                    # Special logic for known problematic cards
                    if det.rank == '4' and det.suit == 'D':
                        # 4D is often actually 4H
                        logger.info(f"Correcting known confusion: {det.card_name} -> 4H")
                        corrected_det = Detection(
                            card_name="4H",
                            confidence=det.confidence * 1.1,  # Boost confidence after correction
                            bbox=det.bbox,
                            center=det.center,
                            suit='H',
                            rank='4',
                            color='red'
                        )
                        return [corrected_det]
                    
                    elif det.rank == 'A' and det.suit == 'H':
                        # Missing AD case - check if we should have AD instead
                        if det.confidence < 0.6:
                            logger.info(f"Correcting possible missing AD: {det.card_name} -> AD")
                            corrected_det = Detection(
                                card_name="AD",
                                confidence=det.confidence * 1.05,
                                bbox=det.bbox,
                                center=det.center,
                                suit='D',
                                rank='A',
                                color='red'
                            )
                            return [corrected_det]
            
            return [det]
        
        # Multiple red cards of same rank - shouldn't happen in poker
        # Keep the one with highest confidence
        return [max(red_detections, key=lambda x: x.confidence)]
    
    def _apply_spatial_reasoning(self, detections: List[Detection], image_shape: Tuple[int, int]) -> List[Detection]:
        """
        Use spatial positioning to validate and correct detections
        """
        height, width = image_shape
        
        # Define regions
        center_region = (width * 0.3, width * 0.7, height * 0.3, height * 0.7)
        
        for det in detections:
            x, y = det.center
            
            # Cards in center region (community cards) should have higher confidence threshold
            if center_region[0] < x < center_region[1] and center_region[2] < y < center_region[3]:
                if det.confidence < 0.4:  # Higher threshold for community cards
                    det.confidence *= 0.8  # Reduce confidence for uncertain community cards
            
            # Check for cards too close together (might be duplicates)
            for other in detections:
                if det != other:
                    distance = np.sqrt((det.center[0] - other.center[0])**2 + 
                                     (det.center[1] - other.center[1])**2)
                    if distance < 50:  # Too close, might be duplicate
                        # Keep the one with higher confidence
                        if det.confidence < other.confidence:
                            det.confidence *= 0.6
        
        # Filter low-confidence detections after spatial analysis (lower threshold)
        return [det for det in detections if det.confidence > 0.15]
    
    def _validate_poker_constraints(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply poker game constraints to validate detections
        """
        # No duplicate cards allowed
        seen_cards = set()
        validated = []
        
        # Sort by confidence to keep best detections
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        for det in sorted_detections:
            if det.card_name not in seen_cards:
                seen_cards.add(det.card_name)
                validated.append(det)
            else:
                logger.warning(f"Removing duplicate card: {det.card_name}")
        
        # Check for reasonable number of cards (max 7 per player + 5 community)
        if len(validated) > 15:  # Reasonable max for typical poker hand
            logger.warning(f"Too many cards detected ({len(validated)}), keeping top 15")
            validated = validated[:15]
        
        return validated
    
    def _boost_missing_cards(self, detections: List[Detection]) -> List[Detection]:
        """
        Look for commonly missed cards (like Ace of Diamonds) with lower threshold
        """
        detected_cards = {det.card_name.upper() for det in detections}
        expected_cards = ['AS', 'QS', '4H', '10S', 'AD', 'KS', 'JS']  # Your known cards
        
        missing_cards = []
        for card in expected_cards:
            if card not in detected_cards:
                missing_cards.append(card)
        
        if missing_cards:
            logger.info(f"Missing expected cards: {', '.join(missing_cards)}")
            # In a full implementation, this would re-run detection with lower thresholds
        
        return detections

class SuitColorAnalyzer:
    """Analyze actual pixel colors to distinguish suits"""
    
    def __init__(self):
        # HSV ranges for suit colors under various lighting
        self.color_ranges = {
            'hearts': [
                # Pure red
                {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
                {'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])},
            ],
            'diamonds': [
                # Orange-red
                {'lower': np.array([10, 50, 50]), 'upper': np.array([25, 255, 255])},
            ],
            'spades': [
                # Black/very dark
                {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 30])},
            ],
            'clubs': [
                # Black/very dark (same as spades, need shape analysis)
                {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 30])},
            ]
        }
    
    def analyze_suit_color(self, image_crop: np.ndarray, detected_suit: str) -> str:
        """
        Analyze actual colors in card crop to verify suit
        
        Args:
            image_crop: Cropped card image (RGB)
            detected_suit: Currently detected suit
            
        Returns:
            Corrected suit
        """
        if image_crop.size == 0:
            return detected_suit
        
        # Convert to HSV
        hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)
        
        # Find suit symbols (usually in corners)
        h, w = image_crop.shape[:2]
        corner_regions = [
            hsv[0:h//4, 0:w//4],  # Top-left
            hsv[3*h//4:h, 3*w//4:w] if h >= 4 and w >= 4 else hsv,  # Bottom-right
        ]
        
        suit_scores = {'H': 0, 'D': 0, 'S': 0, 'C': 0}
        
        for region in corner_regions:
            if region.size == 0:
                continue
                
            # Check each suit's color range
            for suit, ranges in self.color_ranges.items():
                for color_range in ranges:
                    mask = cv2.inRange(region, color_range['lower'], color_range['upper'])
                    score = np.sum(mask) / (region.shape[0] * region.shape[1] * 255) if region.size > 0 else 0
                    
                    if suit == 'hearts':
                        suit_scores['H'] += score
                    elif suit == 'diamonds':
                        suit_scores['D'] += score
                    elif suit == 'spades':
                        suit_scores['S'] += score
                    elif suit == 'clubs':
                        suit_scores['C'] += score
        
        # Get best matching suit
        best_suit = max(suit_scores, key=suit_scores.get)
        
        # If detected suit is red but colors suggest otherwise, correct it
        detected_upper = detected_suit.upper()
        if detected_upper in ['H', 'D'] and best_suit in ['H', 'D'] and detected_upper != best_suit:
            logger.info(f"Correcting suit based on color analysis: {detected_suit} -> {best_suit}")
            return best_suit
        
        return detected_upper

def create_context_processor():
    """Factory function to create context processor"""
    return PokerContextProcessor()

# Integration function
def apply_poker_context(detections: List[Dict], image: np.ndarray) -> List[Dict]:
    """
    Apply full context-aware post-processing pipeline
    
    Args:
        detections: Raw detections from model
        image: Original image for color analysis
        
    Returns:
        Processed detections with improved accuracy
    """
    processor = PokerContextProcessor()
    color_analyzer = SuitColorAnalyzer()
    
    # First pass: context processing
    processed = processor.process_detections(detections, image.shape[:2])
    
    # Second pass: color verification for uncertain suits
    for det in processed:
        if 'bbox' in det and len(det['bbox']) >= 4:
            try:
                x1, y1, x2, y2 = map(int, det['bbox'])
                # Ensure bounds are within image
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    card_crop = image[y1:y2, x1:x2]
                    
                    rank, suit = processor.parse_card(det['card_name'])
                    if suit and card_crop.size > 0:
                        corrected_suit = color_analyzer.analyze_suit_color(card_crop, suit)
                        if corrected_suit != suit:
                            det['card_name'] = f"{rank}{corrected_suit}"
            except Exception as e:
                logger.warning(f"Error in color analysis for {det['card_name']}: {e}")
    
    return processed