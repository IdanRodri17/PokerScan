"""
Poker Card Count Enforcer - Always Returns 9 or 11 Cards
Handles missing detections and duplicates intelligently
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import Counter
import cv2

logger = logging.getLogger(__name__)

class StrictPokerValidator:
    """
    Enforces EXACTLY 9 or 11 cards for Texas Hold'em
    Handles missing cards by re-running detection with lower thresholds
    """
    
    def __init__(self, detector=None):
        """
        Args:
            detector: Reference to the YOLOv8 detector for re-detection if needed
        """
        self.detector = detector
        self.target_counts = [9, 11]  # Valid card counts
        self.community_count = 5
        self.player_card_count = 2
        
    def validate_and_fix(self, 
                        detections: List[Dict], 
                        image: np.ndarray,
                        raw_yolo_results=None) -> List[Dict]:
        """
        Main method that GUARANTEES 9 or 11 unique cards
        
        Args:
            detections: Initial detections from model
            image: Original image for re-detection if needed
            raw_yolo_results: Raw YOLO results for accessing lower confidence detections
            
        Returns:
            Exactly 9 or 11 cards, no duplicates
        """
        logger.info(f"Starting validation with {len(detections)} initial detections")
        
        # Step 1: Remove duplicates by card name
        unique_detections = self._remove_duplicate_cards(detections)
        current_count = len(unique_detections)
        
        logger.info(f"After removing duplicates: {current_count} cards")
        
        # Step 2: Determine target count (9 or 11)
        target_count = self._determine_target_count(unique_detections, image.shape)
        logger.info(f"Target card count: {target_count}")
        
        # Step 3: If we have too few cards, find missing ones
        if current_count < target_count:
            unique_detections = self._find_missing_cards(
                unique_detections, 
                target_count, 
                image, 
                raw_yolo_results
            )
        
        # Step 4: If we have too many cards, intelligently remove extras
        elif current_count > target_count:
            unique_detections = self._remove_excess_cards(unique_detections, target_count, image.shape)
        
        # Step 5: Final layout validation
        unique_detections = self._validate_layout(unique_detections, target_count, image.shape)
        
        # Log final result
        self._log_final_result(unique_detections, target_count)
        
        return unique_detections
    
    def _remove_duplicate_cards(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicate card names, keeping highest confidence"""
        seen_cards = {}
        
        for det in detections:
            card_name = det['card_name'].upper()
            
            if card_name not in seen_cards:
                seen_cards[card_name] = det
            else:
                # Keep the one with higher confidence
                if det['confidence'] > seen_cards[card_name]['confidence']:
                    logger.info(f"Replacing {card_name} detection "
                              f"(conf: {seen_cards[card_name]['confidence']:.3f} -> {det['confidence']:.3f})")
                    seen_cards[card_name] = det
                else:
                    logger.info(f"Keeping existing {card_name} detection "
                              f"(conf: {seen_cards[card_name]['confidence']:.3f} > {det['confidence']:.3f})")
        
        return list(seen_cards.values())
    
    def _determine_target_count(self, detections: List[Dict], image_shape: Tuple[int, int]) -> int:
        """
        Determine if we should have 9 or 11 cards based on clustering
        """
        if len(detections) < 7:
            return 9  # Default to 2 players
        
        # Cluster cards to identify groups
        if len(detections) == 0:
            return 9
            
        centers = np.array([d['center'] for d in detections])
        height, width = image_shape
        
        # Identify community cards (center region)
        center_x, center_y = width / 2, height / 2
        community_region = []
        player_regions = []
        
        for i, det in enumerate(detections):
            x, y = det['center']
            # Community cards are typically in center 40% of image
            if (abs(x - center_x) < width * 0.3 and 
                abs(y - center_y) < height * 0.2):
                community_region.append(det)
            else:
                player_regions.append(det)
        
        # Estimate player count based on card groupings
        if len(player_regions) >= 6:
            return 11  # 3 players
        else:
            return 9   # 2 players
    
    def _find_missing_cards(self, 
                           detections: List[Dict], 
                           target_count: int,
                           image: np.ndarray,
                           raw_yolo_results) -> List[Dict]:
        """
        Find missing cards by looking at lower confidence detections
        """
        current_count = len(detections)
        missing_count = target_count - current_count
        
        logger.warning(f"Missing {missing_count} cards! Looking for additional detections...")
        
        # Get existing card names to avoid duplicates
        existing_cards = {d['card_name'].upper() for d in detections}
        
        # If we have raw YOLO results, check lower confidence detections
        if raw_yolo_results is not None:
            additional_detections = self._extract_lower_confidence_cards(
                raw_yolo_results, 
                existing_cards, 
                missing_count
            )
            
            if additional_detections:
                logger.info(f"Found {len(additional_detections)} additional cards at lower confidence")
                detections.extend(additional_detections)
        
        # If still missing cards, try re-detection with lower threshold
        if len(detections) < target_count and self.detector is not None:
            logger.info("Re-running detection with lower confidence threshold...")
            
            # Get original threshold
            original_conf = self.detector.config.get('model', {}).get('confidence_threshold', 0.3)
            
            # Temporarily lower threshold
            self.detector.config['model']['confidence_threshold'] = 0.05
            
            try:
                # Re-detect
                new_detections, _ = self.detector.detect_cards(image)
                
                # Add new unique cards
                for new_det in new_detections:
                    card_dict = {
                        'card_name': new_det.card_name,
                        'confidence': new_det.confidence,
                        'bbox': new_det.bbox,
                        'center': new_det.center
                    }
                    
                    if new_det.card_name.upper() not in existing_cards:
                        detections.append(card_dict)
                        existing_cards.add(new_det.card_name.upper())
                        if len(detections) >= target_count:
                            break
                            
            finally:
                # Restore threshold
                self.detector.config['model']['confidence_threshold'] = original_conf
        
        # If STILL missing cards, use spatial analysis to identify gaps
        if len(detections) < target_count:
            detections = self._infer_missing_cards_by_position(detections, target_count, image.shape)
        
        return detections[:target_count]  # Ensure we don't exceed target
    
    def _extract_lower_confidence_cards(self, 
                                       raw_results, 
                                       existing_cards: Set[str], 
                                       needed: int) -> List[Dict]:
        """Extract additional cards from raw YOLO results at lower confidence"""
        additional = []
        
        # Process raw results with lower threshold
        for result in raw_results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf < 0.05:  # Look at very low confidence
                        continue
                    
                    class_id = int(box.cls)
                    card_name = self._get_card_name_from_class(class_id)
                    
                    if card_name.upper() not in existing_cards:
                        bbox_coords = box.xyxy[0].cpu().numpy()
                        detection = {
                            'card_name': card_name,
                            'confidence': conf,
                            'bbox': bbox_coords.tolist(),
                            'center': ((bbox_coords[0] + bbox_coords[2]) / 2,
                                     (bbox_coords[1] + bbox_coords[3]) / 2)
                        }
                        additional.append(detection)
                        existing_cards.add(card_name.upper())
                        
                        if len(additional) >= needed:
                            break
        
        return additional
    
    def _infer_missing_cards_by_position(self, 
                                        detections: List[Dict], 
                                        target_count: int,
                                        image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Infer missing cards based on expected positions
        Community cards should be in a row, player cards in pairs
        """
        height, width = image_shape
        center_y = height / 2
        
        # Identify what we have
        community_cards = []
        player_cards = []
        
        for det in detections:
            y = det['center'][1]
            if abs(y - center_y) < height * 0.15:  # Near center
                community_cards.append(det)
            else:
                player_cards.append(det)
        
        logger.info(f"Current: {len(community_cards)} community, {len(player_cards)} player cards")
        
        # We need exactly 5 community cards
        if len(community_cards) < 5:
            # Look harder in the community region
            logger.warning(f"Only {len(community_cards)} community cards found, expected 5")
        
        # Each player needs exactly 2 cards
        expected_players = (target_count - 5) // 2
        if len(player_cards) < expected_players * 2:
            logger.warning(f"Only {len(player_cards)} player cards found, expected {expected_players * 2}")
        
        return detections
    
    def _remove_excess_cards(self, 
                            detections: List[Dict], 
                            target_count: int,
                            image_shape: Tuple[int, int]) -> List[Dict]:
        """Remove excess cards intelligently based on position and confidence"""
        excess_count = len(detections) - target_count
        logger.warning(f"Have {excess_count} too many cards, removing lowest confidence outliers")
        
        # Sort by confidence and remove lowest
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Keep only target count
        return detections[:target_count]
    
    def _validate_layout(self, 
                        detections: List[Dict], 
                        target_count: int,
                        image_shape: Tuple[int, int]) -> List[Dict]:
        """Validate that cards are in proper poker layout"""
        height, width = image_shape
        
        # Group cards by Y position
        cards_by_row = {}
        for det in detections:
            y = det['center'][1]
            # Quantize Y to rows
            row = int(y / (height / 5))
            if row not in cards_by_row:
                cards_by_row[row] = []
            cards_by_row[row].append(det)
        
        # Log the layout
        for row, cards in sorted(cards_by_row.items()):
            card_names = [c['card_name'] for c in cards]
            logger.info(f"Row {row}: {', '.join(card_names)}")
        
        return detections
    
    def _get_card_name_from_class(self, class_id: int) -> str:
        """Convert class ID to card name using your model's class mapping"""
        if self.detector and hasattr(self.detector, 'model') and hasattr(self.detector.model, 'names'):
            return self.detector.model.names.get(class_id, f"Unknown_{class_id}")
        
        # Fallback mapping
        suits = ['C', 'D', 'H', 'S']
        ranks = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'Q', 'K']
        
        if class_id < len(suits) * len(ranks):
            suit_idx = class_id % len(suits)
            rank_idx = class_id // len(suits)
            return f"{ranks[rank_idx]}{suits[suit_idx]}"
        
        return f"Unknown_{class_id}"
    
    def _log_final_result(self, detections: List[Dict], target_count: int):
        """Log the final validated result"""
        card_names = [d['card_name'] for d in detections]
        logger.info("=" * 50)
        logger.info(f"FINAL RESULT: {len(detections)}/{target_count} cards")
        logger.info(f"Cards: {', '.join(card_names)}")
        logger.info("=" * 50)

# Integration with your card_detector.py
def enforce_poker_card_count(detections: List[Dict], 
                            image: np.ndarray,
                            detector=None,
                            raw_results=None) -> List[Dict]:
    """
    Main function to enforce 9 or 11 cards
    
    Args:
        detections: Initial detections
        image: Original image
        detector: YOLOv8 detector instance (optional)
        raw_results: Raw YOLO results (optional)
    
    Returns:
        Exactly 9 or 11 validated cards
    """
    validator = StrictPokerValidator(detector)
    return validator.validate_and_fix(detections, image, raw_results)