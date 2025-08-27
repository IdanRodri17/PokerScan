"""
Duplicate Card Detection and Removal System for PokerVision

This module handles the critical task of removing duplicate card detections,
which is essential for poker applications where each card can only exist once.
"""

import logging
from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class DuplicateCardHandler:
    """
    Handles duplicate card detection and removal for poker applications.
    
    In poker, each card can only exist once in the entire deck, so duplicate
    detections must be intelligently resolved by keeping the best detection.
    """
    
    def __init__(self, iou_threshold: float = 0.3, confidence_weight: float = 0.7):
        """
        Initialize duplicate handler
        
        Args:
            iou_threshold: IOU threshold for considering detections as duplicates
            confidence_weight: Weight for confidence vs area when choosing best detection
        """
        self.iou_threshold = iou_threshold
        self.confidence_weight = confidence_weight
        
        # Track which cards we've seen (for global uniqueness)
        self.seen_cards: Set[str] = set()
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Get intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # No intersection
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_bbox_area(self, bbox: List[float]) -> float:
        """Calculate bounding box area"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def detection_score(self, detection: Dict) -> float:
        """
        Calculate composite score for detection ranking
        
        Args:
            detection: Detection dictionary with confidence and bbox
            
        Returns:
            Composite score combining confidence and area
        """
        confidence = detection['confidence']
        area = self.calculate_bbox_area(detection['bbox'])
        
        # Normalize area (assume max reasonable card area is 20000 pixels)
        normalized_area = min(area / 20000.0, 1.0)
        
        # Composite score: weighted combination of confidence and area
        score = (self.confidence_weight * confidence + 
                (1 - self.confidence_weight) * normalized_area)
        
        return score
    
    def remove_spatial_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """
        Remove spatially overlapping duplicate detections of the same card
        
        Args:
            detections: List of card detections
            
        Returns:
            List of detections with spatial duplicates removed
        """
        if not detections:
            return detections
        
        # Group detections by card name
        card_groups = defaultdict(list)
        for i, detection in enumerate(detections):
            card_name = detection['card_name']
            detection['original_index'] = i
            card_groups[card_name].append(detection)
        
        filtered_detections = []
        
        for card_name, group_detections in card_groups.items():
            if len(group_detections) == 1:
                # Single detection, keep it
                filtered_detections.extend(group_detections)
                continue
            
            # Multiple detections of same card - remove spatial duplicates
            unique_detections = []
            
            # Sort by detection score (best first)
            group_detections.sort(key=self.detection_score, reverse=True)
            
            for detection in group_detections:
                is_duplicate = False
                
                # Check against already accepted detections
                for unique_detection in unique_detections:
                    iou = self.calculate_iou(detection['bbox'], unique_detection['bbox'])
                    
                    if iou > self.iou_threshold:
                        # This is a spatial duplicate
                        is_duplicate = True
                        logger.debug(f"Removing spatial duplicate {card_name}: IoU={iou:.3f}")
                        break
                
                if not is_duplicate:
                    unique_detections.append(detection)
            
            filtered_detections.extend(unique_detections)
            
            if len(unique_detections) < len(group_detections):
                removed_count = len(group_detections) - len(unique_detections)
                logger.info(f"Removed {removed_count} spatial duplicates of {card_name}")
        
        return filtered_detections
    
    def enforce_card_uniqueness(self, detections: List[Dict], 
                              allow_duplicates: bool = False) -> List[Dict]:
        """
        Enforce poker rule: each card can only appear once in the entire game
        
        Args:
            detections: List of card detections
            allow_duplicates: If True, allows multiple instances (for training/testing)
            
        Returns:
            List of detections with card uniqueness enforced
        """
        if allow_duplicates:
            return detections
        
        # Count occurrences of each card
        card_counts = Counter(detection['card_name'] for detection in detections)
        
        # Find cards that appear multiple times
        duplicate_cards = {card: count for card, count in card_counts.items() if count > 1}
        
        if not duplicate_cards:
            # No duplicates found
            return detections
        
        logger.warning(f"Found duplicate cards (impossible in poker): {duplicate_cards}")
        
        # For each duplicate card, keep only the best detection
        card_best_detections = {}
        remaining_detections = []
        
        for detection in detections:
            card_name = detection['card_name']
            
            if card_name in duplicate_cards:
                # This card has duplicates - choose the best one
                if card_name not in card_best_detections:
                    card_best_detections[card_name] = detection
                else:
                    # Compare with current best
                    current_best = card_best_detections[card_name]
                    if self.detection_score(detection) > self.detection_score(current_best):
                        card_best_detections[card_name] = detection
            else:
                # No duplicates for this card
                remaining_detections.append(detection)
        
        # Add the best detections for duplicate cards
        remaining_detections.extend(card_best_detections.values())
        
        # Log the cleanup
        removed_count = len(detections) - len(remaining_detections)
        logger.info(f"Removed {removed_count} duplicate card instances")
        
        return remaining_detections
    
    def validate_poker_deck(self, detections: List[Dict]) -> Dict[str, any]:
        """
        Validate detections against poker deck rules
        
        Args:
            detections: List of card detections
            
        Returns:
            Validation report dictionary
        """
        card_names = [detection['card_name'] for detection in detections]
        
        # Check for duplicates
        card_counts = Counter(card_names)
        duplicates = {card: count for card, count in card_counts.items() if count > 1}
        
        # Valid poker cards (52 cards)
        valid_cards = {
            'As', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'Ts', 'Js', 'Qs', 'Ks',
            'Ah', '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', 'Th', 'Jh', 'Qh', 'Kh',
            'Ad', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', 'Td', 'Jd', 'Qd', 'Kd',
            'Ac', '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', 'Tc', 'Jc', 'Qc', 'Kc'
        }
        
        # Check for invalid cards
        invalid_cards = [card for card in card_names if card not in valid_cards]
        
        # Calculate statistics
        total_cards = len(detections)
        unique_cards = len(set(card_names))
        
        validation_report = {
            'total_detections': total_cards,
            'unique_cards': unique_cards,
            'duplicates': duplicates,
            'invalid_cards': invalid_cards,
            'is_valid_poker': len(duplicates) == 0 and len(invalid_cards) == 0,
            'average_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
        }
        
        return validation_report
    
    def process_detections(self, detections: List[Dict], 
                         remove_spatial_duplicates: bool = True,
                         enforce_uniqueness: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Complete duplicate processing pipeline
        
        Args:
            detections: Raw detections from YOLO model
            remove_spatial_duplicates: Remove overlapping detections of same card
            enforce_uniqueness: Enforce poker card uniqueness rule
            
        Returns:
            Tuple of (processed_detections, processing_report)
        """
        original_count = len(detections)
        
        # Step 1: Remove spatial duplicates
        if remove_spatial_duplicates:
            detections = self.remove_spatial_duplicates(detections)
            after_spatial = len(detections)
        else:
            after_spatial = original_count
        
        # Step 2: Enforce card uniqueness
        if enforce_uniqueness:
            detections = self.enforce_card_uniqueness(detections)
            after_uniqueness = len(detections)
        else:
            after_uniqueness = after_spatial
        
        # Step 3: Validate final result
        validation_report = self.validate_poker_deck(detections)
        
        # Create processing report
        processing_report = {
            'original_detections': original_count,
            'after_spatial_dedup': after_spatial,
            'after_uniqueness': after_uniqueness,
            'final_detections': len(detections),
            'spatial_duplicates_removed': original_count - after_spatial,
            'card_duplicates_removed': after_spatial - after_uniqueness,
            'validation': validation_report
        }
        
        logger.info(f"Duplicate processing: {original_count} → {len(detections)} detections")
        
        return detections, processing_report


def create_duplicate_handler(iou_threshold: float = 0.3, 
                           confidence_weight: float = 0.7) -> DuplicateCardHandler:
    """
    Factory function to create a DuplicateCardHandler instance
    
    Args:
        iou_threshold: IOU threshold for spatial duplicate detection
        confidence_weight: Weight for confidence in scoring
        
    Returns:
        Configured DuplicateCardHandler instance
    """
    return DuplicateCardHandler(iou_threshold=iou_threshold, 
                              confidence_weight=confidence_weight)