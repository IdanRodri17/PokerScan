"""
Optimal Threshold Testing for Poker Card Detection
Based on Claude Opus's recommendations for domain gap issues
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import itertools

logger = logging.getLogger(__name__)


class OptimalThresholdFinder:
    """
    Find optimal detection thresholds specifically for poker table domain gap issues
    Addresses suit confusion and missing card problems
    """
    
    def __init__(self):
        """Initialize threshold finder"""
        pass
    
    def test_threshold_configurations(self, detector, test_image, known_cards: List[str] = None) -> List[Dict]:
        """
        Test different threshold configurations as recommended by Claude Opus
        
        Args:
            detector: Card detector instance
            test_image: Test image (PIL or numpy array)
            known_cards: List of actual card names in the image
            
        Returns:
            List of configuration results sorted by performance
        """
        # Claude Opus's recommended configurations for domain gap issues
        configurations = [
            {"name": "Ultra Low Confidence", "conf": 0.15, "iou": 0.35, "imgsz": 1024},
            {"name": "Opus Recommended", "conf": 0.22, "iou": 0.38, "imgsz": 1024},
            {"name": "Low Confidence", "conf": 0.25, "iou": 0.40, "imgsz": 1024},
            {"name": "Balanced", "conf": 0.30, "iou": 0.45, "imgsz": 1024},
            {"name": "Multi-Scale Low", "conf": 0.20, "iou": 0.35, "imgsz": 1280},
            {"name": "High Resolution", "conf": 0.25, "iou": 0.38, "imgsz": 1536},
            {"name": "Conservative", "conf": 0.35, "iou": 0.50, "imgsz": 1024}
        ]
        
        results = []
        
        for config in configurations:
            try:
                logger.info(f"Testing configuration: {config['name']}")
                
                # Temporarily update detector configuration
                original_config = detector.config['model'].copy()
                detector.config['model']['confidence_threshold'] = config['conf']
                detector.config['model']['iou_threshold'] = config['iou']
                detector.config['model']['input_size'] = config['imgsz']
                
                # Run detection with enhanced preprocessing
                detections, inference_time, processing_report = detector.detect_cards_from_pil_poker(test_image)
                
                # Extract detected card names
                detected_cards = [det.card_name for det in detections]
                
                # Calculate performance metrics
                metrics = self._calculate_performance_metrics(detected_cards, known_cards)
                
                # Store results
                result = {
                    'configuration': config,
                    'detected_cards': detected_cards,
                    'inference_time': inference_time,
                    'processing_report': processing_report,
                    'metrics': metrics
                }
                
                results.append(result)
                
                # Log results
                logger.info(f"  Detected: {len(detected_cards)} cards")
                if known_cards:
                    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
                    logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
                
                # Restore original configuration
                detector.config['model'] = original_config
                
            except Exception as e:
                logger.error(f"Configuration test failed for {config['name']}: {e}")
                continue
        
        # Sort by F1-score (best overall metric for this problem)
        if known_cards:
            results.sort(key=lambda x: x['metrics']['f1_score'], reverse=True)
        
        return results
    
    def _calculate_performance_metrics(self, detected_cards: List[str], 
                                     known_cards: List[str] = None) -> Dict:
        """
        Calculate performance metrics for detection results
        
        Args:
            detected_cards: List of detected card names
            known_cards: List of actual card names (if known)
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'total_detected': len(detected_cards),
            'unique_detected': len(set(detected_cards)),
            'duplicate_count': len(detected_cards) - len(set(detected_cards))
        }
        
        if known_cards is None:
            return metrics
        
        # Calculate accuracy metrics when ground truth is available
        actual_set = set(known_cards)
        detected_set = set(detected_cards)
        
        # Basic metrics
        true_positives = len(actual_set & detected_set)
        false_positives = len(detected_set - actual_set)
        false_negatives = len(actual_set - detected_set)
        
        # Performance metrics
        precision = true_positives / len(detected_set) if detected_set else 0
        recall = true_positives / len(actual_set) if actual_set else 0
        accuracy = true_positives / len(actual_set) if actual_set else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Suit confusion analysis
        suit_confusion = self._analyze_suit_confusion(detected_cards, known_cards)
        
        metrics.update({
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'suit_confusion': suit_confusion
        })
        
        return metrics
    
    def _analyze_suit_confusion(self, detected_cards: List[str], known_cards: List[str]) -> Dict:
        """
        Analyze suit confusion patterns as identified by Claude Opus
        
        Args:
            detected_cards: Detected card names
            known_cards: Actual card names
            
        Returns:
            Dictionary with suit confusion analysis
        """
        confusion_patterns = []
        
        # Check for common suit confusion patterns
        suit_map = {'s': 'spades', 'h': 'hearts', 'd': 'diamonds', 'c': 'clubs'}
        
        for actual_card in known_cards:
            if len(actual_card) >= 2:
                actual_rank = actual_card[:-1]  # All except last character
                actual_suit = actual_card[-1].lower()  # Last character
                
                # Look for same rank with different suit in detected cards
                for detected_card in detected_cards:
                    if len(detected_card) >= 2:
                        detected_rank = detected_card[:-1]
                        detected_suit = detected_card[-1].lower()
                        
                        if actual_rank == detected_rank and actual_suit != detected_suit:
                            confusion_patterns.append({
                                'actual': actual_card,
                                'detected': detected_card,
                                'type': f"{suit_map.get(actual_suit, actual_suit)} → {suit_map.get(detected_suit, detected_suit)}"
                            })
        
        return {
            'patterns': confusion_patterns,
            'count': len(confusion_patterns),
            'most_common': Counter([p['type'] for p in confusion_patterns]).most_common(3)
        }
    
    def find_optimal_configuration(self, results: List[Dict]) -> Dict:
        """
        Find the optimal configuration from test results
        
        Args:
            results: List of test results from test_threshold_configurations
            
        Returns:
            Best configuration with detailed analysis
        """
        if not results:
            return {"error": "No results to analyze"}
        
        # Best result is already first (sorted by F1-score)
        best = results[0]
        
        analysis = {
            'optimal_config': best['configuration'],
            'performance': best['metrics'],
            'detected_cards': best['detected_cards'],
            'improvement_analysis': self._analyze_improvements(results),
            'recommendations': self._generate_recommendations(best)
        }
        
        return analysis
    
    def _analyze_improvements(self, results: List[Dict]) -> Dict:
        """Analyze improvement patterns across configurations"""
        if len(results) < 2:
            return {}
        
        best = results[0]
        worst = results[-1]
        
        return {
            'best_f1': best['metrics'].get('f1_score', 0),
            'worst_f1': worst['metrics'].get('f1_score', 0),
            'improvement': best['metrics'].get('f1_score', 0) - worst['metrics'].get('f1_score', 0),
            'best_config_name': best['configuration']['name'],
            'suit_confusion_reduction': (
                worst['metrics']['suit_confusion']['count'] - 
                best['metrics']['suit_confusion']['count']
            )
        }
    
    def _generate_recommendations(self, best_result: Dict) -> List[str]:
        """Generate actionable recommendations based on best result"""
        recommendations = []
        
        config = best_result['configuration']
        metrics = best_result['metrics']
        
        # Configuration recommendations
        recommendations.append(f"Use confidence_threshold: {config['conf']}")
        recommendations.append(f"Use iou_threshold: {config['iou']}")
        recommendations.append(f"Use input_size: {config['imgsz']}")
        
        # Performance-based recommendations
        if metrics.get('f1_score', 0) > 0.8:
            recommendations.append("✅ Excellent performance achieved!")
        elif metrics.get('suit_confusion', {}).get('count', 0) > 0:
            recommendations.append("⚠️ Still has suit confusion - consider domain-specific fine-tuning")
        
        if metrics.get('false_negatives', 0) > 2:
            recommendations.append("📊 Consider even lower confidence threshold to catch missing cards")
        
        if metrics.get('false_positives', 0) > 3:
            recommendations.append("🎯 Consider higher confidence threshold to reduce false positives")
        
        return recommendations


def create_threshold_finder() -> OptimalThresholdFinder:
    """
    Factory function to create OptimalThresholdFinder instance
    
    Returns:
        Configured OptimalThresholdFinder instance
    """
    return OptimalThresholdFinder()