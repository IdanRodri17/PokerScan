"""
Claude Opus's Optimal threshold configuration and testing for poker card detection
"""

import itertools
import numpy as np
from collections import Counter
from .opus_preprocessing import preprocess_poker_image

def test_threshold_configurations(model, test_image, known_cards=None):
    """
    Test different threshold combinations to find optimal settings
    
    Args:
        model: YOLOv8 model
        test_image: Your poker table image
        known_cards: List of actual cards in image (for evaluation)
        
    Returns:
        Dict of results for each configuration
    """
    
    # Configuration matrix for systematic testing
    confidence_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    iou_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    input_sizes = [832, 1024, 1280]
    
    results = []
    
    print("Testing threshold configurations...")
    print("=" * 60)
    
    for conf, iou, size in itertools.product(confidence_thresholds, iou_thresholds, input_sizes):
        try:
            # Apply enhanced preprocessing
            processed_img = preprocess_poker_image(test_image)
            
            # Run detection
            detections = model(processed_img, conf=conf, iou=iou, imgsz=size)
            
            if detections and len(detections) > 0:
                boxes = detections[0].boxes
                detected_cards = []
                confidences = []
                
                if boxes is not None:
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        # Map class ID to card name (using your class mapping)
                        card_name = get_card_name_from_class_id(cls_id)
                        detected_cards.append(card_name)
                        confidences.append(confidence)
                
                # Calculate metrics if known cards provided
                metrics = {}
                if known_cards:
                    metrics = calculate_detection_metrics(detected_cards, known_cards)
                
                result = {
                    'conf': conf,
                    'iou': iou, 
                    'size': size,
                    'detections': len(detected_cards),
                    'cards': detected_cards,
                    'avg_confidence': np.mean(confidences) if confidences else 0,
                    'unique_cards': len(set(detected_cards)),
                    'duplicates': len(detected_cards) - len(set(detected_cards)),
                    'metrics': metrics
                }
                
                results.append(result)
                
                # Print result
                print(f"Conf:{conf:4.2f} IoU:{iou:4.2f} Size:{size:4d} → "
                      f"{len(detected_cards):2d} cards, "
                      f"{len(set(detected_cards)):2d} unique, "
                      f"avg_conf:{np.mean(confidences):.3f}" + 
                      (f", F1:{metrics.get('f1', 0):.3f}" if known_cards else ""))
                
        except Exception as e:
            print(f"Failed conf:{conf}, iou:{iou}, size:{size} - {e}")
    
    return results

def calculate_detection_metrics(detected_cards, actual_cards):
    """Calculate precision, recall, F1 for detection results"""
    detected_set = set(detected_cards)
    actual_set = set(actual_cards)
    
    true_positives = len(detected_set.intersection(actual_set))
    false_positives = len(detected_set - actual_set)
    false_negatives = len(actual_set - detected_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def get_card_name_from_class_id(class_id):
    """Convert class ID to card name using your mapping"""
    # Your class mapping from config
    class_mapping = {
        0: "As", 1: "2s", 2: "3s", 3: "4s", 4: "5s", 5: "6s", 6: "7s", 7: "8s", 8: "9s", 9: "Ts", 10: "Js", 11: "Qs", 12: "Ks",
        13: "Ah", 14: "2h", 15: "3h", 16: "4h", 17: "5h", 18: "6h", 19: "7h", 20: "8h", 21: "9h", 22: "Th", 23: "Jh", 24: "Qh", 25: "Kh",
        26: "Ad", 27: "2d", 28: "3d", 29: "4d", 30: "5d", 31: "6d", 32: "7d", 33: "8d", 34: "9d", 35: "Td", 36: "Jd", 37: "Qd", 38: "Kd",
        39: "Ac", 40: "2c", 41: "3c", 42: "4c", 43: "5c", 44: "6c", 45: "7c", 46: "8c", 47: "9c", 48: "Tc", 49: "Jc", 50: "Qc", 51: "Kc"
    }
    return class_mapping.get(class_id, f"unknown_{class_id}")

def find_optimal_configuration(results, priority='f1'):
    """Find the best configuration based on specified metric"""
    if not results:
        return None
        
    # Filter results with metrics (if known cards were provided)
    results_with_metrics = [r for r in results if 'metrics' in r and r['metrics']]
    
    if results_with_metrics:
        best_result = max(results_with_metrics, key=lambda x: x['metrics'].get(priority, 0))
        print(f"\n🎯 OPTIMAL CONFIGURATION (based on {priority}):")
        print(f"   Confidence: {best_result['conf']}")
        print(f"   IoU: {best_result['iou']}")  
        print(f"   Input Size: {best_result['size']}")
        print(f"   F1 Score: {best_result['metrics']['f1']:.3f}")
        print(f"   Precision: {best_result['metrics']['precision']:.3f}")
        print(f"   Recall: {best_result['metrics']['recall']:.3f}")
        return best_result
    else:
        # Fallback: balance detection count and confidence
        best_result = max(results, key=lambda x: x['unique_cards'] * x['avg_confidence'])
        print(f"\n🎯 BEST CONFIGURATION (based on unique detections × confidence):")
        print(f"   Confidence: {best_result['conf']}")
        print(f"   IoU: {best_result['iou']}")
        print(f"   Input Size: {best_result['size']}")
        print(f"   Unique Cards: {best_result['unique_cards']}")
        print(f"   Avg Confidence: {best_result['avg_confidence']:.3f}")
        return best_result