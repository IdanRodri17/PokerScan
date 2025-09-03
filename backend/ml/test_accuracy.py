"""
Accuracy Testing Script for Poker Card Detection

This script helps test and optimize detection accuracy with different configurations
as recommended by Claude Opus for troubleshooting detection issues.
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from ml.card_detector import create_card_detector
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_accuracy_improvements(image_path: str, actual_cards: list):
    """
    Test different detection configurations to find optimal settings
    
    Args:
        image_path: Path to test image
        actual_cards: List of actual card names in the image
    """
    print(f"🃏 Testing Accuracy Improvements")
    print(f"Image: {image_path}")
    print(f"Actual cards: {actual_cards}")
    print("=" * 60)
    
    # Load test image
    try:
        image = Image.open(image_path)
        print(f"✅ Loaded image: {image.size}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # Test configurations as suggested by Claude Opus
    configs_to_test = [
        {
            "name": "Original Settings",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "input_size": 832
        },
        {
            "name": "Lower Confidence",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "input_size": 832
        },
        {
            "name": "Lower IOU",
            "confidence_threshold": 0.35,
            "iou_threshold": 0.40,
            "input_size": 832
        },
        {
            "name": "Balanced (Claude Opus Recommended)",
            "confidence_threshold": 0.30,
            "iou_threshold": 0.50,
            "input_size": 832
        },
        {
            "name": "Very Low Confidence",
            "confidence_threshold": 0.20,
            "iou_threshold": 0.45,
            "input_size": 832
        },
        {
            "name": "Large Input Size",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.40,
            "input_size": 1024
        },
        {
            "name": "Max Input Size",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.40,
            "input_size": 1280
        }
    ]
    
    best_config = None
    best_f1_score = 0
    
    for config in configs_to_test:
        print(f"\n🧪 Testing: {config['name']}")
        print(f"   Confidence: {config['confidence_threshold']}")
        print(f"   IOU: {config['iou_threshold']}")
        print(f"   Input Size: {config['input_size']}")
        
        try:
            # Create detector with test configuration
            detector = create_card_detector()
            
            # Temporarily update config for this test
            detector.config['model']['confidence_threshold'] = config['confidence_threshold']
            detector.config['model']['iou_threshold'] = config['iou_threshold']
            detector.config['model']['input_size'] = config['input_size']
            
            # Load model if not already loaded
            if detector.model is None:
                model_path = "ml/models/poker_cards_best.pt"
                success = detector.load_model(model_path)
                if not success:
                    print(f"   ❌ Failed to load model")
                    continue
            
            # Run detection with enhancement
            detections, inference_time, report = detector.detect_cards_from_pil_poker(image)
            
            # Extract detected card names
            detected_cards = [det.card_name for det in detections]
            
            # Calculate accuracy metrics
            actual_set = set(actual_cards)
            detected_set = set(detected_cards)
            
            true_positives = len(actual_set & detected_set)
            false_positives = len(detected_set - actual_set)
            false_negatives = len(actual_set - detected_set)
            
            precision = true_positives / len(detected_set) if detected_set else 0
            recall = true_positives / len(actual_set) if actual_set else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"   📊 Results:")
            print(f"      Detected: {len(detected_cards)} cards")
            print(f"      Correct: {true_positives}/{len(actual_cards)}")
            print(f"      Precision: {precision:.3f}")
            print(f"      Recall: {recall:.3f}")
            print(f"      F1-Score: {f1_score:.3f}")
            print(f"      Inference: {inference_time*1000:.1f}ms")
            
            if detected_cards:
                print(f"      Cards: {', '.join(detected_cards)}")
            
            # Track best configuration
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_config = config.copy()
                best_config['results'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'detected_cards': detected_cards,
                    'inference_time': inference_time
                }
            
            # Show mismatches for debugging
            if false_positives > 0:
                missed = detected_set - actual_set
                print(f"      False Positives: {', '.join(missed)}")
            if false_negatives > 0:
                missed = actual_set - detected_set
                print(f"      Missed Cards: {', '.join(missed)}")
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            continue
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"🏆 BEST CONFIGURATION RESULTS")
    print(f"=" * 60)
    
    if best_config:
        print(f"Configuration: {best_config['name']}")
        print(f"Settings: conf={best_config['confidence_threshold']}, "
              f"iou={best_config['iou_threshold']}, "
              f"size={best_config['input_size']}")
        print(f"F1-Score: {best_config['results']['f1_score']:.3f}")
        print(f"Precision: {best_config['results']['precision']:.3f}")
        print(f"Recall: {best_config['results']['recall']:.3f}")
        print(f"Detected Cards: {', '.join(best_config['results']['detected_cards'])}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"Update your model_config.yaml with these settings:")
        print(f"  confidence_threshold: {best_config['confidence_threshold']}")
        print(f"  iou_threshold: {best_config['iou_threshold']}")
        print(f"  input_size: {best_config['input_size']}")
        
    else:
        print("❌ No successful configurations found")
        print("Check your model file and image path")


def quick_test():
    """Quick test with sample data"""
    print("🚀 Running Quick Accuracy Test")
    print("This tests your current configuration")
    
    # You can modify these for your specific test case
    test_image_path = "test_poker_image.jpg"  # Update this path
    actual_cards = ["As", "Qs", "4h", "Ts", "4d", "Ks", "Js"]  # Update with your actual cards
    
    # Check if test image exists
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("Please update the test_image_path in the script")
        return
    
    test_accuracy_improvements(test_image_path, actual_cards)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        # Command line usage: python test_accuracy.py image_path "As,Qs,4h,Ts,4d,Ks,Js"
        image_path = sys.argv[1]
        actual_cards = sys.argv[2].split(',')
        test_accuracy_improvements(image_path, actual_cards)
    else:
        print("Usage examples:")
        print("1. python ml/test_accuracy.py")
        print("2. python ml/test_accuracy.py image.jpg \"As,Qs,4h,Ts,4d,Ks,Js\"")
        print("")
        quick_test()