"""
Test Claude Opus's domain gap fixes
Run this to test his exact preprocessing and threshold optimization
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from ml.card_detector import create_card_detector
from ml.opus_threshold_tester import test_threshold_configurations, find_optimal_configuration
from ml.opus_preprocessing import preprocess_poker_image
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_opus_fixes(image_path: str):
    """
    Test Claude Opus's exact fixes for domain gap
    
    Args:
        image_path: Path to your poker table image
    """
    # Your actual cards as identified by Claude Opus
    known_cards = ["As", "Qs", "4h", "Ts", "Ad", "Ks", "Js"]  # Note: "Ts" for 10 of spades
    
    print("🃏 TESTING CLAUDE OPUS'S DOMAIN GAP FIXES")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Actual cards: {', '.join(known_cards)}")
    print("=" * 60)
    
    try:
        # Load test image
        image = Image.open(image_path)
        print(f"✅ Loaded image: {image.size}")
        
        # Create detector
        detector = create_card_detector()
        model_path = "ml/models/poker_cards_best.pt"
        success = detector.load_model(model_path)
        if not success:
            print("❌ Failed to load model")
            return
        
        print("✅ Model loaded successfully")
        
        # Test 1: Current settings with Opus preprocessing
        print("\n🧪 TEST 1: Current settings with Opus preprocessing")
        try:
            detections, inference_time, report = detector.detect_cards_from_pil_poker(image)
            detected_cards = [det.card_name for det in detections]
            
            print(f"Current result: {', '.join(detected_cards)}")
            print(f"Detected: {len(detected_cards)} cards")
            
            # Calculate accuracy
            actual_set = set(known_cards)
            detected_set = set(detected_cards)
            correct = len(actual_set & detected_set)
            accuracy = correct / len(actual_set)
            
            print(f"Accuracy: {correct}/{len(actual_set)} = {accuracy:.1%}")
            
        except Exception as e:
            print(f"❌ Test 1 failed: {e}")
        
        # Test 2: Opus threshold optimization
        print("\n🧪 TEST 2: Opus threshold optimization")
        print("This will test multiple configurations to find optimal settings...")
        
        try:
            # Run Opus's threshold testing (simplified version)
            best_configs = [
                {"conf": 0.15, "iou": 0.30, "size": 1024},
                {"conf": 0.20, "iou": 0.35, "size": 1024},
                {"conf": 0.25, "iou": 0.40, "size": 1024},
                {"conf": 0.22, "iou": 0.38, "size": 1024},  # Opus recommended
                {"conf": 0.20, "iou": 0.35, "size": 1280},  # Multi-scale
            ]
            
            best_result = None
            best_accuracy = 0
            
            for config in best_configs:
                try:
                    # Apply Opus preprocessing
                    processed_img = preprocess_poker_image(image)
                    
                    # Test this configuration
                    results = detector.model(processed_img, 
                                           conf=config["conf"], 
                                           iou=config["iou"], 
                                           imgsz=config["size"])
                    
                    if results and len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes
                        detected_cards = []
                        
                        for i in range(len(boxes)):
                            cls_id = int(boxes.cls[i].cpu().numpy())
                            confidence = float(boxes.conf[i].cpu().numpy())
                            
                            # Get card name using class mapping from detector
                            card_name = detector.class_names.get(cls_id, f"unknown_{cls_id}")
                            detected_cards.append(card_name)
                        
                        # Remove duplicates (poker rule)
                        unique_cards = list(set(detected_cards))
                        
                        # Calculate accuracy
                        actual_set = set(known_cards)
                        detected_set = set(unique_cards)
                        correct = len(actual_set & detected_set)
                        accuracy = correct / len(actual_set)
                        
                        print(f"Config conf={config['conf']:.2f}, iou={config['iou']:.2f}, size={config['size']}: "
                              f"{correct}/{len(actual_set)} = {accuracy:.1%}")
                        print(f"  Cards: {', '.join(unique_cards)}")
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_result = {
                                'config': config,
                                'accuracy': accuracy,
                                'detected_cards': unique_cards,
                                'correct': correct
                            }
                            
                except Exception as e:
                    print(f"Config failed: {e}")
                    continue
            
            # Show best result
            if best_result:
                print(f"\n🏆 BEST RESULT:")
                print(f"Configuration: {best_result['config']}")
                print(f"Accuracy: {best_result['correct']}/{len(known_cards)} = {best_result['accuracy']:.1%}")
                print(f"Cards detected: {', '.join(best_result['detected_cards'])}")
                
                print(f"\n💡 RECOMMENDED SETTINGS:")
                print(f"Update your model_config.yaml with:")
                print(f"  confidence_threshold: {best_result['config']['conf']}")
                print(f"  iou_threshold: {best_result['config']['iou']}")
                print(f"  input_size: {best_result['config']['size']}")
                
                if best_result['accuracy'] >= 0.7:
                    print("✅ Good accuracy achieved with Opus fixes!")
                else:
                    print("⚠️ Still need improvement - consider domain-specific fine-tuning")
            
        except Exception as e:
            print(f"❌ Threshold optimization failed: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_opus_fixes(image_path)
    else:
        print("Usage: python test_opus_fix.py your_poker_image.jpg")
        print("")
        print("This script will:")
        print("1. Test Claude Opus's preprocessing fixes")
        print("2. Find optimal threshold settings")
        print("3. Show expected vs actual accuracy improvements")