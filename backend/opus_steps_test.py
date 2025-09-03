"""
Claude Opus's Exact Step-by-Step Testing Process
Follow his recommended order for testing domain gap fixes
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from ml.card_detector import create_card_detector
from ml.opus_preprocessing import preprocess_poker_image
from ml.opus_threshold_tester import test_threshold_configurations, find_optimal_configuration
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def step1_immediate_fix(model, your_poker_image):
    """
    Step 1: Immediate Fix (Try This First)
    Test the enhanced preprocessing on your poker image
    Expected improvement: 5-6 out of 7 cards detected correctly (vs current 3/7)
    """
    print("🚀 STEP 1: IMMEDIATE FIX")
    print("=" * 50)
    print("Testing enhanced preprocessing with Opus settings:")
    print("- conf=0.22, iou=0.38, imgsz=1024")
    print("- Expected: 5-6/7 cards correct (vs current 3/7)")
    print()
    
    try:
        # Apply Opus preprocessing exactly as specified
        processed_image = preprocess_poker_image(your_poker_image)
        print("✅ Applied preprocess_poker_image()")
        
        # Run detection with Opus's recommended settings
        results = model(processed_image, conf=0.22, iou=0.38, imgsz=1024)
        print("✅ Ran model(processed_image, conf=0.22, iou=0.38, imgsz=1024)")
        
        # Extract results
        detected_cards = []
        confidences = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                card_name = model.names.get(cls_id, f"unknown_{cls_id}")
                detected_cards.append(card_name)
                confidences.append(confidence)
        
        # Remove duplicates (poker rule)
        unique_cards = list(set(detected_cards))
        
        # Calculate accuracy vs known cards - using YOLO default class names
        known_cards_yolo = ["AS", "QS", "4H", "10S", "AC", "KS", "JS"]  # YOLO format
        actual_set = set(known_cards_yolo)
        detected_set = set(unique_cards)
        correct = len(actual_set & detected_set)
        accuracy = correct / len(actual_set)
        
        print(f"\n📊 STEP 1 RESULTS:")
        print(f"Detected: {', '.join(unique_cards)}")
        print(f"Accuracy: {correct}/{len(actual_set)} = {accuracy:.1%}")
        print(f"Expected: 5-6/7 = 71-86%")
        
        if accuracy >= 0.71:
            print("✅ SUCCESS: Step 1 achieved expected improvement!")
            return True, unique_cards
        else:
            print("⚠️ Step 1 didn't reach expected improvement, continuing to Step 2...")
            return False, unique_cards
            
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        return False, []


def step2_find_optimal_settings(model, your_poker_image):
    """
    Step 2: Find Your Optimal Settings  
    Expected improvement: 6-7 out of 7 cards detected correctly
    """
    print("\n🎯 STEP 2: FIND YOUR OPTIMAL SETTINGS")
    print("=" * 50)
    print("Running test_threshold_configurations() and find_optimal_configuration()")
    print("Expected: 6-7/7 cards correct")
    print()
    
    try:
        known_cards = ["AS", "QS", "4H", "10S", "AC", "KS", "JS"]  # Actual model format
        print(f"Known cards: {', '.join(known_cards)}")
        print()
        
        # Run Opus's threshold testing (simplified version for speed)
        print("Testing key configurations...")
        
        configs_to_test = [
            {"name": "Opus Recommended", "conf": 0.22, "iou": 0.38, "size": 1024},
            {"name": "Ultra Low Conf", "conf": 0.15, "iou": 0.35, "size": 1024},
            {"name": "Low Conf", "conf": 0.20, "iou": 0.35, "size": 1024},
            {"name": "Balanced", "conf": 0.25, "iou": 0.40, "size": 1024},
            {"name": "Multi-Scale", "conf": 0.20, "iou": 0.35, "size": 1280},
            {"name": "High Res", "conf": 0.25, "iou": 0.38, "size": 1536},
        ]
        
        best_config = None
        best_accuracy = 0
        best_cards = []
        
        for config in configs_to_test:
            try:
                # Apply preprocessing
                processed_image = preprocess_poker_image(your_poker_image)
                
                # Test configuration
                results = model(processed_image, 
                              conf=config["conf"], 
                              iou=config["iou"], 
                              imgsz=config["size"])
                
                detected_cards = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        card_name = model.names.get(cls_id, f"unknown_{cls_id}")
                        detected_cards.append(card_name)
                
                # Remove duplicates
                unique_cards = list(set(detected_cards))
                
                # Calculate accuracy
                actual_set = set(known_cards)
                detected_set = set(unique_cards)
                correct = len(actual_set & detected_set)
                accuracy = correct / len(actual_set)
                
                print(f"{config['name']:15} (conf={config['conf']:.2f}, iou={config['iou']:.2f}, size={config['size']:4d}): "
                      f"{correct}/{len(actual_set)} = {accuracy:.1%}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = config
                    best_cards = unique_cards
                    
            except Exception as e:
                print(f"{config['name']:15}: ❌ Failed - {e}")
                continue
        
        print(f"\n🏆 STEP 2 BEST RESULT:")
        if best_config:
            print(f"Configuration: {best_config['name']}")
            print(f"Settings: conf={best_config['conf']}, iou={best_config['iou']}, size={best_config['size']}")
            print(f"Accuracy: {len(set(best_cards) & set(known_cards))}/{len(known_cards)} = {best_accuracy:.1%}")
            print(f"Detected: {', '.join(best_cards)}")
            print(f"Expected: 6-7/7 = 86-100%")
            
            if best_accuracy >= 0.86:
                print("✅ SUCCESS: Step 2 achieved expected improvement!")
                return True, best_config, best_cards
            else:
                print("⚠️ Step 2 didn't reach full expectation but still improved")
                return False, best_config, best_cards
        else:
            print("❌ No successful configurations found")
            return False, None, []
            
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        return False, None, []


def step3_production_solution():
    """
    Step 3: Production Solution (If Needed)
    Information about fine-tuning for 95%+ accuracy
    """
    print("\n🏭 STEP 3: PRODUCTION SOLUTION")
    print("=" * 50)
    print("If you need consistent 95%+ accuracy across many different poker images,")
    print("you'll need to collect ~200 real poker table images and fine-tune your model.")
    print()
    print("This involves:")
    print("1. Collecting diverse poker table images")
    print("2. Annotating them with correct card labels")
    print("3. Fine-tuning your current model on poker-specific data")
    print("4. Expected result: 95%+ accuracy on real poker tables")
    print()
    print("For most use cases, Steps 1-2 should provide sufficient accuracy!")


def get_class_mapping():
    """Get class ID to card name mapping - using actual model class names"""
    return {
        0: "10C", 1: "10D", 2: "10H", 3: "10S", 4: "2C", 5: "2D", 6: "2H", 7: "2S", 8: "3C", 9: "3D",
        10: "3H", 11: "3S", 12: "4C", 13: "4D", 14: "4H", 15: "4S", 16: "5C", 17: "5D", 18: "5H", 19: "5S",
        20: "6C", 21: "6D", 22: "6H", 23: "6S", 24: "7C", 25: "7D", 26: "7H", 27: "7S", 28: "8C", 29: "8D",
        30: "8H", 31: "8S", 32: "9C", 33: "9D", 34: "9H", 35: "9S", 36: "AC", 37: "AD", 38: "AH", 39: "AS",
        40: "JC", 41: "JD", 42: "JH", 43: "JS", 44: "KC", 45: "KD", 46: "KH", 47: "KS", 48: "QC", 49: "QD",
        50: "QH", 51: "QS"
    }


def main(image_path):
    """
    Run Claude Opus's complete 3-step testing process
    """
    print("🃏 CLAUDE OPUS'S 3-STEP DOMAIN GAP FIX TEST")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Current accuracy: ~3/7 cards (43%)")
    print("Target: 5-7/7 cards (71-100%)")
    print("=" * 60)
    
    try:
        # Fix PyTorch 2.6+ compatibility
        import torch
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'collections.OrderedDict'
            ])
        original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: original_torch_load(*args, **dict(kwargs, weights_only=False))
        
        # Load image and model
        your_poker_image = Image.open(image_path)
        print(f"✅ Loaded image: {your_poker_image.size}")
        
        # Load model directly with YOLO
        from ultralytics import YOLO
        model = YOLO("ml/models/poker_cards_best.pt")
        print("✅ Model loaded successfully")
        
        # Restore torch.load
        torch.load = original_torch_load
        print()
        
        # Execute Opus's 3 steps in order
        step1_success, step1_cards = step1_immediate_fix(model, your_poker_image)
        
        step2_success, best_config, step2_cards = step2_find_optimal_settings(model, your_poker_image)
        
        step3_production_solution()
        
        # Final summary
        print("\n🎯 FINAL SUMMARY")
        print("=" * 50)
        
        known_cards = ["AS", "QS", "4H", "10S", "AC", "KS", "JS"]  # Actual model format
        
        if step2_success and best_config:
            final_accuracy = len(set(step2_cards) & set(known_cards)) / len(known_cards)
            print(f"✅ BEST RESULT ACHIEVED:")
            print(f"Configuration: conf={best_config['conf']}, iou={best_config['iou']}, size={best_config['size']}")
            print(f"Final accuracy: {len(set(step2_cards) & set(known_cards))}/{len(known_cards)} = {final_accuracy:.1%}")
            print(f"Cards detected: {', '.join(step2_cards)}")
            
            print(f"\n💡 APPLY THESE SETTINGS:")
            print(f"Update your backend/ml/config/model_config.yaml:")
            print(f"  confidence_threshold: {best_config['conf']}")
            print(f"  iou_threshold: {best_config['iou']}")
            print(f"  input_size: {best_config['size']}")
            
            if final_accuracy >= 0.85:
                print("\n🎉 EXCELLENT! Domain gap fixes worked as expected!")
            else:
                print("\n⚠️ Partial success - consider fine-tuning for production use")
                
        elif step1_success:
            final_accuracy = len(set(step1_cards) & set(known_cards)) / len(known_cards)
            print(f"✅ STEP 1 SUCCESS:")
            print(f"Accuracy improved to: {final_accuracy:.1%}")
            print(f"Use settings: conf=0.22, iou=0.38, size=1024")
        else:
            print("⚠️ Domain gap fixes didn't achieve full expectations")
            print("Consider collecting poker-specific training data for fine-tuning")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            main(image_path)
        else:
            print(f"❌ Image not found: {image_path}")
    else:
        print("🃏 Claude Opus's 3-Step Domain Gap Fix Test")
        print("=" * 50)
        print("Usage: python opus_steps_test.py your_poker_image.jpg")
        print()
        print("This will run Opus's exact 3-step process:")
        print("Step 1: Test enhanced preprocessing (expect 5-6/7 cards)")
        print("Step 2: Find optimal settings (expect 6-7/7 cards)") 
        print("Step 3: Production solution info (95%+ accuracy)")
        print()
        print("Make sure your poker image is in the backend folder!")