"""
Systematic debugging script to identify the root cause of detection issues
Run this to understand what's really happening with your model
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def comprehensive_model_diagnosis(model, test_image_path, known_cards=None):
    """
    Complete diagnostic analysis of your YOLOv8 model
    This will tell us exactly what's wrong and how to fix it
    """
    
    print("🔍 COMPREHENSIVE MODEL DIAGNOSIS")
    print("=" * 60)
    
    # Load test image
    test_image = Image.open(test_image_path)
    print(f"📸 Test image loaded: {test_image.size}")
    
    # 1. VERIFY MODEL LOADING
    print("\n1️⃣ MODEL LOADING VERIFICATION")
    print("-" * 40)
    
    try:
        # Check if model loaded correctly
        print(f"✅ Model type: {type(model)}")
        print(f"✅ Model device: {model.device if hasattr(model, 'device') else 'Unknown'}")
        
        # Check model classes
        if hasattr(model, 'names'):
            print(f"✅ Model has {len(model.names)} classes")
            print(f"✅ First 10 classes: {list(model.names.values())[:10]}")
            
            # Verify specific cards we're looking for
            if known_cards:
                print(f"\n🎯 Checking for our target cards:")
                for card in known_cards:
                    if card in model.names.values():
                        print(f"   ✅ {card} found in model classes")
                    else:
                        print(f"   ❌ {card} NOT found in model classes")
        else:
            print("❌ Model.names not available")
            
    except Exception as e:
        print(f"❌ Model loading issue: {e}")
        return False
    
    # 2. RAW DETECTION TEST
    print("\n2️⃣ RAW DETECTION TEST")
    print("-" * 40)
    
    try:
        # Test with very low thresholds to see what model actually detects
        raw_results = model(test_image, conf=0.01, iou=0.1, verbose=False)
        
        if raw_results and len(raw_results) > 0:
            boxes = raw_results[0].boxes
            if boxes is not None:
                print(f"✅ Raw detections found: {len(boxes)}")
                
                # Show all detections with confidences
                detections = []
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())
                    class_name = model.names.get(cls_id, f"class_{cls_id}")
                    detections.append((class_name, conf))
                
                # Sort by confidence
                detections.sort(key=lambda x: x[1], reverse=True)
                
                print("🎯 Top 15 detections (class, confidence):")
                for card, conf in detections[:15]:
                    print(f"   {card}: {conf:.3f}")
                    
                # Analysis
                print(f"\n📊 Detection Analysis:")
                print(f"   • Total detections: {len(detections)}")
                print(f"   • Unique classes: {len(set([d[0] for d in detections]))}")
                print(f"   • Confidence range: {min([d[1] for d in detections]):.3f} - {max([d[1] for d in detections]):.3f}")
                
            else:
                print("❌ No boxes found in results")
        else:
            print("❌ No detections found at all")
            
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return False
    
    # 3. CLASS MAPPING VERIFICATION
    print("\n3️⃣ CLASS MAPPING VERIFICATION")
    print("-" * 40)
    
    # Expected class mapping from your config
    expected_mapping = {
        0: "As", 1: "2s", 2: "3s", 3: "4s", 4: "5s", 5: "6s", 6: "7s", 7: "8s", 8: "9s", 9: "Ts", 10: "Js", 11: "Qs", 12: "Ks",
        13: "Ah", 14: "2h", 15: "3h", 16: "4h", 17: "5h", 18: "6h", 19: "7h", 20: "8h", 21: "9h", 22: "Th", 23: "Jh", 24: "Qh", 25: "Kh",
        26: "Ad", 27: "2d", 28: "3d", 29: "4d", 30: "5d", 31: "6d", 32: "7d", 33: "8d", 34: "9d", 35: "Td", 36: "Jd", 37: "Qd", 38: "Kd",
        39: "Ac", 40: "2c", 41: "3c", 42: "4c", 43: "5c", 44: "6c", 45: "7c", 46: "8c", 47: "9c", 48: "Tc", 49: "Jc", 50: "Qc", 51: "Kc"
    }
    
    if hasattr(model, 'names'):
        mapping_correct = True
        for class_id, expected_name in expected_mapping.items():
            actual_name = model.names.get(class_id, "MISSING")
            if actual_name != expected_name:
                print(f"❌ Class {class_id}: Expected '{expected_name}', got '{actual_name}'")
                mapping_correct = False
        
        if mapping_correct:
            print("✅ Class mapping is correct")
        else:
            print("❌ Class mapping mismatch found!")
    
    # 4. CONFIDENCE ANALYSIS
    print("\n4️⃣ CONFIDENCE DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    if known_cards:
        print(f"🎯 Looking for these cards: {known_cards}")
        
        # Test different confidence thresholds
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        for thresh in thresholds:
            results = model(test_image, conf=thresh, iou=0.45, verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                detected_cards = []
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    class_name = model.names.get(cls_id, f"class_{cls_id}")
                    detected_cards.append(class_name)
                
                matches = len(set(known_cards).intersection(set(detected_cards)))
                print(f"   Conf {thresh:.2f}: {len(detected_cards)} detections, {matches}/{len(known_cards)} correct cards found")
    
    # 5. IMAGE PREPROCESSING TEST
    print("\n5️⃣ IMAGE PREPROCESSING IMPACT")
    print("-" * 40)
    
    # Test different preprocessing approaches
    preprocessing_tests = [
        ("Original", lambda x: x),
        ("CLAHE Enhanced", apply_clahe),
        ("Green Corrected", correct_green_bias),
        ("Full Enhancement", full_preprocessing)
    ]
    
    for name, preprocess_func in preprocessing_tests:
        try:
            processed_img = preprocess_func(np.array(test_image))
            results = model(processed_img, conf=0.25, iou=0.45, verbose=False)
            
            detection_count = 0
            if results and len(results) > 0 and results[0].boxes is not None:
                detection_count = len(results[0].boxes)
            
            print(f"   {name}: {detection_count} detections")
            
        except Exception as e:
            print(f"   {name}: Failed - {e}")
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")
    
    return True

def apply_clahe(image):
    """Apply CLAHE enhancement"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image

def correct_green_bias(image):
    """Correct green bias from poker felt"""
    if len(image.shape) == 3:
        # Reduce green channel slightly
        corrected = image.copy()
        corrected[:,:,1] = np.clip(corrected[:,:,1] * 0.9, 0, 255).astype(np.uint8)
        return corrected
    return image

def full_preprocessing(image):
    """Apply full preprocessing pipeline"""
    # Convert to BGR for OpenCV
    if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        # Convert back to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return image

def create_simple_test_case():
    """
    Create a simple test to verify basic functionality
    """
    print("🧪 SIMPLE TEST CASE")
    print("-" * 40)
    
    # Instructions for user
    test_instructions = """
    STEP-BY-STEP DIAGNOSIS:
    
    1. Save this script as 'diagnosis.py' in your project folder
    
    2. Run with your model and test image:
       ```python
       from diagnosis import comprehensive_model_diagnosis
       from ultralytics import YOLO
       
       # Load your model
       model = YOLO('ml/models/poker_cards_best.pt')
       
       # Your known cards from the poker image
       known_cards = ["As", "Qs", "4h", "Ts", "Ad", "Ks", "Js"]
       
       # Run diagnosis
       comprehensive_model_diagnosis(model, 'poker_image.jpg', known_cards)
       ```
    
    3. This will tell us:
       ✅ If your model is loaded correctly
       ✅ If class mapping is correct
       ✅ What the model actually detects (even at low confidence)
       ✅ Which preprocessing helps
       ✅ Optimal confidence thresholds
    
    4. Share the output with me - I'll give you the exact fix needed!
    """
    
    return test_instructions

# Example usage and next steps
if __name__ == "__main__":
    print(create_simple_test_case())