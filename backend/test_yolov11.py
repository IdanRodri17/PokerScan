#!/usr/bin/env python3
"""
Test script to verify YOLOv11 model integration
Run this with: python test_yolov11.py
"""

import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Testing YOLOv11 Model Integration")
print("=" * 60)

try:
    print("\n1. Importing ImageProcessor...")
    from services.image_processor import ImageProcessor
    print("   ✅ ImageProcessor imported successfully")

    print("\n2. Initializing ImageProcessor...")
    processor = ImageProcessor()
    print("   ✅ ImageProcessor initialized")

    print("\n3. Getting model status...")
    status = processor.get_model_status()

    print("\n" + "=" * 60)
    print("MODEL STATUS")
    print("=" * 60)

    print(f"\n ML Available: {status.get('ml_available', False)}")
    print(f" ML Enabled: {status.get('ml_enabled', False)}")
    print(f" Model Loaded: {status.get('model_loaded', False)}")
    print(f" Using Mock Detection: {status.get('using_mock_detection', True)}")

    if status.get('model_loaded'):
        print(f" Model Device: {status.get('model_device', 'unknown')}")

        # Try to get model info
        if hasattr(processor, 'card_detector') and processor.card_detector.model:
            model_info = processor.card_detector.model
            print(f"\n Model Type: {type(model_info).__name__}")

            # Check class names
            if hasattr(processor.card_detector, 'MODEL_CLASS_NAMES'):
                num_classes = len(processor.card_detector.MODEL_CLASS_NAMES)
                print(f" Number of Classes: {num_classes}")
                if num_classes > 0:
                    print(f" First 5 Classes: {list(processor.card_detector.MODEL_CLASS_NAMES.values())[:5]}")

    print("\n" + "=" * 60)

    if status.get('ml_enabled') and status.get('model_loaded'):
        print("\n✅ SUCCESS: YOLOv11 model is loaded and ready!")
    elif status.get('ml_available'):
        print("\n⚠️  WARNING: ML available but model not loaded")
    else:
        print("\n❌ ERROR: ML components not available")

    print("=" * 60)

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nMake sure you have installed all dependencies:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Test completed!\n")
