"""
Run comprehensive diagnosis on your poker card detection model
This will identify the exact issue and provide targeted fixes
"""

from diagnosis import comprehensive_model_diagnosis
from ultralytics import YOLO
import torch
import sys

def main():
    print("🔬 POKER CARD MODEL DIAGNOSTIC TOOL")
    print("=" * 50)
    
    try:
        # Fix PyTorch 2.6+ compatibility issue
        print("Fixing PyTorch 2.6+ compatibility...")
        
        # Add safe globals for YOLO components
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'ultralytics.nn.tasks.SegmentationModel', 
                'ultralytics.nn.tasks.ClassificationModel',
                'ultralytics.nn.tasks.PoseModel',
                'collections.OrderedDict',
                'torch.nn.modules.container.ModuleList',
                'torch.nn.modules.container.Sequential'
            ])
        
        # Patch torch.load temporarily
        original_torch_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_load
        
        print("✅ PyTorch compatibility fix applied")
        
        # Load your model
        print("Loading model...")
        model = YOLO('ml/models/poker_cards_best.pt')
        print("✅ Model loaded successfully")
        
        # Restore original torch.load
        torch.load = original_torch_load
        
        # Your known cards from the poker image (based on previous test output)
        known_cards = ["As", "Qs", "4h", "Ts", "Ad", "Ks", "Js"]  # Expected cards
        
        # Run comprehensive diagnosis
        print("\nStarting diagnostic analysis...")
        success = comprehensive_model_diagnosis(
            model=model, 
            test_image_path='poker_image.jpg', 
            known_cards=known_cards
        )
        
        if success:
            print("\n🎯 DIAGNOSIS COMPLETED SUCCESSFULLY!")
            print("\nNEXT STEPS:")
            print("1. Share this output with me")
            print("2. I'll provide the exact fix needed")
            print("3. Your model will detect cards correctly!")
        else:
            print("\n❌ DIAGNOSIS ENCOUNTERED ISSUES")
            print("Please check if:")
            print("- Model file 'ml/models/poker_cards_best.pt' exists")
            print("- Image file 'poker_image.jpg' exists")
            print("- All dependencies are installed")
        
    except FileNotFoundError as e:
        print(f"\n❌ FILE NOT FOUND: {e}")
        print("\nPlease ensure:")
        print("- Model file: ml/models/poker_cards_best.pt")
        print("- Image file: poker_image.jpg")
        print("- Both files are in the backend directory")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("\nPlease share this error message for debugging assistance")

if __name__ == "__main__":
    main()