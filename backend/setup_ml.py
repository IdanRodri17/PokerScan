#!/usr/bin/env python3
"""
PokerVision ML Setup Script

This script helps set up the ML environment for poker card detection,
including dependencies, model preparation, and initial testing.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    logger.info(f"Python version check passed: {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing ML dependencies...")
    
    try:
        # Install main requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("✓ Main dependencies installed")
        
        # Install additional ML dependencies if needed
        additional_deps = [
            "kaggle",  # For dataset download
            "wandb",   # For experiment tracking (optional)
        ]
        
        for dep in additional_deps:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep
                ])
                logger.info(f"✓ {dep} installed")
            except subprocess.CalledProcessError:
                logger.warning(f"⚠ Failed to install {dep} (optional)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    logger.info("Creating directory structure...")
    
    directories = [
        "ml/models",
        "ml/training/runs",
        "data/raw",
        "data/processed",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")


def setup_kaggle_config():
    """Help setup Kaggle configuration"""
    logger.info("Setting up Kaggle configuration...")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        logger.info("✓ Kaggle configuration already exists")
        return True
    
    logger.warning("⚠ Kaggle configuration not found")
    logger.info("To download datasets from Kaggle:")
    logger.info("1. Go to https://www.kaggle.com/account")
    logger.info("2. Create a new API token")
    logger.info("3. Place kaggle.json in ~/.kaggle/")
    logger.info("4. chmod 600 ~/.kaggle/kaggle.json")
    
    return False


def test_ml_imports():
    """Test if ML libraries can be imported"""
    logger.info("Testing ML library imports...")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("ultralytics", "YOLOv8"),
        ("cv2", "OpenCV"),
        ("sklearn", "Scikit-learn"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
    ]
    
    all_passed = True
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            logger.info(f"✓ {display_name} import successful")
        except ImportError as e:
            logger.error(f"✗ {display_name} import failed: {e}")
            all_passed = False
    
    return all_passed


def test_ml_components():
    """Test ML components"""
    logger.info("Testing ML components...")
    
    try:
        from ml.card_detector import YOLOv8CardDetector
        from ml.spatial_analyzer import PokerSpatialAnalyzer
        
        # Test card detector initialization
        detector = YOLOv8CardDetector()
        logger.info("✓ Card detector component loaded")
        
        # Test spatial analyzer
        analyzer = PokerSpatialAnalyzer()
        logger.info("✓ Spatial analyzer component loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ML component test failed: {e}")
        return False


def test_api_endpoints():
    """Test if the API endpoints work"""
    logger.info("Testing API endpoints...")
    
    try:
        from services.image_processor import ImageProcessor
        
        # Initialize image processor
        processor = ImageProcessor()
        status = processor.get_model_status()
        
        logger.info(f"✓ Image processor initialized")
        logger.info(f"  ML enabled: {status['ml_enabled']}")
        logger.info(f"  Using mock detection: {status['using_mock_detection']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ API endpoint test failed: {e}")
        return False


def download_sample_model():
    """Download a sample YOLOv8 model for testing"""
    logger.info("Downloading sample YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        
        # Download YOLOv8n model for testing
        model = YOLO('yolov8n.pt')
        logger.info("✓ Sample YOLOv8n model downloaded")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download sample model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup PokerVision ML environment")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle setup")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    
    args = parser.parse_args()
    
    logger.info("🃏 PokerVision ML Setup")
    logger.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    if not args.test_only:
        # Create directories
        create_directories()
        
        # Install dependencies
        if not args.skip_deps and not install_dependencies():
            logger.error("Dependency installation failed")
            sys.exit(1)
        
        # Setup Kaggle
        if not args.skip_kaggle:
            setup_kaggle_config()
        
        # Download sample model
        download_sample_model()
    
    # Run tests
    logger.info("Running system tests...")
    
    tests_passed = 0
    total_tests = 3
    
    if test_ml_imports():
        tests_passed += 1
    
    if test_ml_components():
        tests_passed += 1
    
    if test_api_endpoints():
        tests_passed += 1
    
    logger.info("=" * 50)
    logger.info(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        logger.info("🎉 Setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Start the FastAPI server: python main.py")
        logger.info("2. Test with the frontend at http://localhost:3000")
        logger.info("3. To train a custom model: python ml/training/train_model.py")
        logger.info("4. Check model status at: http://localhost:8000/model/status")
    else:
        logger.error("❌ Setup completed with errors")
        logger.info("Check the logs above for specific issues")
        sys.exit(1)


if __name__ == "__main__":
    main()