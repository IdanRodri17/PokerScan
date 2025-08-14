#!/usr/bin/env python3
"""
Simple test script to verify PokerVision project setup
"""

import os
import sys
import json
from pathlib import Path

def test_project_structure():
    """Test that all required files and directories exist"""
    print("🔍 Testing project structure...")
    
    required_files = [
        "README.md",
        "docker-compose.yml",
        ".gitignore",
        "backend/main.py",
        "backend/requirements.txt",
        "backend/Dockerfile",
        "backend/models/schemas.py",
        "backend/services/image_processor.py",
        "frontend/package.json",
        "frontend/src/App.jsx",
        "frontend/src/components/ImageUpload.jsx",
        "frontend/src/components/CameraCapture.jsx",
        "frontend/src/components/ResultsDisplay.jsx",
        "frontend/src/services/api.js",
        "frontend/src/hooks/useCamera.js",
        "frontend/Dockerfile",
        ".github/workflows/ci.yml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_backend_imports():
    """Test that backend modules can be imported"""
    print("🐍 Testing backend imports...")
    try:
        sys.path.append('backend')
        from models.schemas import ImageUploadResponse, HealthCheckResponse
        from services.image_processor import ImageProcessor
        print("✅ Backend modules import successfully")
        return True
    except Exception as e:
        print(f"❌ Backend import failed: {e}")
        return False

def test_frontend_config():
    """Test frontend configuration files"""
    print("⚛️ Testing frontend configuration...")
    try:
        # Check package.json
        with open('frontend/package.json', 'r') as f:
            package_data = json.load(f)
            required_deps = ['react', 'axios', 'tailwindcss']
            missing_deps = [dep for dep in required_deps if dep not in package_data.get('dependencies', {})]
            if missing_deps:
                print(f"❌ Missing dependencies: {missing_deps}")
                return False
        
        # Check if build artifacts exist
        if Path('frontend/dist').exists():
            print("✅ Frontend build artifacts found")
        else:
            print("⚠️ Frontend not built yet - run 'npm run build' in frontend/")
        
        print("✅ Frontend configuration valid")
        return True
    except Exception as e:
        print(f"❌ Frontend config test failed: {e}")
        return False

def test_docker_config():
    """Test Docker configuration"""
    print("🐳 Testing Docker configuration...")
    try:
        # Check docker-compose.yml
        docker_compose_exists = Path('docker-compose.yml').exists()
        backend_dockerfile_exists = Path('backend/Dockerfile').exists()
        frontend_dockerfile_exists = Path('frontend/Dockerfile').exists()
        
        if all([docker_compose_exists, backend_dockerfile_exists, frontend_dockerfile_exists]):
            print("✅ Docker configuration files present")
            return True
        else:
            print("❌ Missing Docker configuration files")
            return False
    except Exception as e:
        print(f"❌ Docker config test failed: {e}")
        return False

def main():
    print("🃏 PokerVision Project Setup Test")
    print("=" * 40)
    
    tests = [
        test_project_structure,
        test_backend_imports,
        test_frontend_config,
        test_docker_config
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your PokerVision project is ready.")
        print("\n🚀 Next steps:")
        print("   1. Run 'docker-compose up --build' to start both services")
        print("   2. Visit http://localhost:3000 to see the frontend")
        print("   3. API will be available at http://localhost:8000")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)