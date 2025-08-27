#!/usr/bin/env python3
"""
PokerVision Backend Server Startup Script

This script ensures proper Python path setup and starts the FastAPI server
with all dependencies correctly loaded.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path to enable relative imports
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

print(f"🐍 Python version: {sys.version}")
print(f"📁 Backend directory: {backend_dir}")
print(f"🔧 Python path includes backend: {str(backend_dir) in sys.path}")

# Now we can safely import the main app
try:
    from main import app
    import uvicorn
    
    print("✅ FastAPI app imported successfully")
    print("🚀 Starting PokerVision backend server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("🏥 Health check: http://localhost:8000/health")
    print("🤖 Model status: http://localhost:8000/model/status")
    print("\n" + "="*50)
    
    # Start the server
    uvicorn.run(
        "main:app",  # Import string instead of app object for reload to work
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Failed to import FastAPI app: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Make sure all dependencies are installed:")
    print("   python -m pip install fastapi uvicorn python-multipart pillow numpy pydantic ultralytics torch torchvision opencv-python")
    print("2. Check that you're in the backend directory")
    print("3. Verify all files are present in the backend directory")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Failed to start server: {e}")
    sys.exit(1)