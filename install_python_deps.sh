#!/bin/bash

echo "🐍 Setting up Python environment for PokerVision..."

# Check if we're on Windows/WSL and suggest using Windows Python
if [[ $(uname -r) == *"microsoft"* ]]; then
    echo "📝 Detected WSL environment. Consider using Windows Python instead:"
    echo "   1. Install Python from python.org (make sure to check 'Add to PATH')"
    echo "   2. Use Windows Command Prompt or PowerShell instead of Git Bash"
    echo "   3. Run: python -m pip install fastapi uvicorn python-multipart pillow numpy pydantic"
    echo ""
fi

# Try to install pip
echo "📦 Attempting to install pip..."
if command -v python3 &> /dev/null; then
    # Try downloading get-pip.py
    if command -v curl &> /dev/null; then
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py --user
        rm get-pip.py
        
        # Add user bin to PATH
        export PATH="$HOME/.local/bin:$PATH"
        
        # Install requirements
        cd backend
        python3 -m pip install --user fastapi==0.104.1 uvicorn[standard]==0.24.0 python-multipart==0.0.6 pillow==10.1.0 numpy==1.26.2 pydantic==2.5.0
        
        echo "✅ Python dependencies installed!"
        echo "🚀 To start the backend, run:"
        echo "   cd backend && python3 main.py"
    else
        echo "❌ curl not found. Please install pip manually."
    fi
else
    echo "❌ Python3 not found. Please install Python first."
fi