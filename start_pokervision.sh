#!/bin/bash

echo "🃏 Starting PokerVision Setup..."
echo "=================================="

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Please run this script from the PokerVision root directory"
    echo "Expected structure: PokerVision/backend/ and PokerVision/frontend/"
    exit 1
fi

echo "✅ Found backend and frontend directories"

# Step 1: Check for trained model
echo ""
echo "🔍 Checking for trained model..."
if [ -f "backend/ml/models/poker_cards_best.pt" ]; then
    echo "✅ Found trained model: backend/ml/models/poker_cards_best.pt"
    MODEL_SIZE=$(du -h "backend/ml/models/poker_cards_best.pt" | cut -f1)
    echo "   Model size: $MODEL_SIZE"
else
    echo "❌ Trained model not found!"
    echo "📋 Please ensure you have:"
    echo "   1. Downloaded pokervision_trained_model.zip from Colab"
    echo "   2. Extracted poker_cards_best.pt"
    echo "   3. Placed it in: backend/ml/models/poker_cards_best.pt"
    echo ""
    echo "🔧 Creating models directory..."
    mkdir -p backend/ml/models
    echo "✅ Created backend/ml/models/ directory"
    echo "⚠️  Please place your trained model there and re-run this script"
    exit 1
fi

# Step 2: Setup Backend
echo ""
echo "🐍 Setting up Backend..."
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Quick test of ML components
echo ""
echo "🧪 Testing ML components..."
python -c "
try:
    from ml.card_detector import create_card_detector
    from ml.spatial_analyzer import PokerSpatialAnalyzer
    print('✅ ML components import successful')
    
    # Test model loading
    detector = create_card_detector()
    print('✅ Card detector initialized')
    
    analyzer = PokerSpatialAnalyzer()
    print('✅ Spatial analyzer initialized')
    
    print('🎯 Backend ML system is ready!')
    
except Exception as e:
    print(f'❌ ML test failed: {e}')
    print('⚠️  Will use mock detection mode')
"

# Start backend server in background
echo ""
echo "🚀 Starting Backend Server..."
echo "   URL: http://localhost:8000"
echo "   Health Check: http://localhost:8000/health"
echo "   Model Status: http://localhost:8000/model/status"

# Run server
python main.py &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# Give backend time to start
sleep 3

# Test backend health
echo ""
echo "🏥 Testing backend health..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is healthy!"
else
    echo "❌ Backend health check failed"
    echo "📋 Check backend logs above for errors"
fi

# Step 3: Setup Frontend
echo ""
echo "⚛️  Setting up Frontend..."
cd ../frontend

# Install npm dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Start frontend
echo ""
echo "🚀 Starting Frontend..."
echo "   URL: http://localhost:3000"
echo "   Or: http://localhost:5173 (if using Vite)"

# Start frontend in background
npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

# Give frontend time to start
sleep 5

# Final status
echo ""
echo "🎉 PokerVision is now running!"
echo "=================================="
echo "🔗 URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   Health Check: http://localhost:8000/health"
echo "   Model Status: http://localhost:8000/model/status"
echo ""
echo "🃏 Ready to detect poker cards!"
echo ""
echo "📋 To stop the servers:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "🧪 Test your model by:"
echo "   1. Open http://localhost:3000"
echo "   2. Upload a poker card image"
echo "   3. See instant AI detection results!"

# Keep script running
echo "Press Ctrl+C to stop all servers..."
wait