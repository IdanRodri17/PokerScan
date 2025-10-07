# 🃏 PokerVision - AI-Powered Poker Card Detection

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://pokervision.netlify.app)
[![Backend API](https://img.shields.io/badge/API-Hugging%20Face-yellow)](https://rodri17-pokervision-backend.hf.space)

**PokerVision** is an AI-powered poker card detection system that analyzes poker game images in real-time and automatically determines the winner. Built with cutting-edge computer vision and machine learning technologies.

![PokerVision Demo](https://via.placeholder.com/800x400/1a1a2e/ffffff?text=Add+Screenshot+Here)

---

## 🌟 Features

- **🎯 Real-time Card Detection** - Powered by custom-trained YOLOv8 model
- **🎨 Beautiful UI** - Modern, responsive design with smooth animations
- **✏️ Manual Correction** - Interactive modal to fix any misdetected cards
- **🏆 Automatic Winner Determination** - Smart hand evaluation for 2-player Texas Hold'em
- **📱 Mobile Responsive** - Works seamlessly on all devices
- **⚡ Fast Analysis** - ~5 second average processing time

---

## 🚀 Live Demo

**Try it now:** [https://pokervision.netlify.app](https://pokervision.netlify.app)

1. Upload a poker game image
2. Wait for AI detection
3. Fix any incorrect cards (if needed)
4. See the winner instantly!

---

## 🛠️ Tech Stack

### Frontend
- **React** - UI framework
- **Vite** - Build tool for fast development
- **TailwindCSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Lucide React** - Modern icon library
- **React Dropzone** - Drag-and-drop file upload

### Backend
- **FastAPI** - High-performance Python web framework
- **YOLOv8 (Ultralytics)** - Object detection model
- **PyTorch** - Deep learning framework
- **OpenCV** - Image processing
- **scikit-learn** - Hand evaluation algorithms
- **Pillow** - Image manipulation

### Deployment
- **Frontend:** Netlify (automatic CI/CD)
- **Backend:** Hugging Face Spaces (Docker)
- **Model Hosting:** Git LFS for large model files

---

## 🎮 How It Works

### 1. **Card Detection**
- Custom-trained YOLOv8 model detects all 52 playing cards
- Trained on thousands of poker game images
- Achieves high accuracy even with challenging lighting/angles

### 2. **Spatial Analysis**
- Analyzes card positions to categorize:
  - Community cards (flop, turn, river)
  - Player 1 hole cards
  - Player 2 hole cards

### 3. **Hand Evaluation**
- Implements poker hand ranking algorithm
- Evaluates best 5-card combination for each player
- Supports all poker hands (Royal Flush to High Card)

### 4. **Winner Determination**
- Compares hand strengths using numerical scoring
- Handles ties correctly
- Displays winning hand and cards

---

## 🏗️ Local Development

### Prerequisites
- **Node.js** (v18 or higher)
- **Python** (3.11)
- **Git LFS** (for model files)

### Frontend Setup

```bash
# Clone the repository
git clone https://github.com/IdanRodri17/PokerScan.git
cd PokerScan/frontend

# Install dependencies
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Start development server
npm run dev
```

Frontend will run at: `http://localhost:5173`

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --port 8000
```

Backend API will run at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

---

## 📡 API Endpoints

### `POST /upload`
Upload poker game image for analysis

**Parameters:**
- `file` (form-data): Image file
- `create_visualization` (query, optional): Boolean

**Response:**
```json
{
  "success": true,
  "filename": "poker_game.jpg",
  "detection_results": [...],
  "cards_detected": ["AS", "KH", "QD", ...],
  "processing_time": 5.2,
  "game_analysis": {
    "community_cards": ["AS", "KH", "QD", "JC", "10S"],
    "players": [...],
    "winner": {
      "id": 1,
      "name": "Player 1",
      "winning_hand": "Royal Flush"
    }
  }
}
```

### `POST /evaluate-winner`
Manually evaluate winner from corrected cards

**Request Body:**
```json
{
  "player1_cards": ["AS", "KS"],
  "community_cards": ["QS", "JS", "10S", "9H", "2C"],
  "player2_cards": ["AH", "KH"]
}
```

### `GET /health`
Health check endpoint

---

## 🎯 Model Training

The YOLOv8 model was custom-trained on:
- **Dataset:** 10000+ poker game images
- **Classes:** 52 playing cards (Ace-King, all suits)
- **Training time:** ~6 hours on GPU
- **Final model:** `poker_cards_best.pt` (50 MB)

Training configuration in: `backend/ml/config/model_config.yaml`

---

## 🧪 Testing

### Frontend Tests
```bash
cd frontend
npm run test
```

### Backend Tests
```bash
cd backend
pytest
```

---

## 🚢 Deployment

### Deploy Frontend to Netlify

1. Connect GitHub repository to Netlify
2. Build settings:
   - **Base directory:** `frontend`
   - **Build command:** `npm run build`
   - **Publish directory:** `frontend/dist`
3. Add environment variable:
   - `VITE_API_URL`: Your backend URL

### Deploy Backend to Hugging Face Spaces

1. Create new Space with Docker SDK
2. Clone Space repository
3. Copy backend files
4. Configure Git LFS for model files:
   ```bash
   git lfs track "*.pt"
   ```
5. Push to Hugging Face

Detailed guides:
- [Netlify Deployment Guide](NETLIFY_DEPLOY_GUIDE.md)
- [Hugging Face Deployment Guide](HUGGINGFACE_DEPLOY_GUIDE.md)

---

## 📁 Project Structure

```
PokerScan/
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── App.jsx        # Main app component
│   │   └── main.jsx       # Entry point
│   ├── public/            # Static assets
│   └── package.json
│
├── backend/               # FastAPI backend
│   ├── main.py           # FastAPI app
│   ├── requirements.txt  # Python dependencies
│   ├── Dockerfile        # Docker configuration
│   ├── models/           # Pydantic schemas
│   ├── services/         # Business logic
│   │   └── image_processor.py
│   └── ml/               # Machine learning
│       ├── card_detector.py
│       ├── hand_evaluator.py
│       ├── spatial_analyzer.py
│       └── models/
│           └── poker_cards_best.pt  # Trained model
│
├── README.md
└── LICENSE
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8 framework
- **Hugging Face** - Model hosting
- **Netlify** - Frontend hosting
- **React Team** - Amazing frontend framework

---

## 📧 Contact

**Idan Rodriguez**

- LinkedIn: [LinkedIn](https://www.linkedin.com/in/idanrodrigez/)
- Email: idan101012@gmail.com
- Portfolio: [Portfolio](https://idanportfolio.netlify.app/)

---
## 🎓 Project Status

**Status:** ✅ Production Ready

**Version:** 1.0.0

**Last Updated:** October 2025

---

⭐ **If you found this project helpful, please give it a star!** ⭐
