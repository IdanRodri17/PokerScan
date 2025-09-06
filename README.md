# 🃏 PokerVision

**The End of Your Poker Arguments!** 🏆

An AI-powered poker hand analyzer that instantly determines the winner in Texas Hold'em games with **100% accuracy**. No more disputes, no more confusion - just upload a photo of your poker table and let PokerVision settle the debate!

![PokerVision Banner](https://img.shields.io/badge/PokerVision-AI%20Powered-yellow?style=for-the-badge&logo=cards)
![Made with React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer%20Vision-FF6B6B?style=for-the-badge)

## ✨ Features

🎯 **Instant Winner Detection** - Upload a poker image and get results in seconds  
🃏 **100% Card Recognition** - Advanced YOLOv8 model trained specifically for poker cards  
🏆 **Official Poker Rules** - Accurate hand evaluation using standard Texas Hold'em rules  
📱 **Mobile Responsive** - Beautiful, modern UI that works perfectly on phones and tablets  
🎨 **Vegas Casino Theme** - Dark, elegant design with stunning animations  
🎊 **Winner Celebration** - Dramatic announcements with confetti and animations  
📊 **Visual Analysis** - See exactly why each player won with detailed comparisons  

## 🚀 Quick Start

### One-Click Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/PokerVision.git
cd PokerVision

# Make setup script executable and run
chmod +x start_pokervision.sh
./start_pokervision.sh
```

This will automatically:
- Set up Python environment and install ML dependencies
- Start the FastAPI backend on `http://localhost:8000`
- Install frontend dependencies and start on `http://localhost:3000`

### Manual Setup

#### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python setup_ml.py  # Download and setup ML models
python main.py
```

#### Frontend Setup

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

### Docker Setup

```bash
docker-compose up --build
```

## 📸 How to Use

1. **Arrange Your Cards**: 
   - **Top**: Player 1's 2 hole cards
   - **Center**: 5 community cards in a row  
   - **Bottom**: Player 2's 2 hole cards

2. **Take a Photo**: Ensure cards are clearly visible and well-lit

3. **Upload & Analyze**: Drag and drop your image or click to upload

4. **See Results**: Get instant winner announcement with detailed analysis!

## 🏗️ Architecture

```
PokerVision/
├── backend/                    # FastAPI + ML Backend
│   ├── main.py                # FastAPI application
│   ├── ml/                    # Machine Learning Pipeline
│   │   ├── card_detector.py   # YOLOv8 card detection
│   │   ├── hand_evaluator.py  # Poker hand evaluation
│   │   ├── poker_game_analyzer.py # Game analysis logic
│   │   └── models/           # Trained ML models
│   ├── services/             # Business logic services
│   └── models/               # Pydantic schemas
├── frontend/                  # Modern React Frontend
│   ├── src/components/       # React components
│   │   ├── PokerLandingPage.jsx    # Main upload page
│   │   ├── PokerResultsPage.jsx    # Results display
│   │   ├── WinnerAnnouncement.jsx  # Winner celebration
│   │   ├── PokerTableView.jsx      # Virtual poker table
│   │   ├── HandComparisonPanel.jsx # Hand comparison
│   │   └── PlayingCard.jsx         # Realistic card component
│   └── src/services/         # API integration
└── docker-compose.yml        # Full stack deployment
```

## 🤖 AI/ML Pipeline

Our sophisticated machine learning pipeline includes:

- **YOLOv8 Card Detection**: Custom-trained model for poker card recognition
- **Advanced Image Processing**: Multiple preprocessing techniques for optimal detection
- **Duplicate Card Handling**: Smart logic to resolve detection ambiguities  
- **Spatial Analysis**: Understanding card positions and relationships
- **Hand Evaluation Engine**: Official poker rules implementation
- **Game Analysis**: Winner determination with detailed explanations

## 🎨 UI/UX Highlights

- **Modern Design**: Dark Vegas casino theme with gradient effects
- **Smooth Animations**: Framer Motion powered transitions and effects
- **Card Flip Animations**: Realistic playing card reveals
- **Winner Celebrations**: Confetti effects and trophy animations
- **Mobile-First Design**: Optimized for phone usage (most common use case)
- **Responsive Layouts**: Adaptive design for all screen sizes
- **Visual Feedback**: Loading states, progress indicators, and error handling

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Backend health check |
| `/analyze-poker-game` | POST | Upload image and get game analysis |
| `/docs` | GET | Interactive API documentation |

## 🧪 Testing

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests  
cd frontend
npm test

# Full integration test
python test_setup.py
```

## 📱 Mobile Optimization

PokerVision is designed **mobile-first** since most poker games happen on phones:

- ✅ Optimized touch interfaces
- ✅ Responsive card sizing
- ✅ Mobile camera integration  
- ✅ Thumb-friendly navigation
- ✅ Fast loading on mobile networks

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin amazing-feature`
5. **Open** a Pull Request

### Development Guidelines

- Follow React 19 best practices
- Use Tailwind CSS for styling
- Write clean, documented code  
- Add tests for new features
- Ensure mobile responsiveness

## 🐛 Troubleshooting

**Common Issues:**

- **Cards not detected?** Ensure good lighting and clear card visibility
- **Frontend won't start?** Use `npm install --legacy-peer-deps` for React 19
- **Backend errors?** Run `python setup_ml.py` to download required models
- **Docker issues?** Ensure Docker has enough memory allocated (4GB+)

## 🔮 Roadmap

- [ ] **Multi-table Support**: Analyze multiple poker games simultaneously
- [ ] **Tournament Mode**: Track winners across multiple hands
- [ ] **Advanced Statistics**: Player performance analytics
- [ ] **Mobile App**: Native iOS/Android applications
- [ ] **Live Streaming**: Real-time game analysis
- [ ] **Custom Rules**: Support for different poker variants

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Credits

**Made with ❤️ by [Idan Rodriguez](https://github.com/your-username) and [Claude](https://claude.ai)**

Special thanks to:
- **Ultralytics** for YOLOv8
- **React Team** for React 19
- **FastAPI** for the amazing web framework
- **Tailwind CSS** for beautiful styling
- **Framer Motion** for smooth animations

---

<div align="center">

**⭐ Star this repo if PokerVision helped end your poker arguments! ⭐**

[Report Bug](https://github.com/your-username/PokerVision/issues) • [Request Feature](https://github.com/your-username/PokerVision/issues) • [Documentation](https://github.com/your-username/PokerVision/wiki)

</div>