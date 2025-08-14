# 🃏 PokerVision

AI-powered poker card detection application built with React, FastAPI, and computer vision.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- Docker and Docker Compose (optional)

### Local Development

#### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`

#### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:3000`

### Docker Setup

Run the entire stack with Docker Compose:

```bash
docker-compose up --build
```

This will start:
- Backend API at `http://localhost:8000`
- Frontend at `http://localhost:3000`

## 🏗️ Project Structure

```
PokerVision/
├── backend/                 # FastAPI backend
│   ├── app/                # Application package
│   ├── models/             # Pydantic models
│   ├── services/           # Business logic
│   ├── main.py            # FastAPI app entry point
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # Backend Docker config
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── hooks/         # Custom React hooks
│   │   ├── services/      # API services
│   │   └── App.jsx        # Main app component
│   ├── package.json       # Node.js dependencies
│   └── Dockerfile         # Frontend Docker config
├── .github/workflows/      # GitHub Actions CI/CD
├── docker-compose.yml      # Multi-service Docker setup
└── README.md              # This file
```

## 🔧 API Endpoints

- `GET /health` - Health check endpoint
- `POST /upload` - Upload and process poker card images

## 🎯 Features

- **Image Upload**: Drag-and-drop file upload interface
- **Camera Capture**: Real-time camera integration for photo capture
- **Card Detection**: AI-powered poker card recognition (mock implementation)
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS
- **Real-time Health Monitoring**: Backend connection status indicator
- **Error Handling**: Comprehensive error management and user feedback

## 🧪 Testing

### Backend Tests
```bash
cd backend
python -m pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Docker Tests
```bash
docker-compose up --build -d
# Test endpoints
curl http://localhost:8000/health
curl http://localhost:3000
docker-compose down
```

## 🚀 Deployment

The project includes GitHub Actions workflows for:
- Automated testing
- Docker image building
- Integration testing

## 🔮 Future Enhancements

This project is designed as a foundation for poker card detection. Future plans include:

- **YOLOv8 Integration**: Real computer vision model for card detection
- **Advanced Card Recognition**: Support for different card types and orientations
- **Batch Processing**: Multiple image upload and processing
- **Result Analytics**: Statistics and insights from detected cards
- **Model Training Pipeline**: Custom model training capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.