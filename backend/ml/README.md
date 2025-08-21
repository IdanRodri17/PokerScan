# PokerVision ML System

This directory contains the complete machine learning pipeline for poker card detection using YOLOv8.

## 📁 Directory Structure

```
ml/
├── __init__.py
├── README.md                    # This file
├── card_detector.py            # YOLOv8 model wrapper
├── spatial_analyzer.py         # Poker table spatial analysis
├── config/
│   └── model_config.yaml      # Model configuration
├── models/                     # Trained model storage
├── training/
│   ├── __init__.py
│   ├── train_model.py         # Complete training pipeline
│   ├── dataset_utils.py       # Data loading and preprocessing
│   ├── evaluate_model.py      # Model evaluation and metrics
│   └── runs/                  # Training run outputs
└── data/                      # Dataset storage (created during setup)
    ├── raw/                   # Raw downloaded dataset
    └── processed/             # Processed YOLOv8 format dataset
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies and setup ML environment
cd backend
python setup_ml.py
```

This script will:
- Install all required ML dependencies
- Create necessary directories
- Test ML component imports
- Download sample YOLOv8 model
- Setup Kaggle configuration (optional)

### 2. Test the System

```bash
# Start the FastAPI server
python main.py

# In another terminal, check model status
curl http://localhost:8000/model/status

# Test with the frontend
# Open http://localhost:3000 and upload a poker image
```

## 🤖 Components Overview

### Card Detector (`card_detector.py`)

YOLOv8-based poker card detection with:
- Support for all 52 playing cards
- Confidence-based filtering
- Multiple export formats (PyTorch, ONNX, TorchScript)
- Performance monitoring
- Flexible configuration

**Usage:**
```python
from ml.card_detector import create_card_detector

# Initialize detector
detector = create_card_detector()

# Detect cards in PIL image
detections, inference_time = detector.detect_cards_from_pil(image)

# Get performance stats
stats = detector.get_performance_stats()
```

### Spatial Analyzer (`spatial_analyzer.py`)

Poker-specific spatial analysis for:
- Community card identification (flop/turn/river)
- Player hand detection (2 cards per player)
- Game stage determination
- Confidence scoring
- Clustering algorithms (DBSCAN, K-Means, Agglomerative)

**Usage:**
```python
from ml.spatial_analyzer import PokerSpatialAnalyzer

analyzer = PokerSpatialAnalyzer()
table_analysis = analyzer.analyze_table(detections, image_shape)

# Get structured results
summary = analyzer.get_analysis_summary(table_analysis)
```

## 📊 Training Pipeline

### Dataset Preparation

The system uses the Kaggle "Playing Cards Object Detection Dataset" with 20k images:

```bash
# Setup Kaggle API credentials first
# Then run dataset download and processing
python ml/training/dataset_utils.py
```

### Training

```bash
# Run complete training pipeline
python ml/training/train_model.py

# With Weights & Biases tracking
python ml/training/train_model.py --use-wandb

# Resume from checkpoint
python ml/training/train_model.py --resume path/to/checkpoint.pt
```

### Evaluation

```bash
# Evaluate trained model
python ml/training/evaluate_model.py
```

This generates:
- Test set metrics (mAP50, mAP50-95, precision, recall, F1)
- Per-class performance analysis
- Speed benchmarks
- Visualization plots
- Comprehensive evaluation report

## ⚙️ Configuration

### Model Configuration (`config/model_config.yaml`)

Key settings:
- **Model**: YOLOv8 variant (n/s/m/l/x)
- **Training**: Epochs, batch size, learning rate
- **Detection**: Confidence and IoU thresholds
- **Classes**: 52 playing card mappings
- **Spatial**: Clustering parameters
- **Performance**: Target metrics

### Class Mapping

The system recognizes all 52 playing cards:
- **Format**: Rank + Suit (e.g., "As" = Ace of Spades)
- **Ranks**: A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K
- **Suits**: s (spades), h (hearts), d (diamonds), c (clubs)

## 🎯 Performance Targets

- **Accuracy**: mAP50 > 90%
- **Speed**: < 100ms inference time
- **Memory**: < 2GB GPU memory
- **Size**: < 50MB model file

## 📈 Integration with Backend

### API Endpoints

1. **Health Check** (`GET /health`): Includes model status
2. **Upload** (`POST /upload`): Enhanced with structured detection results
3. **Model Status** (`GET /model/status`): Detailed ML system status

### Response Format

```json
{
  "success": true,
  "detection_results": [
    {
      "type": "community_cards",
      "stage": "flop",
      "cards": [
        {
          "name": "As",
          "confidence": 0.95,
          "bbox": [320, 280, 380, 320],
          "center": [350, 300]
        }
      ],
      "count": 3
    },
    {
      "type": "player_hand",
      "player_id": 1,
      "cards": [...],
      "confidence": 0.89
    },
    {
      "type": "analysis_summary",
      "total_cards": 5,
      "confidence_score": 0.92,
      "game_stage": "flop"
    }
  ]
}
```

## 🔧 Development

### Adding New Features

1. **New Card Types**: Update `model_config.yaml` classes
2. **New Clustering**: Add method in `spatial_analyzer.py`
3. **New Metrics**: Extend `evaluate_model.py`
4. **New Exports**: Add format in `card_detector.py`

### Testing

```bash
# Test ML components
python -m pytest ml/tests/ -v

# Test integration
python setup_ml.py --test-only

# Benchmark performance
python ml/training/evaluate_model.py --benchmark
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA not available**
   - Install PyTorch with CUDA support
   - Check GPU drivers

2. **Kaggle API errors**
   - Setup `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Memory errors during training**
   - Reduce batch size in config
   - Use gradient accumulation

4. **Low detection accuracy**
   - Check image quality
   - Verify card classes in config
   - Retrain with more data

### Fallback Mode

If ML components fail to load, the system automatically falls back to mock detection, ensuring the API remains functional during development.

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [OpenCV Python](https://opencv.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

For questions or issues, please open a GitHub issue with the "ML" label.