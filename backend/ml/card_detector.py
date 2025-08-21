"""
YOLOv8 Card Detector Wrapper for PokerVision

This module provides a wrapper around YOLOv8 for poker card detection,
including model loading, inference, and result processing.
"""

import os
import time
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import yaml
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class CardDetection:
    """Represents a single card detection result"""
    
    def __init__(self, card_name: str, confidence: float, bbox: List[float], center: Tuple[float, float]):
        self.card_name = card_name
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.center = center  # (x, y)
        
    def to_dict(self) -> Dict:
        return {
            'card': self.card_name,
            'confidence': float(self.confidence),
            'bbox': self.bbox,
            'center': list(self.center)
        }


class YOLOv8CardDetector:
    """YOLOv8-based poker card detector"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the card detector
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.model = None
        self.class_names = self._load_class_names()
        self.device = self._get_device()
        
        # Performance tracking
        self.inference_times = []
        
        logger.info(f"Initialized YOLOv8CardDetector with device: {self.device}")
    
    def _get_default_config_path(self) -> str:
        """Get default config file path"""
        current_dir = Path(__file__).parent
        return str(current_dir / "config" / "model_config.yaml")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if file loading fails"""
        return {
            'model': {
                'name': 'yolov8n',
                'confidence_threshold': 0.3,
                'iou_threshold': 0.45,
                'max_detections': 300,
                'input_size': 640
            }
        }
    
    def _load_class_names(self) -> Dict[int, str]:
        """Load class names from configuration"""
        try:
            return self.config.get('classes', {})
        except Exception as e:
            logger.error(f"Failed to load class names: {e}")
            return {}
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        device_config = self.config.get('model', {}).get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        else:
            device = device_config
            
        logger.info(f"Using device: {device}")
        return device
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load YOLOv8 model
        
        Args:
            model_path: Path to trained model file. If None, uses pretrained model.
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading trained model from {model_path}")
                self.model = YOLO(model_path)
            else:
                # Use pretrained model for initial testing
                model_name = self.config.get('model', {}).get('name', 'yolov8n')
                logger.info(f"Loading pretrained {model_name} model")
                self.model = YOLO(f"{model_name}.pt")
            
            # Move model to appropriate device
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
                
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect_cards(self, image: np.ndarray, return_raw: bool = False) -> Tuple[List[CardDetection], float]:
        """
        Detect poker cards in an image
        
        Args:
            image: Input image as numpy array (BGR format)
            return_raw: Whether to return raw YOLO results
            
        Returns:
            Tuple of (card detections, inference time)
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return [], 0.0
        
        start_time = time.time()
        
        try:
            # Get model configuration
            model_config = self.config.get('model', {})
            conf_threshold = model_config.get('confidence_threshold', 0.3)
            iou_threshold = model_config.get('iou_threshold', 0.45)
            max_det = model_config.get('max_detections', 300)
            
            # Run inference
            results = self.model(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Process results
            detections = []
            if results and len(results) > 0:
                result = results[0]  # Single image
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Extract box information
                        box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Get card name from class ID
                        card_name = self.class_names.get(cls_id, f"unknown_{cls_id}")
                        
                        # Calculate center point
                        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        
                        # Create detection object
                        detection = CardDetection(
                            card_name=card_name,
                            confidence=conf,
                            bbox=box.tolist(),
                            center=center
                        )
                        
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} cards in {inference_time:.3f}s")
            
            if return_raw:
                return detections, inference_time, results
            else:
                return detections, inference_time
                
        except Exception as e:
            logger.error(f"Card detection failed: {e}")
            return [], 0.0
    
    def detect_cards_from_pil(self, pil_image: Image.Image) -> Tuple[List[CardDetection], float]:
        """
        Detect cards from PIL Image
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Tuple of (card detections, inference time)
        """
        # Convert PIL to numpy array (RGB to BGR)
        image_array = np.array(pil_image)
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return self.detect_cards(image_array)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {'avg_inference_time': 0.0, 'total_inferences': 0}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_inferences': len(self.inference_times),
            'target_time_ms': self.config.get('performance', {}).get('target_inference_time', 100)
        }
    
    def clear_performance_stats(self):
        """Clear performance statistics"""
        self.inference_times = []
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        info = {
            'model_loaded': self.is_model_loaded(),
            'device': self.device,
            'config_path': self.config_path,
            'class_count': len(self.class_names),
        }
        
        if self.model is not None:
            try:
                info['model_type'] = str(type(self.model))
                if hasattr(self.model, 'info'):
                    info.update(self.model.info())
            except Exception as e:
                logger.warning(f"Could not get model info: {e}")
        
        return info


def create_card_detector(config_path: Optional[str] = None, model_path: Optional[str] = None) -> YOLOv8CardDetector:
    """
    Factory function to create and initialize a card detector
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model file
        
    Returns:
        Initialized YOLOv8CardDetector instance
    """
    detector = YOLOv8CardDetector(config_path)
    
    if not detector.load_model(model_path):
        logger.error("Failed to load model in card detector")
        raise RuntimeError("Could not initialize card detector")
    
    return detector