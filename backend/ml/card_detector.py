import logging
import time
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import cv2
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)

@dataclass
class CardDetection:
    """Data class for card detection results"""
    card_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    center: Tuple[float, float]  # (x, y)

class YOLOv8CardDetector:
    """Fixed YOLOv8-based poker card detector with correct class mapping"""
    
    # Class names will be loaded directly from model - no hardcoded mapping needed
    MODEL_CLASS_NAMES = {}  # Will be populated from model
    MODEL_TO_CONFIG = {}    # No longer needed
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the fixed card detector"""
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.model = None
        self.device = self._get_device()
        self.inference_times = []
        # Add class_names for compatibility - now using correct mapping
        self.class_names = self.MODEL_CLASS_NAMES
        
        logger.info(f"Initialized Fixed YOLOv8CardDetector with device: {self.device}")
    
    def convert_card_name(self, model_name: str) -> str:
        """
        Convert from model format (10C, AS) to app format (Tc, As)
        Handles the naming convention mismatch
        """
        if not model_name or len(model_name) < 2:
            return model_name
            
        # Handle 10 -> T conversion
        if model_name.startswith('10'):
            rank = 'T'
            suit = model_name[2].lower() if len(model_name) > 2 else model_name[-1].lower()
        else:
            rank = model_name[0]
            suit = model_name[1].lower()
        
        return f"{rank}{suit}"
    
    def _get_default_config_path(self) -> str:
        """Get default config file path"""
        current_dir = Path(__file__).parent
        return str(current_dir / "config" / "model_config.yaml")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            # OPTIMIZED: Balanced thresholds for real-world poker images
            config['model']['confidence_threshold'] = 0.25  # Balanced - catches real cards, filters noise
            config['model']['iou_threshold'] = 0.30         # Better NMS
            config['model']['input_size'] = 1280            # Keep high resolution
            logger.info(f"Loaded configuration with optimized settings (conf=0.25)")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config with optimized settings
            return {
                'model': {
                    'confidence_threshold': 0.25,  # Balanced threshold
                    'iou_threshold': 0.30,
                    'input_size': 1280,
                    'max_detections': 100  # Reduced from 300 - poker game has max 9 cards
                }
            }
    
    def _get_device(self) -> str:
        """Determine computation device"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_model(self, model_path: str) -> bool:
        """Load YOLOv8 model with PyTorch 2.6 compatibility fix"""
        print(f"🔧 DEBUG: Loading YOLO model from: {model_path}")
        try:
            # Check if ultralytics is available
            print("🔍 DEBUG: Importing ultralytics...")
            from ultralytics import YOLO
            print("✅ DEBUG: ultralytics imported successfully")
            
            # Check if file exists
            import os
            if not os.path.exists(model_path):
                print(f"❌ DEBUG: Model file does not exist: {model_path}")
                return False
            
            print(f"✅ DEBUG: Model file exists, size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
            
            # Fix for PyTorch 2.6+
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'collections.OrderedDict'
                ])
            
            # Temporarily override torch.load for compatibility
            original_torch_load = torch.load
            torch.load = lambda *args, **kwargs: original_torch_load(*args, **dict(kwargs, weights_only=False))
            
            # Load model
            print("🔧 DEBUG: Creating YOLO model object...")
            self.model = YOLO(model_path)
            print(f"✅ DEBUG: YOLO model created: {self.model is not None}")
            
            # CRITICAL FIX: Get class names directly from the model (native format)
            print("🔧 DEBUG: Getting class names from model...")
            if hasattr(self.model, 'names') and self.model.names:
                # Use model's native class names directly - no conversion needed
                self.MODEL_CLASS_NAMES = self.model.names  # Direct reference to model's names
                self.class_names = self.model.names        # Same reference
                print(f"✅ DEBUG: Loaded {len(self.MODEL_CLASS_NAMES)} class names from model")
                print(f"🔍 DEBUG: First 10 classes: {[self.model.names[i] for i in range(min(10, len(self.model.names)))]}")
                print(f"🔍 DEBUG: All class names: {list(self.model.names.values())}")
            else:
                print("⚠️ DEBUG: Could not get class names from model, using fallback")
            
            print(f"🔧 DEBUG: Moving model to device: {self.device}")
            self.model.to(self.device)
            print("✅ DEBUG: Model moved to device")
            
            # Restore original torch.load
            torch.load = original_torch_load
            
            # Test the model with a dummy prediction
            print("🔧 DEBUG: Testing model with dummy prediction...")
            import numpy as np
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_results = self.model(dummy_img, verbose=False)
            print(f"✅ DEBUG: Model inference test successful! Detected {len(test_results[0].boxes) if test_results[0].boxes else 0} objects")
            
            logger.info(f"Successfully loaded model from {model_path}")
            print(f"✅ DEBUG: Model loaded successfully from {model_path}")
            return True
            
        except ImportError as e:
            print(f"❌ DEBUG: Import error - ultralytics not available: {e}")
            logger.error(f"Import error loading model: {e}")
            return False
        except Exception as e:
            print(f"❌ DEBUG: Failed to load model: {e}")
            logger.error(f"Failed to load model: {e}")
            import traceback
            print("❌ DEBUG: Full traceback:")
            traceback.print_exc()
            return False
    
    def preprocess_poker_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Enhanced preprocessing for poker table images (Opus's fix)"""
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # 1. Remove green felt bias
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_lab[:, :, 1] = cv2.subtract(img_lab[:, :, 1], 10)
        
        # 2. Enhance contrast for cards
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        
        # 3. Sharpen cards
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 8, 2,-1],
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0
        img = cv2.filter2D(img, -1, kernel)
        
        # 4. Enhance red colors for hearts/diamonds
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)) + \
                   cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        hsv[:, :, 1] = np.where(red_mask > 0, 
                                np.clip(hsv[:, :, 1] * 1.2, 0, 255).astype(np.uint8),
                                hsv[:, :, 1])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 5. Normalize lighting
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        return img
    
    def detect_cards_from_pil(self, pil_image: Image.Image) -> Tuple[List[CardDetection], float]:
        """Detect cards from PIL image with all fixes applied"""
        if self.model is None:
            logger.error("Model not loaded")
            return [], 0.0
        
        start_time = time.time()
        
        try:
            # Skip preprocessing - use image directly as tests showed it works fine
            img_np = np.array(pil_image)
            
            # FIXED: Use production-ready thresholds to avoid false positives
            conf_threshold = self.config.get('model', {}).get('confidence_threshold', 0.50)
            iou_threshold = self.config.get('model', {}).get('iou_threshold', 0.30)

            results = self.model(
                img_np,
                conf=conf_threshold,   # High confidence threshold (0.50)
                iou=iou_threshold,     # Proper NMS threshold (0.30)
                imgsz=1280,            # High resolution for card details
                verbose=False
            )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Process results with correct class mapping
            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy().item() if hasattr(boxes.conf[i].cpu().numpy(), 'item') else boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy().item() if hasattr(boxes.cls[i].cpu().numpy(), 'item') else boxes.cls[i].cpu().numpy())
                    
                    # Get card name from model's class mapping (model's native format)
                    model_card_name = self.MODEL_CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
                    
                    # DEBUG: Print raw predictions to verify class mapping
                    print(f"🔍 DEBUG: Class {cls_id} -> {model_card_name} (conf: {conf:.3f})")
                    
                    # SIMPLIFIED: Use model's native format, convert only for app compatibility
                    app_card_name = self.convert_card_name(model_card_name)
                    print(f"🔄 DEBUG: Converted {model_card_name} -> {app_card_name}")
                    
                    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    
                    detection = CardDetection(
                        card_name=app_card_name,
                        confidence=conf,
                        bbox=box.tolist(),
                        center=center
                    )
                    detections.append(detection)

            # PRODUCTION: Apply aggressive duplicate removal
            print(f"🔍 DEBUG: {len(detections)} raw detections before filtering:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det.card_name}: {det.confidence:.3f} at {det.center}")

            # Remove duplicates aggressively
            unique_detections = self._remove_duplicate_cards(detections)

            # ADAPTIVE POST-FILTERING: Smart card count management
            unique_detections = self._apply_adaptive_filtering(unique_detections)

            logger.info(f"Detected {len(unique_detections)} unique cards in {inference_time:.3f}s")
            return unique_detections, inference_time
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return [], 0.0
    
    def detect_cards_from_pil_poker(self, pil_image: Image.Image) -> Tuple[List[CardDetection], float, Dict]:
        """
        Poker-specific detection method required by the service
        Returns detection results plus processing report
        """
        if self.model is None:
            logger.error("Model not loaded")
            return [], 0.0, {}
        
        start_time = time.time()
        
        # CRITICAL DEBUG: Check image preprocessing
        print(f"📐 DEBUG: Input image size: {pil_image.size}")
        print(f"📐 DEBUG: Input image mode: {pil_image.mode}")
        
        # Test different inference settings to debug the low confidence issue
        print("🔧 DEBUG: Testing inference with training-matched settings...")
        
        try:
            # Convert PIL to numpy for YOLO
            img_np = np.array(pil_image)
            
            # EMERGENCY DEBUG: Use EXACT same settings as validation script
            print(f"🔧 DEBUG: Image shape before inference: {img_np.shape}")
            print(f"🔧 DEBUG: Image dtype: {img_np.dtype}")
            print(f"🔧 DEBUG: Image min/max values: {img_np.min()}/{img_np.max()}")
            
            # PRODUCTION: Use optimized confidence threshold (0.25 = balanced)
            conf_threshold = self.config.get('model', {}).get('confidence_threshold', 0.25)
            iou_threshold = self.config.get('model', {}).get('iou_threshold', 0.30)
            imgsz = self.config.get('model', {}).get('input_size', 1280)

            results = self.model(
                img_np,
                conf=conf_threshold,   # Balanced threshold (0.25)
                iou=iou_threshold,     # Better NMS (0.30)
                imgsz=imgsz,           # High resolution for card details (1280)
                verbose=False
            )
            
            print(f"🔧 DEBUG: Results type: {type(results)}")
            print(f"🔧 DEBUG: Results length: {len(results) if results else 0}")
            if results:
                print(f"🔧 DEBUG: First result type: {type(results[0])}")
                print(f"🔧 DEBUG: First result boxes: {results[0].boxes}")
                print(f"🔧 DEBUG: First result boxes type: {type(results[0].boxes)}")
            
            print(f"🎯 DEBUG: Raw detections count: {len(results[0].boxes) if results and results[0].boxes else 0}")
            
            if results and results[0].boxes is not None:
                # Show all raw confidences
                raw_confs = [float(conf) for conf in results[0].boxes.conf.cpu().numpy()]
                if raw_confs:
                    print(f"📊 DEBUG: Raw confidence range: {min(raw_confs):.3f} - {max(raw_confs):.3f}")
                    print(f"📊 DEBUG: Raw confidences: {raw_confs[:10]}...")  # First 10
                else:
                    print("📊 DEBUG: No confidence scores found")
            
            inference_time = time.time() - start_time
            
            # Process results with correct class mapping
            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get card name from model's class mapping (model's native format)
                    model_card_name = self.MODEL_CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
                    
                    # DEBUG: Print raw predictions to verify class mapping
                    print(f"🔍 DEBUG: Class {cls_id} -> {model_card_name} (conf: {conf:.3f})")
                    
                    # SIMPLIFIED: Use model's native format, convert only for app compatibility
                    app_card_name = self.convert_card_name(model_card_name)
                    print(f"🔄 DEBUG: Converted {model_card_name} -> {app_card_name}")
                    
                    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    
                    detection = CardDetection(
                        card_name=app_card_name,
                        confidence=conf,
                        bbox=box.tolist(),
                        center=center
                    )
                    detections.append(detection)

            # PRODUCTION: Apply aggressive duplicate removal
            print(f"🔍 DEBUG: {len(detections)} raw detections before filtering:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det.card_name}: {det.confidence:.3f} at {det.center}")

            # Remove duplicates aggressively
            unique_detections = self._remove_duplicate_cards(detections)

            # ADAPTIVE POST-FILTERING: Smart card count management
            unique_detections = self._apply_adaptive_filtering(unique_detections)

            print(f"✅ After filtering: {len(unique_detections)} unique cards")

            # Create detailed processing report
            processing_report = {
                "total_detections": len(unique_detections),
                "unique_cards": len(set(d.card_name for d in unique_detections)),
                "inference_time": inference_time,
                "device": str(self.device),
                "confidence_threshold": conf_threshold,  # Actual threshold used (0.25)
                "iou_threshold": iou_threshold,          # Actual IOU used (0.30)
                "input_size": imgsz,                     # Actual size used (1280)
                "detected_cards": [d.card_name for d in unique_detections],
                "image_size": pil_image.size,
                "raw_detection_count": len(detections) if detections else 0
            }
            
            logger.info(f"Poker detection complete: {len(unique_detections)} unique cards in {inference_time:.3f}s")
            
            return unique_detections, inference_time, processing_report
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return [], 0.0, {"error": str(e)}
    
    def analyze_poker_game_from_pil(self, pil_image: Image.Image) -> Tuple[Dict, List]:
        """
        Complete poker game analysis from PIL image
        Required by the service for game analysis

        Returns:
            Tuple of (game_analysis dict, list of CardDetection objects)
        """
        # Get card detections
        detections, inference_time, report = self.detect_cards_from_pil_poker(pil_image)

        if not detections:
            return {
                "status": "no_cards_detected",
                "message": "No cards detected in image",
                "inference_time": inference_time,
                "report": report
            }, []

        # Convert detections to dict format for analyzer
        detection_dicts = []
        for det in detections:
            detection_dicts.append({
                'card_name': det.card_name,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'center': det.center
            })

        # Get image dimensions for spatial analysis
        img_array = np.array(pil_image)
        image_shape = (img_array.shape[0], img_array.shape[1])  # (height, width)

        try:
            # Try to use the poker game analyzer if available
            from ml.poker_game_analyzer import analyze_poker_game

            # This returns a dictionary, not a GameState object
            game_analysis = analyze_poker_game(detection_dicts, image_shape)

            # Add detection metadata
            game_analysis['detection_report'] = report
            game_analysis['inference_time'] = inference_time
            game_analysis['total_cards_detected'] = len(detections)

            return game_analysis, detections
            
        except ImportError:
            # Fallback: Basic spatial grouping if analyzer not available
            logger.warning("PokerGameAnalyzer not available, using fallback analysis")
            
            height = image_shape[0]
            
            # Group cards by position
            player1_cards = []
            community_cards = []
            player2_cards = []
            
            for det in detection_dicts:
                y = det['center'][1]
                if y < height * 0.33:
                    player1_cards.append(det['card_name'])
                elif y < height * 0.66:
                    community_cards.append(det['card_name'])
                else:
                    player2_cards.append(det['card_name'])
            
            return {
                "status": "success",
                "player1": {
                    "cards": player1_cards,
                    "position": "top",
                    "count": len(player1_cards)
                },
                "community": {
                    "cards": community_cards,
                    "position": "center",
                    "count": len(community_cards)
                },
                "player2": {
                    "cards": player2_cards,
                    "position": "bottom",
                    "count": len(player2_cards)
                },
                "total_cards": len(detections),
                "detection_report": report,
                "inference_time": inference_time
            }, detections
    
    def _apply_adaptive_filtering(self, detections: List[CardDetection]) -> List[CardDetection]:
        """
        ADAPTIVE filtering based on poker game rules

        Strategy:
        1. If 7-12 cards: likely correct, just cap at 12
        2. If 13-20 cards: too many, keep top 9 by confidence
        3. If 20+ cards: way too many, aggressive filtering
        4. If 0-6 cards: possibly missing cards, keep all
        """
        count = len(detections)

        if count == 0:
            logger.warning("⚠️ No cards detected!")
            return detections

        if count <= 6:
            logger.warning(f"⚠️ Only {count} cards detected (expected 7-9 for valid game)")
            return detections

        if 7 <= count <= 12:
            # Good range - poker game should have 7-9 cards (or 4 for preflop)
            logger.info(f"✅ Detected {count} cards - in expected range")
            return detections

        if 13 <= count <= 20:
            # Too many but manageable - keep top 9 by confidence
            logger.warning(f"⚠️ Detected {count} cards (expected 7-12). Keeping top 9 by confidence.")
            sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
            return sorted_dets[:9]

        if count > 20:
            # Way too many - aggressive filtering
            logger.error(f"❌ Detected {count} cards (way too many!). Applying aggressive filtering.")
            sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)

            # Keep only top 9 with confidence > 0.40
            filtered = [d for d in sorted_dets if d.confidence > 0.40][:9]
            logger.info(f"   After aggressive filtering: {len(filtered)} cards")
            return filtered

        return detections

    def _remove_duplicate_cards(self, detections: List[CardDetection]) -> List[CardDetection]:
        """
        AGGRESSIVE duplicate removal for poker cards

        Rules:
        1. Each card can only appear once (poker deck has unique cards)
        2. Spatial duplicates (same location) get merged
        3. Keep highest confidence detection for each card
        """
        if not detections:
            return detections

        logger.info(f"🔧 Starting duplicate removal on {len(detections)} detections")

        # Sort by confidence (highest first) to keep best detections
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        filtered = []
        seen_card_names = set()  # Track card names we've already kept

        for det in sorted_detections:
            is_duplicate = False

            # Check 1: Card name uniqueness (CRITICAL - poker has unique cards)
            if det.card_name in seen_card_names:
                logger.warning(f"❌ Duplicate card name: {det.card_name} (conf: {det.confidence:.3f}) - REMOVED")
                is_duplicate = True
            else:
                # Check 2: Spatial duplicates (same physical card detected multiple times)
                for existing in filtered:
                    # Calculate distance between centers
                    dist = np.sqrt((det.center[0] - existing.center[0])**2 +
                                  (det.center[1] - existing.center[1])**2)

                    # AGGRESSIVE: If within 80 pixels, consider it the same physical card
                    if dist < 80:
                        logger.warning(f"❌ Spatial duplicate: {det.card_name} too close ({dist:.0f}px) to {existing.card_name} - REMOVED")
                        is_duplicate = True
                        break

            if not is_duplicate:
                filtered.append(det)
                seen_card_names.add(det.card_name)
                logger.info(f"✅ Kept: {det.card_name} (conf: {det.confidence:.3f})")

        removed_count = len(detections) - len(filtered)
        if removed_count > 0:
            logger.info(f"🗑️ Removed {removed_count} duplicates (kept {len(filtered)} unique cards)")

        return filtered
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {"message": "No inference times recorded"}
        
        return {
            "average_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "total_inferences": len(self.inference_times),
            "device": self.device
        }


def create_card_detector() -> YOLOv8CardDetector:
    """Factory function to create fixed card detector"""
    return YOLOv8CardDetector()