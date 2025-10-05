import time
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

# Import ML components
try:
    from ml.card_detector import create_card_detector, YOLOv8CardDetector
    from ml.spatial_analyzer import PokerSpatialAnalyzer
    from ml.hand_evaluator import create_hand_evaluator
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML components not available: {e}")
    ML_AVAILABLE = False

class ImageProcessor:
    """Enhanced service for processing poker card images with YOLOv8"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # Initialize ML components
        self.ml_enabled = ML_AVAILABLE and self._initialize_ml_components()
        
        if not self.ml_enabled:
            logger.warning("ML components not available, falling back to mock detection")
    
    def _initialize_ml_components(self) -> bool:
        """Initialize ML components (card detector and spatial analyzer)"""
        import os
        model_path = r"C:\Users\zionn\Desktop\PokerScan\backend\ml\models\poker_ultimate_best.pt"
        
        # Debug: Check file directly
        print(f"🔍 Checking model file...")
        print(f"📁 Path: {model_path}")
        print(f"📊 Exists: {os.path.exists(model_path)}")
        if os.path.exists(model_path):
            print(f"📦 Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
        try:
            # Create the card detector
            self.card_detector = create_card_detector()
            
            # Try to find and load a trained model
            model_dir = Path("ml/models")
            model_loaded = False
            
            if model_dir.exists():
                # UPDATED PRIORITY: Use poker_cards_best.pt (tested and proven best)
                # poker_cards_best.pt > others (ultimate is worst)
                best_model = model_dir / "poker_cards_best.pt"
                best_finetuning = model_dir / "poker_cards_best_finetuning.pt"
                ultimate_model = model_dir / "poker_ultimate_best.pt"

                # Try poker_cards_best.pt first (proven best in testing)
                if best_model.exists():
                    logger.info(f"Loading best tested model: {best_model}")
                    model_loaded = self.card_detector.load_model(str(best_model))
                    if model_loaded:
                        logger.info("✅ Best tested model (poker_cards_best.pt) loaded successfully")
                    else:
                        logger.error("❌ Failed to load best model")

                # Fallback to finetuning model
                if not model_loaded and best_finetuning.exists():
                    logger.info(f"Loading finetuning model: {best_finetuning}")
                    model_loaded = self.card_detector.load_model(str(best_finetuning))
                    if model_loaded:
                        logger.info("✅ Finetuning model loaded successfully")
                    else:
                        logger.error("❌ Failed to load finetuning model")

                # Last resort: try best_model again if exists
                if not model_loaded and best_model.exists():
                    logger.info(f"Loading original best model: {best_model}")
                    model_loaded = self.card_detector.load_model(str(best_model))
                    if model_loaded:
                        logger.info("✅ Original best model (poker_cards_best.pt) loaded successfully")
                    else:
                        logger.error("❌ Failed to load original best model")

                # Last resort: ultimate model (performs poorly)
                if not model_loaded and ultimate_model.exists():
                    logger.warning(f"Loading ultimate model (poor performance): {ultimate_model}")
                    model_loaded = self.card_detector.load_model(str(ultimate_model))
                    if model_loaded:
                        logger.warning("⚠️ Ultimate model loaded - may have poor performance")
                    else:
                        logger.error("❌ Failed to load ultimate model")

                # Try any other .pt file
                if not model_loaded:
                    # Try any .pt file in the models directory
                    pt_files = list(model_dir.glob("*.pt"))
                    for model_file in pt_files:
                        logger.info(f"Trying model: {model_file}")
                        model_loaded = self.card_detector.load_model(str(model_file))
                        if model_loaded:
                            logger.info(f"✅ Model loaded: {model_file}")
                            break
                        else:
                            logger.warning(f"Failed to load: {model_file}")
            
            if not model_loaded:
                logger.error("❌ No model could be loaded! Detection will fail.")
                return False
            
            # Initialize spatial analyzer
            config_path = Path("ml/config/model_config.yaml")
            if config_path.exists():
                self.spatial_analyzer = PokerSpatialAnalyzer(str(config_path))
            else:
                self.spatial_analyzer = PokerSpatialAnalyzer()
            
            # Initialize hand evaluator
            self.hand_evaluator = create_hand_evaluator()
            
            logger.info("✅ All ML components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_image(self, image_data: bytes) -> bool:
        """Validate if the uploaded data is a valid image"""
        try:
            image = Image.open(image_data)
            image.verify()
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            return False
    
    def process_image(self, image_data: bytes, filename: str, analyze_game: bool = True, create_visualization: bool = False) -> Tuple[List[Dict], float, Optional[Dict], Optional[str]]:
        """
        Process the uploaded image to detect poker cards using YOLOv8 and spatial analysis
        Returns structured detection results, processing time, game analysis, and optional visualization
        
        Args:
            image_data: Image data as bytes
            filename: Name of the uploaded file
            analyze_game: Whether to perform complete game analysis
            create_visualization: Whether to create annotated image visualization
            
        Returns:
            Tuple of (detection_results, processing_time, game_analysis, visualization_path)
        """
        start_time = time.time()
        
        try:
            # Open and process the image
            image = Image.open(image_data)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image dimensions for spatial analysis
            image_shape = (image.height, image.width)
            
            visualization_path = None
            
            if self.ml_enabled:
                # Use ML pipeline for detection
                if analyze_game:
                    results, game_analysis = self._ml_card_detection_with_game_analysis(image, image_shape)
                    
                    # Create visualization if requested
                    if create_visualization and game_analysis:
                        visualization_path = self._create_game_visualization(
                            image, results, game_analysis, filename
                        )
                else:
                    results = self._ml_card_detection(image, image_shape)
                    game_analysis = None
            else:
                # Fall back to mock detection
                results = self._mock_card_detection_enhanced(np.array(image))
                game_analysis = None
            
            processing_time = time.time() - start_time
            
            logger.info(f"Processed image {filename} in {processing_time:.3f}s")
            logger.info(f"Detected {len(results)} cards")
            
            if analyze_game and game_analysis:
                winner_info = game_analysis.get('winner') if game_analysis else None
                winner_name = winner_info.get('name') if winner_info else 'No winner determined'
                logger.info(f"Game analysis: {winner_name}")
            
            if visualization_path:
                logger.info(f"Created visualization: {visualization_path}")
            
            return results, processing_time, game_analysis, visualization_path
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise
    
    def _ml_card_detection(self, image: Image.Image, image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Perform ML-based card detection and spatial analysis
        
        Args:
            image: PIL Image object
            image_shape: (height, width) of the image
            
        Returns:
            List of structured detection results
        """
        try:
            # Step 1: Detect individual cards with poker optimization (duplicate removal)
            detections, inference_time, processing_report = self.card_detector.detect_cards_from_pil_poker(image)
            
            if not detections:
                logger.info("No cards detected by ML model")
                return []
            
            logger.info(f"ML model detected {len(detections)} unique cards in {inference_time:.3f}s")
            
            # Convert detections to simple format for return
            results = []
            for detection in detections:
                detection_dict = {
                    'card': detection.card_name,
                    'confidence': float(detection.confidence),
                    'bbox': detection.bbox,
                    'center': list(detection.center)
                }
                results.append(detection_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"ML card detection failed: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to enhanced mock detection
            return self._mock_card_detection_enhanced(np.array(image))
    
    def _ml_card_detection_with_game_analysis(self, image: Image.Image, image_shape: Tuple[int, int]) -> Tuple[List[Dict], Dict]:
        """
        Perform ML-based card detection with complete poker game analysis
        
        Args:
            image: PIL Image object
            image_shape: (height, width) of the image
            
        Returns:
            Tuple of (detection results, game analysis)
        """
        try:
            # Step 1: Detect cards and analyze the poker game
            game_analysis = self.card_detector.analyze_poker_game_from_pil(image)
            
            # Step 2: Get the detections from the game analysis
            detections, inference_time, processing_report = self.card_detector.detect_cards_from_pil_poker(image)
            
            if not detections:
                logger.warning("No cards detected by ML model")
                return [], {}
                
            # Step 3: Convert detection objects to dictionaries
            results = []
            for detection in detections:
                detection_dict = {
                    'card': detection.card_name,
                    'confidence': float(detection.confidence),
                    'bbox': detection.bbox,
                    'center': list(detection.center)
                }
                results.append(detection_dict)
            
            logger.info(f"ML detection complete: {len(results)} cards, inference: {inference_time:.3f}s")
            
            return results, game_analysis
            
        except Exception as e:
            logger.error(f"ML game analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fall back to regular detection
            return self._ml_card_detection(image, image_shape), {}
    
    def _mock_card_detection_enhanced(self, img_array: np.ndarray) -> List[Dict]:
        """
        Enhanced mock card detection function that returns structured results
        This maintains API compatibility while ML components are being set up
        """
        # Simulate processing time
        time.sleep(0.1)
        
        # Return mock structured results based on image characteristics
        height, width = img_array.shape[:2]
        
        # Mock detection results in the new structured format
        if width > 800 and height > 600:
            # Large image - simulate full poker table
            return [
                {
                    "type": "community_cards",
                    "stage": "flop",
                    "cards": [
                        {"name": "As", "confidence": 0.95, "bbox": [320, 280, 380, 320], "center": [350, 300]},
                        {"name": "Kh", "confidence": 0.92, "bbox": [390, 280, 450, 320], "center": [420, 300]},
                        {"name": "Qd", "confidence": 0.89, "bbox": [460, 280, 520, 320], "center": [490, 300]}
                    ],
                    "position": [420, 300],
                    "count": 3
                },
                {
                    "type": "player_hand",
                    "player_id": 1,
                    "cards": [
                        {"name": "Jc", "confidence": 0.87, "bbox": [200, 100, 260, 140], "center": [230, 120]},
                        {"name": "Th", "confidence": 0.84, "bbox": [270, 100, 330, 140], "center": [300, 120]}
                    ],
                    "position": [265, 120],
                    "confidence": 0.855,
                    "count": 2
                },
                {
                    "type": "analysis_summary",
                    "total_cards": 5,
                    "confidence_score": 0.89,
                    "game_stage": "flop",
                    "player_count": 1,
                    "metadata": {"mock_detection": True}
                }
            ]
        elif width > 400:
            # Medium image - simulate player hand only
            return [
                {
                    "type": "player_hand",
                    "player_id": 1,
                    "cards": [
                        {"name": "Jc", "confidence": 0.91, "bbox": [100, 150, 160, 190], "center": [130, 170]},
                        {"name": "Th", "confidence": 0.88, "bbox": [180, 150, 240, 190], "center": [210, 170]}
                    ],
                    "position": [170, 170],
                    "confidence": 0.895,
                    "count": 2
                },
                {
                    "type": "analysis_summary",
                    "total_cards": 2,
                    "confidence_score": 0.895,
                    "game_stage": "preflop",
                    "player_count": 1,
                    "metadata": {"mock_detection": True}
                }
            ]
        else:
            # Small image - single card
            return [
                {
                    "type": "unassigned_cards",
                    "cards": [
                        {"name": "7s", "confidence": 0.82, "bbox": [50, 50, 110, 90], "center": [80, 70]}
                    ],
                    "count": 1
                },
                {
                    "type": "analysis_summary",
                    "total_cards": 1,
                    "confidence_score": 0.82,
                    "game_stage": "unknown",
                    "player_count": 0,
                    "metadata": {"mock_detection": True}
                }
            ]
    
    def get_model_status(self) -> Dict:
        """Get current status of ML components"""
        status = {
            "ml_enabled": self.ml_enabled,
            "ml_available": ML_AVAILABLE,
            "using_mock_detection": not self.ml_enabled
        }
        
        if self.ml_enabled and hasattr(self, 'card_detector'):
            try:
                # Check if model is actually loaded
                if self.card_detector.model is not None:
                    status["model_loaded"] = True
                    status["model_device"] = str(self.card_detector.device)
                    status["performance_stats"] = self.card_detector.get_performance_stats()
                else:
                    status["model_loaded"] = False
                    status["error"] = "Model not loaded"
            except Exception as e:
                logger.error(f"Error getting model status: {e}")
                status["error"] = str(e)
        
        return status
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats"""
        return list(self.supported_formats)
    
    def _create_game_visualization(self, image: Image.Image, detection_results: List[Dict], 
                                 game_analysis: Dict, filename: str) -> Optional[str]:
        """
        Create a visualization of the poker game with winner announcement
        
        Args:
            image: PIL Image object
            detection_results: Detection results from ML processing
            game_analysis: Complete game analysis results
            filename: Original filename for naming the visualization
            
        Returns:
            Path to the created visualization image, or None if failed
        """
        try:
            # Import visualizer
            from .poker_visualizer import PokerGameVisualizer
            
            # Convert PIL image to OpenCV format
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image_array
            
            # Create visualizer and generate annotated image
            visualizer = PokerGameVisualizer()
            annotated_image = visualizer.visualize_game_result(
                image_cv, detection_results, game_analysis
            )
            
            # Generate output filename
            base_name = os.path.splitext(filename)[0] if filename else "poker_game"
            output_filename = f"{base_name}_annotated.jpg"
            output_path = os.path.join("visualizations", output_filename)
            
            # Create output directory if it doesn't exist
            os.makedirs("visualizations", exist_ok=True)
            
            # Save the annotated image
            cv2.imwrite(output_path, annotated_image)
            
            logger.info(f"Created game visualization: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create game visualization: {str(e)}")
            return None