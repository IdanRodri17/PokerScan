import time
import os
from pathlib import Path
from PIL import Image
import numpy as np
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
        try:
            # Try to find a trained model with fallback options
            model_dir = Path("ml/models")
            
            if model_dir.exists():
                # Priority order: .pt (PyTorch), .onnx (ONNX), pretrained
                pt_files = list(model_dir.glob("*.pt"))
                onnx_files = list(model_dir.glob("*.onnx"))
                
                model_loaded = False
                
                # First try PyTorch models
                for model_file in pt_files:
                    try:
                        model_path = str(model_file)
                        logger.info(f"Loading trained PyTorch model: {model_path}")
                        self.card_detector = create_card_detector(model_path=model_path)
                        model_loaded = True
                        logger.info("✅ PyTorch model loaded successfully")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load PyTorch model {model_file}: {e}")
                        continue
                
                # If PyTorch failed, try ONNX models
                if not model_loaded and onnx_files:
                    for onnx_file in onnx_files:
                        try:
                            logger.info(f"Trying ONNX model as fallback: {onnx_file}")
                            # Note: ONNX loading would need separate implementation
                            # For now, log the attempt and continue to pretrained
                            logger.warning("ONNX model support not yet implemented, using pretrained")
                            break
                        except Exception as e:
                            logger.warning(f"ONNX model loading failed: {e}")
                            continue
                
                # If all custom models failed, use pretrained
                if not model_loaded:
                    logger.info("Custom model loading failed, using pretrained YOLOv8")
                    self.card_detector = create_card_detector()
                    model_loaded = True
            else:
                # Use pretrained YOLOv8 for initial testing
                logger.info("No trained model directory found, using pretrained YOLOv8")
                self.card_detector = create_card_detector()
            
            # Initialize spatial analyzer
            config_path = Path("ml/config/model_config.yaml")
            if config_path.exists():
                self.spatial_analyzer = PokerSpatialAnalyzer(str(config_path))
            else:
                self.spatial_analyzer = PokerSpatialAnalyzer()
            
            # Initialize hand evaluator
            self.hand_evaluator = create_hand_evaluator()
            
            logger.info("ML components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
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
    
    def process_image(self, image_data: bytes, filename: str) -> Tuple[List[Dict], float]:
        """
        Process the uploaded image to detect poker cards using YOLOv8 and spatial analysis
        Returns structured detection results and processing time
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
            
            if self.ml_enabled:
                # Use ML pipeline for detection
                results = self._ml_card_detection(image, image_shape)
            else:
                # Fall back to mock detection
                results = self._mock_card_detection_enhanced(np.array(image))
            
            processing_time = time.time() - start_time
            
            logger.info(f"Processed image {filename} in {processing_time:.3f}s")
            logger.info(f"Detected {len(results)} cards")
            
            return results, processing_time
            
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
            if processing_report.get('spatial_duplicates_removed', 0) > 0:
                logger.info(f"Removed {processing_report['spatial_duplicates_removed']} spatial duplicates")
            if processing_report.get('card_duplicates_removed', 0) > 0:
                logger.info(f"Removed {processing_report['card_duplicates_removed']} card duplicates (poker rule)")
            
            # Step 2: Perform spatial analysis
            table_analysis = self.spatial_analyzer.analyze_table(detections, image_shape)
            
            # Step 3: Structure results for frontend
            structured_results = self._structure_detection_results(table_analysis, detections)
            
            return structured_results
            
        except Exception as e:
            logger.error(f"ML card detection failed: {e}")
            # Fall back to enhanced mock detection
            return self._mock_card_detection_enhanced(np.array(image))
    
    def _structure_detection_results(self, table_analysis, raw_detections) -> List[Dict]:
        """
        Structure detection results for the API response
        
        Args:
            table_analysis: PokerTableAnalysis object
            raw_detections: List of CardDetection objects
            
        Returns:
            List of structured detection dictionaries
        """
        results = []
        
        # Add community cards
        if table_analysis.community_cards:
            community_section = {
                "type": "community_cards",
                "stage": table_analysis.community_cards.stage.value,
                "cards": [],
                "position": list(table_analysis.community_cards.position),
                "count": len(table_analysis.community_cards.cards)
            }
            
            for card in table_analysis.community_cards.cards:
                community_section["cards"].append({
                    "name": card.card_name,
                    "confidence": card.confidence,
                    "bbox": card.bbox,
                    "center": list(card.center)
                })
            
            results.append(community_section)
        
        # Add player hands
        for player_hand in table_analysis.player_hands:
            player_section = {
                "type": "player_hand",
                "player_id": player_hand.player_id,
                "cards": [],
                "position": list(player_hand.position),
                "confidence": player_hand.confidence,
                "count": len(player_hand.cards)
            }
            
            for card in player_hand.cards:
                player_section["cards"].append({
                    "name": card.card_name,
                    "confidence": card.confidence,
                    "bbox": card.bbox,
                    "center": list(card.center)
                })
            
            results.append(player_section)
        
        # Add unassigned cards
        if table_analysis.unassigned_cards:
            unassigned_section = {
                "type": "unassigned_cards",
                "cards": [],
                "count": len(table_analysis.unassigned_cards)
            }
            
            for card in table_analysis.unassigned_cards:
                unassigned_section["cards"].append({
                    "name": card.card_name,
                    "confidence": card.confidence,
                    "bbox": card.bbox,
                    "center": list(card.center)
                })
            
            results.append(unassigned_section)
        
        # Evaluate poker hands
        hand_evaluations = self._evaluate_poker_hands(table_analysis)
        
        # Add analysis metadata
        analysis_summary = {
            "type": "analysis_summary",
            "total_cards": table_analysis.total_cards,
            "confidence_score": table_analysis.confidence_score,
            "game_stage": table_analysis.community_cards.stage.value if table_analysis.community_cards else "preflop",
            "player_count": len(table_analysis.player_hands),
            "hand_evaluations": hand_evaluations,
            "metadata": table_analysis.analysis_metadata
        }
        
        results.append(analysis_summary)
        
        return results
    
    def _evaluate_poker_hands(self, table_analysis) -> Dict:
        """
        Evaluate poker hands for all players using detected cards
        
        Args:
            table_analysis: PokerTableAnalysis object
            
        Returns:
            Dictionary containing hand evaluations for all players
        """
        hand_evaluations = {
            "community_cards_available": False,
            "player_hands": {},
            "best_possible_hand": None
        }
        
        try:
            # Get community cards
            community_card_names = []
            if table_analysis.community_cards and table_analysis.community_cards.cards:
                community_card_names = [card.card_name for card in table_analysis.community_cards.cards]
                hand_evaluations["community_cards_available"] = True
            
            # Evaluate each player's hand
            for i, player_hand in enumerate(table_analysis.player_hands):
                player_id = player_hand.player_id or f"player_{i+1}"
                hole_card_names = [card.card_name for card in player_hand.cards]
                
                if len(hole_card_names) >= 2:  # Need at least 2 hole cards
                    if community_card_names:
                        # Texas Hold'em style evaluation
                        evaluation = self.hand_evaluator.evaluate_community_and_hole_cards(
                            community_card_names, hole_card_names
                        )
                    else:
                        # Evaluate just the hole cards
                        evaluation = self.hand_evaluator.evaluate_best_hand(hole_card_names)
                        if evaluation:
                            evaluation = {
                                'valid': True,
                                'hand_rank': evaluation.hand_rank.display_name,
                                'hand_rank_value': evaluation.hand_rank.rank_value,
                                'hand_strength': evaluation.hand_strength,
                                'best_cards': [str(card) for card in evaluation.cards],
                                'hole_cards': hole_card_names,
                                'total_cards': len(hole_card_names)
                            }
                        else:
                            evaluation = {'valid': False, 'error': 'Not enough cards'}
                    
                    hand_evaluations["player_hands"][player_id] = evaluation
                else:
                    hand_evaluations["player_hands"][player_id] = {
                        'valid': False,
                        'error': f'Not enough hole cards: {len(hole_card_names)}'
                    }
            
            # Find best possible hand if we have community cards
            if community_card_names and len(community_card_names) >= 3:
                best_hand = self.hand_evaluator.evaluate_best_hand(community_card_names)
                if best_hand:
                    hand_evaluations["best_possible_hand"] = {
                        'hand_rank': best_hand.hand_rank.display_name,
                        'hand_rank_value': best_hand.hand_rank.rank_value,
                        'cards': [str(card) for card in best_hand.cards[:5]]
                    }
            
        except Exception as e:
            logger.error(f"Hand evaluation failed: {e}")
            hand_evaluations["error"] = str(e)
        
        return hand_evaluations
    
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
        
        if self.ml_enabled:
            try:
                status["card_detector"] = self.card_detector.get_model_info()
                status["performance_stats"] = self.card_detector.get_performance_stats()
            except Exception as e:
                logger.error(f"Error getting model status: {e}")
                status["error"] = str(e)
        
        return status
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats"""
        return list(self.supported_formats)