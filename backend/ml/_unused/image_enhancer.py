"""
Image Enhancement Pipeline for Poker Card Detection

This module provides specialized image preprocessing for poker card detection,
addressing common issues like poor lighting, blurry cards, and low contrast.
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple

logger = logging.getLogger(__name__)


class PokerImageEnhancer:
    """
    Specialized image enhancement for poker card detection
    
    Addresses common poker image issues:
    - Poor lighting on poker tables
    - Blurry card details
    - Low contrast between cards and table
    - Shadows and reflections
    """
    
    def __init__(self):
        """Initialize the image enhancer"""
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.sharpening_kernel = np.array([[-1, -1, -1], 
                                          [-1, 9, -1], 
                                          [-1, -1, -1]])
    
    def enhance_for_poker_detection(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply comprehensive enhancement pipeline for poker card detection
        Includes domain-specific fixes for poker table images vs training data
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Enhanced image as numpy array (BGR format for YOLO)
        """
        # Convert to OpenCV format if needed
        if isinstance(image, Image.Image):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Assume RGB numpy array
            img_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_cv = image.copy()
        
        # Apply Claude Opus's domain-specific enhancement pipeline
        enhanced = self._apply_green_felt_correction(img_cv)      # Fix green background bias
        enhanced = self._apply_suit_color_normalization(enhanced)  # Improve suit recognition  
        enhanced = self._apply_contrast_enhancement(enhanced)       # Original contrast fix
        enhanced = self._apply_card_edge_enhancement(enhanced)      # Better card boundaries
        enhanced = self._apply_lighting_normalization(enhanced)     # Consistent lighting
        enhanced = self._apply_sharpening(enhanced)                # Original sharpening
        enhanced = self._apply_noise_reduction(enhanced)           # Original noise reduction
        
        return enhanced
    
    def _apply_green_felt_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Correct green felt background bias that affects suit color recognition
        Addresses the domain gap between training data (clean backgrounds) and poker tables
        """
        try:
            # Convert to HSV to work with hue/saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create mask for green regions (poker table felt)
            # Green hue range in HSV: approximately 40-80 degrees  
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Reduce green saturation to minimize color cast on cards
            hsv[:, :, 1] = np.where(green_mask > 0, 
                                   hsv[:, :, 1] * 0.7,  # Reduce green saturation
                                   hsv[:, :, 1])
            
            # Convert back to BGR
            corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            logger.debug("Applied green felt correction")
            return corrected
            
        except Exception as e:
            logger.warning(f"Green felt correction failed: {e}")
            return image
    
    def _apply_suit_color_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize suit colors to improve red/black discrimination
        Critical for fixing 4h→4s, Ad→Ah type confusion
        """
        try:
            # Convert to LAB color space for better color manipulation
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Enhance A and B channels (color information) while preserving lightness
            lab[:, :, 1] = cv2.convertScaleAbs(lab[:, :, 1], alpha=1.2, beta=0)  # Green-Red
            lab[:, :, 2] = cv2.convertScaleAbs(lab[:, :, 2], alpha=1.2, beta=0)  # Blue-Yellow
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Additional red/black separation enhancement
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            
            # Boost saturation for red cards (hearts/diamonds) 
            red_lower = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_upper2 = np.array([170, 255, 255])
            red_mask1 = cv2.inRange(hsv, red_lower, red_upper1)
            red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Enhance red card visibility
            hsv[:, :, 1] = np.where(red_mask > 0,
                                   np.clip(hsv[:, :, 1] * 1.3, 0, 255).astype(np.uint8),
                                   hsv[:, :, 1])
            
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            logger.debug("Applied suit color normalization")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Suit color normalization failed: {e}")
            return image
    
    def _apply_card_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance card edges and boundaries for better detection
        Helps with angled cards and overlapping detection
        """
        try:
            # Create edge-enhanced version
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges slightly
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Convert edges back to 3-channel
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Blend with original (subtle enhancement)
            enhanced = cv2.addWeighted(image, 0.85, edges_bgr, 0.15, 0)
            
            logger.debug("Applied card edge enhancement")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Card edge enhancement failed: {e}")
            return image
    
    def _apply_lighting_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize lighting to match training data conditions
        Addresses uneven poker table lighting
        """
        try:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            # Apply histogram equalization to Y channel (luminance)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # Convert back to BGR
            normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # Blend with original to avoid over-processing
            enhanced = cv2.addWeighted(image, 0.6, normalized, 0.4, 0)
            
            logger.debug("Applied lighting normalization")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Lighting normalization failed: {e}")
            return image
    
    def _apply_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Particularly effective for poker tables with uneven lighting
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            logger.debug("Applied CLAHE contrast enhancement")
            return enhanced
        
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter to enhance card details
        Helps with slightly blurry cards from phone cameras
        """
        try:
            sharpened = cv2.filter2D(image, -1, self.sharpening_kernel)
            
            # Blend with original to avoid over-sharpening
            enhanced = cv2.addWeighted(image, 0.6, sharpened, 0.4, 0)
            
            logger.debug("Applied sharpening filter")
            return enhanced
        
        except Exception as e:
            logger.warning(f"Sharpening failed: {e}")
            return image
    
    def _apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gentle noise reduction while preserving card details
        """
        try:
            # Use bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            logger.debug("Applied noise reduction")
            return denoised
        
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return image
    
    def _apply_brightness_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply automatic brightness correction for dark poker table images
        """
        try:
            # Calculate average brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            # Target brightness (0-255 scale)
            target_brightness = 128
            
            # Calculate brightness adjustment
            brightness_diff = target_brightness - avg_brightness
            
            # Apply adjustment only if significant difference
            if abs(brightness_diff) > 20:
                # Convert to float to avoid overflow
                adjusted = image.astype(np.float32)
                adjusted += brightness_diff
                
                # Clip values to valid range
                adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
                
                logger.debug(f"Applied brightness correction: {brightness_diff:.1f}")
                return adjusted
            else:
                return image
        
        except Exception as e:
            logger.warning(f"Brightness correction failed: {e}")
            return image
    
    def preprocess_for_yolo(self, image: Union[np.ndarray, Image.Image], 
                           target_size: int = 1024) -> np.ndarray:
        """
        Complete preprocessing pipeline for YOLO inference
        
        Args:
            image: Input image
            target_size: Target image size for YOLO model
            
        Returns:
            Preprocessed image ready for YOLO inference
        """
        # Apply enhancement
        enhanced = self.enhance_for_poker_detection(image)
        
        # Resize for YOLO while maintaining aspect ratio
        h, w = enhanced.shape[:2]
        
        # Calculate scaling factor
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (square)
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # Gray padding
        
        # Calculate padding offsets
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        logger.debug(f"Preprocessed image: {w}x{h} → {target_size}x{target_size}")
        
        return padded
    
    def test_enhancement_pipeline(self, image: Union[np.ndarray, Image.Image]) -> dict:
        """
        Test the enhancement pipeline and return before/after statistics
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with enhancement statistics
        """
        # Convert to cv2 format
        if isinstance(image, Image.Image):
            original_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            original_cv = image.copy()
        
        # Apply enhancement
        enhanced = self.enhance_for_poker_detection(original_cv)
        
        # Calculate statistics
        original_gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        stats = {
            'original_brightness': float(np.mean(original_gray)),
            'enhanced_brightness': float(np.mean(enhanced_gray)),
            'original_contrast': float(np.std(original_gray)),
            'enhanced_contrast': float(np.std(enhanced_gray)),
            'brightness_improvement': float(np.mean(enhanced_gray) - np.mean(original_gray)),
            'contrast_improvement': float(np.std(enhanced_gray) - np.std(original_gray))
        }
        
        return stats


def create_poker_enhancer() -> PokerImageEnhancer:
    """
    Factory function to create a PokerImageEnhancer instance
    
    Returns:
        Configured PokerImageEnhancer instance
    """
    return PokerImageEnhancer()