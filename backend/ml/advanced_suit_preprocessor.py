"""
Advanced Preprocessing for Poker Card Suit Recognition
Specifically designed to handle red suit confusion and green felt background issues
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class PokerSuitPreprocessor:
    """Advanced preprocessing to solve suit recognition issues"""
    
    def __init__(self):
        self.debug_mode = False
        
    def preprocess_for_suits(self, image: Image.Image) -> np.ndarray:
        """
        Main preprocessing pipeline optimized for suit recognition
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed numpy array (BGR format)
        """
        # Convert to numpy array
        img = np.array(image)
        if img.shape[2] == 4:  # Remove alpha channel if present
            img = img[:, :, :3]
        
        # Apply multi-stage preprocessing
        img = self._neutralize_green_background(img)
        img = self._enhance_red_suit_separation(img)
        img = self._adaptive_color_correction(img)
        img = self._enhance_suit_symbols(img)
        
        # Convert RGB to BGR for YOLOv8
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    
    def _neutralize_green_background(self, img: np.ndarray) -> np.ndarray:
        """
        Neutralize green felt background influence on card colors
        """
        # Convert to LAB color space for better color manipulation
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Create green mask (poker felt)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        green_lower = np.array([35, 30, 20])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Reduce green channel influence
        green_mask_inv = cv2.bitwise_not(green_mask)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Enhance a channel (red-green) to separate suits better
        a_enhanced = cv2.addWeighted(a, 1.3, np.zeros_like(a), 0, 0)
        
        # Reconstruct image
        lab_enhanced = cv2.merge([l, a_enhanced, b])
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return img_enhanced
    
    def _enhance_red_suit_separation(self, img: np.ndarray) -> np.ndarray:
        """
        Specifically enhance distinction between hearts and diamonds
        """
        # Work in HSV space for better color control
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Hearts: Pure red (hue ~0-10 or ~170-180)
        # Diamonds: Orange-red (hue ~10-25)
        
        # Create masks for different red ranges
        hearts_mask1 = cv2.inRange(h, 0, 10)
        hearts_mask2 = cv2.inRange(h, 170, 180)
        hearts_mask = cv2.bitwise_or(hearts_mask1, hearts_mask2)
        
        diamonds_mask = cv2.inRange(h, 10, 25)
        
        # Shift hues to make them more distinct
        h_modified = h.copy()
        
        # Make hearts more pure red (shift toward 0)
        h_modified[hearts_mask > 0] = np.clip(h[hearts_mask > 0] * 0.7, 0, 180)
        
        # Make diamonds more orange (shift toward 20)
        h_modified[diamonds_mask > 0] = np.clip(h[diamonds_mask > 0] * 1.3 + 5, 0, 180)
        
        # Boost saturation for red suits
        red_suits_mask = cv2.bitwise_or(hearts_mask, diamonds_mask)
        s_modified = s.copy()
        s_modified[red_suits_mask > 0] = np.clip(s[red_suits_mask > 0] * 1.4, 0, 255)
        
        # Reconstruct
        hsv_enhanced = cv2.merge([h_modified, s_modified, v])
        img_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
        
        return img_enhanced
    
    def _adaptive_color_correction(self, img: np.ndarray) -> np.ndarray:
        """
        Apply adaptive color correction based on lighting conditions
        """
        # Estimate white balance
        result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        
        # Correct color cast
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        
        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # Apply subtle gamma correction for better contrast
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(result, table)
        
        return result
    
    def _enhance_suit_symbols(self, img: np.ndarray) -> np.ndarray:
        """
        Sharpen and enhance suit symbols for better detection
        """
        # Create high-pass filter for edge enhancement
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Apply sharpening
        sharpened = cv2.filter2D(img, -1, kernel * 0.3)
        
        # Blend with original
        result = cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
        
        # Apply bilateral filter to reduce noise while preserving edges
        result = cv2.bilateralFilter(result, 9, 50, 50)
        
        return result

def create_advanced_preprocessor():
    """Factory function to create preprocessor instance"""
    return PokerSuitPreprocessor()

# Integration with your existing card_detector.py
def preprocess_poker_image_advanced(pil_image: Image.Image) -> np.ndarray:
    """
    Advanced preprocessing specifically for poker tables with suit recognition
    
    Args:
        pil_image: PIL Image
        
    Returns:
        Preprocessed image as numpy array (BGR)
    """
    preprocessor = PokerSuitPreprocessor()
    return preprocessor.preprocess_for_suits(pil_image)