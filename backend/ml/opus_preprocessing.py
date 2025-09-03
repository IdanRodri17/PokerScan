"""
Claude Opus's Enhanced preprocessing for poker table images to reduce domain gap
Focus on suit color normalization and card visibility enhancement
"""

import cv2
import numpy as np
from PIL import Image
import colorsys

def preprocess_poker_image(image):
    """
    Enhanced preprocessing specifically for poker table images
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        numpy array: Preprocessed image optimized for card detection
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    # 1. Remove green felt bias (critical for suit recognition)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Reduce green channel dominance
    img_lab[:, :, 1] = cv2.subtract(img_lab[:, :, 1], 10)  # Reduce green-magenta axis
    
    # 2. Enhance contrast for card regions (white cards on dark background)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpen cards (reduces blur from angle/distance)
    kernel = np.array([[-1,-1,-1,-1,-1],
                       [-1, 2, 2, 2,-1],
                       [-1, 2, 8, 2,-1],
                       [-1, 2, 2, 2,-1],
                       [-1,-1,-1,-1,-1]]) / 8.0
    img = cv2.filter2D(img, -1, kernel)
    
    # 4. Enhance red colors (for hearts/diamonds vs black suits)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Boost red hue regions
    red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)) + \
               cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    
    hsv[:, :, 1] = np.where(red_mask > 0, 
                            np.clip(hsv[:, :, 1] * 1.2, 0, 255).astype(np.uint8),
                            hsv[:, :, 1])
    
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 5. Normalize lighting (compensate for table lighting)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    return img

def detect_with_enhanced_preprocessing(model, image, conf=0.25, iou=0.40):
    """
    Run detection with enhanced preprocessing
    
    Args:
        model: YOLOv8 model
        image: Input image (PIL or numpy)
        conf: Confidence threshold
        iou: IoU threshold
        
    Returns:
        Detection results
    """
    # Apply enhanced preprocessing
    processed_img = preprocess_poker_image(image)
    
    # Run detection on processed image
    results = model(processed_img, conf=conf, iou=iou, imgsz=1024)
    
    return results

def multi_scale_detection(model, image, scales=[832, 1024, 1280]):
    """
    Run detection at multiple scales and combine results
    
    Args:
        model: YOLOv8 model
        image: Input image
        scales: List of input sizes to try
        
    Returns:
        Combined detection results
    """
    all_detections = []
    
    for scale in scales:
        processed_img = preprocess_poker_image(image)
        results = model(processed_img, conf=0.20, iou=0.35, imgsz=scale)
        
        if results and len(results) > 0:
            for detection in results[0].boxes:
                all_detections.append({
                    'bbox': detection.xyxy[0].cpu().numpy(),
                    'conf': detection.conf[0].cpu().numpy(),
                    'class': int(detection.cls[0].cpu().numpy()),
                    'scale': scale
                })
    
    # Remove duplicates using Non-Maximum Suppression
    # This would need proper NMS implementation
    return all_detections