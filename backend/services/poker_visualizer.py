"""
Integration code for poker game analysis with winner display
Visualizes poker game results with winner announcement
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)

class PokerGameVisualizer:
    """
    Visualizes poker game results with winner announcement
    """
    
    def __init__(self):
        self.colors = {
            'player1': (0, 255, 0),      # Green
            'player2': (255, 0, 0),      # Blue
            'community': (255, 255, 0),   # Cyan
            'winner': (0, 255, 255),      # Yellow
            'tie': (255, 0, 255)          # Magenta
        }
    
    def visualize_game_result(self, 
                             image: np.ndarray, 
                             detections: List[Dict],
                             game_result: Dict) -> np.ndarray:
        """
        Create visualization with winner announcement
        
        Args:
            image: Original image
            detections: Card detections with positions
            game_result: Result from analyze_poker_game
            
        Returns:
            Image with annotations
        """
        # Copy image for annotation
        annotated = image.copy()
        height, width = annotated.shape[:2]
        
        # Draw card detections with player colors
        annotated = self._draw_card_groups(annotated, detections, game_result)
        
        # Add text overlay with game information
        annotated = self._add_game_info_overlay(annotated, game_result)
        
        # Add winner announcement
        annotated = self._add_winner_announcement(annotated, game_result)
        
        return annotated
    
    def _draw_card_groups(self, image: np.ndarray, detections: List[Dict], game_result: Dict) -> np.ndarray:
        """Draw bounding boxes colored by player/community"""
        height, width = image.shape[:2]
        
        for det in detections:
            # Determine which group this card belongs to
            y_ratio = det['center'][1] / height
            
            if y_ratio < 0.33:
                color = self.colors['player1']
                label = "P1"
            elif y_ratio < 0.67:
                color = self.colors['community']
                label = "Community"
            else:
                color = self.colors['player2']
                label = "P2"
            
            # Draw bounding box
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add card name and group label
            card_text = f"{label}: {det['card_name']}"
            cv2.putText(image, card_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def _add_game_info_overlay(self, image: np.ndarray, game_result: Dict) -> np.ndarray:
        """Add semi-transparent overlay with game information"""
        height, width = image.shape[:2]
        
        # Create overlay
        overlay = image.copy()
        
        # Add semi-transparent rectangles for text backgrounds
        # Top area for community cards
        cv2.rectangle(overlay, (10, 10), (width - 10, 60), (0, 0, 0), -1)
        
        # Bottom area for player hands
        cv2.rectangle(overlay, (10, height - 120), (width - 10, height - 10), (0, 0, 0), -1)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Add text
        # Community cards
        community_text = "Community: " + ", ".join(game_result['community_cards'])
        cv2.putText(image, community_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Player hands
        y_offset = height - 90
        for player in game_result['players']:
            player_text = f"{player['name']}: {', '.join(player['hole_cards'])} - {player['best_hand']}"
            cv2.putText(image, player_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        return image
    
    def _add_winner_announcement(self, image: np.ndarray, game_result: Dict) -> np.ndarray:
        """Add large winner announcement"""
        height, width = image.shape[:2]
        
        # Create winner text
        if game_result['winner']:
            winner_text = f"WINNER: {game_result['winner']['name']}"
            hand_text = game_result['winner']['winning_hand']
            color = self.colors['winner']
        elif game_result['tie']:
            winner_text = "IT'S A TIE!"
            tied_names = ", ".join([p['name'] for p in game_result['tied_players']])
            hand_text = f"Between: {tied_names}"
            color = self.colors['tie']
        else:
            return image
        
        # Calculate text position (center of image)
        font_scale = 1.5
        thickness = 3
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            winner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Center position
        text_x = (width - text_width) // 2
        text_y = height // 2
        
        # Add background rectangle
        padding = 20
        cv2.rectangle(image, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + padding + 30),
                     (0, 0, 0), -1)
        
        # Add winner text
        cv2.putText(image, winner_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Add hand description
        cv2.putText(image, hand_text, (text_x, text_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return image

def process_poker_image_with_winner(image_path: str, detector) -> Dict:
    """
    Complete pipeline: detect cards → analyze game → determine winner
    
    Args:
        image_path: Path to poker image
        detector: Your YOLOv8 card detector
        
    Returns:
        Complete game analysis with winner
    """
    # Load image
    image = cv2.imread(image_path)
    pil_image = Image.open(image_path)
    
    # Detect cards
    detections, inference_time, report = detector.detect_cards_poker_optimized(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    
    # Convert to dict format
    detection_dicts = []
    for det in detections:
        detection_dicts.append({
            'card_name': det.card_name,
            'confidence': det.confidence,
            'bbox': det.bbox,
            'center': det.center
        })
    
    logger.info(f"Detected {len(detection_dicts)} cards")
    
    # Analyze game
    try:
        from ml.poker_game_analyzer import analyze_poker_game
        game_result = analyze_poker_game(detection_dicts, image.shape[:2])
    except ImportError:
        logger.error("Could not import poker_game_analyzer")
        return {}
    
    # Create visualization
    visualizer = PokerGameVisualizer()
    annotated_image = visualizer.visualize_game_result(image, detection_dicts, game_result)
    
    # Save or display result
    cv2.imwrite('poker_result.jpg', annotated_image)
    
    # Return complete analysis
    return {
        'detections': detection_dicts,
        'game_analysis': game_result,
        'annotated_image_path': 'poker_result.jpg',
        'inference_time': inference_time
    }

# Simple text output for testing
def print_game_result(game_result: Dict):
    """Print game result in a nice format"""
    print("\n" + "=" * 60)
    print("🃏 POKER GAME ANALYSIS 🃏")
    print("=" * 60)
    
    print(f"\n📍 Community Cards: {', '.join(game_result['community_cards'])}")
    
    print("\n👥 Players:")
    for player in game_result['players']:
        print(f"\n  {player['name']} ({player['position']})")
        print(f"    Cards: {', '.join(player['hole_cards'])}")
        print(f"    Best Hand: {player['best_hand']}")
        print(f"    Description: {player['hand_description']}")
    
    print("\n" + "-" * 60)
    
    if game_result['winner']:
        print(f"\n🏆 WINNER: {game_result['winner']['name']} 🏆")
        print(f"   Winning Hand: {game_result['winner']['winning_hand']}")
    elif game_result['tie']:
        tied_names = ", ".join([p['name'] for p in game_result['tied_players']])
        print(f"\n🤝 TIE between: {tied_names}")
    
    print("\n" + "=" * 60)

def create_poker_visualizer() -> PokerGameVisualizer:
    """Factory function to create a poker visualizer"""
    return PokerGameVisualizer()

def enhance_image_processor_with_winner_detection():
    """
    Integration helper that can be used to add winner detection to existing processors
    This is automatically integrated in the main ImageProcessor class
    """
    def process_image_with_winner(processor_self, image_data, filename: str):
        """
        Enhanced processing that includes winner determination
        """
        # Call the existing detection with game analysis
        results, processing_time, game_analysis, visualization_path = processor_self.process_image(
            image_data, filename, analyze_game=True, create_visualization=True
        )
        
        # Print game result to console for debugging
        if game_analysis:
            print_game_result(game_analysis)
        
        return results, processing_time, game_analysis, visualization_path
    
    return process_image_with_winner