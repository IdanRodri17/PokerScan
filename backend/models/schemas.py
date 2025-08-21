from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

class CardDetection(BaseModel):
    """Individual card detection result"""
    name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    center: List[float]  # [x, y]

class CommunityCardsSection(BaseModel):
    """Community cards detection section"""
    type: str = "community_cards"
    stage: str  # preflop, flop, turn, river
    cards: List[CardDetection]
    position: List[float]  # [x, y] center position
    count: int

class PlayerHandSection(BaseModel):
    """Player hand detection section"""
    type: str = "player_hand"
    player_id: int
    cards: List[CardDetection]
    position: List[float]  # [x, y] center position
    confidence: float
    count: int

class UnassignedCardsSection(BaseModel):
    """Unassigned cards detection section"""
    type: str = "unassigned_cards"
    cards: List[CardDetection]
    count: int

class AnalysisSummary(BaseModel):
    """Analysis summary section"""
    type: str = "analysis_summary"
    total_cards: int
    confidence_score: float
    game_stage: str
    player_count: int
    metadata: Dict[str, Any]

class ImageUploadResponse(BaseModel):
    """Enhanced response with structured card detection results"""
    success: bool
    message: str
    filename: Optional[str] = None
    timestamp: datetime
    detection_results: Optional[List[Dict[str, Any]]] = None  # Flexible structure for different section types
    processing_time: Optional[float] = None
    
    # Backward compatibility
    cards_detected: Optional[List[str]] = None  # Simple card names for backward compatibility

class ModelStatusResponse(BaseModel):
    """Model status information"""
    ml_enabled: bool
    ml_available: bool
    using_mock_detection: bool
    card_detector: Optional[Dict[str, Any]] = None
    performance_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    model_status: Optional[ModelStatusResponse] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime