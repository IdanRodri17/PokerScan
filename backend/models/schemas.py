from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import json

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

class PlayerInfo(BaseModel):
    """Player information from game analysis"""
    id: int
    name: str
    position: str
    hole_cards: List[str]
    best_hand: Optional[str] = None
    hand_description: Optional[str] = None

class WinnerInfo(BaseModel):
    """Winner information"""
    id: int
    name: str
    winning_hand: str

class GameAnalysis(BaseModel):
    """Complete poker game analysis results"""
    community_cards: List[str]
    players: List[PlayerInfo]
    winner: Optional[WinnerInfo] = None
    tie: bool = False
    tied_players: Optional[List[Dict[str, Any]]] = None

class ImageUploadResponse(BaseModel):
    """Enhanced response with structured card detection results"""
    success: bool
    message: str
    filename: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    detection_results: Optional[List[Dict[str, Any]]] = None  # Flexible structure for different section types
    processing_time: Optional[float] = None
    game_analysis: Optional[GameAnalysis] = None  # Complete poker game analysis
    visualization_path: Optional[str] = None  # Path to annotated visualization image
    
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
    model_config = {"protected_namespaces": ()}
    
    status: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str
    model_status: Optional[ModelStatusResponse] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class EvaluateWinnerRequest(BaseModel):
    """Request schema for manual winner evaluation"""
    player1_cards: List[str] = Field(..., description="Player 1 hole cards (2 cards)")
    community_cards: List[str] = Field(..., description="Community cards (3-5 cards)")
    player2_cards: List[str] = Field(..., description="Player 2 hole cards (2 cards)")

class EvaluateWinnerResponse(BaseModel):
    """Response schema for winner evaluation"""
    success: bool
    message: str
    game_analysis: GameAnalysis
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())