from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
from datetime import datetime
from io import BytesIO
from PIL import UnidentifiedImageError

from models.schemas import (
    ImageUploadResponse, HealthCheckResponse, ErrorResponse, ModelStatusResponse,
    EvaluateWinnerRequest, EvaluateWinnerResponse, GameAnalysis, PlayerInfo, WinnerInfo
)
from services.image_processor import ImageProcessor
from ml.hand_evaluator import create_hand_evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PokerVision API",
    description="AI-powered poker card detection API",
    version="1.0.0"
)

# Configure CORS. Allowed origins come from the comma-separated CORS_ORIGINS env
# var; if unset we default to local dev + the deployed Netlify frontend.
# NOTE: a wildcard "*" together with allow_credentials=True is rejected by
# browsers (the CORS spec forbids credentialed requests when the server returns
# Access-Control-Allow-Origin: "*"), so we never default to "*".
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",            # frontend dev (docker-compose)
    "http://localhost:5173",            # frontend dev (Vite default)
    "https://pokervision.netlify.app",  # deployed frontend
]
cors_env = os.getenv("CORS_ORIGINS", "")
allow_origins_setting = [o.strip() for o in cors_env.split(",") if o.strip()] or DEFAULT_CORS_ORIGINS
if "*" in allow_origins_setting:
    logger.warning(
        "CORS_ORIGINS contains '*', which browsers reject together with "
        "allow_credentials=True. Set explicit origins instead."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_setting,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
image_processor = ImageProcessor()
hand_evaluator = create_hand_evaluator()

# Maximum accepted upload size (server-side guard; returns 413 if exceeded).
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Create visualizations directory if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

# Mount static files for visualizations
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint with model status"""
    # Get model status
    model_status = image_processor.get_model_status()
    
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        model_status=ModelStatusResponse(**model_status)
    )

@app.post("/upload", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    create_visualization: bool = Query(False, description="Create annotated visualization with winner announcement")
):
    """
    Upload and process poker card image
    """
    try:
        # Read the bytes first so we can enforce a server-side size limit.
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image is too large (max {MAX_UPLOAD_MB} MB)"
            )

        # content_type is unreliable from mobile clients (HEIC often arrives as
        # application/octet-stream, or missing), so it is only a soft check -- the
        # real gate is whether PIL can actually decode the bytes below.
        content_type = (file.content_type or "").lower()
        if content_type and not (
            content_type.startswith("image/") or content_type == "application/octet-stream"
        ):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Reject unreadable images with a clear 400 (instead of a later 500).
        if not image_processor.validate_image(BytesIO(content)):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a readable image"
            )
        
        # Process the image with game analysis and optional visualization
        detection_results, processing_time, game_analysis, visualization_path = image_processor.process_image(
            BytesIO(content), 
            file.filename,
            analyze_game=True,
            create_visualization=create_visualization
        )
        
        # Extract simple card names for backward compatibility
        simple_card_names = []
        for section in detection_results:
            if isinstance(section, dict):
                if section.get("type") in ["community_cards", "player_hand", "unassigned_cards"]:
                    # Structured format with sections
                    cards = section.get("cards", [])
                    simple_card_names.extend([card["name"] for card in cards])
                elif "card" in section:
                    # Direct detection format
                    simple_card_names.append(section["card"])
        
        # If no cards were extracted from structured format, try direct format
        if not simple_card_names and detection_results:
            for result in detection_results:
                if isinstance(result, dict) and "card" in result:
                    simple_card_names.append(result["card"])
        
        logger.info(f"Successfully processed image: {file.filename}")
        logger.info(f"Detection results type: {type(detection_results)}")
        logger.info(f"Detection results: {detection_results}")
        logger.info(f"Simple card names extracted: {simple_card_names}")
        
        # Convert game analysis to proper format if available, or create fallback
        game_analysis_response = None
        if game_analysis:
            from models.schemas import GameAnalysis, PlayerInfo, WinnerInfo
            
            players = []
            for player_data in game_analysis.get('players', []):
                players.append(PlayerInfo(**player_data))
            
            winner_info = None
            if game_analysis.get('winner'):
                winner_info = WinnerInfo(**game_analysis['winner'])
            
            game_analysis_response = GameAnalysis(
                community_cards=game_analysis.get('community_cards', []),
                players=players,
                winner=winner_info,
                tie=game_analysis.get('tie', False),
                tied_players=game_analysis.get('tied_players')
            )
        elif simple_card_names:
            # Create fallback game analysis when ML analysis fails but we have detected cards
            from models.schemas import GameAnalysis, PlayerInfo
            
            # Simple fallback: assume first 2 cards are player 1, rest are community
            fallback_players = []
            community_cards = simple_card_names[2:] if len(simple_card_names) > 2 else []
            
            if len(simple_card_names) >= 2:
                fallback_players.append(PlayerInfo(
                    id=1,
                    name="Player 1",
                    position="Button",
                    hole_cards=simple_card_names[:2],
                    hand_description="Cards detected (analysis unavailable)"
                ))
            
            game_analysis_response = GameAnalysis(
                community_cards=community_cards,
                players=fallback_players,
                winner=None,  # No winner determination in fallback
                tie=False
            )
            
            logger.info(f"Created fallback game analysis with {len(fallback_players)} player(s) and {len(community_cards)} community cards")
        
        return ImageUploadResponse(
            success=True,
            message="Image processed successfully",
            filename=file.filename,
            detection_results=detection_results,
            cards_detected=simple_card_names,  # Backward compatibility
            processing_time=processing_time,
            game_analysis=game_analysis_response,
            visualization_path=visualization_path
        )
        
    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get detailed model status information"""
    try:
        status = image_processor.get_model_status()
        return ModelStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get model status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )

@app.get("/model-status")
async def get_simple_model_status():
    """Simple model status endpoint for debugging (as suggested by Opus)"""
    try:
        status = image_processor.get_model_status()
        return {
            "ml_enabled": status.get("ml_enabled", False),
            "model_loaded": status.get("model_loaded", False),
            "using_mock_detection": status.get("using_mock_detection", True),
            "device": status.get("model_device", "unknown"),
            "debug_info": status
        }
    except Exception as e:
        return {
            "error": str(e),
            "ml_enabled": False,
            "model_loaded": False,
            "using_mock_detection": True
        }

@app.get("/version")
async def get_version():
    """App version, the loaded model file, and the device (for diagnostics)."""
    status = image_processor.get_model_status()
    return {
        "version": app.version,
        "model_file": status.get("model_file"),
        "model_loaded": status.get("model_loaded", False),
        "device": status.get("model_device", "unknown"),
    }

@app.post("/evaluate-winner", response_model=EvaluateWinnerResponse)
async def evaluate_winner(request: EvaluateWinnerRequest):
    """
    Evaluate winner from manually corrected cards

    Args:
        request: Contains player1_cards, community_cards, player2_cards

    Returns:
        Complete game analysis with winner information
    """
    try:
        # Validate input
        if len(request.player1_cards) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Player 1 must have exactly 2 cards, got {len(request.player1_cards)}"
            )

        if len(request.player2_cards) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Player 2 must have exactly 2 cards, got {len(request.player2_cards)}"
            )

        if not (3 <= len(request.community_cards) <= 5):
            raise HTTPException(
                status_code=400,
                detail=f"Community cards must be 3-5 cards, got {len(request.community_cards)}"
            )

        logger.info(f"🎯 Evaluating winner - Player 1: {request.player1_cards}, Community: {request.community_cards}, Player 2: {request.player2_cards}")

        # Evaluate both players' hands
        logger.info(f"🔵 Evaluating Player 1...")
        player1_result = hand_evaluator.evaluate_community_and_hole_cards(
            request.community_cards, request.player1_cards
        )
        logger.info(f"   Player 1 result: {player1_result['hand_rank']} (strength: {player1_result['hand_strength']})")

        logger.info(f"🔴 Evaluating Player 2...")
        player2_result = hand_evaluator.evaluate_community_and_hole_cards(
            request.community_cards, request.player2_cards
        )
        logger.info(f"   Player 2 result: {player2_result['hand_rank']} (strength: {player2_result['hand_strength']})")

        # Validate both hands
        if not player1_result.get('valid') or not player2_result.get('valid'):
            raise HTTPException(
                status_code=400,
                detail="Invalid cards or insufficient cards for evaluation"
            )

        # Determine winner
        player1_strength = player1_result['hand_strength']
        player2_strength = player2_result['hand_strength']

        logger.info(f"🏆 Comparing hands:")
        logger.info(f"   Player 1: {player1_result['hand_rank']} (strength: {player1_strength})")
        logger.info(f"   Player 2: {player2_result['hand_rank']} (strength: {player2_strength})")

        is_tie = player1_strength == player2_strength
        winner_id = None
        winner_name = None
        winning_hand = None

        if not is_tie:
            if player1_strength > player2_strength:
                winner_id = 1
                winner_name = "Player 1"
                winning_hand = player1_result['hand_rank']
                logger.info(f"✅ Winner: Player 1 with {winning_hand}")
            else:
                winner_id = 2
                winner_name = "Player 2"
                winning_hand = player2_result['hand_rank']
                logger.info(f"✅ Winner: Player 2 with {winning_hand}")
        else:
            logger.info(f"🤝 It's a TIE!")

        # Create player info objects
        players = [
            PlayerInfo(
                id=1,
                name="Player 1",
                position="Button",
                hole_cards=request.player1_cards,
                best_hand=', '.join(player1_result['best_5_cards']),
                hand_description=player1_result['hand_rank']
            ),
            PlayerInfo(
                id=2,
                name="Player 2",
                position="Big Blind",
                hole_cards=request.player2_cards,
                best_hand=', '.join(player2_result['best_5_cards']),
                hand_description=player2_result['hand_rank']
            )
        ]

        # Create winner info if not a tie
        winner_info = None
        tied_players = None

        if is_tie:
            tied_players = [
                {"id": 1, "name": "Player 1", "hand": player1_result['hand_rank']},
                {"id": 2, "name": "Player 2", "hand": player2_result['hand_rank']}
            ]
        else:
            winner_info = WinnerInfo(
                id=winner_id,
                name=winner_name,
                winning_hand=winning_hand
            )

        # Create game analysis
        game_analysis = GameAnalysis(
            community_cards=request.community_cards,
            players=players,
            winner=winner_info,
            tie=is_tie,
            tied_players=tied_players
        )

        logger.info(f"Winner evaluation complete: {winner_name if not is_tie else 'TIE'}")

        return EvaluateWinnerResponse(
            success=True,
            message="Winner evaluated successfully" if not is_tie else "Game is a tie",
            game_analysis=game_analysis
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Winner evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate winner: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )