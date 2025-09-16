from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
from datetime import datetime
from io import BytesIO

from models.schemas import ImageUploadResponse, HealthCheckResponse, ErrorResponse, ModelStatusResponse
from services.image_processor import ImageProcessor

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

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
image_processor = ImageProcessor()

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
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate image data
        if not image_processor.validate_image(BytesIO(content)):
            raise HTTPException(
                status_code=400,
                detail="Invalid image format"
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
        
        # Convert game analysis to proper format if available
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