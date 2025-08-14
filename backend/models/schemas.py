from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class ImageUploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    timestamp: datetime
    cards_detected: Optional[List[str]] = None
    processing_time: Optional[float] = None

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime