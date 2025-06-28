# backend/api/routes/audio.py - UPDATED TO USE NEW PATTERNS
"""
Audio processing API routes - UPDATED to use centralized patterns
"""

from fastapi import APIRouter, Depends, File, Form, UploadFile
from ...middleware.error_handling import handle_api_errors, validate_input
from ...services.mock_database import audio_service
from ..models import AudioUploadResponse, SuccessResponse
from ...utils import get_current_timestamp, validate_file_extension, get_file_size_mb
from ...config.settings import get_settings

router = APIRouter()

@router.post("/upload", response_model=AudioUploadResponse)
@handle_api_errors("audio_upload")
@validate_input(
    file={"required": True, "type": UploadFile},
    session_id={"required": False, "type": str}
)
async def upload_audio_file(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    customer_id: str = Form(None),
    agent_id: str = Form(None),
    settings = Depends(get_settings)
):
    """Upload an audio file for fraud detection analysis"""
    
    # Validate file using centralized utilities
    if not validate_file_extension(file.filename, settings.supported_audio_formats):
        raise ValueError(f"Unsupported file format. Supported: {settings.supported_audio_formats}")
    
    # Read and validate file size
    content = await file.read()
    if len(content) > settings.get_max_file_size_bytes():
        raise ValueError(f"File too large. Maximum size: {settings.max_audio_file_size_mb}MB")
    
    # Generate file ID and store metadata using centralized service
    file_id = f"audio_{get_current_timestamp().replace(':', '').replace('-', '')[:14]}"
    
    metadata = {
        "file_id": file_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "size_mb": get_file_size_mb(len(content)),
        "format": file.filename.split('.')[-1].lower(),
        "session_id": session_id,
        "customer_id": customer_id,
        "agent_id": agent_id,
        "status": "uploaded"
    }
    
    # Store using centralized audio service
    audio_service.store_audio_metadata(file_id, metadata)
    
    return AudioUploadResponse(
        status="success",
        file_id=file_id,
        filename=file.filename,
        size_bytes=len(content),
        format=metadata["format"],
        message="File uploaded successfully"
    )
