# backend/api/routes/audio.py - UPDATED TO USE NEW PATTERNS
"""
Audio processing API routes - UPDATED to use centralized patterns
"""
import os
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any

from ...middleware.error_handling import handle_api_errors, validate_input
from ...services.mock_database import audio_service
from ..models import AudioUploadResponse, SuccessResponse
from ...utils import get_current_timestamp, validate_file_extension, get_file_size_mb
from ...config.settings import get_settings

from pydantic import BaseModel

class AudioProcessRequest(BaseModel):
    filename: str
    session_id: str = None
    
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

@router.get("/sample-files")
async def get_sample_audio_files():
    """Get list of available sample audio files"""
    
    audio_dir = Path("data/sample_audio")
    
    if not audio_dir.exists():
        raise HTTPException(status_code=404, detail="Sample audio directory not found")
    
    sample_files = []
    
    # Scan for audio files
    for file_path in audio_dir.glob("*.wav"):
        if file_path.is_file():
            # Determine call type and metadata from filename
            filename = file_path.name
            call_info = determine_call_info(filename)
            
            sample_files.append({
                "id": file_path.stem,
                "filename": filename,
                "title": call_info["title"],
                "description": call_info["description"],
                "icon": call_info["icon"],
                "file_path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "duration": call_info.get("duration", 30)  # Default duration
            })
    
    return {
        "status": "success",
        "files": sample_files,
        "count": len(sample_files)
    }

@router.get("/sample-files/{filename}")
async def get_sample_audio_file(filename: str):
    """Serve a sample audio file"""
    
    audio_path = Path(f"data/sample_audio/{filename}")
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=filename
    )

def determine_call_info(filename: str) -> Dict[str, Any]:
    """Determine call information from filename"""
    
    filename_lower = filename.lower()
    
    if "investment" in filename_lower:
        return {
            "title": "Investment Scam Call",
            "description": "Suspicious investment opportunity with guaranteed returns",
            "icon": "💰",
            "scam_type": "investment_scam",
            "duration": 45
        }
    elif "romance" in filename_lower:
        return {
            "title": "Romance Scam Call", 
            "description": "Online relationship emergency payment request",
            "icon": "💕",
            "scam_type": "romance_scam",
            "duration": 38
        }
    elif "impersonation" in filename_lower:
        return {
            "title": "Impersonation Scam",
            "description": "Fake bank security team requesting credentials",
            "icon": "🎭", 
            "scam_type": "impersonation_scam",
            "duration": 35
        }
    elif "legitimate" in filename_lower:
        return {
            "title": "Legitimate Call",
            "description": "Normal banking inquiry and account services",
            "icon": "✅",
            "scam_type": "none", 
            "duration": 25
        }
    elif "app" in filename_lower:
        return {
            "title": "APP Scam",
            "description": "Authorised Push Payment scam call",
            "icon": "💰",
            "scam_type": "app_scam", 
            "duration": 35
        }        
    else:
        return {
            "title": "Unknown Call Type",
            "description": "Audio file analysis pending",
            "icon": "🎵",
            "scam_type": "unknown",
            "duration": 30
        }

@router.post("/process-audio-realtime")
async def process_audio_realtime(request: AudioProcessRequest):
    filename = request.filename
    session_id = request.session_id or generate_session_id("realtime_audio")
    """Process audio file through the real-time multi-agent pipeline"""
    
    from ...agents.fraud_detection_system import fraud_detection_system
    
    try:
        # Generate session ID if not provided
        if not session_id:
            from ...utils import generate_session_id
            session_id = generate_session_id("realtime_audio")
        
        # Process through multi-agent system
        audio_path = f"data/sample_audio/{filename}"
        result = await fraud_detection_system.process_audio_file(audio_path, session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "analysis": result,
            "message": "Audio processed through multi-agent system"
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "session_id": session_id
        }
