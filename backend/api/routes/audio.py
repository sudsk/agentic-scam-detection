"""
Audio processing API routes for HSBC Scam Detection Agent
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator

from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()

class AudioUploadResponse(BaseModel):
    """Response model for audio upload"""
    file_id: str
    filename: str
    size_bytes: int
    duration_seconds: Optional[float] = None
    format: str
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    upload_timestamp: str
    status: str
    message: str

class AudioProcessingStatus(BaseModel):
    """Model for audio processing status"""
    file_id: str
    status: str  # uploaded, processing, transcribing, analyzing, completed, failed
    progress_percentage: int
    current_step: str
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None

class TranscriptionResult(BaseModel):
    """Model for transcription results"""
    file_id: str
    transcript_text: str
    confidence_score: float
    speaker_segments: List[Dict]
    word_timestamps: List[Dict]
    processing_time_seconds: float
    language_detected: str

class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis"""
    file_id: str
    analysis_type: str = "full"  # full, quick, custom
    include_transcription: bool = True
    include_fraud_detection: bool = True
    include_sentiment: bool = False
    speaker_diarization: bool = True

class AudioMetadata(BaseModel):
    """Model for audio file metadata"""
    file_id: str
    filename: str
    size_bytes: int
    duration_seconds: Optional[float]
    format: str
    sample_rate: Optional[int]
    channels: Optional[int]
    bitrate: Optional[int]
    upload_timestamp: str
    last_accessed: str

@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    customer_id: Optional[str] = Form(None),
    agent_id: Optional[str] = Form(None),
    settings: Settings = Depends(get_settings)
):
    """
    Upload an audio file for fraud detection analysis
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Check file format
        if not settings.is_audio_format_supported(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Supported formats: {settings.supported_audio_formats}"
            )
        
        # Read and validate file size
        content = await file.read()
        if len(content) > settings.get_max_file_size_bytes():
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.max_audio_file_size_mb}MB"
            )
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        stored_filename = f"{file_id}{file_extension}"
        
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = upload_dir / stored_filename
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        
        # Get audio metadata (mock implementation for demo)
        audio_metadata = await _get_audio_metadata(file_path, content)
        
        # Create response
        response = AudioUploadResponse(
            file_id=file_id,
            filename=file.filename,
            size_bytes=len(content),
            duration_seconds=audio_metadata.get("duration"),
            format=audio_metadata.get("format", file_extension.lstrip(".")),
            sample_rate=audio_metadata.get("sample_rate"),
            channels=audio_metadata.get("channels"),
            upload_timestamp=datetime.now().isoformat(),
            status="uploaded",
            message="File uploaded successfully"
        )
        
        # Log upload
        logger.info(f"📁 Audio file uploaded: {file.filename} ({len(content)} bytes) - ID: {file_id}")
        
        # Store metadata (in production, this would go to Firestore)
        await _store_audio_metadata(file_id, response, session_id, customer_id, agent_id)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading audio file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )

@router.post("/analyze/{file_id}")
async def analyze_audio(
    file_id: str,
    request: AudioAnalysisRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Start analysis of an uploaded audio file
    """
    try:
        # Verify file exists
        file_path = Path("uploads") / f"{file_id}.*"
        matching_files = list(Path("uploads").glob(f"{file_id}.*"))
        
        if not matching_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audio file not found"
            )
        
        file_path = matching_files[0]
        
        # Start async analysis
        analysis_task = asyncio.create_task(
            _perform_audio_analysis(file_id, file_path, request, settings)
        )
        
        return {
            "file_id": file_id,
            "status": "analysis_started",
            "message": "Audio analysis started. Use /status/{file_id} to check progress."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting audio analysis for {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start analysis: {str(e)}"
        )

@router.get("/status/{file_id}", response_model=AudioProcessingStatus)
async def get_processing_status(file_id: str):
    """
    Get the processing status of an audio file
    """
    try:
        # In production, this would query Firestore or Redis
        # For demo, return mock status
        status_data = await _get_processing_status(file_id)
        
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found or not being processed"
            )
        
        return AudioProcessingStatus(**status_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing status"
        )

@router.get("/transcription/{file_id}", response_model=TranscriptionResult)
async def get_transcription(file_id: str):
    """
    Get transcription results for an audio file
    """
    try:
        # In production, this would query the transcription database
        transcription_data = await _get_transcription_result(file_id)
        
        if not transcription_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transcription not found. File may not be processed yet."
            )
        
        return TranscriptionResult(**transcription_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transcription for {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get transcription"
        )

@router.get("/download/{file_id}")
async def download_audio_file(file_id: str):
    """
    Download an uploaded audio file
    """
    try:
        # Find the file
        matching_files = list(Path("uploads").glob(f"{file_id}.*"))
        
        if not matching_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audio file not found"
            )
        
        file_path = matching_files[0]
        
        # Return file
        return FileResponse(
            path=str(file_path),
            filename=f"audio_{file_id}{file_path.suffix}",
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download file"
        )

@router.delete("/delete/{file_id}")
async def delete_audio_file(file_id: str):
    """
    Delete an uploaded audio file and its associated data
    """
    try:
        # Find and delete the file
        matching_files = list(Path("uploads").glob(f"{file_id}.*"))
        deleted_files = 0
        
        for file_path in matching_files:
            try:
                os.remove(file_path)
                deleted_files += 1
                logger.info(f"🗑️ Deleted audio file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
        
        if deleted_files == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audio file not found"
            )
        
        # Delete metadata (in production, remove from Firestore)
        await _delete_audio_metadata(file_id)
        
        return {
            "file_id": file_id,
            "status": "deleted",
            "message": f"Successfully deleted {deleted_files} file(s)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete file"
        )

@router.get("/list")
async def list_audio_files(
    limit: int = 50,
    offset: int = 0,
    session_id: Optional[str] = None
):
    """
    List uploaded audio files with optional filtering
    """
    try:
        # In production, this would query Firestore with proper pagination
        files_data = await _list_audio_files(limit, offset, session_id)
        
        return {
            "files": files_data,
            "total_count": len(files_data),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing audio files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list files"
        )

# Helper functions (mock implementations for demo)

async def _get_audio_metadata(file_path: Path, content: bytes) -> Dict:
    """Get audio file metadata (mock implementation)"""
    # In production, use librosa or similar library
    file_extension = file_path.suffix.lower()
    
    # Mock metadata based on file extension
    metadata = {
        "format": file_extension.lstrip("."),
        "size_bytes": len(content),
        "duration": 30.0,  # Mock 30 seconds
        "sample_rate": 16000,
        "channels": 1,
        "bitrate": 128000
    }
    
    return metadata

async def _store_audio_metadata(
    file_id: str, 
    response: AudioUploadResponse, 
    session_id: Optional[str],
    customer_id: Optional[str],
    agent_id: Optional[str]
):
    """Store audio metadata (mock implementation)"""
    # In production, store in Firestore
    metadata = {
        "file_id": file_id,
        "filename": response.filename,
        "size_bytes": response.size_bytes,
        "upload_timestamp": response.upload_timestamp,
        "session_id": session_id,
        "customer_id": customer_id,
        "agent_id": agent_id,
        "status": "uploaded"
