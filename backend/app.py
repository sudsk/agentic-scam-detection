# backend/app.py - UPDATED WITH FRAUD DETECTION ORCHESTRATOR
"""
Fraud Detection Agent - FastAPI Backend
UPDATED: Now uses FraudDetectionOrchestrator instead of fraud_detection_system
Eliminates redundant fraud_detection_system.py
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path if not already there
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import aiofiles

from .orchestrator.fraud_detection_orchestrator import FraudDetectionOrchestrator, FraudDetectionWebSocketHandler
from .websocket.connection_manager import ConnectionManager
from .api.routes import audio, fraud, cases
from .config.settings import get_settings
from .middleware.error_handling import error_handling_middleware, handle_api_errors
from .services.mock_database import mock_db, audio_service, case_service, session_service
from .utils import (
    get_current_timestamp, generate_session_id, create_success_response,
    create_error_response, log_agent_activity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ===== SUPPRESS ADK VERBOSE LOGGING =====
# Suppress Google ADK internal logging to reduce noise
logging.getLogger('google_adk').setLevel(logging.WARNING)
logging.getLogger('google_adk.google.adk.models.google_llm').setLevel(logging.WARNING)
logging.getLogger('google_adk.google.adk.runners').setLevel(logging.WARNING)
logging.getLogger('google_adk.google.adk.sessions').setLevel(logging.WARNING)
logging.getLogger('google_adk.google.adk.agents').setLevel(logging.WARNING)

# Also suppress related Google Cloud logging
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)
logging.getLogger('google.api_core').setLevel(logging.WARNING)
logging.getLogger('googleapis').setLevel(logging.WARNING)

# Suppress HTTP request logging
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Load centralized settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection Agent API",
    description="Real-time fraud detection using FraudDetectionOrchestrator with multi-agent coordination",
    version="3.0.0",  # UPDATED: Version bump for orchestrator architecture
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add consolidated error handling middleware
app.middleware("http")(error_handling_middleware)

# Configure CORS using centralized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# UPDATED: Initialize orchestrator and WebSocket components
connection_manager = ConnectionManager()
orchestrator = FraudDetectionOrchestrator(connection_manager)
fraud_ws_handler = FraudDetectionWebSocketHandler(connection_manager)

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup using FraudDetectionOrchestrator"""
    
    log_agent_activity("system", "Starting Fraud Detection System with FraudDetectionOrchestrator")
    
    try:
        # Create uploads directory
        Path("uploads").mkdir(exist_ok=True)
        
        # Initialize database with sample data (already done in mock_db)
        db_info = mock_db.get_database_info()
        logger.info(f"📊 Database initialized with {db_info['total_records']} records across {db_info['total_collections']} collections")
        
        # UPDATED: Get orchestrator status instead of fraud_detection_system
        orchestrator_status = orchestrator.get_orchestrator_status()
        
        logger.info("✅ Fraud Detection Orchestrator system initialized successfully")
        logger.info(f"🎭 Orchestrator ready with {len(orchestrator_status['agents_managed'])} managed agents")
        logger.info(f"🔌 WebSocket handler ready for real-time communication")
        
        # Log each managed agent
        for agent_name in orchestrator_status['agents_managed']:
            log_agent_activity("orchestrator", f"Managing agent: {agent_name}")
        
        # Log architecture change
        logger.info("🏗️ Architecture: fraud_detection_system.py eliminated - using FraudDetectionOrchestrator")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    log_agent_activity("system", "Shutting down orchestrator system")
    
    # Close all WebSocket connections
    await connection_manager.disconnect_all()
    
    # Cleanup orchestrator sessions
    active_count = len(orchestrator.active_sessions)
    if active_count > 0:
        logger.info(f"🧹 Cleaning up {active_count} active orchestrator sessions")
        for session_id in list(orchestrator.active_sessions.keys()):
            await orchestrator._cleanup_session(session_id)
    
    # Create backup of mock database
    backup = mock_db.backup_data()
    logger.info(f"💾 Created system backup with {len(backup['collections'])} collections")
    
    logger.info("✅ Orchestrator system shutdown complete")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with updated system information"""
    # Get WebSocket status
    ws_status = "Connected" if connection_manager.get_connection_count() > 0 else "Ready"
    active_sessions = fraud_ws_handler.get_active_sessions_count()
    orchestrator_status = orchestrator.get_orchestrator_status()
    
    return f"""
    <html>
        <head>
            <title>Fraud Detection Agent API - ORCHESTRATOR ARCHITECTURE</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ background: white; padding: 30px; border-radius: 8px; max-width: 900px; }}
                .header {{ color: #dc2626; border-bottom: 2px solid #dc2626; padding-bottom: 10px; }}
                .feature {{ background: #f0f9ff; padding: 15px; margin: 10px 0; border-radius: 6px; }}
                .orchestrator {{ background: #ecfdf5; padding: 15px; margin: 10px 0; border-radius: 6px; }}
                .websocket {{ background: #fef3c7; padding: 15px; margin: 10px 0; border-radius: 6px; }}
                .status {{ background: #ddd6fe; padding: 10px; border-radius: 4px; margin: 5px 0; }}
                .architecture {{ background: #fdf2f8; padding: 15px; margin: 10px 0; border-radius: 6px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="header">🏦 Fraud Detection Agent API 🎭</h1>
                <p><strong>FraudDetectionOrchestrator Architecture with Multi-Agent Coordination!</strong></p>
                
                <div class="orchestrator">
                    <h3>🎭 FraudDetectionOrchestrator Status</h3>
                    <div class="status">
                        <p><strong>Orchestrator Status:</strong> {orchestrator_status.get('system_status', 'unknown')}</p>
                        <p><strong>Active Sessions:</strong> {orchestrator_status.get('active_sessions', 0)}</p>
                        <p><strong>Agents Available:</strong> {orchestrator_status.get('agents_available', False)}</p>
                        <p><strong>Managed Agents:</strong> {len(orchestrator_status.get('agents_managed', []))}</p>
                    </div>
                </div>
                
                <div class="websocket">
                    <h3>🔌 WebSocket Integration Status</h3>
                    <div class="status">
                        <p><strong>WebSocket Status:</strong> {ws_status}</p>
                        <p><strong>Active Connections:</strong> {connection_manager.get_connection_count()}</p>
                        <p><strong>Processing Sessions:</strong> {active_sessions}</p>
                        <p><strong>Handler Type:</strong> Orchestrator Delegation</p>
                    </div>
                </div>
                
                <div class="feature">
                    <h3>🤖 Orchestrator-Managed Agents</h3>
                    <p>All agents coordinated by FraudDetectionOrchestrator:</p>
                    <ul>
                        <li>🎵 Audio Processing Agent</li>
                        <li>🔍 Scam Detection Agent</li>
                        <li>📚 Policy Guidance Agent</li>
                        <li>🎯 Decision Agent</li>
                        <li>📝 Summarization Agent</li>
                        <li>📋 Case Management Agent</li>
                    </ul>
                </div>
                
                <div class="feature">
                    <h3>🔗 API Endpoints</h3>
                    <ul>
                        <li><a href="/docs">📚 Interactive API Documentation</a></li>
                        <li><a href="/health">💚 System Health Check</a></li>
                        <li><a href="/api/v1/fraud/orchestrator/status">🎭 Orchestrator Status</a></li>
                        <li><a href="/api/v1/database/info">🗄️ Database Information</a></li>
                        <li><a href="/api/v1/websocket/status">🔌 WebSocket Status</a></li>
                    </ul>
                </div>
                
                <div class="feature">
                    <h3>🔗 WebSocket Endpoints</h3>
                    <ul>
                        <li><strong>Fraud Detection:</strong> <code>ws://localhost:8000/ws/fraud-detection/{{client_id}}</code></li>
                        <li><strong>General Purpose:</strong> <code>ws://localhost:8000/ws/{{client_id}}</code></li>
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/health")
@handle_api_errors("health_check")
async def health_check():
    """System health check using FraudDetectionOrchestrator"""
    
    orchestrator_status = orchestrator.get_orchestrator_status()
    db_info = mock_db.get_database_info()
    
    # Add WebSocket health info
    ws_health = {
        "active_connections": connection_manager.get_connection_count(),
        "active_sessions": fraud_ws_handler.get_active_sessions_count(),
        "handler_status": "operational"
    }
    
    return create_success_response(
        data={
            "status": "healthy",
            "timestamp": get_current_timestamp(),
            "orchestrator_active": orchestrator_status.get('agents_available', False),
            "orchestrator_sessions": orchestrator_status.get('active_sessions', 0),
            "managed_agents": len(orchestrator_status.get('agents_managed', [])),
            "version": "3.0.0",  # UPDATED: Version bump
            "architecture": "FraudDetectionOrchestrator",
            "database_records": db_info.get('total_records', 0),
            "websocket_health": ws_health,
            "fraud_detection_system_eliminated": True  # UPDATED: Note the architectural change
        },
        message="System is healthy with FraudDetectionOrchestrator architecture"
    )

@app.get("/api/v1/system/status")
@handle_api_errors("system_status")
async def get_system_status():
    """Get detailed system status using FraudDetectionOrchestrator"""
    
    orchestrator_status = orchestrator.get_orchestrator_status()
    db_info = mock_db.get_database_info()
    
    # Add WebSocket status
    ws_status = {
        "handler_active": True,
        "handler_type": "orchestrator_delegation",
        "active_connections": connection_manager.get_connection_count(),
        "active_sessions": fraud_ws_handler.get_active_sessions_count(),
        "connection_manager": connection_manager.__class__.__name__,
        "fraud_handler": fraud_ws_handler.__class__.__name__
    }
    
    return create_success_response(
        data={
            "system_architecture": "FraudDetectionOrchestrator",
            "orchestrator_status": orchestrator_status,
            "framework": "Multi-Agent System with Orchestrator Coordination",
            "agents_managed": orchestrator_status.get('agents_managed', []),
            "agents_available": orchestrator_status.get('agents_available', False),
            "last_updated": orchestrator_status.get('timestamp', get_current_timestamp()),
            "database_info": db_info,
            "websocket_status": ws_status,
            "architectural_changes": {
                "fraud_detection_system_eliminated": True,
                "orchestrator_agent_renamed_to_decision_agent": True,
                "websocket_handler_delegates_to_orchestrator": True,
                "rest_apis_use_orchestrator": True
            }
        },
        message="System status retrieved from FraudDetectionOrchestrator"
    )

@app.get("/api/v1/database/info")
@handle_api_errors("database_info")
async def get_database_info():
    """Get consolidated database information"""
    
    db_info = mock_db.get_database_info()
    
    # Add service-specific statistics
    audio_stats = mock_db.get_statistics("audio_files")
    case_stats = mock_db.get_statistics("fraud_cases")
    session_stats = mock_db.get_statistics("fraud_sessions")
    
    return create_success_response(
        data={
            "database_overview": db_info,
            "collection_statistics": {
                "audio_files": audio_stats,
                "fraud_cases": case_stats,
                "fraud_sessions": session_stats
            },
            "specialized_services": {
                "audio_service": "AudioFileService - handles audio metadata",
                "case_service": "CaseService - manages fraud cases", 
                "session_service": "SessionService - tracks sessions"
            },
            "orchestrator_integration": {
                "sessions_managed_by_orchestrator": True,
                "fraud_detection_system_data_migrated": True
            }
        },
        message="Consolidated database information retrieved"
    )

@app.get("/api/v1/websocket/status")
@handle_api_errors("websocket_status")
async def get_websocket_status():
    """Get WebSocket system status with orchestrator info"""
    
    active_connections = connection_manager.get_connection_count()
    active_sessions = fraud_ws_handler.get_active_sessions_count()
    orchestrator_sessions = len(orchestrator.active_sessions)
    
    # Get detailed connection info
    connection_details = []
    for connection in connection_manager.active_connections:
        connection_details.append({
            "client_id": connection.client_id,
            "connected_at": connection.connected_at,
            "is_active": True
        })
    
    return create_success_response(
        data={
            "websocket_system": {
                "status": "operational",
                "handler_class": fraud_ws_handler.__class__.__name__,
                "manager_class": connection_manager.__class__.__name__,
                "orchestrator_integration": True
            },
            "connections": {
                "active_count": active_connections,
                "processing_sessions": active_sessions,
                "orchestrator_sessions": orchestrator_sessions,
                "details": connection_details
            },
            "orchestrator_info": {
                "class": orchestrator.__class__.__name__,
                "agents_managed": len(orchestrator.orchestrator.get_orchestrator_status().get('agents_managed', [])) if hasattr(orchestrator, 'orchestrator') else 0,
                "delegation_model": "WebSocket Handler delegates to Orchestrator"
            },
            "capabilities": [
                "Real-time fraud detection via orchestrator",
                "Live transcription streaming", 
                "Multi-agent coordination",
                "Session management through orchestrator",
                "Error recovery",
                "Orchestrator delegation pattern"
            ]
        },
        message="WebSocket status retrieved with orchestrator integration"
    )

@app.post("/upload-audio")
@handle_api_errors("upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file using consolidated patterns"""
    
    # Validate file using centralized utilities
    from .utils import validate_file_extension, get_file_size_mb, sanitize_filename
    
    if not validate_file_extension(file.filename, settings.supported_audio_formats):
        raise ValueError("Invalid file type. Only WAV, MP3, and M4A files are supported.")
    
    # Read file content
    content = await file.read()
    
    # Validate size using centralized settings
    if len(content) > settings.get_max_file_size_bytes():
        raise ValueError(f"File too large. Maximum size: {settings.max_audio_file_size_mb}MB")
    
    # Save uploaded file with sanitized name
    sanitized_name = sanitize_filename(file.filename)
    file_path = f"uploads/{sanitized_name}"
    
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)
    
    # Store metadata using consolidated service
    file_id = f"upload_{get_current_timestamp().replace(':', '').replace('-', '')[:14]}"
    metadata = {
        "file_id": file_id,
        "filename": file.filename,
        "sanitized_filename": sanitized_name,
        "size_bytes": len(content),
        "size_mb": get_file_size_mb(len(content)),
        "format": file.filename.split('.')[-1].lower(),
        "status": "uploaded",
        "orchestrator_compatible": True  # UPDATED: Note orchestrator compatibility
    }
    
    audio_service.store_audio_metadata(file_id, metadata)
    
    log_agent_activity("upload", f"File uploaded: {file.filename} ({len(content)} bytes) - Orchestrator compatible")
    
    return create_success_response(
        data={
            "file_id": file_id,
            "filename": file.filename,
            "size_bytes": len(content),
            "size_mb": metadata["size_mb"],
            "orchestrator_compatible": True,
            "processing_note": "File ready for FraudDetectionOrchestrator processing"
        },
        message="File uploaded successfully - ready for orchestrator processing"
    )

@app.post("/process-audio-orchestrator")
@handle_api_errors("process_audio_orchestrator")
async def process_audio_with_orchestrator(filename: str, session_id: str = None):
    """Process audio file directly through FraudDetectionOrchestrator (REST API)"""
    
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = generate_session_id("rest_orchestrator")
        
        logger.info(f"🎭 REST API processing audio via orchestrator: {filename}")
        
        # Process through orchestrator
        result = await orchestrator.orchestrate_audio_processing(
            filename=filename,
            session_id=session_id
        )
        
        if result.get('success'):
            return create_success_response(
                data={
                    "session_id": session_id,
                    "orchestrator_result": result,
                    "message": "Audio processing started via FraudDetectionOrchestrator",
                    "api_type": "rest",
                    "orchestrator": "FraudDetectionOrchestrator"
                },
                message="Audio processed through FraudDetectionOrchestrator"
            )
        else:
            return create_error_response(
                error=result.get('error', 'Orchestrator processing failed'),
                code="ORCHESTRATOR_PROCESSING_ERROR"
            )
        
    except Exception as e:
        logger.error(f"❌ REST orchestrator processing error: {e}")
        return create_error_response(
            error=str(e),
            code="ORCHESTRATOR_ERROR"
        )

# WebSocket endpoints - UPDATED to note orchestrator delegation

@app.websocket("/ws/fraud-detection/{client_id}")
async def fraud_detection_websocket(websocket: WebSocket, client_id: str):
    """
    Dedicated WebSocket endpoint for fraud detection via orchestrator delegation
    """
    await websocket.accept()
    await connection_manager.connect(websocket, client_id)
    logger.info(f"🔌 Fraud detection WebSocket connected: {client_id} (Orchestrator delegation)")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process through fraud detection handler (which delegates to orchestrator)
            await fraud_ws_handler.handle_message(websocket, client_id, message_data)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        fraud_ws_handler.cleanup_client_sessions(client_id)
        logger.info(f"🔌 Fraud detection client {client_id} disconnected")
    except Exception as e:
        logger.error(f"❌ Fraud detection WebSocket error for {client_id}: {e}")
        connection_manager.disconnect(client_id)
        fraud_ws_handler.cleanup_client_sessions(client_id)

@app.websocket("/ws/{client_id}")
async def general_websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    General WebSocket endpoint with orchestrator awareness
    """
    await websocket.accept()
    await connection_manager.connect(websocket, client_id)
    logger.info(f"🔌 General WebSocket connected: {client_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle basic message types
            await handle_general_websocket_message(websocket, client_id, message_data)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        logger.info(f"🔌 General client {client_id} disconnected")
    except Exception as e:
        logger.error(f"❌ General WebSocket error for {client_id}: {e}")
        connection_manager.disconnect(client_id)

async def handle_general_websocket_message(websocket: WebSocket, client_id: str, message_data: dict):
    """Handle general WebSocket messages with orchestrator awareness"""
    try:
        message_type = message_data.get('type')
        
        if message_type == 'ping':
            await connection_manager.send_personal_message(
                json.dumps({
                    'type': 'pong', 
                    'timestamp': get_current_timestamp(),
                    'server': 'fraud_detection_orchestrator_system'  # UPDATED
                }),
                client_id
            )
        elif message_type == 'get_system_status':
            # Send orchestrator status via WebSocket
            status = orchestrator.get_orchestrator_status()
            await connection_manager.send_personal_message(
                json.dumps({
                    'type': 'system_status',
                    'data': status,
                    'timestamp': get_current_timestamp()
                }),
                client_id
            )
        elif message_type == 'fraud_detection':
            # Redirect to fraud detection handler (which uses orchestrator)
            await fraud_ws_handler.handle_message(websocket, client_id, message_data)
        else:
            logger.warning(f"❓ Unknown general message type: {message_type} from {client_id}")
            
    except Exception as e:
        logger.error(f"❌ Error handling general WebSocket message from {client_id}: {e}")
        error_response = create_error_response(
            error=f'Error processing message: {str(e)}',
            code="WEBSOCKET_PROCESSING_ERROR"
        )
        await connection_manager.send_personal_message(
            json.dumps({'type': 'error', 'data': error_response}),
            client_id
        )

# Include API routes (now using orchestrator)
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(fraud.router, prefix="/api/v1/fraud", tags=["fraud"])  # UPDATED: Now uses orchestrator
app.include_router(cases.router, prefix="/api/v1/cases", tags=["cases"])

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
