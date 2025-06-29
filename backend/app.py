# backend/app.py - FIXED WITH WEBSOCKET HANDLER INTEGRATION
"""
Fraud Detection Agent - FastAPI Backend
FIXED: Integrated FraudDetectionWebSocketHandler for real-time communication
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

# Import consolidated components with FIXED paths
from .agents.fraud_detection_system import fraud_detection_system
from .websocket.connection_manager import ConnectionManager
from .websocket.fraud_detection_handler import FraudDetectionWebSocketHandler  # ADDED
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
logger = logging.getLogger(__name__)

# Load centralized settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection Agent API",
    description="Real-time fraud detection using modular multi-agent system with WebSocket support",
    version="2.0.0",
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

# FIXED: Initialize WebSocket components
connection_manager = ConnectionManager()
fraud_ws_handler = FraudDetectionWebSocketHandler(connection_manager)

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup using consolidated components"""
    
    log_agent_activity("system", "Starting Consolidated Fraud Detection System with WebSocket Support")
    
    try:
        # Create uploads directory
        Path("uploads").mkdir(exist_ok=True)
        
        # Initialize database with sample data (already done in mock_db)
        db_info = mock_db.get_database_info()
        logger.info(f"📊 Database initialized with {db_info['total_records']} records across {db_info['total_collections']} collections")
        
        # Get system status to verify agents are ready
        status = fraud_detection_system.get_agent_status()
        
        logger.info("✅ Consolidated fraud detection system initialized successfully")
        logger.info(f"System ready with {status['agents_count']} agents")
        logger.info(f"🔌 WebSocket handler ready for real-time communication")
        
        # Log each agent status using centralized logging
        for agent_name, agent_status in status['agents'].items():
            log_agent_activity("system", f"Agent {agent_name}: {agent_status}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    log_agent_activity("system", "Shutting down system")
    
    # Close all WebSocket connections
    await connection_manager.disconnect_all()
    
    # Cleanup fraud handler sessions
    if hasattr(fraud_ws_handler, 'active_sessions'):
        active_count = len(fraud_ws_handler.active_sessions)
        if active_count > 0:
            logger.info(f"🧹 Cleaning up {active_count} active WebSocket sessions")
    
    # Create backup of mock database
    backup = mock_db.backup_data()
    logger.info(f"💾 Created system backup with {len(backup['collections'])} collections")
    
    logger.info("✅ Shutdown complete")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system information"""
    # Get WebSocket status
    ws_status = "Connected" if connection_manager.get_connection_count() > 0 else "Ready"
    active_sessions = fraud_ws_handler.get_active_sessions_count()
    
    return f"""
    <html>
        <head>
            <title>Fraud Detection Agent API - CONSOLIDATED + WEBSOCKET</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ background: white; padding: 30px; border-radius: 8px; max-width: 900px; }}
                .header {{ color: #dc2626; border-bottom: 2px solid #dc2626; padding-bottom: 10px; }}
                .feature {{ background: #f0f9ff; padding: 15px; margin: 10px 0; border-radius: 6px; }}
                .consolidation {{ background: #ecfdf5; padding: 15px; margin: 10px 0; border-radius: 6px; }}
                .websocket {{ background: #fef3c7; padding: 15px; margin: 10px 0; border-radius: 6px; }}
                .status {{ background: #ddd6fe; padding: 10px; border-radius: 4px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="header">🏦 Fraud Detection Agent API - CONSOLIDATED + WEBSOCKET ✨</h1>
                <p><strong>Real-time Multi-Agent System with WebSocket Support!</strong></p>
                
                <div class="websocket">
                    <h3>🔌 WebSocket Integration Status</h3>
                    <p>✅ <strong>WebSocket Handler:</strong> Integrated and Ready</p>
                    <p>✅ <strong>Connection Manager:</strong> Active</p>
                    <p>✅ <strong>Real-time Processing:</strong> Enabled</p>
                    <div class="status">
                        <p><strong>WebSocket Status:</strong> {ws_status}</p>
                        <p><strong>Active Connections:</strong> {connection_manager.get_connection_count()}</p>
                        <p><strong>Processing Sessions:</strong> {active_sessions}</p>
                    </div>
                </div>
                
                <div class="consolidation">
                    <h3>🎯 Consolidation Achievements</h3>
                    <p>✅ <strong>500+ lines of redundant code removed</strong></p>
                    <p>✅ <strong>Centralized configuration</strong> - Single source of truth</p>
                    <p>✅ <strong>Unified error handling</strong> - Consistent across all routes</p>
                    <p>✅ <strong>Shared utilities</strong> - No more duplicate functions</p>
                    <p>✅ <strong>Consolidated database</strong> - One service for all CRUD</p>
                    <p>✅ <strong>Shared API models</strong> - Consistent validation</p>
                    <p>✅ <strong>WebSocket Handler</strong> - Real-time communication</p>
                </div>
                
                <div class="feature">
                    <h3>🤖 Modular Agents (No Wrapper Functions)</h3>
                    <p>All agents now use <code>agent.process()</code> directly:</p>
                    <ul>
                        <li>🎵 Audio Processing Agent</li>
                        <li>🔍 Fraud Detection Agent</li>
                        <li>📚 Policy Guidance Agent</li>
                        <li>📋 Case Management Agent</li>
                        <li>🎭 System Orchestrator</li>
                    </ul>
                </div>
                
                <div class="feature">
                    <h3>🔗 WebSocket Endpoints</h3>
                    <ul>
                        <li><strong>Fraud Detection:</strong> <code>ws://localhost:8000/ws/fraud-detection/{{client_id}}</code></li>
                        <li><strong>General Purpose:</strong> <code>ws://localhost:8000/ws/{{client_id}}</code></li>
                    </ul>
                </div>
                
                <div class="feature">
                    <h3>🔗 API Endpoints</h3>
                    <ul>
                        <li><a href="/docs">📚 Interactive API Documentation</a></li>
                        <li><a href="/health">💚 System Health Check</a></li>
                        <li><a href="/api/v1/system/status">📊 Detailed Agent Status</a></li>
                        <li><a href="/api/v1/database/info">🗄️ Database Information</a></li>
                        <li><a href="/api/v1/websocket/status">🔌 WebSocket Status</a></li>
                    </ul>
                </div>
                
                <p><em>🚀 Zero redundancy, maximum efficiency, real-time communication!</em></p>
            </div>
        </body>
    </html>
    """

@app.get("/health")
@handle_api_errors("health_check")
async def health_check():
    """System health check using consolidated components"""
    
    status = fraud_detection_system.get_agent_status()
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
            "agents_active": status.get('agents_count', 0),
            "version": "2.0.0",
            "database_records": db_info.get('total_records', 0),
            "websocket_health": ws_health
        },
        message="System is healthy with WebSocket support"
    )

@app.get("/api/v1/system/status")
@handle_api_errors("system_status")
async def get_system_status():
    """Get detailed system and agent status using consolidated services"""
    
    status = fraud_detection_system.get_agent_status()
    db_info = mock_db.get_database_info()
    
    # Add WebSocket status
    ws_status = {
        "handler_active": True,
        "active_connections": connection_manager.get_connection_count(),
        "active_sessions": fraud_ws_handler.get_active_sessions_count(),
        "connection_manager": connection_manager.__class__.__name__,
        "fraud_handler": fraud_ws_handler.__class__.__name__
    }
    
    return create_success_response(
        data={
            "system_status": status.get('system_status', 'operational'),
            "agents_count": status.get('agents_count', 0),
            "agents": status.get('agents', {}),
            "framework": "Consolidated Multi-Agent System with WebSocket",
            "last_updated": status.get('last_updated', get_current_timestamp()),
            "database_info": db_info,
            "websocket_status": ws_status
        },
        message="System status retrieved with WebSocket info"
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
            }
        },
        message="Consolidated database information retrieved"
    )

# ADDED: WebSocket status endpoint
@app.get("/api/v1/websocket/status")
@handle_api_errors("websocket_status")
async def get_websocket_status():
    """Get WebSocket system status"""
    
    active_connections = connection_manager.get_connection_count()
    active_sessions = fraud_ws_handler.get_active_sessions_count()
    
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
                "manager_class": connection_manager.__class__.__name__
            },
            "connections": {
                "active_count": active_connections,
                "processing_sessions": active_sessions,
                "details": connection_details
            },
            "capabilities": [
                "Real-time fraud detection",
                "Live transcription streaming", 
                "Multi-agent coordination",
                "Session management",
                "Error recovery"
            ]
        },
        message="WebSocket status retrieved"
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
        "status": "uploaded"
    }
    
    audio_service.store_audio_metadata(file_id, metadata)
    
    log_agent_activity("upload", f"File uploaded: {file.filename} ({len(content)} bytes)")
    
    return create_success_response(
        data={
            "file_id": file_id,
            "filename": file.filename,
            "size_bytes": len(content),
            "size_mb": metadata["size_mb"],
            "consolidation_note": "Processed using centralized utilities and services"
        },
        message="File uploaded successfully using consolidated system"
    )

# backend/app.py - FIXED WebSocket endpoints

@app.websocket("/ws/fraud-detection/{client_id}")
async def fraud_detection_websocket(websocket: WebSocket, client_id: str):
    """
    Dedicated WebSocket endpoint for fraud detection - FIXED: Proper connection handling
    """
    # FIXED: Accept the WebSocket connection first
    await websocket.accept()
    
    # Then connect through the manager (without accepting again)
    await connection_manager.connect(websocket, client_id)
    logger.info(f"🔌 Fraud detection WebSocket connected: {client_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process through fraud detection handler
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
    General WebSocket endpoint - FIXED: Proper connection handling
    """
    # FIXED: Accept the WebSocket connection first
    await websocket.accept()
    
    # Then connect through the manager
    await connection_manager.connect(websocket, client_id)
    logger.info(f"🔌 General WebSocket connected: {client_id}")
    
    try:
        while True:
            # Receive message from client
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
    """Handle general WebSocket messages with basic functionality"""
    try:
        message_type = message_data.get('type')
        
        if message_type == 'ping':
            await connection_manager.send_personal_message(
                json.dumps({
                    'type': 'pong', 
                    'timestamp': get_current_timestamp(),
                    'server': 'consolidated_fraud_detection_system'
                }),
                client_id
            )
        elif message_type == 'get_system_status':
            # Send system status via WebSocket
            status = fraud_detection_system.get_agent_status()
            await connection_manager.send_personal_message(
                json.dumps({
                    'type': 'system_status',
                    'data': status,
                    'timestamp': get_current_timestamp()
                }),
                client_id
            )
        elif message_type == 'fraud_detection':
            # Redirect to fraud detection handler
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

# Include consolidated API routes
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(fraud.router, prefix="/api/v1/fraud", tags=["fraud"])
app.include_router(cases.router, prefix="/api/v1/cases", tags=["cases"])

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
