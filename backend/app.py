# backend/app.py
"""
Fraud Detection Agent - FastAPI Backend
Real-time fraud detection with modular 4-agent system
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import aiofiles

# Import the modular fraud detection system
from agents.fraud_detection_system import (
    fraud_detection_system,
    coordinate_fraud_analysis
)
from websocket.connection_manager import ConnectionManager
from api.routes import audio, fraud, cases
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection Agent API",
    description="Real-time fraud detection using modular multi-agent system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state
connection_manager = ConnectionManager()

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agents_active: int
    version: str

class SystemStatusResponse(BaseModel):
    system_status: str
    agents_count: int
    agents: Dict[str, str]
    framework: str
    last_updated: str

class ProcessAudioRequest(BaseModel):
    filename: str
    session_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize modular fraud detection system on startup"""
    
    logger.info("🚀 Starting Modular Fraud Detection Agent System...")
    
    try:
        # Create uploads directory
        Path("uploads").mkdir(exist_ok=True)
        
        # Get system status to verify agents are ready
        status = fraud_detection_system.get_agent_status()
        
        logger.info("✅ Modular fraud detection system initialized successfully")
        logger.info(f"System ready with {status['agents_count']} modular agents")
        
        # Log each agent status
        for agent_name, agent_status in status['agents'].items():
            logger.info(f"  - {agent_name}: {agent_status}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize modular fraud detection system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔄 Shutting down Modular Fraud Detection Agent System...")
    
    # Close all WebSocket connections
    await connection_manager.disconnect_all()
    
    logger.info("✅ Shutdown complete")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system information"""
    return """
    <html>
        <head>
            <title>Fraud Detection Agent API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { background: white; padding: 30px; border-radius: 8px; max-width: 800px; }
                .header { color: #dc2626; border-bottom: 2px solid #dc2626; padding-bottom: 10px; }
                .feature { background: #f0f9ff; padding: 15px; margin: 10px 0; border-radius: 6px; }
                .agent { background: #ecfdf5; padding: 10px; margin: 5px 0; border-radius: 4px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="header">🏦 Fraud Detection Agent API</h1>
                <p><strong>Powered by Modular Multi-Agent Architecture</strong></p>
                
                <div class="feature">
                    <h3>🤖 Modular Agents</h3>
                    <div class="agent">🎵 Audio Processing Agent - Transcription & speaker diarization</div>
                    <div class="agent">🔍 Fraud Detection Agent - Pattern recognition & risk scoring</div>
                    <div class="agent">📚 Policy Guidance Agent - Response procedures & escalation</div>
                    <div class="agent">📋 Case Management Agent - Investigation case creation</div>
                    <div class="agent">🎭 System Orchestrator - Workflow coordination</div>
                </div>
                
                <div class="feature">
                    <h3>🔗 API Endpoints</h3>
                    <ul>
                        <li><a href="/docs">📚 Interactive API Documentation (Swagger)</a></li>
                        <li><a href="/health">💚 System Health Check</a></li>
                        <li><a href="/api/v1/system/status">📊 Detailed Agent Status</a></li>
                        <li><a href="/api/v1/agents/list">🤖 Agent Details</a></li>
                        <li><a href="/api/v1/fraud/demo">🧪 Demo Fraud Analysis</a></li>
                    </ul>
                </div>
                
                <div class="feature">
                    <h3>🧪 Demo Instructions</h3>
                    <p>Upload sample audio files to see the modular multi-agent system in action:</p>
                    <ul>
                        <li><code>investment_scam_live_call.wav</code> - Investment fraud detection</li>
                        <li><code>romance_scam_live_call.wav</code> - Romance scam analysis</li>
                        <li><code>impersonation_scam_live_call.wav</code> - Authority impersonation</li>
                        <li><code>legitimate_call.wav</code> - Normal banking call</li>
                    </ul>
                </div>
                
                <div class="feature">
                    <h3>🏗️ Architecture</h3>
                    <p>Modular design with specialized agents:</p>
                    <ul>
                        <li><strong>Audio Processing:</strong> agents/audio_processor/agent.py</li>
                        <li><strong>Fraud Detection:</strong> agents/scam_detection/agent.py</li>
                        <li><strong>Policy Guidance:</strong> agents/policy_guidance/agent.py</li>
                        <li><strong>Case Management:</strong> agents/case_management/agent.py</li>
                        <li><strong>System Coordinator:</strong> agents/fraud_detection_system.py</li>
                    </ul>
                </div>
                
                <p><em>Real-time fraud detection protecting customers from scams ⚡</em></p>
            </div>
        </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint"""
    
    status = fraud_detection_system.get_agent_status()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agents_active=status.get('agents_count', 0),
        version="2.0.0"
    )

@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get detailed system and agent status"""
    
    status = fraud_detection_system.get_agent_status()
    
    return SystemStatusResponse(
        system_status=status.get('system_status', 'unknown'),
        agents_count=status.get('agents_count', 0),
        agents=status.get('agents', {}),
        framework=status.get('framework', 'Modular Multi-Agent System'),
        last_updated=status.get('last_updated', datetime.now().isoformat())
    )

@app.get("/api/v1/agents/list")
async def list_agents():
    """List all agents with detailed information"""
    
    status = fraud_detection_system.get_agent_status()
    
    return {
        "total_agents": status.get('agents_count', 0),
        "framework": "Modular Multi-Agent Architecture",
        "agents": status.get('agent_details', {}),
        "case_statistics": status.get('case_statistics', {}),
        "performance_metrics": fraud_detection_system.get_agent_performance_metrics(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/fraud/process-audio")
async def process_audio_through_pipeline(request: ProcessAudioRequest):
    """Process audio file through modular multi-agent pipeline"""
    
    try:
        logger.info(f"🔄 Processing {request.filename} through modular pipeline...")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process through modular pipeline
        result = await fraud_detection_system.process_audio_file(
            file_path=request.filename,
            session_id=session_id
        )
        
        logger.info(f"✅ Modular pipeline completed for {request.filename}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in modular pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/v1/fraud/demo")
async def demo_fraud_analysis():
    """Demo endpoint showing modular agent capabilities"""
    demo_cases = [
        {
            "case_type": "Investment Scam",
            "transcript": "My investment advisor just called saying there's a margin call on my trading account. I need to transfer fifteen thousand pounds immediately or I'll lose all my investments. He's been guaranteeing thirty-five percent monthly returns.",
            "expected_risk": "85-95%",
            "agents_involved": ["audio_processing", "fraud_detection", "policy_guidance", "case_management", "orchestrator"],
            "decision": "BLOCK_AND_ESCALATE",
            "modular_flow": "Audio → Fraud Analysis → Policy Guidance → Case Creation → Decision"
        },
        {
            "case_type": "Romance Scam", 
            "transcript": "I need to send four thousand pounds to Turkey. My partner Alex is stuck in Istanbul and needs money for accommodation. We've been together for seven months online.",
            "expected_risk": "70-80%",
            "agents_involved": ["audio_processing", "fraud_detection", "policy_guidance", "case_management"],
            "decision": "VERIFY_AND_MONITOR",
            "modular_flow": "Audio → Fraud Analysis → Policy Guidance → Case Creation"
        },
        {
            "case_type": "Authority Impersonation",
            "transcript": "Someone from bank security called saying my account has been compromised. They asked me to confirm my card details and PIN to verify my identity immediately.",
            "expected_risk": "90-95%", 
            "agents_involved": ["audio_processing", "fraud_detection", "policy_guidance", "case_management"],
            "decision": "BLOCK_AND_ESCALATE",
            "modular_flow": "Audio → Fraud Analysis → Policy Guidance → Case Creation"
        },
        {
            "case_type": "Legitimate Call",
            "transcript": "Hi, I'd like to check my account balance and see if my salary has been deposited yet. I'm also interested in opening a savings account.",
            "expected_risk": "0-15%",
            "agents_involved": ["audio_processing", "fraud_detection"],
            "decision": "CONTINUE_NORMAL",
            "modular_flow": "Audio → Fraud Analysis (low risk, pipeline ends)"
        }
    ]
    
    return {
        "demo_title": "Fraud Detection - Modular Multi-Agent System Demo",
        "description": "Sample cases showing modular agent capabilities and workflow",
        "architecture": "Modular design with specialized agents",
        "total_agents": len(fraud_detection_system.get_agent_status()['agents']),
        "demo_cases": demo_cases,
        "agent_details": {
            "audio_processing": "Handles transcription and speaker identification",
            "fraud_detection": "Analyzes patterns and calculates risk scores", 
            "policy_guidance": "Provides response procedures and escalation advice",
            "case_management": "Creates and manages investigation cases",
            "orchestrator": "Coordinates workflow and makes final decisions"
        },
        "instructions": {
            "upload_audio": "Upload sample audio files to see real-time analysis",
            "supported_formats": [".wav", ".mp3", ".m4a"],
            "processing_time": "Typically 2-5 seconds for complete analysis",
            "output": "Comprehensive fraud analysis with modular agent coordination"
        }
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time modular agent communication"""
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message through modular agents
            await handle_websocket_message(client_id, message_data)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        connection_manager.disconnect(client_id)

async def handle_websocket_message(client_id: str, message_data: dict):
    """Handle incoming WebSocket messages through modular agents"""
    try:
        message_type = message_data.get('type')
        
        if message_type == 'audio_upload':
            await handle_realtime_audio_processing(client_id, message_data['data'])
        elif message_type == 'agent_status':
            await send_agent_status(client_id)
        elif message_type == 'get_session_data':
            await send_session_data(client_id, message_data.get('session_id'))
        elif message_type == 'ping':
            await connection_manager.send_personal_message(
                json.dumps({'type': 'pong', 'timestamp': datetime.now().isoformat()}),
                client_id
            )
        else:
            logger.warning(f"Unknown message type: {message_type} from {client_id}")
            
    except Exception as e:
        logger.error(f"Error handling WebSocket message from {client_id}: {e}")
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'error',
                'data': {'message': f'Error processing message: {str(e)}'}
            }),
            client_id
        )

async def handle_realtime_audio_processing(client_id: str, audio_data: dict):
    """Process audio through modular agents in real-time"""
    try:
        # Send status update
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'status',
                'data': {'status': 'processing', 'message': 'Processing through modular agents...'}
            }),
            client_id
        )
        
        # Get filename from audio data
        filename = audio_data.get('filename', 'unknown.wav')
        session_id = f"ws_{client_id}_{datetime.now().strftime('%H%M%S')}"
        
        # Process through modular pipeline
        result = await fraud_detection_system.process_audio_file(
            file_path=filename,
            session_id=session_id
        )
        
        # Send transcription if available
        if 'transcription' in result:
            await connection_manager.send_personal_message(
                json.dumps({
                    'type': 'transcription',
                    'data': result['transcription']
                }),
                client_id
            )
        
        # Send fraud analysis if available
        if 'fraud_analysis' in result:
            await connection_manager.send_personal_message(
                json.dumps({
                    'type': 'fraud_analysis',
                    'data': result['fraud_analysis']
                }),
                client_id
            )
        
        # Send policy guidance if available
        if 'policy_guidance' in result and result['policy_guidance']:
            await connection_manager.send_personal_message(
                json.dumps({
                    'type': 'policy_guidance',
                    'data': result['policy_guidance']
                }),
                client_id
            )
        
        # Send case information if created
        if 'case_info' in result and result['case_info']:
            await connection_manager.send_personal_message(
                json.dumps({
                    'type': 'case_created',
                    'data': result['case_info']
                }),
                client_id
            )
        
        # Send final orchestrator decision
        if 'orchestrator_decision' in result:
            await connection_manager.send_personal_message(
                json.dumps({
                    'type': 'orchestrator_decision',
                    'data': result['orchestrator_decision']
                }),
                client_id
            )
        
        # Send processing summary
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'processing_complete',
                'data': {
                    'session_id': session_id,
                    'agents_involved': result.get('agents_involved', []),
                    'pipeline_complete': result.get('pipeline_complete', False),
                    'processing_summary': result.get('processing_summary', {})
                }
            }),
            client_id
        )
        
        # Send completion status
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'status',
                'data': {'status': 'complete', 'message': 'Modular agent analysis complete'}
            }),
            client_id
        )
        
    except Exception as e:
        logger.error(f"Error in real-time processing for {client_id}: {e}")
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'error',
                'data': {'message': str(e)}
            }),
            client_id
        )

async def send_agent_status(client_id: str):
    """Send current modular agent status to client"""
    try:
        status = fraud_detection_system.get_agent_status()
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'agent_status',
                'data': status
            }),
            client_id
        )
    except Exception as e:
        logger.error(f"Error sending agent status to {client_id}: {e}")

async def send_session_data(client_id: str, session_id: Optional[str]):
    """Send session data to client"""
    try:
        if session_id:
            session_data = fraud_detection_system.get_session_data(session_id)
        else:
            session_data = fraud_detection_system.get_all_sessions()
        
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'session_data',
                'data': session_data
            }),
            client_id
        )
    except Exception as e:
        logger.error(f"Error sending session data to {client_id}: {e}")

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for modular agent processing"""
    try:
        # Validate file type
        if not file.filename.endswith(('.wav', '.mp3', '.m4a')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only WAV, MP3, and M4A files are supported.")
        
        # Read file content
        content = await file.read()
        
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        
        logger.info(f"📁 File uploaded: {file.filename} ({len(content)} bytes)")
        
        return {
            "status": "success",
            "filename": file.filename,
            "size": len(content),
            "message": "File uploaded successfully - ready for modular agent processing"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sessions/{session_id}")
async def get_session_data(session_id: str):
    """Get data for a specific session"""
    session_data = fraud_detection_system.get_session_data(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_data

@app.get("/api/v1/sessions")
async def list_sessions():
    """List all sessions"""
    return {
        "sessions": fraud_detection_system.get_all_sessions(),
        "total_sessions": len(fraud_detection_system.get_all_sessions()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/metrics")
async def get_system_metrics():
    """Get comprehensive system metrics"""
    return {
        "system_metrics": fraud_detection_system.get_agent_performance_metrics(),
        "agent_status": fraud_detection_system.get_agent_status(),
        "timestamp": datetime.now().isoformat()
    }

# Include API routes
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(fraud.router, prefix="/api/v1/fraud", tags=["fraud"])
app.include_router(cases.router, prefix="/api/v1/cases", tags=["cases"])

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
