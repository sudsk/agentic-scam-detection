"""
HSBC Scam Detection Agent - FastAPI Backend
Real-time fraud detection with multi-agent system
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

# Import agents
from agents.live_scam_detection.agent import LiveScamDetectionAgent
from agents.rag_policy.agent import RAGPolicyAgent
from agents.orchestrator.agent import MasterOrchestratorAgent
from agents.shared.message_bus import Message
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
    title="HSBC Scam Detection Agent API",
    description="Real-time fraud detection using multi-agent system",
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
agents = {}
orchestrator = None
connection_manager = ConnectionManager()

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agents_active: int
    orchestrator_active: bool
    version: str

class MetricsResponse(BaseModel):
    timestamp: str
    system_metrics: Dict
    agent_metrics: Dict

@app.on_event("startup")
async def startup_event():
    """Initialize all agents and services on startup"""
    global agents, orchestrator
    
    logger.info("🚀 Starting HSBC Scam Detection Agent System...")
    
    try:
        # Create uploads directory
        Path("uploads").mkdir(exist_ok=True)
        
        # Initialize agents
        logger.info("Initializing agents...")
        agents['live_scam_detection'] = LiveScamDetectionAgent()
        agents['rag_policy'] = RAGPolicyAgent()
        
        # Initialize orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = MasterOrchestratorAgent()
        
        # Register agents with orchestrator
        await orchestrator.register_agent('live_scam_detection', agents['live_scam_detection'])
        await orchestrator.register_agent('rag_policy', agents['rag_policy'])
        
        logger.info("✅ All agents initialized successfully")
        logger.info(f"System ready to handle {settings.max_concurrent_sessions} concurrent sessions")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize agents: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔄 Shutting down HSBC Scam Detection Agent System...")
    
    # Close all WebSocket connections
    await connection_manager.disconnect_all()
    
    # Cleanup agents if needed
    # In production, implement proper agent cleanup
    
    logger.info("✅ Shutdown complete")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system information"""
    return """
    <html>
        <head>
            <title>HSBC Scam Detection Agent API</title>
        </head>
        <body>
            <h1>HSBC Scam Detection Agent API</h1>
            <p>Real-time fraud detection using multi-agent system</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/metrics">System Metrics</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agents_active=len([a for a in agents.values() if a is not None]),
        orchestrator_active=orchestrator is not None,
        version="2.0.0"
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system performance metrics"""
    system_metrics = {
        "active_connections": len(connection_manager.active_connections),
        "uptime_seconds": (datetime.now() - datetime.now()).total_seconds(),  # Placeholder
        "memory_usage": "N/A",  # Could implement psutil for real metrics
        "cpu_usage": "N/A"
    }
    
    agent_metrics = {}
    for agent_name, agent in agents.items():
        if agent:
            agent_metrics[agent_name] = agent.get_performance_metrics()
    
    if orchestrator:
        agent_metrics["orchestrator"] = orchestrator.get_performance_metrics()
    
    return MetricsResponse(
        timestamp=datetime.now().isoformat(),
        system_metrics=system_metrics,
        agent_metrics=agent_metrics
    )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication"""
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message based on type
            await handle_websocket_message(client_id, message_data)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        connection_manager.disconnect(client_id)

async def handle_websocket_message(client_id: str, message_data: dict):
    """Handle incoming WebSocket messages"""
    try:
        message_type = message_data.get('type')
        
        if message_type == 'audio_upload':
            await handle_audio_upload(client_id, message_data['data'])
        elif message_type == 'feedback':
            await handle_agent_feedback(client_id, message_data['data'])
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

async def handle_audio_upload(client_id: str, audio_data: dict):
    """Process audio upload through agent pipeline"""
    try:
        # Send status update
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'status',
                'data': {'status': 'processing', 'message': 'Processing audio...'}
            }),
            client_id
        )
        
        # Simulate processing delay for demo
        await asyncio.sleep(1)
        
        # Get mock transcription based on filename
        transcription = get_mock_transcription(audio_data.get('filename', ''))
        
        # Send transcription to client
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'transcription',
                'data': transcription
            }),
            client_id
        )
        
        # Create transcript message for agent processing
        transcript_message = Message(
            message_id=str(uuid.uuid4()),
            message_type='new_transcript',
            data={
                'session_id': client_id,
                'text': transcription['text'],
                'speaker': transcription['speaker'],
                'confidence': transcription['confidence'],
                'timestamp': transcription['timestamp']
            },
            source_agent='audio_processor'
        )
        
        # Send to orchestrator for processing
        if orchestrator:
            result = await orchestrator.process_message(transcript_message)
            
            # Send results to client
            if result.message_type == 'comprehensive_fraud_analysis':
                await connection_manager.send_personal_message(
                    json.dumps({
                        'type': 'fraud_analysis',
                        'data': result.data
                    }),
                    client_id
                )
            elif result.message_type == 'monitoring_update':
                await connection_manager.send_personal_message(
                    json.dumps({
                        'type': 'monitoring_update',
                        'data': result.data
                    }),
                    client_id
                )
        
        # Send completion status
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'status',
                'data': {'status': 'complete', 'message': 'Analysis complete'}
            }),
            client_id
        )
        
    except Exception as e:
        logger.error(f"Error processing audio for {client_id}: {e}")
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'error',
                'data': {'message': str(e)}
            }),
            client_id
        )

def get_mock_transcription(filename: str) -> dict:
    """Return mock transcription based on filename for demo"""
    transcriptions = {
        'investment_scam_live_call.wav': {
            'text': "My investment advisor just called saying there's a margin call on my trading account. I need to transfer fifteen thousand pounds to Sterling Investment Management immediately or I'll lose all my previous investments. James has been guaranteeing thirty-five percent monthly returns and said this is normal for successful traders.",
            'speaker': 'customer',
            'confidence': 0.94,
            'timestamp': datetime.now().isoformat()
        },
        'romance_scam_live_call.wav': {
            'text': "I need to send four thousand pounds to Turkey. My partner Alex is stuck in Istanbul after a work project went wrong. We've been together for seven months online and Alex needs money for accommodation and flights home. I trust Alex completely because we video call every day.",
            'speaker': 'customer', 
            'confidence': 0.92,
            'timestamp': datetime.now().isoformat()
        },
        'impersonation_scam_live_call.wav': {
            'text': "Someone claiming to be from HSBC security called saying my account has been compromised. They knew my name and address and asked me to confirm my card details and PIN. They said I need to verify immediately or my account will be frozen today.",
            'speaker': 'customer',
            'confidence': 0.96,
            'timestamp': datetime.now().isoformat()
        },
        'legitimate_call.wav': {
            'text': "Hi, I'd like to check my account balance and see if my salary has been deposited yet. I'm also interested in opening a savings account if you have any good rates available.",
            'speaker': 'customer',
            'confidence': 0.98,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    return transcriptions.get(filename, {
        'text': "This is a sample transcription for demonstration purposes.",
        'speaker': 'customer',
        'confidence': 0.85,
        'timestamp': datetime.now().isoformat()
    })

async def handle_agent_feedback(client_id: str, feedback_data: dict):
    """Handle feedback from agents for continuous learning"""
    try:
        # In production, this would feed back to learning agent
        logger.info(f"Received feedback from {client_id}: {feedback_data}")
        
        # Send acknowledgment
        await connection_manager.send_personal_message(
            json.dumps({
                'type': 'feedback_received',
                'data': {'status': 'success', 'message': 'Feedback recorded for model improvement'}
            }),
            client_id
        )
        
    except Exception as e:
        logger.error(f"Error handling feedback from {client_id}: {e}")

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for processing"""
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
        
        return {
            "status": "success",
            "filename": file.filename,
            "size": len(content),
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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