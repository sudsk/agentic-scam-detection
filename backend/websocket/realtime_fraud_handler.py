# backend/websocket/realtime_fraud_handler.py - NEW FILE
"""
Real-time WebSocket handler that processes live audio streams and transcribes in real-time
"""

import asyncio
import json
import logging
import base64
from datetime import datetime
from typing import Dict, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from ..agents.fraud_detection_system import fraud_detection_system
from ..agents.audio_processor.real_time_agent import realtime_audio_agent
from ..websocket.connection_manager import ConnectionManager
from ..utils import get_current_timestamp, log_agent_activity, create_error_response

logger = logging.getLogger(__name__)

class RealTimeFraudDetectionHandler:
    """
    Real-time WebSocket handler that processes live audio and detects fraud as it happens
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_sessions: Dict[str, Dict] = {}
        self.realtime_sessions: Dict[str, str] = {}  # Maps session_id to client_id
        
        logger.info("🎙️ Real-time fraud detection handler initialized")
    
    async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]) -> None:
        """Handle real-time WebSocket messages"""
        
        message_type = message.get('type')
        data = message.get('data', {})
        
        logger.info(f"📨 Real-time message from {client_id}: {message_type}")
        
        try:
            if message_type == 'start_realtime_session':
                await self.start_realtime_session(websocket, client_id, data)
            elif message_type == 'audio_chunk':
                await self.process_audio_chunk(websocket, client_id, data)
            elif message_type == 'analyze_fraud':
                await self.analyze_fraud_text(websocket, client_id, data)
            elif message_type == 'stop_realtime_session':
                await self.stop_realtime_session(websocket, client_id, data)
            elif message_type == 'ping':
                await self.send_message(websocket, client_id, {
                    'type': 'pong',
                    'data': {'timestamp': get_current_timestamp()}
                })
            else:
                logger.warning(f"❓ Unknown real-time message type: {message_type}")
                await self.send_error(websocket, client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling real-time message from {client_id}: {e}")
            await self.send_error(websocket, client_id, f"Message processing error: {str(e)}")
    
    async def start_realtime_session(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Start a real-time transcription and fraud detection session"""
        
        session_id = data.get('session_id')
        audio_config = data.get('audio_config', {})
        
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            logger.info(f"🎙️ Starting real-time session {session_id} for {client_id}")
            
            # Start real-time audio agent session
            audio_result = await realtime_audio_agent.start_realtime_session(session_id, audio_config)
            
            if 'error' in audio_result:
                await self.send_error(websocket, client_id, audio_result['error'])
                return
            
            # Store session mapping
            self.realtime_sessions[session_id] = client_id
            self.active_sessions[session_id] = {
                'client_id': client_id,
                'websocket': websocket,
                'start_time': datetime.now(),
                'audio_config': audio_config,
                'customer_speech_buffer': [],
                'last_fraud_analysis': None,
                'total_transcripts': 0
            }
            
            # Send confirmation
            await self.send_message(websocket, client_id, {
                'type': 'realtime_session_started',
                'data': {
                    'session_id': session_id,
                    'status': 'active',
                    'config': audio_result.get('config', {}),
                    'timestamp': get_current_timestamp()
                }
            })
            
            logger.info(f"✅ Real-time session {session_id} started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting real-time session: {e}")
            await self.send_error(websocket, client_id, f"Failed to start session: {str(e)}")
    
    async def process_audio_chunk(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Process incoming audio chunk for real-time transcription"""
        
        session_id = data.get('session_id')
        audio_data = data.get('audio_data')  # Base64 encoded
        timestamp = data.get('timestamp', 0)
        
        if not session_id or session_id not in self.active_sessions:
            await self.send_error(websocket, client_id, "Invalid session")
            return
        
        if not audio
