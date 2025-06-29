# backend/websocket/fraud_detection_handler.py - NEW FILE

import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import WebSocket

from ..agents.fraud_detection_system import fraud_detection_system
from ..websocket.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

class FraudDetectionWebSocketHandler:
    """Handles real-time fraud detection WebSocket communication"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_sessions: Dict[str, Dict] = {}
    
    async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        
        message_type = message.get('type')
        data = message.get('data', {})
        
        logger.info(f"📨 WebSocket message from {client_id}: {message_type}")
        
        try:
            if message_type == 'process_audio':
                await self.handle_audio_processing(websocket, client_id, data)
            elif message_type == 'ping':
                await self.send_message(websocket, client_id, {
                    'type': 'pong',
                    'data': {'timestamp': datetime.now().isoformat()}
                })
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling WebSocket message: {e}")
            await self.send_error(websocket, client_id, str(e))
    
    async def handle_audio_processing(self, websocket: WebSocket, client_id: str, data: Dict):
        """Process audio through multi-agent system with real-time updates"""
        
        filename = data.get('filename')
        session_id = data.get('session_id')
        
        if not filename or not session_id:
            await self.send_error(websocket, client_id, "Missing filename or session_id")
            return
        
        try:
            # Store session info
            self.active_sessions[session_id] = {
                'client_id': client_id,
                'filename': filename,
                'websocket': websocket,
                'start_time': datetime.now()
            }
            
            # Start processing with real-time callbacks
            audio_path = f"data/sample_audio/{filename}"
            
            # Process through agents with WebSocket callbacks
            await self.process_with_realtime_updates(
                websocket, client_id, session_id, audio_path
            )
            
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            await self.send_error(websocket, client_id, str(e))
    
    async def process_with_realtime_updates(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        audio_path: str
    ):
        """Process audio with real-time WebSocket updates"""
        
        # Step 1: Audio Processing Agent
        await self.send_message(websocket, client_id, {
            'type': 'processing_started',
            'data': {'stage': 'audio_processing', 'agent': 'Audio Processing Agent'}
        })
        
        # Get transcription from audio processor agent
        transcription_result = fraud_detection_system.audio_processor.process(audio_path)
        
        # Send transcription segments in real-time
        if 'speaker_segments' in transcription_result:
            for segment in transcription_result['speaker_segments']:
                await self.send_message(websocket, client_id, {
                    'type': 'transcription_segment',
                    'data': {
                        'speaker': segment['speaker'],
                        'start': segment['start'],
                        'duration': segment['end'] - segment['start'],
                        'text': segment['text'],
                        'confidence': segment.get('confidence', 0.9)
                    }
                })
                # Simulate real-time delay
                await asyncio.sleep(1)
        
        # Step 2: Fraud Detection Agent
        await self.send_message(websocket, client_id, {
            'type': 'processing_started',
            'data': {'stage': 'fraud_detection', 'agent': 'Fraud Detection Agent'}
        })
        
        # Extract customer text for fraud analysis
        customer_text = " ".join([
            seg['text'] for seg in transcription_result.get('speaker_segments', [])
            if seg['speaker'] == 'customer'
        ])
        
        if customer_text:
            fraud_analysis = fraud_detection_system.fraud_detector.process(customer_text)
            
            await self.send_message(websocket, client_id, {
                'type': 'fraud_analysis_update',
                'data': {
                    'risk_score': fraud_analysis['risk_score'],
                    'risk_level': fraud_analysis['risk_level'],
                    'scam_type': fraud_analysis['scam_type'],
                    'detected_patterns': fraud_analysis['detected_patterns'],
                    'confidence': fraud_analysis['confidence']
                }
            })
        
        # Step 3: Policy Guidance Agent (if needed)
        if fraud_analysis.get('risk_score', 0) >= 40:
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {'stage': 'policy_guidance', 'agent': 'Policy Guidance Agent'}
            })
            
            policy_guidance = fraud_detection_system.policy_guide.process({
                'scam_type': fraud_analysis['scam_type'],
                'risk_score': fraud_analysis['risk_score']
            })
            
            await self.send_message(websocket, client_id, {
                'type': 'policy_guidance_ready',
                'data': policy_guidance.get('agent_guidance', {})
            })
        
        # Step 4: Case Management (if high risk)
        if fraud_analysis.get('risk_score', 0) >= 60:
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {'stage': 'case_management', 'agent': 'Case Management Agent'}
            })
            
            case_info = fraud_detection_system.case_manager.process({
                "session_id": session_id,
                "fraud_analysis": fraud_analysis,
                "customer_id": None
            })
            
            await self.send_message(websocket, client_id, {
                'type': 'case_created',
                'data': {
                    'case_id': case_info.get('case_id'),
                    'priority': case_info.get('priority'),
                    'status': case_info.get('status')
                }
            })
        
        # Final completion message
        await self.send_message(websocket, client_id, {
            'type': 'processing_complete',
            'data': {
                'session_id': session_id,
                'total_agents': 4,
                'final_risk_score': fraud_analysis.get('risk_score', 0),
                'case_created': fraud_analysis.get('risk_score', 0) >= 60
            }
        })
        
        # Clean up session
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    async def send_message(self, websocket: WebSocket, client_id: str, message: Dict):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message))
            logger.debug(f"📤 Sent to {client_id}: {message['type']}")
        except Exception as e:
            logger.error(f"❌ Failed to send message to {client_id}: {e}")
    
    async def send_error(self, websocket: WebSocket, client_id: str, error_message: str):
        """Send error message to client"""
        await self.send_message(websocket, client_id, {
            'type': 'error',
            'data': {'error': error_message}
        })

# Add this to your main backend/app.py WebSocket endpoint
fraud_ws_handler = FraudDetectionWebSocketHandler(connection_manager)

@app.websocket("/ws/fraud-detection-client")
async def fraud_detection_websocket(websocket: WebSocket):
    client_id = f"fraud_client_{int(time.time())}"
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await fraud_ws_handler.handle_message(websocket, client_id, message)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        logger.info(f"🔌 Fraud detection client {client_id} disconnected")
