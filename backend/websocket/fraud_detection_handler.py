# backend/websocket/fraud_detection_handler.py - UPDATED FOR OPTION 4
"""
WebSocket handler for server-side real-time fraud detection
Coordinates between server audio processing and fraud detection pipeline
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Set, List
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from ..agents.fraud_detection_system import fraud_detection_system
from ..agents.audio_processor.server_realtime_processor import server_realtime_processor
from ..websocket.connection_manager import ConnectionManager
from ..utils import get_current_timestamp, log_agent_activity, create_error_response

logger = logging.getLogger(__name__)

class FraudDetectionWebSocketHandler:
    """
    WebSocket handler for server-side real-time fraud detection
    Coordinates audio processing, fraud detection, and client updates
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_sessions: Dict[str, Dict] = {}
        self.customer_speech_buffer: Dict[str, List[str]] = {}
        
        logger.info("🔌 Server-side Fraud Detection WebSocket handler initialized")
    
    async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages"""
        
        message_type = message.get('type')
        data = message.get('data', {})
        
        logger.info(f"📨 WebSocket message from {client_id}: {message_type}")
        
        try:
            if message_type == 'process_audio':
                await self.handle_server_audio_processing(websocket, client_id, data)
            elif message_type == 'start_realtime_session':
                await self.handle_server_audio_processing(websocket, client_id, data)
            elif message_type == 'stop_processing':
                await self.handle_stop_processing(websocket, client_id, data)
            elif message_type == 'get_status':
                await self.handle_status_request(websocket, client_id, data)
            elif message_type == 'ping':
                await self.send_message(websocket, client_id, {
                    'type': 'pong',
                    'data': {'timestamp': get_current_timestamp()}
                })
            else:
                logger.warning(f"❓ Unknown message type: {message_type} from {client_id}")
                await self.send_error(websocket, client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling WebSocket message from {client_id}: {e}")
            await self.send_error(websocket, client_id, f"Message processing error: {str(e)}")
    
    async def handle_server_audio_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Start server-side audio processing with fraud detection"""
        
        filename = data.get('filename')
        session_id = data.get('session_id')
        
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
            
        if not session_id:
            session_id = f"server_session_{uuid.uuid4().hex[:8]}"
        
        if session_id in self.active_sessions:
            await self.send_error(websocket, client_id, f"Session {session_id} already active")
            return
        
        try:
            logger.info(f"🎵 Starting server-side processing: {filename} for {client_id}")
            
            # Initialize session tracking
            self.active_sessions[session_id] = {
                'client_id': client_id,
                'filename': filename,
                'websocket': websocket,
                'start_time': datetime.now(),
                'status': 'processing'
            }
            
            self.customer_speech_buffer[session_id] = []
            
            # Create WebSocket callback for server processor
            async def websocket_callback(message_data: Dict) -> None:
                await self.handle_server_message(websocket, client_id, session_id, message_data)
            
            # Start server-side audio processing
            result = await server_realtime_processor.start_realtime_processing(
                session_id=session_id,
                audio_filename=filename,
                websocket_callback=websocket_callback
            )
            
            if 'error' in result:
                await self.send_error(websocket, client_id, result['error'])
                return
            
            # Send initial confirmation to client
            await self.send_message(websocket, client_id, {
                'type': 'server_processing_started',
                'data': {
                    'session_id': session_id,
                    'filename': filename,
                    'processing_mode': 'server_realtime',
                    'audio_duration': result.get('audio_duration', 0),
                    'timestamp': get_current_timestamp()
                }
            })
            
            logger.info(f"✅ Server processing started for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error starting server processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to start processing: {str(e)}")
            
            # Cleanup on error
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.customer_speech_buffer:
                del self.customer_speech_buffer[session_id]
    
    async def handle_server_message(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        message_data: Dict
    ) -> None:
        """Handle messages from the server audio processor"""
        
        message_type = message_data.get('type')
        data = message_data.get('data', {})
        
        try:
            if message_type == 'transcription_segment':
                # Forward transcription to client
                await self.send_message(websocket, client_id, message_data)
                
                # Process customer speech for fraud detection
                speaker = data.get('speaker')
                text = data.get('text', '')
                
                if speaker == 'customer' and text.strip():
                    await self.process_customer_speech(websocket, client_id, session_id, text)
            
            elif message_type == 'processing_complete':
                # Handle completion
                await self.handle_processing_complete(websocket, client_id, session_id, data)
            
            elif message_type == 'error':
                # Forward error to client
                await self.send_message(websocket, client_id, message_data)
            
            else:
                # Forward other messages to client
                await self.send_message(websocket, client_id, message_data)
                
        except Exception as e:
            logger.error(f"❌ Error handling server message: {e}")
    
    async def process_customer_speech(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Process customer speech through fraud detection pipeline"""
        
        try:
            # Add to speech buffer
            if session_id not in self.customer_speech_buffer:
                self.customer_speech_buffer[session_id] = []
            
            self.customer_speech_buffer[session_id].append(customer_text)
            
            # Trigger fraud analysis after accumulating enough speech
            buffer_length = len(self.customer_speech_buffer[session_id])
            
            if buffer_length >= 2 and (buffer_length % 2 == 0 or buffer_length >= 4):
                combined_text = " ".join(self.customer_speech_buffer[session_id])
                
                if len(combined_text.strip()) >= 20:  # Minimum text length
                    await self.run_fraud_analysis(websocket, client_id, session_id, combined_text)
            
        except Exception as e:
            logger.error(f"❌ Error processing customer speech: {e}")
    
    async def run_fraud_analysis(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Run fraud analysis and send results"""
        
        try:
            logger.info(f"🔍 Running fraud analysis for session {session_id}")
            
            # Send analysis started notification
            await self.send_message(websocket, client_id, {
                'type': 'fraud_analysis_started',
                'data': {
                    'session_id': session_id,
                    'text_length': len(customer_text),
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Add realistic processing delay
            await asyncio.sleep(0.6)
            
            # Get policy guidance
            policy_result = fraud_detection_system.policy_guide.process({
                'scam_type': fraud_result.get('scam_type', 'unknown'),
                'risk_score': fraud_result.get('risk_score', 0)
            })
            
            if 'error' not in policy_result:
                await self.send_message(websocket, client_id, {
                    'type': 'policy_guidance_ready',
                    'data': {
                        'session_id': session_id,
                        **policy_result.get('agent_guidance', {}),
                        'timestamp': get_current_timestamp()
                    }
                })
                
                logger.info(f"✅ Policy guidance sent for session {session_id}")
            else:
                logger.error(f"❌ Policy guidance failed: {policy_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error getting policy guidance: {e}")
    
    async def create_fraud_case(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        fraud_result: Dict
    ) -> None:
        """Create fraud case for high-risk sessions"""
        
        try:
            # Send case creation started notification
            await self.send_message(websocket, client_id, {
                'type': 'case_creation_started',
                'data': {
                    'session_id': session_id,
                    'timestamp': get_current_timestamp()
                }
            })
            
            await asyncio.sleep(0.5)
            
            # Create case
            case_result = fraud_detection_system.case_manager.process({
                "session_id": session_id,
                "fraud_analysis": fraud_result,
                "customer_id": f"SERVER_{session_id}"
            })
            
            if 'error' not in case_result:
                await self.send_message(websocket, client_id, {
                    'type': 'case_created',
                    'data': {
                        'session_id': session_id,
                        'case_id': case_result.get('case_id'),
                        'priority': case_result.get('priority'),
                        'status': case_result.get('status'),
                        'assigned_team': case_result.get('assigned_team'),
                        'timestamp': get_current_timestamp()
                    }
                })
                
                logger.info(f"✅ Fraud case created: {case_result.get('case_id')}")
            else:
                logger.error(f"❌ Case creation failed: {case_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error creating fraud case: {e}")
    
    async def handle_processing_complete(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        data: Dict
    ) -> None:
        """Handle processing completion"""
        
        try:
            # Run final fraud analysis if we have customer speech
            if session_id in self.customer_speech_buffer and self.customer_speech_buffer[session_id]:
                final_text = " ".join(self.customer_speech_buffer[session_id])
                await self.run_fraud_analysis(websocket, client_id, session_id, final_text)
            
            # Forward completion message to client
            await self.send_message(websocket, client_id, {
                'type': 'processing_complete',
                'data': {
                    **data,
                    'final_analysis_complete': True,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Cleanup session
            await self.cleanup_session(session_id)
            
            logger.info(f"✅ Processing completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling processing completion: {e}")
    
    async def handle_stop_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Handle request to stop processing"""
        
        session_id = data.get('session_id')
        
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            # Stop server processing
            stop_result = await server_realtime_processor.stop_realtime_processing(session_id)
            
            # Send confirmation to client
            await self.send_message(websocket, client_id, {
                'type': 'processing_stopped',
                'data': {
                    'session_id': session_id,
                    **stop_result,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Cleanup session
            await self.cleanup_session(session_id)
            
            logger.info(f"🛑 Processing stopped for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error stopping processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to stop processing: {str(e)}")
    
    async def handle_status_request(self, websocket: WebSocket, client_id: str, data: Dict = None) -> None:
        """Handle status request"""
        
        try:
            # Get system status
            system_status = fraud_detection_system.get_agent_status()
            server_status = server_realtime_processor.get_all_sessions_status()
            
            status_data = {
                'system_status': system_status,
                'server_processor_status': server_status,
                'websocket_sessions': len(self.active_sessions),
                'timestamp': get_current_timestamp()
            }
            
            await self.send_message(websocket, client_id, {
                'type': 'status_response',
                'data': status_data
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            await self.send_error(websocket, client_id, f"Status error: {str(e)}")
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session data"""
        
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            if session_id in self.customer_speech_buffer:
                del self.customer_speech_buffer[session_id]
            
            logger.info(f"🧹 Cleaned up session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up session {session_id}: {e}")
    
    def cleanup_client_sessions(self, client_id: str) -> None:
        """Clean up all sessions for a disconnected client"""
        
        sessions_to_cleanup = []
        
        for session_id, session in self.active_sessions.items():
            if session.get('client_id') == client_id:
                sessions_to_cleanup.append(session_id)
        
        for session_id in sessions_to_cleanup:
            try:
                # Stop server processing
                asyncio.create_task(server_realtime_processor.stop_realtime_processing(session_id))
                
                # Clean up WebSocket session
                asyncio.create_task(self.cleanup_session(session_id))
                
            except Exception as e:
                logger.error(f"❌ Error cleaning up session {session_id}: {e}")
        
        if sessions_to_cleanup:
            logger.info(f"🧹 Cleaned up {len(sessions_to_cleanup)} sessions for client {client_id}")
    
    async def send_message(self, websocket: WebSocket, client_id: str, message: Dict) -> bool:
        """Send message to specific client"""
        try:
            message_json = json.dumps(message)
            await websocket.send_text(message_json)
            return True
        except WebSocketDisconnect:
            logger.warning(f"🔌 Client {client_id} disconnected during send")
            self.cleanup_client_sessions(client_id)
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send message to {client_id}: {e}")
            return False
    
    async def send_error(self, websocket: WebSocket, client_id: str, error_message: str) -> None:
        """Send error message to client"""
        error_response = create_error_response(
            error=error_message,
            code="SERVER_PROCESSING_ERROR"
        )
        
        await self.send_message(websocket, client_id, {
            'type': 'error',
            'data': error_response
        })
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.active_sessions)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        
        return {
            'handler_type': 'ServerSide_FraudDetectionHandler',
            'active_websocket_sessions': len(self.active_sessions),
            'server_processor_sessions': len(server_realtime_processor.active_sessions),
            'customer_speech_buffers': len(self.customer_speech_buffer),
            'system_status': 'operational',
            'processing_mode': 'server_realtime',
            'timestamp': get_current_timestamp()
        }

# Export the handler class
__all__ = ['FraudDetectionWebSocketHandler']0.8)
            
            # Run fraud detection
            fraud_result = fraud_detection_system.fraud_detector.process(customer_text)
            
            if 'error' not in fraud_result:
                # Send fraud analysis results
                await self.send_message(websocket, client_id, {
                    'type': 'fraud_analysis_update',
                    'data': {
                        'session_id': session_id,
                        'risk_score': fraud_result.get('risk_score', 0),
                        'risk_level': fraud_result.get('risk_level', 'MINIMAL'),
                        'scam_type': fraud_result.get('scam_type', 'unknown'),
                        'detected_patterns': fraud_result.get('detected_patterns', {}),
                        'confidence': fraud_result.get('confidence', 0.0),
                        'explanation': fraud_result.get('explanation', ''),
                        'timestamp': get_current_timestamp()
                    }
                })
                
                risk_score = fraud_result.get('risk_score', 0)
                
                # Get policy guidance for medium/high risk
                if risk_score >= 40:
                    await self.get_policy_guidance(websocket, client_id, session_id, fraud_result)
                
                # Create case for high risk
                if risk_score >= 80:
                    await self.create_fraud_case(websocket, client_id, session_id, fraud_result)
                
                logger.info(f"✅ Fraud analysis complete: {risk_score}% risk")
            else:
                logger.error(f"❌ Fraud analysis failed: {fraud_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error in fraud analysis: {e}")
    
    async def get_policy_guidance(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        fraud_result: Dict
    ) -> None:
        """Get and send policy guidance"""
        
        try:
            # Send policy guidance started notification
            await self.send_message(websocket, client_id, {
                'type': 'policy_analysis_started',
                'data': {
                    'session_id': session_id,
                    'timestamp': get_current_timestamp()
                }
            })
            
            await asyncio.sleep(
