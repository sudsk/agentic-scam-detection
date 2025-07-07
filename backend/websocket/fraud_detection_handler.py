# backend/websocket/fraud_detection_handler.py - SIMPLIFIED VERSION
"""
Simplified WebSocket handler for real-time fraud detection
Works with the improved audio processor
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from ..agents.fraud_detection_system import fraud_detection_system
from ..agents.audio_processor.agent import audio_processor_agent 
from ..websocket.connection_manager import ConnectionManager
from ..utils import get_current_timestamp, create_error_response

logger = logging.getLogger(__name__)

class FraudDetectionWebSocketHandler:
    """Simplified WebSocket handler for fraud detection"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_sessions: Dict[str, Dict] = {}
        self.customer_speech_buffer: Dict[str, List[str]] = {}
        
        logger.info("🔌 Simplified Fraud Detection WebSocket handler initialized")
    
    async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages"""
        
        message_type = message.get('type')
        data = message.get('data', {})
        
        try:
            if message_type == 'process_audio':
                await self.handle_audio_processing(websocket, client_id, data)
            elif message_type == 'start_realtime_session':
                await self.handle_audio_processing(websocket, client_id, data)
            elif message_type == 'stop_processing':
                await self.handle_stop_processing(websocket, client_id, data)
            elif message_type == 'get_status':
                await self.handle_status_request(websocket, client_id)
            elif message_type == 'ping':
                await self.send_message(websocket, client_id, {
                    'type': 'pong',
                    'data': {'timestamp': get_current_timestamp()}
                })
            else:
                await self.send_error(websocket, client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling message from {client_id}: {e}")
            await self.send_error(websocket, client_id, f"Message processing error: {str(e)}")
    
    async def handle_audio_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Start audio processing with fraud detection"""
        
        filename = data.get('filename')
        session_id = data.get('session_id') or f"session_{uuid.uuid4().hex[:8]}"
        
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
        
        if session_id in self.active_sessions:
            await self.send_error(websocket, client_id, f"Session {session_id} already active")
            return
        
        try:
            logger.info(f"🎵 Starting audio processing: {filename} for {client_id}")
            
            # Initialize session tracking
            self.active_sessions[session_id] = {
                'client_id': client_id,
                'filename': filename,
                'websocket': websocket,
                'start_time': datetime.now(),
                'status': 'processing'
            }
            self.customer_speech_buffer[session_id] = []
            
            # Create callback for audio processor
            async def audio_callback(message_data: Dict) -> None:
                await self.handle_audio_message(websocket, client_id, session_id, message_data)
            
            # Start audio processing
            result = await audio_processor_agent.start_realtime_processing(
                session_id=session_id,
                audio_filename=filename,
                websocket_callback=audio_callback
            )
            
            if 'error' in result:
                await self.send_error(websocket, client_id, result['error'])
                self.cleanup_session(session_id)
                return
            
            # Send confirmation
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'session_id': session_id,
                    'filename': filename,
                    'audio_duration': result.get('audio_duration', 0),
                    'channels': result.get('channels', 1),
                    'processing_mode': result.get('processing_mode', 'mono_realtime'),
                    'timestamp': get_current_timestamp()
                }
            })
            
            logger.info(f"✅ Processing started for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error starting processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to start processing: {str(e)}")
            self.cleanup_session(session_id)
    
    async def handle_audio_message(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        message_data: Dict
    ) -> None:
        """Handle messages from audio processor"""
        
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
            
            elif message_type == 'transcription_interim':
                # Forward interim results
                await self.send_message(websocket, client_id, message_data)
            
            elif message_type == 'streaming_started':
                # Forward streaming confirmation
                await self.send_message(websocket, client_id, message_data)
            
            elif message_type == 'error':
                # Forward errors
                await self.send_message(websocket, client_id, message_data)
            
            else:
                # Forward other messages
                await self.send_message(websocket, client_id, message_data)
                
        except Exception as e:
            logger.error(f"❌ Error handling audio message: {e}")
    
    async def process_customer_speech(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Process customer speech for fraud detection"""
        
        try:
            # Add to speech buffer
            if session_id not in self.customer_speech_buffer:
                self.customer_speech_buffer[session_id] = []
            
            self.customer_speech_buffer[session_id].append(customer_text)
            
            # Run fraud analysis after accumulating enough speech
            buffer_length = len(self.customer_speech_buffer[session_id])
            
            # Trigger analysis every 2 customer segments, or when we have 4+ segments
            if buffer_length >= 2 and (buffer_length % 2 == 0 or buffer_length >= 4):
                combined_text = " ".join(self.customer_speech_buffer[session_id])
                
                if len(combined_text.strip()) >= 20:  # Minimum text for analysis
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
            
            # Small delay for realistic processing
            await asyncio.sleep(0.5)
            
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
            await self.send_message(websocket, client_id, {
                'type': 'policy_analysis_started',
                'data': {
                    'session_id': session_id,
                    'timestamp': get_current_timestamp()
                }
            })
            
            await asyncio.sleep(0.3)
            
            # Get policy guidance
            policy_result = fraud_detection_system.policy_guide.process({
                'scam_type': fraud_result.get('scam_type', 'unknown'),
                'risk_score': fraud_result.get('risk_score', 0)
            })
            
            if 'error' not in policy_result:
                primary_policy = policy_result.get('primary_policy', {})
                agent_guidance = policy_result.get('agent_guidance', {})
                
                await self.send_message(websocket, client_id, {
                    'type': 'policy_guidance_ready',
                    'data': {
                        'session_id': session_id,
                        'policy_id': primary_policy.get('policy_id', ''),
                        'policy_title': primary_policy.get('title', ''),
                        'immediate_alerts': agent_guidance.get('immediate_alerts', []),
                        'recommended_actions': agent_guidance.get('recommended_actions', []),
                        'key_questions': agent_guidance.get('key_questions', []),
                        'customer_education': agent_guidance.get('customer_education', []),
                        'timestamp': get_current_timestamp()
                    }
                })
                
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
            await self.send_message(websocket, client_id, {
                'type': 'case_creation_started',
                'data': {
                    'session_id': session_id,
                    'timestamp': get_current_timestamp()
                }
            })
            
            await asyncio.sleep(0.3)
            
            # Create case
            case_result = fraud_detection_system.case_manager.process({
                "session_id": session_id,
                "fraud_analysis": fraud_result,
                "customer_id": f"CUSTOMER_{session_id}"
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
                
        except Exception as e:
            logger.error(f"❌ Error creating fraud case: {e}")
    
    async def handle_stop_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Handle request to stop processing with immediate action"""
        
        session_id = data.get('session_id')
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            logger.info(f"🛑 STOP request received for session {session_id}")
            
            # IMMEDIATE: Stop audio processing
            stop_result = await audio_processor_agent.stop_streaming(session_id)
            
            # IMMEDIATE: Update session status
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "stopped"
            
            # Send immediate confirmation to client
            await self.send_message(websocket, client_id, {
                'type': 'processing_stopped',
                'data': {
                    'session_id': session_id,
                    'status': 'stopped',
                    'message': 'Processing stopped immediately',
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Cleanup
            self.cleanup_session(session_id)
            
            logger.info(f"✅ STOP completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error stopping processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to stop: {str(e)}")
            
            # Force cleanup even on error
            self.cleanup_session(session_id)
    
    async def handle_status_request(self, websocket: WebSocket, client_id: str) -> None:
        """Handle status request"""
        
        try:
            system_status = fraud_detection_system.get_agent_status()
            audio_status = audio_processor_agent.get_all_sessions_status()
            
            await self.send_message(websocket, client_id, {
                'type': 'status_response',
                'data': {
                    'system_status': system_status,
                    'audio_processor_status': audio_status,
                    'websocket_sessions': len(self.active_sessions),
                    'timestamp': get_current_timestamp()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            await self.send_error(websocket, client_id, f"Status error: {str(e)}")
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session data"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.customer_speech_buffer:
                del self.customer_speech_buffer[session_id]
        except Exception as e:
            logger.error(f"❌ Error cleaning up session {session_id}: {e}")
    
    def cleanup_client_sessions(self, client_id: str) -> None:
        """Clean up all sessions for a disconnected client"""
        sessions_to_cleanup = [
            session_id for session_id, session in self.active_sessions.items()
            if session.get('client_id') == client_id
        ]
        
        for session_id in sessions_to_cleanup:
            try:
                asyncio.create_task(audio_processor_agent.stop_streaming(session_id))
                self.cleanup_session(session_id)
            except Exception as e:
                logger.error(f"❌ Error cleaning up session {session_id}: {e}")
        
        if sessions_to_cleanup:
            logger.info(f"🧹 Cleaned up {len(sessions_to_cleanup)} sessions for {client_id}")
    
    async def send_message(self, websocket: WebSocket, client_id: str, message: Dict) -> bool:
        """Send message to client"""
        try:
            await websocket.send_text(json.dumps(message))
            return True
        except WebSocketDisconnect:
            logger.warning(f"🔌 Client {client_id} disconnected")
            self.cleanup_client_sessions(client_id)
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send message to {client_id}: {e}")
            return False
    
    async def send_error(self, websocket: WebSocket, client_id: str, error_message: str) -> None:
        """Send error message to client"""
        await self.send_message(websocket, client_id, {
            'type': 'error',
            'data': create_error_response(error_message, "PROCESSING_ERROR")
        })
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.active_sessions)
    
    def get_connected_clients(self) -> List[str]:
        """Get list of connected client IDs"""
        return [session['client_id'] for session in self.active_sessions.values()]
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is currently active"""
        return session_id in self.active_sessions
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a specific session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "client_id": session.get('client_id'),
            "filename": session.get('filename'),
            "status": session.get('status'),
            "start_time": session.get('start_time', '').isoformat() if session.get('start_time') else '',
            "customer_speech_segments": len(self.customer_speech_buffer.get(session_id, []))
        }
    
    async def get_session_transcription(self, session_id: str) -> Optional[List[Dict]]:
        """Get transcription segments for a session"""
        if session_id in self.active_sessions:
            # Get transcription from audio processor
            return audio_processor_agent.get_session_transcription(session_id)
        return None
    
    async def get_customer_speech_history(self, session_id: str) -> Optional[List[str]]:
        """Get customer speech history for fraud analysis"""
        if session_id in self.customer_speech_buffer:
            return self.customer_speech_buffer[session_id].copy()
        return None
    
    async def cleanup_stale_sessions(self, max_age_hours: int = 2) -> int:
        """Clean up stale sessions"""
        current_time = datetime.now()
        stale_sessions = []
        
        for session_id, session in self.active_sessions.items():
            start_time = session.get("start_time")
            if start_time:
                age_hours = (current_time - start_time).total_seconds() / 3600
                if age_hours > max_age_hours:
                    stale_sessions.append(session_id)
        
        cleaned_count = 0
        for session_id in stale_sessions:
            try:
                await audio_processor_agent.stop_streaming(session_id)
                self.cleanup_session(session_id)
                cleaned_count += 1
            except Exception as e:
                logger.error(f"❌ Error cleaning stale session {session_id}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} stale sessions")
        
        return cleaned_count
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'handler_type': 'SimplifiedFraudDetectionHandler',
            'active_sessions': len(self.active_sessions),
            'audio_processor_sessions': len(audio_processor_agent.active_sessions),
            'customer_speech_buffers': len(self.customer_speech_buffer),
            'system_status': 'operational',
            'timestamp': get_current_timestamp()
        }
