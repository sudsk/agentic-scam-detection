# backend/websocket/realtime_fraud_handler.py - COMPLETED
"""
Real-time WebSocket handler that processes live audio streams and transcribes in real-time
"""

import asyncio
import json
import logging
import base64
from datetime import datetime
from typing import Dict, Any, Optional, Set, List
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
        self.transcription_buffer: Dict[str, List[str]] = {}  # Buffer customer speech
        self.processing_lock = asyncio.Lock()  # Prevent concurrent processing issues
        
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
            elif message_type == 'get_session_status':
                await self.get_session_status(websocket, client_id, data)
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
        
        async with self.processing_lock:
            try:
                logger.info(f"🎙️ Starting real-time session {session_id} for {client_id}")
                
                # Check if session already exists
                if session_id in self.active_sessions:
                    await self.send_error(websocket, client_id, f"Session {session_id} already exists")
                    return
                
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
                    'total_transcripts': 0,
                    'status': 'active',
                    'last_activity': datetime.now()
                }
                
                # Initialize transcription buffer
                self.transcription_buffer[session_id] = []
                
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
        
        if not audio_data:
            await self.send_error(websocket, client_id, "Missing audio data")
            return
        
        try:
            # Decode audio data
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception as e:
                await self.send_error(websocket, client_id, f"Invalid audio data format: {str(e)}")
                return
            
            # Update session activity
            self.active_sessions[session_id]['last_activity'] = datetime.now()
            
            # Process audio chunk through real-time agent
            transcription_result = await realtime_audio_agent.process_audio_chunk(
                session_id, audio_bytes, timestamp
            )
            
            if transcription_result:
                # Send transcription to client
                await self.send_message(websocket, client_id, {
                    'type': 'realtime_transcription',
                    'data': transcription_result
                })
                
                # If it's customer speech, buffer it for fraud analysis
                if transcription_result.get('speaker') == 'customer':
                    await self.buffer_customer_speech(session_id, transcription_result.get('text', ''))
                
                # Update session stats
                self.active_sessions[session_id]['total_transcripts'] += 1
                
        except Exception as e:
            logger.error(f"❌ Error processing audio chunk for {session_id}: {e}")
            await self.send_error(websocket, client_id, f"Audio processing error: {str(e)}")
    
    async def buffer_customer_speech(self, session_id: str, text: str) -> None:
        """Buffer customer speech and trigger fraud analysis when appropriate"""
        
        if session_id not in self.active_sessions:
            return
        
        # Add to buffer
        session = self.active_sessions[session_id]
        session['customer_speech_buffer'].append(text)
        
        # Add to transcription buffer
        if session_id not in self.transcription_buffer:
            self.transcription_buffer[session_id] = []
        self.transcription_buffer[session_id].append(text)
        
        # Trigger fraud analysis if we have enough content
        buffer_length = len(session['customer_speech_buffer'])
        
        # Analyze every 2-3 customer statements or if buffer gets long
        if buffer_length >= 2 and (buffer_length % 2 == 0 or buffer_length >= 5):
            combined_text = " ".join(session['customer_speech_buffer'])
            
            # Only analyze if we have substantial content
            if len(combined_text.strip()) >= 20:  # At least 20 characters
                await self.trigger_fraud_analysis(session_id, combined_text)
    
    async def trigger_fraud_analysis(self, session_id: str, customer_text: str) -> None:
        """Trigger fraud analysis on accumulated customer speech"""
        
        try:
            logger.info(f"🔍 Triggering fraud analysis for session {session_id}")
            
            # Get session info
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            client_id = session['client_id']
            websocket = session['websocket']
            
            # Send analysis started notification
            await self.send_message(websocket, client_id, {
                'type': 'fraud_analysis_started',
                'data': {
                    'session_id': session_id,
                    'text_length': len(customer_text),
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Run fraud analysis using the fraud detection agent
            fraud_result = fraud_detection_system.fraud_detector.process(customer_text)
            
            if 'error' not in fraud_result:
                # Store the latest analysis
                session['last_fraud_analysis'] = fraud_result
                
                # Send fraud analysis update
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
                
                # If risk is high enough, get policy guidance
                risk_score = fraud_result.get('risk_score', 0)
                if risk_score >= 40:
                    await self.get_policy_guidance(session_id, fraud_result)
                
                # If risk is very high, create a case
                if risk_score >= 80:
                    await self.create_fraud_case(session_id, fraud_result)
                
                logger.info(f"✅ Fraud analysis complete for {session_id}: {risk_score}% risk")
            else:
                logger.error(f"❌ Fraud analysis failed for {session_id}: {fraud_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error in fraud analysis for {session_id}: {e}")
    
    async def get_policy_guidance(self, session_id: str, fraud_result: Dict) -> None:
        """Get policy guidance based on fraud analysis"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            client_id = session['client_id']
            websocket = session['websocket']
            
            # Send policy guidance started notification
            await self.send_message(websocket, client_id, {
                'type': 'policy_guidance_started',
                'data': {
                    'session_id': session_id,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Get policy guidance from the policy agent
            policy_result = fraud_detection_system.policy_guide.process({
                'scam_type': fraud_result.get('scam_type', 'unknown'),
                'risk_score': fraud_result.get('risk_score', 0)
            })
            
            if 'error' not in policy_result:
                # Send policy guidance
                await self.send_message(websocket, client_id, {
                    'type': 'policy_guidance_ready',
                    'data': {
                        'session_id': session_id,
                        **policy_result.get('agent_guidance', {}),
                        'timestamp': get_current_timestamp()
                    }
                })
                
                logger.info(f"✅ Policy guidance sent for {session_id}")
            else:
                logger.error(f"❌ Policy guidance failed for {session_id}: {policy_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error getting policy guidance for {session_id}: {e}")
    
    async def create_fraud_case(self, session_id: str, fraud_result: Dict) -> None:
        """Create a fraud case for high-risk sessions"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            client_id = session['client_id']
            websocket = session['websocket']
            
            # Send case creation started notification
            await self.send_message(websocket, client_id, {
                'type': 'case_creation_started',
                'data': {
                    'session_id': session_id,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Create case using the case management agent
            case_result = fraud_detection_system.case_manager.process({
                "session_id": session_id,
                "fraud_analysis": fraud_result,
                "customer_id": f"REALTIME_{session_id}"
            })
            
            if 'error' not in case_result:
                # Send case creation confirmation
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
                
                logger.info(f"✅ Fraud case created for {session_id}: {case_result.get('case_id')}")
            else:
                logger.error(f"❌ Case creation failed for {session_id}: {case_result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error creating fraud case for {session_id}: {e}")
    
    async def analyze_fraud_text(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Analyze specific text for fraud (triggered by frontend)"""
        
        session_id = data.get('session_id')
        text = data.get('text', '')
        speaker = data.get('speaker', 'customer')
        
        if not session_id or session_id not in self.active_sessions:
            await self.send_error(websocket, client_id, "Invalid session")
            return
        
        if not text.strip():
            await self.send_error(websocket, client_id, "No text provided for analysis")
            return
        
        # Only analyze customer speech
        if speaker != 'customer':
            return
        
        try:
            logger.info(f"🔍 Manual fraud analysis requested for {session_id}")
            
            # Trigger fraud analysis
            await self.trigger_fraud_analysis(session_id, text)
            
        except Exception as e:
            logger.error(f"❌ Error in manual fraud analysis for {session_id}: {e}")
            await self.send_error(websocket, client_id, f"Analysis error: {str(e)}")
    
    async def stop_realtime_session(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Stop a real-time session"""
        
        session_id = data.get('session_id')
        
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        async with self.processing_lock:
            try:
                logger.info(f"🛑 Stopping real-time session {session_id}")
                
                # Stop the real-time audio agent session
                if session_id in self.active_sessions:
                    await realtime_audio_agent.stop_realtime_session(session_id)
                
                # Get session info before cleanup
                session_info = self.active_sessions.get(session_id, {})
                start_time = session_info.get('start_time', datetime.now())
                total_transcripts = session_info.get('total_transcripts', 0)
                
                # Cleanup session data
                self.cleanup_session(session_id)
                
                # Calculate session duration
                session_duration = (datetime.now() - start_time).total_seconds()
                
                # Send session stopped confirmation
                await self.send_message(websocket, client_id, {
                    'type': 'realtime_session_stopped',
                    'data': {
                        'session_id': session_id,
                        'session_duration': session_duration,
                        'total_transcripts': total_transcripts,
                        'timestamp': get_current_timestamp()
                    }
                })
                
                logger.info(f"✅ Real-time session {session_id} stopped successfully")
                
            except Exception as e:
                logger.error(f"❌ Error stopping real-time session {session_id}: {e}")
                await self.send_error(websocket, client_id, f"Failed to stop session: {str(e)}")
    
    async def get_session_status(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Get status of a real-time session"""
        
        session_id = data.get('session_id')
        
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Calculate session duration
                session_duration = (datetime.now() - session['start_time']).total_seconds()
                
                # Get transcription buffer length
                buffer_length = len(self.transcription_buffer.get(session_id, []))
                
                # Get last fraud analysis
                last_analysis = session.get('last_fraud_analysis')
                
                status_data = {
                    'session_id': session_id,
                    'status': session.get('status', 'unknown'),
                    'session_duration': session_duration,
                    'total_transcripts': session.get('total_transcripts', 0),
                    'buffer_length': buffer_length,
                    'last_analysis': {
                        'risk_score': last_analysis.get('risk_score', 0) if last_analysis else 0,
                        'scam_type': last_analysis.get('scam_type', 'unknown') if last_analysis else 'unknown'
                    } if last_analysis else None,
                    'timestamp': get_current_timestamp()
                }
            else:
                status_data = {
                    'session_id': session_id,
                    'status': 'not_found',
                    'error': 'Session not found',
                    'timestamp': get_current_timestamp()
                }
            
            await self.send_message(websocket, client_id, {
                'type': 'session_status_response',
                'data': status_data
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting session status for {session_id}: {e}")
            await self.send_error(websocket, client_id, f"Status error: {str(e)}")
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session data"""
        
        try:
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Remove from realtime sessions mapping
            if session_id in self.realtime_sessions:
                del self.realtime_sessions[session_id]
            
            # Clear transcription buffer
            if session_id in self.transcription_buffer:
                del self.transcription_buffer[session_id]
            
            logger.info(f"🧹 Cleaned up session data for {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up session {session_id}: {e}")
    
    def cleanup_client_sessions(self, client_id: str) -> None:
        """Clean up all sessions for a disconnected client"""
        
        sessions_to_cleanup = []
        
        # Find sessions for this client
        for session_id, session in self.active_sessions.items():
            if session.get('client_id') == client_id:
                sessions_to_cleanup.append(session_id)
        
        # Clean up each session
        for session_id in sessions_to_cleanup:
            try:
                # Stop the audio agent session
                asyncio.create_task(realtime_audio_agent.stop_realtime_session(session_id))
                
                # Clean up session data
                self.cleanup_session(session_id)
                
            except Exception as e:
                logger.error(f"❌ Error cleaning up session {session_id} for client {client_id}: {e}")
        
        if sessions_to_cleanup:
            logger.info(f"🧹 Cleaned up {len(sessions_to_cleanup)} sessions for disconnected client {client_id}")
    
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
            code="REALTIME_ERROR"
        )
        
        await self.send_message(websocket, client_id, {
            'type': 'error',
            'data': error_response
        })
    
    def get_active_sessions_count(self) -> int:
        """Get count of active real-time sessions"""
        return len([
            session for session in self.active_sessions.values()
            if session.get('status') == 'active'
        ])
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        
        total_sessions = len(self.active_sessions)
        active_sessions = self.get_active_sessions_count()
        
        # Calculate average session duration for completed sessions
        session_durations = []
        for session in self.active_sessions.values():
            if session.get('status') == 'active':
                duration = (datetime.now() - session['start_time']).total_seconds()
                session_durations.append(duration)
        
        avg_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        # Get transcription stats
        total_transcripts = sum(
            session.get('total_transcripts', 0) 
            for session in self.active_sessions.values()
        )
        
        return {
            'handler_type': 'RealTimeFraudDetectionHandler',
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'average_session_duration': avg_duration,
            'total_transcripts_processed': total_transcripts,
            'transcription_buffers_active': len(self.transcription_buffer),
            'system_status': 'operational',
            'timestamp': get_current_timestamp()
        }
    
    async def periodic_cleanup(self) -> None:
        """Periodic cleanup of stale sessions"""
        
        try:
            stale_sessions = []
            current_time = datetime.now()
            
            # Find stale sessions (inactive for more than 30 minutes)
            for session_id, session in self.active_sessions.items():
                last_activity = session.get('last_activity', session.get('start_time', current_time))
                if (current_time - last_activity).total_seconds() > 1800:  # 30 minutes
                    stale_sessions.append(session_id)
            
            # Clean up stale sessions
            for session_id in stale_sessions:
                logger.info(f"🧹 Cleaning up stale session: {session_id}")
                await realtime_audio_agent.stop_realtime_session(session_id)
                self.cleanup_session(session_id)
            
            if stale_sessions:
                logger.info(f"🧹 Cleaned up {len(stale_sessions)} stale real-time sessions")
                
        except Exception as e:
            logger.error(f"❌ Error in periodic cleanup: {e}")

# Export the handler class
__all__ = ['RealTimeFraudDetectionHandler']
