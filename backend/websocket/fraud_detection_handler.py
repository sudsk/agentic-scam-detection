# backend/websocket/fraud_detection_handler.py - Enhanced debugging version

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from ..agents.fraud_detection_system import fraud_detection_system
from ..websocket.connection_manager import ConnectionManager
from ..utils import get_current_timestamp, log_agent_activity, create_error_response

logger = logging.getLogger(__name__)

class FraudDetectionWebSocketHandler:
    """
    Enhanced WebSocket handler with better debugging and error handling
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_sessions: Dict[str, Dict] = {}
        self.processing_sessions: Set[str] = set()
        
        logger.info("🔌 FraudDetectionWebSocketHandler initialized with debugging")
    
    async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages with enhanced debugging"""
        
        message_type = message.get('type')
        data = message.get('data', {})
        
        logger.info(f"📨 WebSocket message from {client_id}: {message_type}")
        logger.debug(f"📊 Message data: {data}")
        
        try:
            if message_type == 'process_audio':
                await self.handle_audio_processing(websocket, client_id, data)
            elif message_type == 'ping':
                await self.send_message(websocket, client_id, {
                    'type': 'pong',
                    'data': {'timestamp': get_current_timestamp()}
                })
            elif message_type == 'get_status':
                await self.handle_status_request(websocket, client_id)
            elif message_type == 'cancel_processing':
                await self.handle_cancel_processing(websocket, client_id, data)
            else:
                logger.warning(f"❓ Unknown message type: {message_type} from {client_id}")
                await self.send_error(websocket, client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling WebSocket message from {client_id}: {e}")
            await self.send_error(websocket, client_id, f"Message processing error: {str(e)}")
    
    async def handle_audio_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Process audio through multi-agent system with enhanced real-time updates"""
        
        filename = data.get('filename')
        session_id = data.get('session_id')
        
        logger.info(f"🎵 Starting audio processing for {client_id}: {filename}")
        
        # Validate required parameters
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
            
        if not session_id:
            session_id = f"ws_session_{uuid.uuid4().hex[:8]}"
            logger.info(f"🆔 Generated session ID: {session_id}")
        
        # Check if session is already being processed
        if session_id in self.processing_sessions:
            await self.send_error(websocket, client_id, f"Session {session_id} is already being processed")
            return
        
        try:
            # Mark session as processing
            self.processing_sessions.add(session_id)
            
            # Store session info
            self.active_sessions[session_id] = {
                'client_id': client_id,
                'filename': filename,
                'websocket': websocket,
                'start_time': datetime.now(),
                'status': 'processing'
            }
            
            log_agent_activity("websocket", f"Starting audio processing for session {session_id}")
            
            # Send initial confirmation
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'session_id': session_id,
                    'filename': filename,
                    'stage': 'initializing',
                    'agent': 'WebSocket Handler'
                }
            })
            
            # Process through agents with real-time callbacks
            audio_path = f"data/sample_audio/{filename}"
            await self.process_with_realtime_updates(
                websocket, client_id, session_id, audio_path
            )
            
        except Exception as e:
            logger.error(f"❌ Audio processing error for session {session_id}: {e}")
            await self.send_error(websocket, client_id, f"Processing error: {str(e)}")
        finally:
            # Clean up processing session
            self.processing_sessions.discard(session_id)
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'completed'
    
    async def process_with_realtime_updates(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        audio_path: str
    ) -> None:
        """Process audio with enhanced real-time WebSocket updates"""
        
        try:
            logger.info(f"🔄 Starting multi-agent processing for session {session_id}")
            
            # Step 1: Audio Processing Agent
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'stage': 'audio_processing', 
                    'agent': 'Audio Processing Agent',
                    'session_id': session_id
                }
            })
            
            # Use consolidated agent directly
            logger.info(f"🎵 Processing audio with Audio Agent")
            transcription_result = fraud_detection_system.audio_processor.process(audio_path)
            
            if 'error' in transcription_result:
                raise Exception(f"Audio processing failed: {transcription_result['error']}")
            
            logger.info(f"✅ Audio processing complete. Segments: {len(transcription_result.get('speaker_segments', []))}")
            
            # Send transcription segments in real-time with delays
            if 'speaker_segments' in transcription_result:
                for i, segment in enumerate(transcription_result['speaker_segments']):
                    # Check if WebSocket is still connected
                    if client_id not in [conn.client_id for conn in self.connection_manager.active_connections]:
                        logger.warning(f"⚠️ Client {client_id} disconnected during processing")
                        return
                    
                    await self.send_message(websocket, client_id, {
                        'type': 'transcription_segment',
                        'data': {
                            'speaker': segment['speaker'],
                            'start': segment['start'],
                            'duration': segment['end'] - segment['start'],
                            'text': segment['text'],
                            'confidence': segment.get('confidence', 0.9),
                            'segment_index': i,
                            'session_id': session_id
                        }
                    })
                    
                    logger.debug(f"📤 Sent segment {i+1}/{len(transcription_result['speaker_segments'])}")
                    
                    # Realistic delay between segments
                    await asyncio.sleep(1.0 + (i * 0.3))  # Increasing delay for realism
            
            # Extract customer text for fraud analysis
            customer_text = " ".join([
                seg['text'] for seg in transcription_result.get('speaker_segments', [])
                if seg['speaker'] == 'customer'
            ])
            
            if not customer_text.strip():
                await self.send_message(websocket, client_id, {
                    'type': 'processing_complete',
                    'data': {
                        'session_id': session_id,
                        'message': 'No customer speech detected for analysis',
                        'final_risk_score': 0,
                        'total_agents': 1
                    }
                })
                return
            
            logger.info(f"🗣️ Customer text extracted: {len(customer_text)} characters")
            
            # Step 2: Fraud Detection Agent
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'stage': 'fraud_detection', 
                    'agent': 'Fraud Detection Agent',
                    'session_id': session_id
                }
            })
            
            # Add delay for realism
            await asyncio.sleep(1.5)
            
            # Use consolidated fraud detection agent
            logger.info(f"🔍 Analyzing fraud patterns")
            fraud_analysis = fraud_detection_system.fraud_detector.process(customer_text)
            
            if 'error' in fraud_analysis:
                raise Exception(f"Fraud detection failed: {fraud_analysis['error']}")
            
            logger.info(f"✅ Fraud analysis complete. Risk score: {fraud_analysis['risk_score']}%")
            
            await self.send_message(websocket, client_id, {
                'type': 'fraud_analysis_update',
                'data': {
                    'session_id': session_id,
                    'risk_score': fraud_analysis['risk_score'],
                    'risk_level': fraud_analysis['risk_level'],
                    'scam_type': fraud_analysis['scam_type'],
                    'detected_patterns': fraud_analysis['detected_patterns'],
                    'confidence': fraud_analysis['confidence']
                }
            })
            
            # Step 3: Policy Guidance Agent (if needed)
            policy_guidance = None
            if fraud_analysis.get('risk_score', 0) >= 40:
                await self.send_message(websocket, client_id, {
                    'type': 'processing_started',
                    'data': {
                        'stage': 'policy_guidance', 
                        'agent': 'Policy Guidance Agent',
                        'session_id': session_id
                    }
                })
                
                await asyncio.sleep(1.0)
                
                # Use consolidated policy guidance agent
                logger.info(f"📚 Retrieving policy guidance")
                policy_guidance = fraud_detection_system.policy_guide.process({
                    'scam_type': fraud_analysis['scam_type'],
                    'risk_score': fraud_analysis['risk_score']
                })
                
                if 'error' not in policy_guidance:
                    logger.info(f"✅ Policy guidance retrieved")
                    await self.send_message(websocket, client_id, {
                        'type': 'policy_guidance_ready',
                        'data': {
                            'session_id': session_id,
                            **policy_guidance.get('agent_guidance', {})
                        }
                    })
                else:
                    logger.warning(f"⚠️ Policy guidance error: {policy_guidance['error']}")
            
            # Step 4: Case Management (if high risk)
            case_info = None
            if fraud_analysis.get('risk_score', 0) >= 60:
                await self.send_message(websocket, client_id, {
                    'type': 'processing_started',
                    'data': {
                        'stage': 'case_management', 
                        'agent': 'Case Management Agent',
                        'session_id': session_id
                    }
                })
                
                await asyncio.sleep(0.8)
                
                # Use consolidated case management agent
                logger.info(f"📋 Creating case for high-risk session")
                case_info = fraud_detection_system.case_manager.process({
                    "session_id": session_id,
                    "fraud_analysis": fraud_analysis,
                    "customer_id": None
                })
                
                if 'error' not in case_info:
                    logger.info(f"✅ Case created: {case_info.get('case_id')}")
                    await self.send_message(websocket, client_id, {
                        'type': 'case_created',
                        'data': {
                            'session_id': session_id,
                            'case_id': case_info.get('case_id'),
                            'priority': case_info.get('priority'),
                            'status': case_info.get('status')
                        }
                    })
                else:
                    logger.warning(f"⚠️ Case creation error: {case_info['error']}")
            
            # Final completion message
            processing_time = (datetime.now() - self.active_sessions[session_id]['start_time']).total_seconds()
            
            await self.send_message(websocket, client_id, {
                'type': 'processing_complete',
                'data': {
                    'session_id': session_id,
                    'total_agents': len(fraud_detection_system._get_agents_involved(fraud_analysis['risk_score'])),
                    'final_risk_score': fraud_analysis.get('risk_score', 0),
                    'scam_type': fraud_analysis.get('scam_type', 'unknown'),
                    'case_created': case_info is not None,
                    'processing_time': processing_time
                }
            })
            
            logger.info(f"✅ Complete processing pipeline finished for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Processing pipeline error for session {session_id}: {e}")
            await self.send_error(websocket, client_id, f"Pipeline error: {str(e)}")
        finally:
            # Clean up session
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['end_time'] = datetime.now()
                self.active_sessions[session_id]['status'] = 'completed'
    
    async def send_message(self, websocket: WebSocket, client_id: str, message: Dict) -> bool:
        """Send message to specific client with enhanced error handling"""
        try:
            message_json = json.dumps(message)
            await websocket.send_text(message_json)
            logger.debug(f"📤 Sent to {client_id}: {message['type']}")
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
            code="WEBSOCKET_ERROR"
        )
        
        await self.send_message(websocket, client_id, {
            'type': 'error',
            'data': error_response
        })
    
    async def handle_status_request(self, websocket: WebSocket, client_id: str) -> None:
        """Handle status request from client with system information"""
        try:
            # Get system status from fraud detection system
            system_status = fraud_detection_system.get_agent_status()
            
            # Add WebSocket-specific status
            ws_status = {
                'active_sessions': len(self.active_sessions),
                'processing_sessions': len(self.processing_sessions),
                'connected_clients': len(self.connection_manager.active_connections),
                'handler_status': 'operational',
                'sessions': {
                    session_id: {
                        'status': session_data['status'],
                        'filename': session_data['filename'],
                        'start_time': session_data['start_time'].isoformat() if isinstance(session_data['start_time'], datetime) else session_data['start_time']
                    }
                    for session_id, session_data in self.active_sessions.items()
                }
            }
            
            await self.send_message(websocket, client_id, {
                'type': 'status_response',
                'data': {
                    'system_status': system_status,
                    'websocket_status': ws_status,
                    'timestamp': get_current_timestamp()
                }
            })
            
            logger.info(f"📊 Status sent to {client_id}")
            
        except Exception as e:
            logger.error(f"❌ Error getting status for {client_id}: {e}")
            await self.send_error(websocket, client_id, f"Status error: {str(e)}")
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len([
            session for session in self.active_sessions.values()
            if session.get('status') in ['processing', 'active']
        ])
    
    def cleanup_client_sessions(self, client_id: str) -> None:
        """Clean up sessions when client disconnects"""
        sessions_to_remove = []
        
        for session_id, session_data in self.active_sessions.items():
            if session_data.get('client_id') == client_id:
                sessions_to_remove.append(session_id)
                # Remove from processing if still active
                self.processing_sessions.discard(session_id)
        
        for session_id in sessions_to_remove:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'disconnected'
                self.active_sessions[session_id]['end_time'] = datetime.now()
        
        if sessions_to_remove:
            logger.info(f"🧹 Cleaned up {len(sessions_to_remove)} sessions for disconnected client {client_id}")

# Export the handler class
__all__ = ['FraudDetectionWebSocketHandler']
