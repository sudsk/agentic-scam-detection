# backend/websocket/fraud_detection_handler.py - FIXED IMPORTS

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Set, List  # FIXED: Added List import
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from ..agents.fraud_detection_system import fraud_detection_system
from ..websocket.connection_manager import ConnectionManager
from ..utils import get_current_timestamp, log_agent_activity, create_error_response

logger = logging.getLogger(__name__)

class FraudDetectionWebSocketHandler:
    """
    Real-time streaming WebSocket handler that transcribes audio as it plays
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_sessions: Dict[str, Dict] = {}
        self.processing_sessions: Set[str] = set()
        self.streaming_tasks: Dict[str, asyncio.Task] = {}  # Track streaming tasks
        
        logger.info("🔌 Real-time streaming WebSocket handler initialized")
    
    async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]) -> None:
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
        """Start real-time audio processing that follows audio playback timing"""
        
        filename = data.get('filename')
        session_id = data.get('session_id')
        
        logger.info(f"🎵 Starting REAL-TIME audio processing for {client_id}: {filename}")
        
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
            
        if not session_id:
            session_id = f"ws_session_{uuid.uuid4().hex[:8]}"
            logger.info(f"🆔 Generated session ID: {session_id}")
        
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
            
            # Send initial confirmation
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'session_id': session_id,
                    'filename': filename,
                    'stage': 'initializing',
                    'agent': 'Real-time Audio Processor'
                }
            })
            
            # Start streaming transcription task
            streaming_task = asyncio.create_task(
                self.stream_realtime_transcription(websocket, client_id, session_id, filename)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            # Wait for streaming to complete
            await streaming_task
            
        except Exception as e:
            logger.error(f"❌ Real-time processing error for session {session_id}: {e}")
            await self.send_error(websocket, client_id, f"Processing error: {str(e)}")
        finally:
            # Clean up
            self.processing_sessions.discard(session_id)
            if session_id in self.streaming_tasks:
                del self.streaming_tasks[session_id]
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'completed'
    
    async def stream_realtime_transcription(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        filename: str
    ) -> None:
        """Stream transcription in real-time as if listening to live audio"""
        
        try:
            logger.info(f"🎙️ Starting real-time transcription stream for {filename}")
            
            # Get the audio scenario based on filename
            scenario_type = self._determine_scenario_type(filename)
            segments = self._get_audio_segments(scenario_type)
            
            if not segments:
                logger.warning(f"No segments found for {filename}")
                return
            
            # Send processing started message
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'stage': 'audio_processing',
                    'agent': 'Real-time Audio Agent',
                    'session_id': session_id
                }
            })
            
            # Track accumulated customer speech for fraud analysis
            customer_speech_accumulator = []
            
            # Stream each segment with realistic timing
            for i, segment in enumerate(segments):
                # Check if client is still connected
                if client_id not in [conn.client_id for conn in self.connection_manager.active_connections]:
                    logger.warning(f"⚠️ Client {client_id} disconnected during streaming")
                    return
                
                # Wait until the segment's start time (real-time simulation)
                if i == 0:
                    # Small delay before first segment
                    await asyncio.sleep(1.0)
                else:
                    # Wait for the time difference between segments
                    prev_segment = segments[i-1]
                    time_gap = segment['start'] - prev_segment['end']
                    if time_gap > 0:
                        await asyncio.sleep(time_gap)
                
                # Send the transcription segment
                await self.send_message(websocket, client_id, {
                    'type': 'transcription_segment',
                    'data': {
                        'speaker': segment['speaker'],
                        'start': segment['start'],
                        'duration': segment['end'] - segment['start'],
                        'text': segment['text'],
                        'confidence': segment.get('confidence', 0.92),
                        'segment_index': i,
                        'session_id': session_id
                    }
                })
                
                logger.info(f"📤 Streamed segment {i+1}/{len(segments)}: {segment['speaker']} - {segment['text'][:50]}...")
                
                # Accumulate customer speech for fraud analysis
                if segment['speaker'] == 'customer':
                    customer_speech_accumulator.append(segment['text'])
                    
                    # Trigger fraud analysis when we have enough customer speech
                    if len(customer_speech_accumulator) >= 2:  # After 2+ customer segments
                        combined_speech = " ".join(customer_speech_accumulator)
                        
                        # Run fraud analysis on accumulated speech
                        await self.run_fraud_analysis(
                            websocket, client_id, session_id, combined_speech
                        )
                
                # Simulate the duration of the speech segment
                segment_duration = segment['end'] - segment['start']
                await asyncio.sleep(min(segment_duration, 3.0))  # Cap at 3 seconds for demo
            
            # Final fraud analysis on all customer speech
            if customer_speech_accumulator:
                final_customer_speech = " ".join(customer_speech_accumulator)
                await self.run_final_analysis(
                    websocket, client_id, session_id, final_customer_speech
                )
            
            # Complete the processing
            processing_time = (datetime.now() - self.active_sessions[session_id]['start_time']).total_seconds()
            
            await self.send_message(websocket, client_id, {
                'type': 'processing_complete',
                'data': {
                    'session_id': session_id,
                    'total_segments': len(segments),
                    'processing_time': processing_time,
                    'message': 'Real-time transcription completed'
                }
            })
            
            logger.info(f"✅ Real-time streaming completed for session {session_id}")
            
        except asyncio.CancelledError:
            logger.info(f"🛑 Real-time streaming cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"❌ Real-time streaming error for session {session_id}: {e}")
            await self.send_error(websocket, client_id, f"Streaming error: {str(e)}")
    
    async def run_fraud_analysis(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Run fraud analysis on accumulated customer speech"""
        
        try:
            logger.info(f"🔍 Running fraud analysis on: {customer_text[:100]}...")
            
            # Send analysis started message
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'stage': 'fraud_detection',
                    'agent': 'Fraud Detection Agent',
                    'session_id': session_id
                }
            })
            
            # Small delay for realism
            await asyncio.sleep(0.8)
            
            # Use fraud detection agent
            fraud_analysis = fraud_detection_system.fraud_detector.process(customer_text)
            
            if 'error' not in fraud_analysis:
                logger.info(f"✅ Fraud analysis complete. Risk score: {fraud_analysis['risk_score']}%")
                
                # Send fraud analysis update
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
                
                # If risk is high enough, get policy guidance
                if fraud_analysis.get('risk_score', 0) >= 40:
                    await self.get_policy_guidance(
                        websocket, client_id, session_id, fraud_analysis
                    )
            
        except Exception as e:
            logger.error(f"❌ Fraud analysis error: {e}")
    
    async def run_final_analysis(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Run final comprehensive analysis"""
        
        try:
            # Run final fraud analysis
            await self.run_fraud_analysis(websocket, client_id, session_id, customer_text)
            
            # Small delay
            await asyncio.sleep(0.5)
            
            # Get updated fraud analysis for case creation
            fraud_analysis = fraud_detection_system.fraud_detector.process(customer_text)
            
            # Create case if high risk
            if fraud_analysis.get('risk_score', 0) >= 60:
                await self.send_message(websocket, client_id, {
                    'type': 'processing_started',
                    'data': {
                        'stage': 'case_management',
                        'agent': 'Case Management Agent',
                        'session_id': session_id
                    }
                })
                
                await asyncio.sleep(0.5)
                
                case_info = fraud_detection_system.case_manager.process({
                    "session_id": session_id,
                    "fraud_analysis": fraud_analysis,
                    "customer_id": None
                })
                
                if 'error' not in case_info:
                    await self.send_message(websocket, client_id, {
                        'type': 'case_created',
                        'data': {
                            'session_id': session_id,
                            'case_id': case_info.get('case_id'),
                            'priority': case_info.get('priority'),
                            'status': case_info.get('status')
                        }
                    })
            
        except Exception as e:
            logger.error(f"❌ Final analysis error: {e}")
    
    async def get_policy_guidance(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        fraud_analysis: Dict
    ) -> None:
        """Get policy guidance based on fraud analysis"""
        
        try:
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'stage': 'policy_guidance',
                    'agent': 'Policy Guidance Agent',
                    'session_id': session_id
                }
            })
            
            await asyncio.sleep(0.6)
            
            policy_guidance = fraud_detection_system.policy_guide.process({
                'scam_type': fraud_analysis['scam_type'],
                'risk_score': fraud_analysis['risk_score']
            })
            
            if 'error' not in policy_guidance:
                await self.send_message(websocket, client_id, {
                    'type': 'policy_guidance_ready',
                    'data': {
                        'session_id': session_id,
                        **policy_guidance.get('agent_guidance', {})
                    }
                })
                
        except Exception as e:
            logger.error(f"❌ Policy guidance error: {e}")
    
    def _determine_scenario_type(self, filename: str) -> str:
        """Determine scenario type based on filename"""
        filename_lower = filename.lower()
        
        if "investment" in filename_lower:
            return "investment"
        elif "romance" in filename_lower:
            return "romance"
        elif "impersonation" in filename_lower:
            return "impersonation"
        else:
            return "legitimate"
    
    def _get_audio_segments(self, scenario_type: str) -> List[Dict]:
        """Get realistic audio segments for streaming transcription"""
        
        scenarios = {
            "investment": [
                {"speaker": "customer", "start": 1.0, "end": 6.0, 
                 "text": "My investment advisor just called saying there's a margin call on my trading account.", "confidence": 0.94},
                {"speaker": "agent", "start": 6.5, "end": 8.5, 
                 "text": "I see. Can you tell me more about this advisor?", "confidence": 0.96},
                {"speaker": "customer", "start": 9.0, "end": 15.0, 
                 "text": "He's been guaranteeing thirty-five percent monthly returns and says I need to transfer fifteen thousand pounds immediately.", "confidence": 0.91},
                {"speaker": "agent", "start": 15.5, "end": 18.0,
                 "text": "I understand. Can you tell me how you first met this advisor?", "confidence": 0.95},
                {"speaker": "customer", "start": 18.5, "end": 23.0,
                 "text": "He called me initially about cryptocurrency, then moved me to forex trading.", "confidence": 0.93},
                {"speaker": "agent", "start": 23.5, "end": 26.5,
                 "text": "Have you been able to withdraw any profits from this investment?", "confidence": 0.97},
                {"speaker": "customer", "start": 27.0, "end": 32.0,
                 "text": "He says it's better to reinvest the profits for compound gains. But I really need to act fast on this margin call.", "confidence": 0.89}
            ],
            "romance": [
                {"speaker": "customer", "start": 1.0, "end": 5.0,
                 "text": "I need to send four thousand pounds to Turkey urgently.", "confidence": 0.92},
                {"speaker": "agent", "start": 5.5, "end": 7.0,
                 "text": "May I ask who this payment is for?", "confidence": 0.96},
                {"speaker": "customer", "start": 7.5, "end": 12.0,
                 "text": "My partner Alex is stuck in Istanbul. We've been together for seven months online.", "confidence": 0.90},
                {"speaker": "agent", "start": 12.5, "end": 14.5,
                 "text": "Have you and Alex met in person?", "confidence": 0.97},
                {"speaker": "customer", "start": 15.0, "end": 20.0,
                 "text": "Not yet, but we were planning to meet next month before this emergency happened.", "confidence": 0.88},
                {"speaker": "agent", "start": 20.5, "end": 22.5,
                 "text": "What kind of emergency is Alex facing?", "confidence": 0.95},
                {"speaker": "customer", "start": 23.0, "end": 28.0,
                 "text": "He's in hospital and needs money for treatment before they'll help him properly.", "confidence": 0.87}
            ],
            "impersonation": [
                {"speaker": "customer", "start": 1.0, "end": 6.0,
                 "text": "Someone from bank security called saying my account has been compromised.", "confidence": 0.93},
                {"speaker": "agent", "start": 6.5, "end": 8.5,
                 "text": "Did they ask for any personal information?", "confidence": 0.96},
                {"speaker": "customer", "start": 9.0, "end": 15.0,
                 "text": "Yes, they asked me to confirm my card details and PIN to verify my identity immediately.", "confidence": 0.89},
                {"speaker": "agent", "start": 15.5, "end": 19.0,
                 "text": "I need you to know that HSBC will never ask for your PIN over the phone.", "confidence": 0.98},
                {"speaker": "customer", "start": 19.5, "end": 23.0,
                 "text": "But they said fraudsters were trying to steal my money right now!", "confidence": 0.86},
                {"speaker": "agent", "start": 23.5, "end": 26.5,
                 "text": "That person was the fraudster. Did you give them any information?", "confidence": 0.94}
            ],
            "legitimate": [
                {"speaker": "customer", "start": 1.0, "end": 4.0,
                 "text": "Hi, I'd like to check my account balance please.", "confidence": 0.95},
                {"speaker": "agent", "start": 4.5, "end": 6.0,
                 "text": "Certainly, I can help you with that.", "confidence": 0.97},
                {"speaker": "customer", "start": 6.5, "end": 9.5,
                 "text": "I also want to set up a standing order for my rent.", "confidence": 0.94},
                {"speaker": "agent", "start": 10.0, "end": 12.0,
                 "text": "No problem. What's the amount and frequency?", "confidence": 0.96},
                {"speaker": "customer", "start": 12.5, "end": 16.0,
                 "text": "Four hundred and fifty pounds monthly on the first of each month.", "confidence": 0.92}
            ]
        }
        
        return scenarios.get(scenario_type, scenarios["legitimate"])
    
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
            code="WEBSOCKET_ERROR"
        )
        
        await self.send_message(websocket, client_id, {
            'type': 'error',
            'data': error_response
        })
    
    async def handle_status_request(self, websocket: WebSocket, client_id: str) -> None:
        """Handle status request from client"""
        try:
            system_status = fraud_detection_system.get_agent_status()
            
            ws_status = {
                'active_sessions': len(self.active_sessions),
                'processing_sessions': len(self.processing_sessions),
                'streaming_tasks': len(self.streaming_tasks),
                'connected_clients': len(self.connection_manager.active_connections)
            }
            
            await self.send_message(websocket, client_id, {
                'type': 'status_response',
                'data': {
                    'system_status': system_status,
                    'websocket_status': ws_status,
                    'timestamp': get_current_timestamp()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting status for {client_id}: {e}")
            await self.send_error(websocket, client_id, f"Status error: {str(e)}")
    
    async def handle_cancel_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Handle request to cancel processing"""
        session_id = data.get('session_id')
        
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id for cancellation")
            return
        
        # Cancel streaming task if running
        if session_id in self.streaming_tasks:
            self.streaming_tasks[session_id].cancel()
            del self.streaming_tasks[session_id]
        
        self.processing_sessions.discard(session_id)
        
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['status'] = 'cancelled'
            self.active_sessions[session_id]['end_time'] = datetime.now()
        
        await self.send_message(websocket, client_id, {
            'type': 'processing_cancelled',
            'data': {
                'session_id': session_id,
                'timestamp': get_current_timestamp()
            }
        })
        
        logger.info(f"🚫 Real-time processing cancelled for session {session_id}")
    
    def cleanup_client_sessions(self, client_id: str) -> None:
        """Clean up sessions when client disconnects"""
        sessions_to_remove = []
        
        for session_id, session_data in self.active_sessions.items():
            if session_data.get('client_id') == client_id:
                sessions_to_remove.append(session_id)
                
                # Cancel streaming task
                if session_id in self.streaming_tasks:
                    self.streaming_tasks[session_id].cancel()
                    del self.streaming_tasks[session_id]
                
                self.processing_sessions.discard(session_id)
        
        for session_id in sessions_to_remove:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'disconnected'
                self.active_sessions[session_id]['end_time'] = datetime.now()
        
        if sessions_to_remove:
            logger.info(f"🧹 Cleaned up {len(sessions_to_remove)} streaming sessions for {client_id}")
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len([
            session for session in self.active_sessions.values()
            if session.get('status') in ['processing', 'active']
        ])

# Export the handler class
__all__ = ['FraudDetectionWebSocketHandler']
