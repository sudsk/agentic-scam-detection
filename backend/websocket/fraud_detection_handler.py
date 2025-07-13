# backend/websocket/fraud_detection_handler.py - UPDATED WITH ADK CASE MANAGEMENT INTEGRATION
"""
Complete Fraud Detection WebSocket Handler - UPDATED
Now integrates with the new Case Management ADK Agent for ServiceNow incident creation
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from ..agents.audio_processor.agent import audio_processor_agent 
from ..agents.case_management.adk_agent import PureADKCaseManagementAgent
from ..websocket.connection_manager import ConnectionManager
from ..utils import get_current_timestamp, create_error_response
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class FraudDetectionWebSocketHandler:
    """Complete fraud detection WebSocket handler with ADK Case Management integration"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        
        # Session tracking (preserved from original)
        self.active_sessions: Dict[str, Dict] = {}
        self.customer_speech_buffer: Dict[str, List[str]] = {}
        self.session_analysis_history: Dict[str, List[Dict]] = {}
        self.accumulated_patterns: Dict[str, Dict] = {}
        
        # State tracking (preserved from original)
        self.processing_steps: Dict[str, List[str]] = {}
        self.agent_states: Dict[str, Dict] = {}
        
        self.settings = get_settings()
        
        # Import individual ADK agents
        self._import_adk_agents()
        
        # UPDATED: Initialize ADK Case Management Agent
        self.case_management_agent = PureADKCaseManagementAgent()
        
        # Initialize customer profiles (preserved from original)
        self._initialize_customer_profiles()
        
        logger.info("🤖 Complete Fraud Detection WebSocket handler initialized with ADK Case Management")
    
    def _import_adk_agents(self):
        """Import individual ADK agents"""
        try:
            from ..agents.scam_detection.adk_agent import create_scam_detection_agent
            from ..agents.policy_guidance.adk_agent import create_policy_guidance_agent  
            from ..agents.summarization.adk_agent import create_summarization_agent
            from ..agents.orchestrator.adk_agent import create_orchestrator_agent
            
            self.scam_detection_agent = create_scam_detection_agent()
            self.policy_guidance_agent = create_policy_guidance_agent()
            self.summarization_agent = create_summarization_agent()
            self.orchestrator_agent = create_orchestrator_agent()
            
            self.adk_agents_available = True
            logger.info("✅ All ADK agents imported successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to import ADK agents: {e}")
            self.adk_agents_available = False
    
    def _initialize_customer_profiles(self):
        """Initialize customer profiles (preserved from original)"""
        self.customer_profiles = {
            'romance_scam_1.wav': {
                'name': 'Mrs Patricia Williams',
                'account': '****3847',
                'phone': '+44 1202 555123',
                'demographics': {'age': 67, 'location': 'Bournemouth, UK', 'relationship': 'Widow'}
            },
            'investment_scam_live_call.wav': {
                'name': 'Mr David Chen', 
                'account': '****5691',
                'phone': '+44 161 555456',
                'demographics': {'age': 42, 'location': 'Manchester, UK', 'relationship': 'Married'}
            },
            'impersonation_scam_live_call.wav': {
                'name': 'Ms Sarah Thompson',
                'account': '****7234', 
                'phone': '+44 121 555789',
                'demographics': {'age': 34, 'location': 'Birmingham, UK', 'relationship': 'Single'}
            }
        }
    
    async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages (complete functionality preserved)"""
        
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
            elif message_type == 'get_session_analysis':
                await self.handle_session_analysis_request(websocket, client_id, data)
            elif message_type == 'create_manual_case':
                await self.handle_manual_case_creation(websocket, client_id, data)
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
        """Start audio processing with complete session management"""
        
        filename = data.get('filename')
        session_id = data.get('session_id') or f"session_{uuid.uuid4().hex[:8]}"
        
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
        
        if session_id in self.active_sessions:
            await self.send_error(websocket, client_id, f"Session {session_id} already active")
            return
        
        try:
            logger.info(f"🎵 Starting complete audio processing: {filename} for {client_id}")
            
            # Extract scam context from filename (preserved from original)
            scam_context = self._extract_scam_context_from_filename(filename)
            
            # Initialize complete session tracking
            self.active_sessions[session_id] = {
                'client_id': client_id,
                'filename': filename,
                'websocket': websocket,
                'start_time': datetime.now(),
                'status': 'processing',
                'scam_type_context': scam_context,
                'customer_profile': self._get_customer_profile(filename),
                'processing_stopped': False,
                'audio_completed': False,
                'restart_count': 0
            }
            
            # Initialize all tracking structures (preserved from original)
            self.customer_speech_buffer[session_id] = []
            self.session_analysis_history[session_id] = []
            self.accumulated_patterns[session_id] = {}
            self.processing_steps[session_id] = []
            self.agent_states[session_id] = {
                'scam_analysis_active': False,
                'policy_analysis_active': False,
                'summarization_active': False,
                'orchestrator_active': False,
                'case_management_active': False,  # UPDATED: Added case management
                'current_agent': '',
                'agents_completed': []
            }
            
            # Create comprehensive callback
            async def comprehensive_audio_callback(message_data: Dict) -> None:
                await self.handle_audio_message(websocket, client_id, session_id, message_data)
            
            # Start audio processing
            result = await audio_processor_agent.start_realtime_processing(
                session_id=session_id,
                audio_filename=filename,
                websocket_callback=comprehensive_audio_callback
            )
            
            if 'error' in result:
                await self.send_error(websocket, client_id, result['error'])
                self.cleanup_session(session_id)
                return
            
            # Send comprehensive confirmation (preserved structure)
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'session_id': session_id,
                    'filename': filename,
                    'audio_duration': result.get('audio_duration', 0),
                    'channels': result.get('channels', 1),
                    'processing_mode': 'complete_adk_agents_with_case_management',
                    'adk_agents_enabled': self.adk_agents_available,
                    'case_management_enabled': True,  # UPDATED
                    'scam_context': scam_context,
                    'customer_profile': self.active_sessions[session_id]['customer_profile'],
                    'channel_mapping': result.get('channel_mapping', 'Unknown'),
                    'timestamp': get_current_timestamp()
                }
            })
            
            logger.info(f"✅ Complete processing with ADK Case Management started for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error starting complete processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to start processing: {str(e)}")
            self.cleanup_session(session_id)
    
    async def handle_audio_message(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        message_data: Dict
    ) -> None:
        """Handle messages from audio processor with complete functionality"""
        
        message_type = message_data.get('type')
        data = message_data.get('data', {})
        
        try:
            if message_type == 'transcription_segment':
                # Forward transcription to client
                await self.send_message(websocket, client_id, message_data)
                
                # Complete customer speech processing
                speaker = data.get('speaker')
                text = data.get('text', '')
                
                if speaker == 'customer' and text.strip():
                    await self.process_customer_speech_complete(websocket, client_id, session_id, text, data)
            
            elif message_type == 'transcription_interim':
                # Forward interim results
                await self.send_message(websocket, client_id, message_data)
            
            elif message_type == 'ui_player_start':
                # Handle UI synchronization (preserved from original)
                self.active_sessions[session_id]["ui_player_started"] = True
                await self.send_message(websocket, client_id, message_data)
            
            elif message_type == 'ui_player_stop':
                # Handle UI stop (preserved from original)
                await self.send_message(websocket, client_id, message_data)
            
            elif message_type == 'processing_progress':
                # Forward progress updates (preserved from original)
                await self.send_message(websocket, client_id, message_data)
            
            elif message_type == 'processing_complete':
                # Complete processing completion handling
                await self.handle_processing_complete_comprehensive(websocket, client_id, session_id)
                await self.send_message(websocket, client_id, message_data)
            
            elif message_type == 'streaming_started':
                # Handle streaming status (preserved from original)
                await self.send_message(websocket, client_id, message_data)
            
            else:
                # Forward other messages
                await self.send_message(websocket, client_id, message_data)
                
        except Exception as e:
            logger.error(f"❌ Error handling audio message: {e}")
    
    async def process_customer_speech_complete(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str,
        segment_data: Dict
    ) -> None:
        """Complete customer speech processing with all original functionality"""
        
        try:
            # Add to speech buffer with metadata (preserved from original)
            if session_id not in self.customer_speech_buffer:
                self.customer_speech_buffer[session_id] = []
            
            self.customer_speech_buffer[session_id].append({
                'text': customer_text,
                'timestamp': segment_data.get('start', 0),
                'confidence': segment_data.get('confidence', 0.0),
                'segment_id': segment_data.get('segment_id')
            })
            
            # Trigger analysis every customer segment (preserved behavior)
            buffer_length = len(self.customer_speech_buffer[session_id])
            
            if buffer_length >= 1:
                # Use accumulated speech for analysis
                accumulated_speech = " ".join([
                    segment['text'] for segment in self.customer_speech_buffer[session_id]
                ])
                
                if len(accumulated_speech.strip()) >= 15:
                    await self.run_complete_agent_pipeline(websocket, client_id, session_id, accumulated_speech)
            
        except Exception as e:
            logger.error(f"❌ Error in complete customer speech processing: {e}")
    
    async def run_complete_agent_pipeline(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Run complete agent pipeline with all original functionality"""
        
        try:
            if not self.adk_agents_available:
                logger.warning("ADK agents not available - using fallback")
                return
            
            logger.info(f"🤖 Running complete agent pipeline for session {session_id}")
            
            # Get session context (preserved from original)
            session_data = self.active_sessions[session_id]
            scam_context = session_data.get('scam_type_context', 'unknown')
            customer_profile = session_data.get('customer_profile', {})
            
            # Update processing steps (preserved from original)
            self.processing_steps[session_id].append(f"🤖 Starting ADK analysis #{len(self.session_analysis_history[session_id]) + 1}")
            
            # STEP 1: ADK Scam Detection Agent
            await self.send_message(websocket, client_id, {
                'type': 'adk_scam_analysis_started',
                'data': {
                    'session_id': session_id,
                    'agent': 'ADK Scam Detection Agent',
                    'text_length': len(customer_text),
                    'context': scam_context,
                    'analysis_number': len(self.session_analysis_history[session_id]) + 1,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Update agent state
            self.agent_states[session_id]['scam_analysis_active'] = True
            self.agent_states[session_id]['current_agent'] = 'ADK Scam Detection Agent'
            
            # Run ADK scam detection with complete context
            scam_analysis_context = {
                'session_id': session_id,
                'scam_context': scam_context,
                'customer_profile': customer_profile,
                'previous_analysis_count': len(self.session_analysis_history[session_id]),
                'call_duration': (datetime.now() - session_data['start_time']).total_seconds()
            }
            
            scam_analysis_result = await self._run_scam_detection_agent_complete(customer_text, scam_analysis_context)
            
            # Parse and accumulate patterns (preserved from original)
            parsed_scam_analysis = await self._parse_and_accumulate_patterns(session_id, scam_analysis_result)
            
            # Store in history (preserved from original)
            self.session_analysis_history[session_id].append(parsed_scam_analysis)
            
            # Update agent state
            self.agent_states[session_id]['scam_analysis_active'] = False
            self.agent_states[session_id]['agents_completed'].append('scam_detection')
            self.processing_steps[session_id].append(f"✅ ADK Scam Analysis: {parsed_scam_analysis.get('risk_score', 0)}% risk")
            
            # Send comprehensive results (preserved structure)
            await self.send_message(websocket, client_id, {
                'type': 'fraud_analysis_update',
                'data': {
                    'session_id': session_id,
                    'risk_score': parsed_scam_analysis.get('risk_score', 0),
                    'risk_level': parsed_scam_analysis.get('risk_level', 'MINIMAL'),
                    'scam_type': parsed_scam_analysis.get('scam_type', 'unknown'),
                    'detected_patterns': self.accumulated_patterns[session_id],
                    'confidence': parsed_scam_analysis.get('confidence', 0.0),
                    'explanation': parsed_scam_analysis.get('explanation', ''),
                    'pattern_count': len(self.accumulated_patterns[session_id]),
                    'analysis_number': len(self.session_analysis_history[session_id]),
                    'adk_powered': True,
                    'processing_agent': 'ADK Scam Detection Agent',
                    'timestamp': get_current_timestamp()
                }
            })
            
            risk_score = parsed_scam_analysis.get('risk_score', 0)
            
            # STEP 2: ADK Policy Guidance (if medium+ risk) - preserved logic
            if risk_score >= 40:
                await self._run_policy_guidance_step(websocket, client_id, session_id, parsed_scam_analysis)
            
            # STEP 3: ADK Orchestrator Decision (if high risk) - preserved logic
            if risk_score >= 60:
                await self._run_orchestrator_step(websocket, client_id, session_id, parsed_scam_analysis)
            
            logger.info(f"✅ Complete agent pipeline finished: {risk_score}% risk ({len(self.accumulated_patterns[session_id])} patterns)")
            
        except Exception as e:
            logger.error(f"❌ Error in complete agent pipeline: {e}")
            await self.send_error(websocket, client_id, f"Agent pipeline error: {str(e)}")
    
    async def _run_scam_detection_agent_complete(self, customer_text: str, context: Dict) -> str:
        """Run scam detection agent with complete context"""
        try:
            # Prepare comprehensive prompt for ADK agent
            scam_prompt = f"""
FRAUD ANALYSIS REQUEST:

CUSTOMER SPEECH: "{customer_text}"

CONTEXT:
- Session ID: {context.get('session_id')}
- Scam Context: {context.get('scam_context', 'unknown')}
- Customer: {context.get('customer_profile', {}).get('name', 'Unknown')}
- Customer Age: {context.get('customer_profile', {}).get('demographics', {}).get('age', 'Unknown')}
- Call Duration: {context.get('call_duration', 0):.1f} seconds
- Previous Analyses: {context.get('previous_analysis_count', 0)}

INSTRUCTIONS:
Focus analysis on {context.get('scam_context', 'general fraud')} patterns if context suggests this scam type.
Provide enhanced risk assessment considering customer demographics and call progression.
"""
            
            result = await self.scam_detection_agent.run(scam_prompt, context)
            return result
            
        except Exception as e:
            logger.error(f"❌ ADK scam detection agent error: {e}")
            return f"Error in scam detection: {str(e)}"
    
    async def _parse_and_accumulate_patterns(self, session_id: str, analysis_result: str) -> Dict[str, Any]:
        """Parse analysis result and accumulate patterns (preserved from original)"""
        try:
            # Parse the ADK result
            parsed = self.scam_detection_agent.parse_result(analysis_result)
            
            # Accumulate patterns across session (preserved logic)
            current_patterns = parsed.get('detected_patterns', [])
            session_patterns = self.accumulated_patterns.get(session_id, {})
            
            # Enhanced pattern accumulation
            for pattern_name in current_patterns:
                if pattern_name in session_patterns:
                    session_patterns[pattern_name]['count'] += 1
                    session_patterns[pattern_name]['confidence'] = min(
                        session_patterns[pattern_name]['confidence'] + 0.1, 1.0
                    )
                else:
                    session_patterns[pattern_name] = {
                        'pattern_name': pattern_name,
                        'count': 1,
                        'weight': 25,  # Default weight
                        'severity': 'medium',
                        'confidence': 0.7,
                        'matches': [pattern_name]
                    }
            
            self.accumulated_patterns[session_id] = session_patterns
            
            # Update parsed result with accumulated patterns
            parsed['accumulated_patterns'] = session_patterns
            parsed['total_pattern_count'] = len(session_patterns)
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Error parsing and accumulating patterns: {e}")
            return {
                'risk_score': 0.0,
                'risk_level': 'MINIMAL',
                'scam_type': 'unknown',
                'confidence': 0.0,
                'detected_patterns': [],
                'error': str(e)
            }
    
    async def _run_policy_guidance_step(self, websocket: WebSocket, client_id: str, session_id: str, scam_analysis: Dict):
        """Run policy guidance step (preserved from original)"""
        try:
            await self.send_message(websocket, client_id, {
                'type': 'adk_policy_analysis_started',
                'data': {
                    'session_id': session_id,
                    'agent': 'ADK Policy Guidance Agent',
                    'scam_type': scam_analysis.get('scam_type'),
                    'risk_score': scam_analysis.get('risk_score'),
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Update agent state
            self.agent_states[session_id]['policy_analysis_active'] = True
            self.agent_states[session_id]['current_agent'] = 'ADK Policy Guidance Agent'
            
            # Prepare comprehensive context for policy agent
            policy_context = {
                'session_id': session_id,
                'customer_profile': self.active_sessions[session_id].get('customer_profile', {}),
                'scam_analysis': scam_analysis,
                'accumulated_patterns': self.accumulated_patterns[session_id]
            }
            
            policy_result = await self.policy_guidance_agent.run(
                json.dumps(scam_analysis, indent=2), 
                policy_context
            )
            
            # Parse policy result
            parsed_policy = self.policy_guidance_agent.parse_result(policy_result)
            
            # Store policy guidance in session
            self.active_sessions[session_id]['policy_guidance'] = parsed_policy
            
            # Update agent state
            self.agent_states[session_id]['policy_analysis_active'] = False
            self.agent_states[session_id]['agents_completed'].append('policy_guidance')
            self.processing_steps[session_id].append('✅ ADK Policy guidance ready')
            
            await self.send_message(websocket, client_id, {
                'type': 'policy_guidance_ready',
                'data': {
                    'session_id': session_id,
                    'policy_guidance': parsed_policy,
                    'adk_powered': True,
                    'timestamp': get_current_timestamp()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error in policy guidance step: {e}")
    
    async def _run_orchestrator_step(self, websocket: WebSocket, client_id: str, session_id: str, scam_analysis: Dict):
        """Run orchestrator step (preserved from original)"""
        try:
            await self.send_message(websocket, client_id, {
                'type': 'adk_orchestrator_started',
                'data': {
                    'session_id': session_id,
                    'agent': 'ADK Orchestrator Agent',
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Update agent state
            self.agent_states[session_id]['orchestrator_active'] = True
            self.agent_states[session_id]['current_agent'] = 'ADK Orchestrator Agent'
            
            # Prepare comprehensive context for orchestrator
            orchestrator_context = {
                'session_id': session_id,
                'customer_profile': self.active_sessions[session_id].get('customer_profile', {}),
                'policy_guidance': self.active_sessions[session_id].get('policy_guidance', {}),
                'financial_context': {
                    'amount': 'Unknown',  # Could be extracted from speech
                    'destination': 'Unknown'
                },
                'call_duration': (datetime.now() - self.active_sessions[session_id]['start_time']).total_seconds()
            }
            
            # Prepare consolidated analysis for orchestrator
            consolidated_analysis = {
                'scam_analysis': scam_analysis,
                'policy_guidance': self.active_sessions[session_id].get('policy_guidance', {}),
                'accumulated_patterns': self.accumulated_patterns[session_id],
                'session_context': orchestrator_context
            }
            
            orchestrator_result = await self.orchestrator_agent.run(
                json.dumps(consolidated_analysis, indent=2),
                orchestrator_context
            )
            
            # Parse orchestrator result
            parsed_decision = self.orchestrator_agent.parse_result(orchestrator_result)
            
            # Store decision in session
            self.active_sessions[session_id]['orchestrator_decision'] = parsed_decision
            
            # Update agent state
            self.agent_states[session_id]['orchestrator_active'] = False
            self.agent_states[session_id]['agents_completed'].append('orchestrator')
            self.processing_steps[session_id].append(f"✅ ADK Decision: {parsed_decision.get('decision', 'Complete')}")
            
            await self.send_message(websocket, client_id, {
                'type': 'adk_orchestrator_decision',
                'data': {
                    'session_id': session_id,
                    'decision': parsed_decision,
                    'adk_powered': True,
                    'timestamp': get_current_timestamp()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error in orchestrator step: {e}")
    
    async def handle_processing_complete_comprehensive(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str
    ) -> None:
        """Handle processing completion with comprehensive functionality (preserved from original)"""
        
        try:
            # Get final analysis (preserved logic)
            analysis_history = self.session_analysis_history.get(session_id, [])
            if not analysis_history:
                logger.warning(f"No analysis history for session {session_id}")
                return
            
            final_analysis = analysis_history[-1]
            final_risk_score = final_analysis.get('risk_score', 0)
            
            logger.info(f"🏁 Complete processing finished for {session_id}, final risk: {final_risk_score}%")
            
            # UPDATED: Auto-create incident using ADK Case Management Agent for medium+ risk
            if final_risk_score >= 50:
                await self.auto_create_incident_with_adk_case_management(websocket, client_id, session_id, final_analysis)
            
        except Exception as e:
            logger.error(f"❌ Error handling comprehensive processing completion: {e}")
    
    # UPDATED: New method to use ADK Case Management Agent
    async def auto_create_incident_with_adk_case_management(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        final_analysis: Dict
    ) -> None:
        """UPDATED: Auto-create incident using ADK Case Management Agent"""
        
        try:
            logger.info(f"📋 Auto-creating incident using ADK Case Management Agent for session {session_id}")
            
            # Send notification
            await self.send_message(websocket, client_id, {
                'type': 'auto_incident_creation_started',
                'data': {
                    'session_id': session_id,
                    'risk_score': final_analysis.get('risk_score', 0),
                    'reason': 'Medium/High risk detected - ADK Case Management auto-creating incident',
                    'adk_powered': True,
                    'case_management_agent': True,  # UPDATED
                    'timestamp': get_current_timestamp()
                }
            })
            
            # UPDATED: Set agent state for case management
            self.agent_states[session_id]['case_management_active'] = True
            self.agent_states[session_id]['current_agent'] = 'ADK Case Management Agent'
            
            await self.send_message(websocket, client_id, {
                'type': 'adk_case_management_started',
                'data': {
                    'session_id': session_id,
                    'agent': 'ADK Case Management Agent',
                    'timestamp': get_current_timestamp()
                }
            })
            
            # UPDATED: Prepare comprehensive incident data for ADK Case Management Agent
            incident_context = {
                'session_id': session_id,
                'customer_profile': self.active_sessions[session_id].get('customer_profile', {}),
                'customer_speech': [segment['text'] for segment in self.customer_speech_buffer.get(session_id, [])],
                'final_analysis': final_analysis,
                'policy_guidance': self.active_sessions[session_id].get('policy_guidance', {}),
                'orchestrator_decision': self.active_sessions[session_id].get('orchestrator_decision', {}),
                'accumulated_patterns': self.accumulated_patterns.get(session_id, {}),
                'processing_timeline': self.processing_steps.get(session_id, [])
            }
            
            # UPDATED: Prepare incident data for ADK Case Management Agent
            incident_data = {
                'customer_name': incident_context['customer_profile'].get('name', 'Unknown'),
                'customer_account': incident_context['customer_profile'].get('account', 'Unknown'),
                'customer_phone': incident_context['customer_profile'].get('phone', 'Unknown'),
                'risk_score': final_analysis.get('risk_score', 0),
                'scam_type': final_analysis.get('scam_type', 'unknown'),
                'session_id': session_id,
                'confidence_score': final_analysis.get('confidence', 0.0),
                'transcript_segments': [
                    {
                        'text': segment['text'],
                        'timestamp': segment['timestamp'],
                        'confidence': segment['confidence']
                    } for segment in self.customer_speech_buffer.get(session_id, [])
                ],
                'detected_patterns': self.accumulated_patterns.get(session_id, {}),
                'recommended_actions': incident_context['policy_guidance'].get('recommended_actions', []),
                'auto_created': True,
                'creation_method': 'adk_case_management_agent',
                'orchestrator_decision': incident_context['orchestrator_decision'].get('decision', 'REVIEW_REQUIRED'),
                'processing_timeline': self.processing_steps.get(session_id, [])
            }
            
            # UPDATED: Use ADK Case Management Agent to create incident
            case_result = await self.case_management_agent.create_fraud_incident(
                incident_data=incident_data,
                context=incident_context
            )
            
            # Update agent state
            self.agent_states[session_id]['case_management_active'] = False
            self.agent_states[session_id]['agents_completed'].append('case_management')
            
            if case_result.get('success'):
                # Store incident info in session
                self.active_sessions[session_id]['auto_incident'] = {
                    'incident_number': case_result.get('incident_number'),
                    'incident_sys_id': case_result.get('incident_sys_id'),
                    'incident_url': case_result.get('incident_url'),
                    'priority': case_result.get('priority'),
                    'urgency': case_result.get('urgency'),
                    'category': case_result.get('category'),
                    'assignment_group': case_result.get('assignment_group'),
                    'created_at': get_current_timestamp(),
                    'creation_method': 'adk_case_management_agent'
                }
                
                self.processing_steps[session_id].append(f"📋 ADK Case Management: {case_result.get('incident_number')}")
                
                await self.send_message(websocket, client_id, {
                    'type': 'auto_incident_created',
                    'data': {
                        'session_id': session_id,
                        'incident_number': case_result.get('incident_number'),
                        'incident_sys_id': case_result.get('incident_sys_id'),
                        'incident_url': case_result.get('incident_url'),
                        'priority': case_result.get('priority'),
                        'urgency': case_result.get('urgency'),
                        'category': case_result.get('category'),
                        'assignment_group': case_result.get('assignment_group'),
                        'risk_score': final_analysis.get('risk_score', 0),
                        'creation_method': 'adk_case_management_agent',
                        'adk_powered': True,
                        'structured_by_adk': True,
                        'timestamp': get_current_timestamp()
                    }
                })
                
                logger.info(f"✅ ADK Case Management created incident {case_result.get('incident_number')} for session {session_id}")
            else:
                error_msg = case_result.get('error', 'Unknown error in case creation')
                logger.error(f"❌ ADK Case Management failed to create incident: {error_msg}")
                
                await self.send_message(websocket, client_id, {
                    'type': 'auto_incident_creation_failed',
                    'data': {
                        'session_id': session_id,
                        'error': error_msg,
                        'creation_method': 'adk_case_management_agent',
                        'timestamp': get_current_timestamp()
                    }
                })
            
        except Exception as e:
            logger.error(f"❌ Error in ADK Case Management auto-incident creation: {e}")
            await self.send_message(websocket, client_id, {
                'type': 'auto_incident_creation_failed',
                'data': {
                    'session_id': session_id,
                    'error': str(e),
                    'creation_method': 'adk_case_management_agent',
                    'timestamp': get_current_timestamp()
                }
            })
    
    # ===== ADDITIONAL REQUEST HANDLERS (UPDATED) =====
    
    async def handle_session_analysis_request(self, websocket: WebSocket, client_id: str, data: Dict):
        """Handle request for session analysis data (preserved from original)"""
        session_id = data.get('session_id')
        if not session_id or session_id not in self.active_sessions:
            await self.send_error(websocket, client_id, "Invalid session ID")
            return
        
        try:
            analysis_data = {
                'session_id': session_id,
                'analysis_history': self.session_analysis_history.get(session_id, []),
                'accumulated_patterns': self.accumulated_patterns.get(session_id, {}),
                'customer_speech_buffer': self.customer_speech_buffer.get(session_id, []),
                'processing_steps': self.processing_steps.get(session_id, []),
                'agent_states': self.agent_states.get(session_id, {}),
                'session_info': {
                    'start_time': self.active_sessions[session_id]['start_time'].isoformat(),
                    'filename': self.active_sessions[session_id]['filename'],
                    'status': self.active_sessions[session_id]['status'],
                    'customer_profile': self.active_sessions[session_id].get('customer_profile', {})
                },
                'auto_incident': self.active_sessions[session_id].get('auto_incident'),  # UPDATED: Include incident info
                'case_management_enabled': True  # UPDATED
            }
            
            await self.send_message(websocket, client_id, {
                'type': 'session_analysis_response',
                'data': analysis_data
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling session analysis request: {e}")
            await self.send_error(websocket, client_id, f"Failed to get session analysis: {str(e)}")
    
    async def handle_manual_case_creation(self, websocket: WebSocket, client_id: str, data: Dict):
        """UPDATED: Handle manual case creation request using ADK Case Management Agent"""
        session_id = data.get('session_id')
        if not session_id or session_id not in self.active_sessions:
            await self.send_error(websocket, client_id, "Invalid session ID")
            return
        
        try:
            # Get latest analysis data
            latest_analysis = {}
            if self.session_analysis_history.get(session_id):
                latest_analysis = self.session_analysis_history[session_id][-1]
            
            # UPDATED: Prepare manual case data for ADK Case Management Agent
            case_data = {
                'customer_name': self.active_sessions[session_id].get('customer_profile', {}).get('name', 'Unknown'),
                'customer_account': self.active_sessions[session_id].get('customer_profile', {}).get('account', 'Unknown'),
                'customer_phone': self.active_sessions[session_id].get('customer_profile', {}).get('phone', 'Unknown'),
                'risk_score': latest_analysis.get('risk_score', 0),
                'scam_type': latest_analysis.get('scam_type', 'unknown'),
                'session_id': session_id,
                'confidence_score': latest_analysis.get('confidence', 0.0),
                'transcript_segments': [
                    {
                        'text': segment['text'],
                        'timestamp': segment['timestamp'],
                        'confidence': segment['confidence']
                    } for segment in self.customer_speech_buffer.get(session_id, [])
                ],
                'detected_patterns': self.accumulated_patterns.get(session_id, {}),
                'recommended_actions': self.active_sessions[session_id].get('policy_guidance', {}).get('recommended_actions', []),
                'auto_created': False,
                'creation_method': 'manual_via_adk_case_management',
                'orchestrator_decision': self.active_sessions[session_id].get('orchestrator_decision', {}),
                'manual_override': data.get('override_reason', 'Manual case creation requested'),
                'created_by': data.get('agent_id', 'unknown')
            }
            
            # Prepare context for ADK Case Management Agent
            case_context = {
                'session_id': session_id,
                'creation_type': 'manual',
                'customer_profile': self.active_sessions[session_id].get('customer_profile', {}),
                'analysis_data': latest_analysis,
                'accumulated_patterns': self.accumulated_patterns.get(session_id, {}),
                'agent_decision': self.active_sessions[session_id].get('orchestrator_decision', {}),
                'customer_speech': [segment['text'] for segment in self.customer_speech_buffer.get(session_id, [])],
                'processing_timeline': self.processing_steps.get(session_id, [])
            }
            
            # UPDATED: Use ADK Case Management Agent for manual case creation
            case_result = await self.case_management_agent.create_fraud_incident(
                incident_data=case_data,
                context=case_context
            )
            
            if case_result.get('success'):
                await self.send_message(websocket, client_id, {
                    'type': 'manual_case_created',
                    'data': {
                        'session_id': session_id,
                        'case_number': case_result.get('incident_number'),
                        'case_sys_id': case_result.get('incident_sys_id'),
                        'case_url': case_result.get('incident_url'),
                        'priority': case_result.get('priority'),
                        'urgency': case_result.get('urgency'),
                        'category': case_result.get('category'),
                        'assignment_group': case_result.get('assignment_group'),
                        'creation_type': 'manual',
                        'creation_method': 'adk_case_management_agent',
                        'adk_powered': True,
                        'timestamp': get_current_timestamp()
                    }
                })
            else:
                await self.send_error(websocket, client_id, f"Failed to create manual case: {case_result.get('error')}")
            
        except Exception as e:
            logger.error(f"❌ Error handling manual case creation: {e}")
            await self.send_error(websocket, client_id, f"Failed to create manual case: {str(e)}")
    
    # ===== HELPER FUNCTIONS (preserved from original) =====
    
    def _extract_scam_context_from_filename(self, filename: str) -> str:
        """Extract scam type context from filename (preserved from original)"""
        filename_lower = filename.lower()
        if 'romance' in filename_lower:
            return 'romance_scam'
        elif 'investment' in filename_lower:
            return 'investment_scam'
        elif 'impersonation' in filename_lower:
            return 'impersonation_scam'
        else:
            return 'unknown'
    
    def _get_customer_profile(self, filename: str) -> Dict:
        """Get customer profile for filename (preserved from original)"""
        for key, profile in self.customer_profiles.items():
            if key in filename:
                return profile
        
        return {
            'name': 'Unknown Customer',
            'account': '****0000',
            'phone': 'Unknown',
            'demographics': {'age': 0, 'location': 'Unknown', 'relationship': 'Unknown'}
        }
    
    # ===== SESSION MANAGEMENT (preserved from original) =====
    
    async def handle_stop_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Handle request to stop processing (complete functionality preserved)"""
        session_id = data.get('session_id')
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            logger.info(f"🛑 STOP request received for session {session_id}")
            
            # Mark session as stopped
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["processing_stopped"] = True
                self.active_sessions[session_id]["status"] = "stopped"
            
            # Stop audio processing
            stop_result = await audio_processor_agent.stop_streaming(session_id)
            
            # Send confirmation
            await self.send_message(websocket, client_id, {
                'type': 'processing_stopped',
                'data': {
                    'session_id': session_id,
                    'status': 'stopped',
                    'message': 'Complete processing stopped',
                    'final_analysis_count': len(self.session_analysis_history.get(session_id, [])),
                    'final_pattern_count': len(self.accumulated_patterns.get(session_id, {})),
                    'processing_steps': len(self.processing_steps.get(session_id, [])),
                    'case_management_enabled': True,  # UPDATED
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Cleanup (but preserve data for potential retrieval)
            logger.info(f"✅ STOP completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error stopping processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to stop: {str(e)}")
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session data (preserved from original)"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.customer_speech_buffer:
                del self.customer_speech_buffer[session_id]
            if session_id in self.session_analysis_history:
                del self.session_analysis_history[session_id]
            if session_id in self.accumulated_patterns:
                del self.accumulated_patterns[session_id]
            if session_id in self.processing_steps:
                del self.processing_steps[session_id]
            if session_id in self.agent_states:
                del self.agent_states[session_id]
        except Exception as e:
            logger.error(f"❌ Error cleaning up session {session_id}: {e}")
    
    def cleanup_client_sessions(self, client_id: str) -> None:
        """Clean up all sessions for a disconnected client (preserved from original)"""
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
    
    async def handle_status_request(self, websocket: WebSocket, client_id: str):
        """UPDATED: Handle status request (complete functionality preserved with case management info)"""
        try:
            # Calculate comprehensive statistics
            total_sessions = len(self.active_sessions)
            total_analyses = sum(len(history) for history in self.session_analysis_history.values())
            total_patterns = sum(len(patterns) for patterns in self.accumulated_patterns.values())
            
            # Agent status summary
            agent_summary = {}
            for session_id, states in self.agent_states.items():
                for agent in ['scam_detection', 'policy_guidance', 'summarization', 'orchestrator', 'case_management']:  # UPDATED: Added case_management
                    if agent not in agent_summary:
                        agent_summary[agent] = 0
                    if agent in states.get('agents_completed', []):
                        agent_summary[agent] += 1
            
            # Count auto-created incidents
            auto_incidents_count = len([
                s for s in self.active_sessions.values() 
                if s.get('auto_incident') is not None
            ])
            
            status = {
                'handler_type': 'CompleteFraudDetectionHandlerWithADKCaseManagement',  # UPDATED
                'active_sessions': total_sessions,
                'adk_agents_available': self.adk_agents_available,
                'case_management_agent_available': True,  # UPDATED
                'total_analyses_performed': total_analyses,
                'total_patterns_detected': total_patterns,
                'agent_completion_summary': agent_summary,
                'customer_profiles_loaded': len(self.customer_profiles),
                'sessions_with_incidents': auto_incidents_count,  # UPDATED
                'auto_incidents_created': auto_incidents_count,  # UPDATED
                'system_status': 'operational',
                'features': [
                    'Complete session tracking',
                    'Pattern accumulation across segments',
                    'ADK agent pipeline',
                    'ADK Case Management Agent integration',  # UPDATED
                    'Auto-incident creation via ADK',  # UPDATED
                    'Manual case creation via ADK',  # UPDATED
                    'ServiceNow integration',  # UPDATED
                    'Customer profile management',
                    'Processing step tracking'
                ],
                'case_management_features': [  # UPDATED: New section
                    'ADK-structured incident creation',
                    'Priority and urgency mapping',
                    'Automatic assignment group routing',
                    'ServiceNow API integration',
                    'Evidence preservation',
                    'Regulatory compliance notes'
                ],
                'timestamp': get_current_timestamp()
            }
            
            await self.send_message(websocket, client_id, {
                'type': 'status_response',
                'data': status
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling status request: {e}")
            await self.send_error(websocket, client_id, f"Status request failed: {str(e)}")
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions (preserved from original)"""
        return len(self.active_sessions)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """UPDATED: Get comprehensive system statistics (with case management info)"""
        return {
            'active_sessions': len(self.active_sessions),
            'total_customer_speech_segments': sum(len(buffer) for buffer in self.customer_speech_buffer.values()),
            'total_analysis_history': sum(len(history) for history in self.session_analysis_history.values()),
            'total_accumulated_patterns': sum(len(patterns) for patterns in self.accumulated_patterns.values()),
            'total_processing_steps': sum(len(steps) for steps in self.processing_steps.values()),
            'sessions_with_auto_incidents': len([
                s for s in self.active_sessions.values() 
                if s.get('auto_incident') is not None
            ]),
            'agent_states_tracking': len(self.agent_states),
            'customer_profiles_available': len(self.customer_profiles),
            'adk_agents_available': self.adk_agents_available,
            'case_management_agent_available': True,  # UPDATED
            'case_management_features': {  # UPDATED: New section
                'adk_structured_incidents': True,
                'servicenow_integration': True,
                'auto_priority_mapping': True,
                'evidence_preservation': True,
                'regulatory_compliance': True
            }
        }
    
    # ===== WEBSOCKET COMMUNICATION (preserved from original) =====
    
    async def send_message(self, websocket: WebSocket, client_id: str, message: Dict) -> bool:
        """Send message to client (preserved from original)"""
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
        """Send error message to client (preserved from original)"""
        await self.send_message(websocket, client_id, {
            'type': 'error',
            'data': create_error_response(error_message, "COMPLETE_PROCESSING_ERROR")
        })

# Create the handler class for export
FraudDetectionWebSocketHandler = FraudDetectionWebSocketHandler
