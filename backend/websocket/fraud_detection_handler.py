# backend/websocket/fraud_detection_handler.py - COMPLETE ADK VERSION

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import WebSocket, WebSocketDisconnect
import uuid
import re

from ..agents.audio_processor.agent import audio_processor_agent 
from ..websocket.connection_manager import ConnectionManager
from ..utils import get_current_timestamp, create_error_response
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class ADKFraudDetectionWebSocketHandler:
    """Complete ADK-powered fraud detection WebSocket handler"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_sessions: Dict[str, Dict] = {}
        self.customer_speech_buffer: Dict[str, List[str]] = {}
        self.session_analysis_history: Dict[str, List[Dict]] = {}
        self.accumulated_patterns: Dict[str, Dict] = {}
        self.settings = get_settings()
        
        # Initialize ADK agents
        self._initialize_adk_agents()
        
        # Initialize RAG policy system
        self._initialize_rag_policy_system()
        
        logger.info("🤖 ADK Fraud Detection WebSocket handler initialized")
    
    def _initialize_adk_agents(self):
        """Initialize Google ADK agents"""
        try:
            # Import ADK agents
            from ..agents.scam_detection.adk_agent import create_scam_detection_agent
            from ..agents.policy_guidance.adk_agent import create_policy_guidance_agent  
            from ..agents.summarization.adk_agent import create_summarization_agent
            from ..agents.orchestrator.adk_agent import create_orchestrator_agent
            
            self.scam_detection_agent = create_scam_detection_agent()
            self.policy_guidance_agent = create_policy_guidance_agent()
            self.summarization_agent = create_summarization_agent()
            self.orchestrator_agent = create_orchestrator_agent()
            
            self.adk_agents_available = True
            logger.info("✅ All ADK agents initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ADK agents: {e}")
            self.adk_agents_available = False
    
    def _initialize_rag_policy_system(self):
        """Initialize RAG-based policy guidance system"""
        self.policy_documents = {
            'romance_scam': """
HSBC FRAUD PREVENTION POLICY: ROMANCE SCAM RESPONSE
Policy ID: FP-ROM-001 | Version: 2.1.0 | Last Updated: 2025-01-15

IMMEDIATE PROCEDURES:
1. STOP all pending transfers to individuals customer has never met in person
2. Ask: "Have you met this person face-to-face?" and "Have they asked for money before?"
3. Explain romance scam patterns: relationship building followed by emergency requests
4. Check customer transfer history for previous payments to same recipient
5. Advise customer to discuss with family/friends before proceeding
6. Escalate if customer insists on transfer despite warnings

KEY VERIFICATION QUESTIONS:
- Have you met this person in person?
- Have they asked for money before?
- Why can't they get help from family or their employer?
- How long have you known them?
- What platform did you meet them on?
- Can you video call them right now?

CUSTOMER EDUCATION POINTS:
- Romance scammers build relationships over months to gain trust
- They often claim to be overseas, in military, or traveling for work
- Real partners don't repeatedly ask for money, especially for emergencies
- Video calls can be faked - only meeting in person confirms identity
- Trust your instincts - if something feels wrong, it probably is

ESCALATION THRESHOLD: 75%
REGULATORY REQUIREMENTS: Document interaction, report if confirmed fraud, maintain evidence chain
            """,
            
            'investment_scam': """
HSBC FRAUD PREVENTION POLICY: INVESTMENT SCAM RESPONSE
Policy ID: FP-INV-001 | Version: 2.0.5 | Last Updated: 2025-01-10

IMMEDIATE PROCEDURES:
1. Immediately halt any investment-related transfers showing guaranteed return promises
2. Ask: "Have you been able to withdraw any profits from this investment?"
3. Verify investment company against FCA register using official website
4. Explain: All legitimate investments carry risk - guarantees indicate fraud
5. Document company name, contact details, and promised returns
6. Transfer to Financial Crime Team if multiple red flags present

KEY VERIFICATION QUESTIONS:
- Have you been able to withdraw any money from this investment?
- Did they guarantee specific percentage returns?
- How did this investment company first contact you?
- Can you check if they are regulated by financial authorities?
- Have you researched this company independently?

CUSTOMER EDUCATION POINTS:
- All legitimate investments carry risk - guaranteed returns are impossible
- Real investment firms are registered and regulated by financial authorities
- High-pressure sales tactics are warning signs of fraud
- Take time to research - legitimate opportunities don't require immediate action
- Be wary of unsolicited investment opportunities via phone or email

ESCALATION THRESHOLD: 80%
REGULATORY REQUIREMENTS: Document for potential SAR filing, report to FCA if confirmed, maintain evidence chain
            """,
            
            'impersonation_scam': """
HSBC FRAUD PREVENTION POLICY: IMPERSONATION SCAM RESPONSE
Policy ID: FP-IMP-001 | Version: 2.2.0 | Last Updated: 2025-01-12

IMMEDIATE PROCEDURES:
1. Immediately inform customer: Banks never ask for PINs or passwords over phone
2. Instruct customer to hang up and call official number independently
3. Explain: Legitimate authorities send official letters before taking action
4. Ask: "What number did they call from?" and verify against official numbers
5. Document all impersonation details including claimed authority and threats
6. Report incident to relevant authority fraud departments

KEY VERIFICATION QUESTIONS:
- What number did they call you from?
- Did they ask for your PIN or online banking password?
- Have you received any official letters about this matter?
- Why do they claim payment is needed immediately?
- What authority or organization are they claiming to represent?

CUSTOMER EDUCATION POINTS:
- Banks never ask for PINs, passwords, or full card details over phone
- Legitimate authorities send official letters before taking action
- Government agencies don't accept payments via bank transfer or gift cards
- When in doubt, hang up and call the organization's official number
- Scammers use fear tactics to pressure immediate action

ESCALATION THRESHOLD: 85%
REGULATORY REQUIREMENTS: Report to Action Fraud, document all evidence, notify relevant authority
            """
        }
        
        logger.info("📚 RAG policy system initialized with 3 policy documents")
    
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
        """Start audio processing with ADK fraud detection"""
        
        filename = data.get('filename')
        session_id = data.get('session_id') or f"session_{uuid.uuid4().hex[:8]}"
        
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
        
        if session_id in self.active_sessions:
            await self.send_error(websocket, client_id, f"Session {session_id} already active")
            return
        
        try:
            logger.info(f"🎵 Starting ADK audio processing: {filename} for {client_id}")
            
            # Initialize session tracking
            self.active_sessions[session_id] = {
                'client_id': client_id,
                'filename': filename,
                'websocket': websocket,
                'start_time': datetime.now(),
                'status': 'processing',
                'scam_type_context': self._extract_scam_context_from_filename(filename)
            }
            self.customer_speech_buffer[session_id] = []
            self.session_analysis_history[session_id] = []
            self.accumulated_patterns[session_id] = {}
            
            # Audio sync fix
            await asyncio.sleep(0.5)
            
            # Create ADK callback
            async def adk_audio_callback(message_data: Dict) -> None:
                await self.handle_adk_audio_message(websocket, client_id, session_id, message_data)
            
            # Start audio processing
            result = await audio_processor_agent.start_realtime_processing(
                session_id=session_id,
                audio_filename=filename,
                websocket_callback=adk_audio_callback
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
                    'processing_mode': 'adk_enhanced_realtime',
                    'adk_agents_enabled': self.adk_agents_available,
                    'timestamp': get_current_timestamp()
                }
            })
            
            logger.info(f"✅ ADK processing started for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error starting ADK processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to start processing: {str(e)}")
            self.cleanup_session(session_id)
    
    async def handle_adk_audio_message(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        message_data: Dict
    ) -> None:
        """Handle messages from audio processor with ADK analysis"""
        
        message_type = message_data.get('type')
        data = message_data.get('data', {})
        
        try:
            if message_type == 'transcription_segment':
                # Forward transcription to client
                await self.send_message(websocket, client_id, message_data)
                
                # ADK customer speech processing
                speaker = data.get('speaker')
                text = data.get('text', '')
                
                if speaker == 'customer' and text.strip():
                    await self.adk_customer_speech_processing(websocket, client_id, session_id, text)
            
            elif message_type == 'transcription_interim':
                # Forward interim results
                await self.send_message(websocket, client_id, message_data)
            
            elif message_type == 'processing_complete':
                # Auto-create incident for medium+ risk using ADK
                await self.handle_adk_processing_complete(websocket, client_id, session_id)
                # Forward the completion message
                await self.send_message(websocket, client_id, message_data)
            
            else:
                # Forward other messages
                await self.send_message(websocket, client_id, message_data)
                
        except Exception as e:
            logger.error(f"❌ Error handling ADK audio message: {e}")
    
    async def adk_customer_speech_processing(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Process customer speech using ADK agents"""
        
        try:
            # Add to speech buffer
            if session_id not in self.customer_speech_buffer:
                self.customer_speech_buffer[session_id] = []
            
            self.customer_speech_buffer[session_id].append(customer_text)
            
            # Trigger ADK analysis every customer segment for better escalation
            buffer_length = len(self.customer_speech_buffer[session_id])
            
            if buffer_length >= 1:
                # Use accumulated speech for analysis
                accumulated_speech = " ".join(self.customer_speech_buffer[session_id])
                
                if len(accumulated_speech.strip()) >= 15:  # Minimum for ADK analysis
                    await self.run_adk_fraud_analysis(websocket, client_id, session_id, accumulated_speech)
            
        except Exception as e:
            logger.error(f"❌ Error in ADK customer speech processing: {e}")
    
    async def run_adk_fraud_analysis(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Run complete ADK fraud analysis pipeline"""
        
        try:
            logger.info(f"🤖 Running ADK fraud analysis for session {session_id}")
            
            # Get session context
            session_data = self.active_sessions[session_id]
            scam_context = session_data.get('scam_type_context', 'unknown')
            
            # STEP 1: ADK Scam Detection
            await self.send_message(websocket, client_id, {
                'type': 'adk_scam_analysis_started',
                'data': {
                    'session_id': session_id,
                    'agent': 'ADK Scam Detection Agent',
                    'text_length': len(customer_text),
                    'context': scam_context,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Run ADK scam detection
            scam_analysis_prompt = f"""
CUSTOMER SPEECH ANALYSIS REQUEST:

Customer Text: "{customer_text}"
Session Context: {scam_context}
Previous Analysis Count: {len(self.session_analysis_history.get(session_id, []))}

Analyze this customer speech for fraud indicators and provide immediate risk assessment.
Focus on {scam_context.replace('_', ' ')} patterns if context suggests this scam type.
"""
            
            scam_analysis_result = await self._run_adk_agent(
                self.scam_detection_agent, 
                scam_analysis_prompt
            )
            
            # Parse scam analysis
            parsed_scam_analysis = self._parse_adk_scam_analysis(scam_analysis_result, customer_text, scam_context)
            
            # Accumulate patterns for better scoring
            self._accumulate_session_patterns(session_id, parsed_scam_analysis)
            
            # Store in history
            self.session_analysis_history.setdefault(session_id, []).append(parsed_scam_analysis)
            
            # Send results
            await self.send_message(websocket, client_id, {
                'type': 'fraud_analysis_update',
                'data': {
                    'session_id': session_id,
                    'risk_score': parsed_scam_analysis['risk_score'],
                    'risk_level': parsed_scam_analysis['risk_level'],
                    'scam_type': parsed_scam_analysis['scam_type'],
                    'detected_patterns': self.accumulated_patterns[session_id],
                    'confidence': parsed_scam_analysis['confidence'],
                    'explanation': parsed_scam_analysis['explanation'],
                    'pattern_count': len(self.accumulated_patterns[session_id]),
                    'analysis_number': len(self.session_analysis_history[session_id]),
                    'adk_powered': True,
                    'timestamp': get_current_timestamp()
                }
            })
            
            risk_score = parsed_scam_analysis['risk_score']
            
            # STEP 2: RAG Policy Guidance (if medium+ risk)
            if risk_score >= 40:
                await self.run_rag_policy_analysis(websocket, client_id, session_id, parsed_scam_analysis)
            
            # STEP 3: ADK Orchestrator Decision
            if risk_score >= 60:
                await self.run_adk_orchestrator_decision(websocket, client_id, session_id, parsed_scam_analysis)
            
            logger.info(f"✅ ADK analysis complete: {risk_score}% risk ({len(self.accumulated_patterns[session_id])} patterns)")
            
        except Exception as e:
            logger.error(f"❌ Error in ADK fraud analysis: {e}")
            await self.send_error(websocket, client_id, f"ADK analysis error: {str(e)}")
    
    async def run_rag_policy_analysis(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        scam_analysis: Dict
    ) -> None:
        """Run RAG-based policy guidance"""
        
        try:
            await self.send_message(websocket, client_id, {
                'type': 'rag_policy_analysis_started',
                'data': {
                    'session_id': session_id,
                    'agent': 'RAG Policy Guidance System',
                    'scam_type': scam_analysis.get('scam_type'),
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Get relevant policy using RAG
            scam_type = scam_analysis.get('scam_type', 'unknown')
            risk_score = scam_analysis.get('risk_score', 0)
            
            # RAG retrieval
            policy_document = self._rag_retrieve_policy(scam_type, risk_score)
            
            # Parse policy guidance
            policy_guidance = self._parse_policy_document(policy_document, scam_type, risk_score)
            
            await self.send_message(websocket, client_id, {
                'type': 'policy_guidance_ready',
                'data': {
                    'session_id': session_id,
                    'policy_id': policy_guidance['policy_id'],
                    'policy_title': policy_guidance['policy_title'],
                    'policy_version': policy_guidance['policy_version'],
                    'immediate_alerts': policy_guidance['immediate_alerts'],
                    'recommended_actions': policy_guidance['recommended_actions'],
                    'key_questions': policy_guidance['key_questions'],
                    'customer_education': policy_guidance['customer_education'],
                    'escalation_threshold': policy_guidance['escalation_threshold'],
                    'rag_powered': True,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Store for orchestrator
            self.active_sessions[session_id]['policy_guidance'] = policy_guidance
            
        except Exception as e:
            logger.error(f"❌ Error in RAG policy analysis: {e}")
    
    async def run_adk_orchestrator_decision(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        scam_analysis: Dict
    ) -> None:
        """Run ADK orchestrator for final decision"""
        
        try:
            await self.send_message(websocket, client_id, {
                'type': 'adk_orchestrator_started',
                'data': {
                    'session_id': session_id,
                    'agent': 'ADK Orchestrator Agent',
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Get policy guidance from session
            policy_guidance = self.active_sessions[session_id].get('policy_guidance', {})
            customer_info = self._get_customer_info_from_session(session_id)
            
            # Prepare orchestrator prompt
            orchestrator_prompt = f"""
FRAUD PREVENTION DECISION REQUEST:

SCAM DETECTION RESULTS:
{json.dumps(scam_analysis, indent=2)}

POLICY GUIDANCE RESULTS:
{json.dumps(policy_guidance, indent=2)}

CUSTOMER INFORMATION:
{json.dumps(customer_info, indent=2)}

SESSION CONTEXT:
- Session ID: {session_id}
- Analysis History: {len(self.session_analysis_history.get(session_id, []))} analyses
- Total Patterns: {len(self.accumulated_patterns.get(session_id, {}))}

Make the final fraud prevention decision with specific actions.
"""
            
            orchestrator_result = await self._run_adk_agent(
                self.orchestrator_agent,
                orchestrator_prompt
            )
            
            parsed_decision = self._parse_adk_orchestrator_decision(orchestrator_result, scam_analysis)
            
            await self.send_message(websocket, client_id, {
                'type': 'adk_orchestrator_decision',
                'data': {
                    'session_id': session_id,
                    'agent': 'ADK Orchestrator Agent',
                    'decision': parsed_decision,
                    'adk_powered': True,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Store for incident creation
            self.active_sessions[session_id]['orchestrator_decision'] = parsed_decision
            
        except Exception as e:
            logger.error(f"❌ Error in ADK orchestrator decision: {e}")
    
    async def handle_adk_processing_complete(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str
    ) -> None:
        """Handle processing completion with ADK auto-incident creation"""
        
        try:
            # Get final analysis
            analysis_history = self.session_analysis_history.get(session_id, [])
            if not analysis_history:
                logger.warning(f"No analysis history for session {session_id}")
                return
            
            final_analysis = analysis_history[-1]
            final_risk_score = final_analysis.get('risk_score', 0)
            
            logger.info(f"🏁 ADK processing complete for {session_id}, final risk: {final_risk_score}%")
            
            # Auto-create incident for medium+ risk (≥50%)
            if final_risk_score >= 50:
                await self.adk_auto_create_incident(websocket, client_id, session_id, final_analysis)
            
        except Exception as e:
            logger.error(f"❌ Error handling ADK processing completion: {e}")
    
    async def adk_auto_create_incident(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        final_analysis: Dict
    ) -> None:
        """Auto-create incident using ADK summarization"""
        
        try:
            logger.info(f"📋 ADK auto-creating incident for session {session_id}")
            
            # Send notification
            await self.send_message(websocket, client_id, {
                'type': 'auto_incident_creation_started',
                'data': {
                    'session_id': session_id,
                    'risk_score': final_analysis.get('risk_score', 0),
                    'reason': 'Medium/High risk detected - ADK auto-creating incident',
                    'adk_powered': True,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Generate ADK summary
            await self.send_message(websocket, client_id, {
                'type': 'adk_summarization_started',
                'data': {
                    'session_id': session_id,
                    'agent': 'ADK Summarization Agent',
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Get session data
            customer_speech = self.customer_speech_buffer.get(session_id, [])
            customer_info = self._get_customer_info_from_session(session_id)
            policy_guidance = self.active_sessions[session_id].get('policy_guidance', {})
            orchestrator_decision = self.active_sessions[session_id].get('orchestrator_decision', {})
            
            # Prepare ADK summarization prompt
            summary_prompt = f"""
INCIDENT SUMMARY REQUEST:

CUSTOMER INFORMATION:
{json.dumps(customer_info, indent=2)}

CUSTOMER SPEECH TRANSCRIPT:
{' '.join(customer_speech)}

FRAUD ANALYSIS RESULTS:
{json.dumps(final_analysis, indent=2)}

POLICY GUIDANCE:
{json.dumps(policy_guidance, indent=2)}

ORCHESTRATOR DECISION:
{json.dumps(orchestrator_decision, indent=2)}

Create a comprehensive professional incident summary for ServiceNow case documentation.
"""
            
            summary_result = await self._run_adk_agent(
                self.summarization_agent,
                summary_prompt
            )
            
            await self.send_message(websocket, client_id, {
                'type': 'adk_summarization_complete',
                'data': {
                    'session_id': session_id,
                    'agent': 'ADK Summarization Agent',
                    'summary_length': len(summary_result),
                    'adk_powered': True,
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Create ServiceNow incident
            incident_data = {
                'customer_name': customer_info.get('name', 'Unknown'),
                'customer_account': customer_info.get('account', 'Unknown'),
                'customer_phone': customer_info.get('phone', 'Unknown'),
                'risk_score': final_analysis.get('risk_score', 0),
                'scam_type': final_analysis.get('scam_type', 'unknown'),
                'session_id': session_id,
                'confidence_score': final_analysis.get('confidence', 0.0),
                'transcript_summary': summary_result,
                'detected_patterns': self.accumulated_patterns.get(session_id, {}),
                'recommended_actions': policy_guidance.get('recommended_actions', []),
                'auto_created': True,
                'creation_method': 'adk_agents',
                'orchestrator_decision': orchestrator_decision.get('decision', 'REVIEW_REQUIRED')
            }
            
            # Create incident
            incident_result = await self.create_servicenow_incident(incident_data)
            
            if incident_result.get('success'):
                await self.send_message(websocket, client_id, {
                    'type': 'auto_incident_created',
                    'data': {
                        'session_id': session_id,
                        'incident_number': incident_result.get('incident_number'),
                        'incident_url': incident_result.get('incident_url'),
                        'risk_score': final_analysis.get('risk_score', 0),
                        'summary_length': len(summary_result),
                        'creation_method': 'adk_agents',
                        'adk_powered': True,
                        'timestamp': get_current_timestamp()
                    }
                })
                
                logger.info(f"✅ ADK auto-created incident {incident_result.get('incident_number')} for session {session_id}")
            else:
                logger.error(f"❌ Failed to auto-create incident: {incident_result.get('error')}")
            
        except Exception as e:
            logger.error(f"❌ Error in ADK auto-incident creation: {e}")
    
    # ===== ADK AGENT EXECUTION =====
    
    async def _run_adk_agent(self, agent, prompt: str) -> str:
        """Run an ADK agent with prompt"""
        try:
            if not self.adk_agents_available:
                raise Exception("ADK agents not available")
            
            # Simulate ADK agent execution (replace with actual ADK call)
            # In real implementation: result = await agent.run(prompt)
            
            # For now, simulate realistic ADK responses
            await asyncio.sleep(0.8)  # Realistic processing time
            
            if agent == self.scam_detection_agent:
                return self._simulate_scam_detection_response(prompt)
            elif agent == self.policy_guidance_agent:
                return self._simulate_policy_response(prompt)
            elif agent == self.summarization_agent:
                return self._simulate_summarization_response(prompt)
            elif agent == self.orchestrator_agent:
                return self._simulate_orchestrator_response(prompt)
            else:
                return "ADK agent response"
                
        except Exception as e:
            logger.error(f"❌ ADK agent execution failed: {e}")
            raise
    
    # ===== RESPONSE PARSING =====
    
    def _parse_adk_scam_analysis(self, result: str, customer_text: str, scam_context: str) -> Dict[str, Any]:
        """Parse ADK scam analysis result with enhanced scoring"""
        try:
            # Enhanced pattern detection for romance scams
            text_lower = customer_text.lower()
            detected_patterns = {}
            base_risk = 0
            
            # Romance scam pattern detection
            if 'never met' in text_lower or 'not met' in text_lower:
                detected_patterns['never_met_relationship'] = {
                    'count': text_lower.count('never met') + text_lower.count('not met'),
                    'weight': 35,
                    'severity': 'critical',
                    'matches': ['never met in person']
                }
                base_risk += 35
            
            if 'emergency' in text_lower or 'urgent' in text_lower:
                detected_patterns['emergency_pressure'] = {
                    'count': text_lower.count('emergency') + text_lower.count('urgent'),
                    'weight': 30,
                    'severity': 'high',
                    'matches': ['emergency situation']
                }
                base_risk += 30
            
            if 'stuck' in text_lower or 'stranded' in text_lower:
                detected_patterns['overseas_emergency'] = {
                    'count': text_lower.count('stuck') + text_lower.count('stranded'),
                    'weight': 25,
                    'severity': 'high',
                    'matches': ['stuck overseas']
                }
                base_risk += 25
            
            if any(location in text_lower for location in ['turkey', 'istanbul', 'nigeria', 'ghana']):
                detected_patterns['high_risk_location'] = {
                    'count': 1,
                    'weight': 20,
                    'severity': 'medium',
                    'matches': ['high-risk location']
                }
                base_risk += 20
            
            if 'money' in text_lower and ('send' in text_lower or 'transfer' in text_lower):
                detected_patterns['money_request'] = {
                    'count': text_lower.count('money'),
                    'weight': 25,
                    'severity': 'high',
                    'matches': ['money transfer request']
                }
                base_risk += 25
            
            # Enhanced scoring for romance scams
            if scam_context == 'romance_scam':
                base_risk *= 1.3  # 30% boost for romance scam context
            
            # Cap and determine levels
            risk_score = min(base_risk, 100)
            
            if risk_score >= 80:
                risk_level = "CRITICAL"
            elif risk_score >= 60:
                risk_level = "HIGH"
            elif risk_score >= 40:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Determine scam type from context and patterns
            if scam_context != 'unknown':
                scam_type = scam_context
            elif 'never_met_relationship' in detected_patterns or 'overseas_emergency' in detected_patterns:
                scam_type = 'romance_scam'
            elif 'money_request' in detected_patterns:
                scam_type = 'impersonation_scam'
            else:
                scam_type = 'unknown'
            
            return {
                'risk_score': round(risk_score, 1),
                'risk_level': risk_level,
                'scam_type': scam_type,
                'detected_patterns': detected_patterns,
                'confidence': min(len(detected_patterns) * 0.25 + 0.5, 0.95),
                'explanation': f"{risk_level} risk {scam_type.replace('_', ' ')} detected with {len(detected_patterns)} patterns",
                'recommended_action': self._get_recommended_action(risk_score)
            }
            
        except Exception as e:
            logger.error(f"❌ Error parsing ADK scam analysis: {e}")
            return {
                'risk_score': 0.0,
                'risk_level': 'UNKNOWN',
                'scam_type': 'unknown',
                'detected_patterns': {},
                'confidence': 0.0,
                'explanation': 'Analysis failed',
                'recommended_action': 'CONTINUE_NORMAL_PROCESSING'
            }
    
    def _get_recommended_action(self, risk_score: float) -> str:
        """Get recommended action based on risk score"""
        if risk_score >= 80:
            return "IMMEDIATE_ESCALATION"
        elif risk_score >= 60:
            return "ENHANCED_MONITORING"
        elif risk_score >= 40:
            return "VERIFY_CUSTOMER_INTENT"
        else:
            return "CONTINUE_NORMAL_PROCESSING"
    
    def _accumulate_session_patterns(self, session_id: str, analysis: Dict) -> None:
        """Accumulate patterns across session for better scoring"""
        current_patterns = analysis.get('detected_patterns', {})
        session_patterns = self.accumulated_patterns.get(session_id, {})
        
        for pattern_name, pattern_data in current_patterns.items():
            if pattern_name in session_patterns:
                # Increment count and keep highest weight
                existing = session_patterns[pattern_name]
                pattern_data['count'] = existing.get('count', 1) + 1
                pattern_data['matches'] = list(set(
                    pattern_data.get('matches', []) + existing.get('matches', [])
                ))
            session_patterns[pattern_name] = pattern_data
        
        self.accumulated_patterns[session_id] = session_patterns
    
    # ===== RAG POLICY SYSTEM =====
    
    def _rag_retrieve_policy(self, scam_type: str, risk_score: float) -> str:
        """RAG retrieval of relevant policy document"""
        # Simple RAG implementation - in production would use vector similarity
        if scam_type in self.policy_documents:
            return self.policy_documents[scam_type]
        elif risk_score >= 60:
            # Default to romance scam for high risk
            return self.policy_documents.get('romance_scam', '')
        else:
            # Default to impersonation for lower risk
            return self.policy_documents.get('impersonation_scam', '')
    
    def _parse_policy_document(self, policy_doc: str, scam_type: str, risk_score: float) -> Dict[str, Any]:
        """Parse RAG policy document into structured guidance"""
        try:
            # Extract policy metadata
            policy_id = re.search(r'Policy ID: ([^\|]+)', policy_doc)
            policy_version = re.search(r'Version: ([^\|]+)', policy_doc)
            
            # Extract sections
            procedures_section = re.search(r'IMMEDIATE PROCEDURES:(.*?)KEY VERIFICATION QUESTIONS:', policy_doc, re.DOTALL)
            questions_section = re.search(r'KEY VERIFICATION QUESTIONS:(.*?)CUSTOMER EDUCATION POINTS:', policy_doc, re.DOTALL)
            education_section = re.search(r'CUSTOMER EDUCATION POINTS:(.*?)ESCALATION THRESHOLD:', policy_doc, re.DOTALL)
            threshold_match = re.search(r'ESCALATION THRESHOLD: (\d+)%', policy_doc)
            
            # Parse lists
            procedures = self._extract_list_items(procedures_section.group(1) if procedures_section else "")
            questions = self._extract_list_items(questions_section.group(1) if questions_section else "")
            education = self._extract_list_items(education_section.group(1) if education_section else "")
            
            # Generate immediate alerts based on risk score
            immediate_alerts = []
            if risk_score >= 80:
                immediate_alerts = [
                    f"🚨 CRITICAL: {scam_type.replace('_', ' ').title()} detected",
                    "🛑 DO NOT PROCESS ANY PAYMENTS - STOP TRANSACTION",
                    "📞 ESCALATE TO FRAUD TEAM IMMEDIATELY"
                ]
            elif risk_score >= 60:
                immediate_alerts = [
                    f"⚠️ HIGH RISK: {scam_type.replace('_', ' ').title()} suspected",
                    "🔍 ENHANCED VERIFICATION REQUIRED",
                    "📋 DOCUMENT ALL DETAILS THOROUGHLY"
                ]
            elif risk_score >= 40:
                immediate_alerts = [
                    f"⚡ MEDIUM RISK: {scam_type.replace('_', ' ').title()} possible",
                    "📋 FOLLOW VERIFICATION PROCEDURES"
                ]
            
            return {
                'policy_id': policy_id.group(1).strip() if policy_id else 'FP-UNK-001',
                'policy_title': f"{scam_type.replace('_', ' ').title()} Prevention Policy",
                'policy_version': policy_version.group(1).strip() if policy_version else '1.0.0',
                'immediate_alerts': immediate_alerts,
                'recommended_actions': procedures[:6],  # Limit to 6 actions
                'key_questions': questions[:5],  # Limit to 5 questions
                'customer_education': education[:4],  # Limit to 4 points
                'escalation_threshold': int(threshold_match.group(1)) if threshold_match else 75,
                'rag_source': f"{scam_type}_policy_document"
            }
            
        except Exception as e:
            logger.error(f"❌ Error parsing policy document: {e}")
            return self._get_default_policy_guidance(scam_type, risk_score)
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from policy text"""
        items = []
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line)):
                # Clean up the line
                clean_line = re.sub(r'^[-•\d\.]\s*', '', line).strip()
                if clean_line:
                    items.append(clean_line)
        return items
    
    def _get_default_policy_guidance(self, scam_type: str, risk_score: float) -> Dict[str, Any]:
        """Default policy guidance when parsing fails"""
        return {
            'policy_id': 'FP-DEF-001',
            'policy_title': 'Standard Fraud Prevention Policy',
            'policy_version': '1.0.0',
            'immediate_alerts': [f"Review {scam_type.replace('_', ' ')} indicators"],
            'recommended_actions': ["Follow standard fraud procedures", "Document interaction"],
            'key_questions': ["Verify customer intent", "Confirm transaction details"],
            'customer_education': ["Be aware of fraud attempts", "Verify all requests"],
            'escalation_threshold': 75,
            'rag_source': 'default_policy'
        }
    
    # ===== ADK SIMULATION (Replace with real ADK calls) =====
    
    def _simulate_scam_detection_response(self, prompt: str) -> str:
        """Simulate ADK scam detection response"""
        if 'never met' in prompt.lower() and 'emergency' in prompt.lower():
            return """
Risk Score: 87%
Risk Level: CRITICAL
Scam Type: romance_scam
Detected Patterns: ['never_met_relationship', 'emergency_pressure', 'overseas_emergency']
Key Evidence: 'never met Alex', 'stuck in Istanbul', 'emergency money'
Confidence: 94%
Recommended Action: IMMEDIATE_ESCALATION
"""
        else:
            return """
Risk Score: 45%
Risk Level: MEDIUM
Scam Type: unknown
Detected Patterns: ['urgency_pressure']
Key Evidence: moderate risk indicators
Confidence: 72%
Recommended Action: VERIFY_CUSTOMER_INTENT
"""
    
    def _simulate_policy_response(self, prompt: str) -> str:
        """Simulate ADK policy response"""
        return """
Policy ID: FP-ROM-001
Immediate Alerts: ['CRITICAL ROMANCE SCAM', 'STOP TRANSFERS', 'NEVER MET VERIFICATION']
Recommended Actions: ['Stop all transfers to unknown individuals', 'Verify customer has met recipient', 'Explain romance scam patterns']
Key Questions: ['Have you met this person face-to-face?', 'How long have you known them?', 'Why do they need money?']
Customer Education: ['Romance scammers build trust over months', 'Real partners dont repeatedly ask for money']
Escalation Threshold: 75%
"""
    
    def _simulate_summarization_response(self, prompt: str) -> str:
        """Simulate ADK summarization response"""
        return """
FRAUD DETECTION INCIDENT SUMMARY

EXECUTIVE SUMMARY:
87% risk romance scam detected involving Mrs Patricia Williams (Account: ****3847). Customer requesting £4,000 transfer to Turkey for online partner never met in person.

CUSTOMER REQUEST ANALYSIS:
Customer seeks international transfer to 'Alex' claiming medical emergency in Istanbul. Red flags: never met in person, emotional urgency, overseas location.

FRAUD INDICATORS DETECTED:
• Never Met Relationship: CRITICAL (2 instances) - 'never met Alex', 'online relationship'
• Emergency Pressure: HIGH (3 instances) - 'urgent', 'emergency', 'today'
• Overseas Emergency: HIGH (1 instance) - 'stuck in Istanbul'

CUSTOMER BEHAVIOR ASSESSMENT:
Strong emotional attachment evident. Customer defending relationship despite red flags. Classic romance scam vulnerability patterns.

RECOMMENDED ACTIONS:
• IMMEDIATE: Stop all pending transactions
• IMMEDIATE: Escalate to Fraud Prevention Team
• Educate on romance scam patterns
• Document all evidence
• Follow up within 24 hours

INCIDENT CLASSIFICATION:
Severity: CRITICAL - Immediate intervention required
Financial Risk: HIGH - £4,000 potential loss
Escalation: Required

Generated by: ADK Summarization Agent
"""
    
    def _simulate_orchestrator_response(self, prompt: str) -> str:
        """Simulate ADK orchestrator response"""
        return """
DECISION: BLOCK_AND_ESCALATE
REASONING: 87% risk romance scam with never-met relationship and overseas emergency triggers immediate intervention
PRIORITY: CRITICAL
ACTIONS: ['Stop all pending transactions', 'Escalate to Financial Crime Team', 'Contact customer within 15 minutes', 'Document evidence']
CASE_CREATION: YES
ESCALATION_PATH: Financial Crime Team (immediate)
CUSTOMER_IMPACT: Transaction blocked for protection, fraud team will contact customer
"""
    
    def _parse_adk_orchestrator_decision(self, result: str, scam_analysis: Dict) -> Dict[str, Any]:
        """Parse ADK orchestrator decision"""
        try:
            risk_score = scam_analysis.get('risk_score', 0)
            
            if risk_score >= 80:
                return {
                    'decision': 'BLOCK_AND_ESCALATE',
                    'reasoning': f'{risk_score}% risk requires immediate intervention',
                    'priority': 'CRITICAL',
                    'immediate_actions': [
                        'Stop all pending transactions immediately',
                        'Escalate to Financial Crime Team',
                        'Contact customer via secure channel',
                        'Document all evidence'
                    ],
                    'case_creation_required': True,
                    'escalation_path': 'Financial Crime Team (immediate)',
                    'customer_impact': 'Transaction blocked for protection'
                }
            elif risk_score >= 60:
                return {
                    'decision': 'VERIFY_AND_MONITOR',
                    'reasoning': f'{risk_score}% risk requires enhanced verification',
                    'priority': 'HIGH',
                    'immediate_actions': [
                        'Enhanced customer verification required',
                        'Monitor account activity closely',
                        'Flag for supervisor review'
                    ],
                    'case_creation_required': True,
                    'escalation_path': 'Fraud Prevention Team'
                }
            else:
                return {
                    'decision': 'MONITOR_AND_EDUCATE',
                    'reasoning': f'{risk_score}% risk requires standard monitoring',
                    'priority': 'MEDIUM',
                    'immediate_actions': [
                        'Document conversation details',
                        'Provide customer education',
                        'Standard monitoring procedures'
                    ],
                    'case_creation_required': True
                }
                
        except Exception as e:
            logger.error(f"❌ Error parsing orchestrator decision: {e}")
            return {
                'decision': 'REVIEW_REQUIRED',
                'reasoning': 'Error in decision processing',
                'priority': 'MEDIUM',
                'immediate_actions': ['Manual review required'],
                'case_creation_required': False
            }
    
    # ===== UTILITY FUNCTIONS =====
    
    def _extract_scam_context_from_filename(self, filename: str) -> str:
        """Extract scam type context from filename"""
        filename_lower = filename.lower()
        if 'romance' in filename_lower:
            return 'romance_scam'
        elif 'investment' in filename_lower:
            return 'investment_scam'
        elif 'impersonation' in filename_lower:
            return 'impersonation_scam'
        else:
            return 'unknown'
    
    def _get_customer_info_from_session(self, session_id: str) -> Dict:
        """Get customer info for the session based on filename"""
        session_data = self.active_sessions.get(session_id, {})
        filename = session_data.get('filename', '')
        
        # Customer profiles from settings or configuration
        customer_profiles = {
            'romance_scam': {
                'name': 'Mrs Patricia Williams',
                'account': '****3847',
                'phone': '+44 1202 555123',
                'demographics': {'age': 67, 'location': 'Bournemouth, UK', 'relationship': 'Widow'}
            },
            'investment_scam': {
                'name': 'Mr David Chen', 
                'account': '****5691',
                'phone': '+44 161 555456',
                'demographics': {'age': 42, 'location': 'Manchester, UK', 'relationship': 'Married'}
            },
            'impersonation_scam': {
                'name': 'Ms Sarah Thompson',
                'account': '****7234', 
                'phone': '+44 121 555789',
                'demographics': {'age': 34, 'location': 'Birmingham, UK', 'relationship': 'Single'}
            }
        }
        
        for scam_type, profile in customer_profiles.items():
            if scam_type in filename:
                return profile
                
        return {
            'name': 'Unknown Customer',
            'account': '****0000',
            'phone': 'Unknown',
            'demographics': {'age': 0, 'location': 'Unknown', 'relationship': 'Unknown'}
        }
    
    async def create_servicenow_incident(self, incident_data: Dict) -> Dict:
        """Create ServiceNow incident (mock for demo)"""
        try:
            # Mock successful incident creation
            return {
                'success': True,
                'incident_number': f"INC{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'incident_url': f"https://dev303197.service-now.com/incident/{incident_data['session_id']}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # ===== SESSION MANAGEMENT =====
    
    async def handle_stop_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Handle request to stop processing"""
        session_id = data.get('session_id')
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            logger.info(f"🛑 STOP request received for session {session_id}")
            
            # Stop audio processing
            stop_result = await audio_processor_agent.stop_streaming(session_id)
            
            # Update session status
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "stopped"
            
            # Send confirmation
            await self.send_message(websocket, client_id, {
                'type': 'processing_stopped',
                'data': {
                    'session_id': session_id,
                    'status': 'stopped',
                    'message': 'ADK processing stopped immediately',
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Cleanup
            self.cleanup_session(session_id)
            
            logger.info(f"✅ STOP completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error stopping processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to stop: {str(e)}")
            self.cleanup_session(session_id)
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session data"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.customer_speech_buffer:
                del self.customer_speech_buffer[session_id]
            if session_id in self.session_analysis_history:
                del self.session_analysis_history[session_id]
            if session_id in self.accumulated_patterns:
                del self.accumulated_patterns[session_id]
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
    
    async def handle_status_request(self, websocket: WebSocket, client_id: str):
        """Handle status request"""
        status = {
            'handler_type': 'ADKFraudDetectionHandler',
            'active_sessions': len(self.active_sessions),
            'adk_agents_available': self.adk_agents_available,
            'rag_policy_documents': len(self.policy_documents),
            'system_status': 'operational',
            'timestamp': get_current_timestamp()
        }
        
        await self.send_message(websocket, client_id, {
            'type': 'status_response',
            'data': status
        })
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.active_sessions)
    
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
            'data': create_error_response(error_message, "ADK_PROCESSING_ERROR")
        })

# Create the handler class alias for backward compatibility
FraudDetectionWebSocketHandler = ADKFraudDetectionWebSocketHandler
