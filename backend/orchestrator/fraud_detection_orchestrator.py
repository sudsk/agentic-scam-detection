# backend/orchestrator/fraud_detection_orchestrator.py - COMPLETELY FIXED VERSION
"""
Fixed Fraud Detection Orchestrator with proper ADK integration
Based on successful orchestrator patterns from the TradeSage project
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from fastapi import WebSocket, WebSocketDisconnect
import uuid
import warnings
import os

# Suppress warnings at the top
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'

from ..agents.audio_processor.agent import audio_processor_agent 
from ..agents.case_management.adk_agent import PureADKCaseManagementAgent
from ..websocket.connection_manager import ConnectionManager
from ..utils import get_current_timestamp, create_error_response
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class WarningSuppressionContext:
    """Context manager to suppress ADK warnings during operations"""
    
    def __init__(self):
        self.original_warnings = warnings.filters[:]
        
    def __enter__(self):
        warnings.filterwarnings('ignore')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filters[:] = self.original_warnings

class FraudDetectionOrchestrator:
    """
    FIXED Fraud Detection Orchestrator with proper ADK integration
    Uses working patterns from TradeSage orchestrator
    """
    
    def __init__(self, connection_manager: Optional[ConnectionManager] = None):
        self.connection_manager = connection_manager
        
        # Session tracking and state management
        self.active_sessions: Dict[str, Dict] = {}
        self.customer_speech_buffer: Dict[str, List[str]] = {}
        self.session_analysis_history: Dict[str, List[Dict]] = {}
        self.accumulated_patterns: Dict[str, Dict] = {}
        self.processing_steps: Dict[str, List[str]] = {}
        self.agent_states: Dict[str, Dict] = {}
        
        self.settings = get_settings()
        
        # Initialize all ADK agents with proper error handling
        self._initialize_agents()
        
        # Initialize customer profiles
        self._initialize_customer_profiles()
        
        logger.info("🎭 FIXED Fraud Detection Orchestrator initialized")
    
    def _initialize_agents(self):
        """Initialize ADK agents with proper error handling and fallback"""
        try:
            with WarningSuppressionContext():
                from ..agents.scam_detection.adk_agent import create_scam_detection_agent
                from ..agents.policy_guidance.adk_agent import create_policy_guidance_agent  
                from ..agents.summarization.adk_agent import create_summarization_agent
                from ..agents.decision.adk_agent import create_decision_agent
                
                self.scam_detection_agent = create_scam_detection_agent()
                self.policy_guidance_agent = create_policy_guidance_agent()
                self.summarization_agent = create_summarization_agent()
                self.decision_agent = create_decision_agent()
                self.case_management_agent = PureADKCaseManagementAgent()
                
                self.adk_agents_available = True
                logger.info("✅ All ADK agents initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize ADK agents: {e}")
            self.adk_agents_available = False
            logger.info("🔄 Falling back to simplified pattern-based analysis")
    
    def _initialize_customer_profiles(self):
        """Initialize customer profiles for demo scenarios"""
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
    
    # ===== MAIN ORCHESTRATION METHODS =====
    
    async def orchestrate_audio_processing(
        self, 
        filename: str, 
        session_id: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Main orchestration method for audio processing pipeline"""
        session_id = session_id or f"orchestrator_session_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"🎭 FIXED Orchestrator starting audio processing pipeline: {filename}")
            
            # Initialize session
            await self._initialize_session(session_id, filename, callback)
            
            # Create audio processing callback
            audio_callback = self._create_audio_callback(session_id, callback)
            
            # Start audio processing
            result = await audio_processor_agent.start_realtime_processing(
                session_id=session_id,
                audio_filename=filename,
                websocket_callback=audio_callback
            )
            
            if 'error' in result:
                await self._cleanup_session(session_id)
                return {'success': False, 'error': result['error']}
            
            return {
                'success': True,
                'session_id': session_id,
                'orchestrator': 'FIXED_FraudDetectionOrchestrator',
                'pipeline_mode': 'complete_multi_agent',
                'agents_available': self.adk_agents_available,
                'audio_duration': result.get('audio_duration', 0),
                'processing_mode': result.get('processing_mode', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"❌ FIXED Orchestrator error in audio processing: {e}")
            await self._cleanup_session(session_id)
            return {'success': False, 'error': str(e)}
    
    async def orchestrate_text_analysis(
        self,
        customer_text: str,
        session_id: Optional[str] = None,
        context: Optional[Dict] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Orchestrate text-only fraud analysis with proper ADK integration"""
        session_id = session_id or f"text_analysis_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"🎭 FIXED Orchestrator starting text analysis pipeline: {session_id}")
            
            # Initialize lightweight session
            await self._initialize_text_session(session_id, customer_text, context, callback)
            
            # Run the agent pipeline
            result = await self._run_agent_pipeline(session_id, customer_text, callback)
            
            return {
                'success': True,
                'session_id': session_id,
                'orchestrator': 'FIXED_FraudDetectionOrchestrator',
                'analysis_result': result,
                'agents_involved': self._get_agents_involved(result.get('risk_score', 0))
            }
            
        except Exception as e:
            logger.error(f"❌ FIXED Orchestrator error in text analysis: {e}")
            return {'success': False, 'error': str(e)}
    
    # ===== SESSION MANAGEMENT =====
    
    async def _initialize_session(self, session_id: str, filename: str, callback: Optional[Callable]):
        """Initialize session with proper tracking"""
        scam_context = self._extract_scam_context_from_filename(filename)
        
        self.active_sessions[session_id] = {
            'session_id': session_id,
            'filename': filename,
            'start_time': datetime.now(),
            'status': 'processing',
            'scam_type_context': scam_context,
            'customer_profile': self._get_customer_profile(filename),
            'callback': callback,
            'processing_stopped': False,
            'audio_completed': False
        }
        
        # Initialize tracking structures
        self.customer_speech_buffer[session_id] = []
        self.session_analysis_history[session_id] = []
        self.accumulated_patterns[session_id] = {}
        self.processing_steps[session_id] = []
        self.agent_states[session_id] = {
            'scam_analysis_active': False,
            'policy_analysis_active': False,
            'decision_active': False,
            'case_management_active': False,
            'current_agent': '',
            'agents_completed': []
        }
        
        logger.info(f"🎭 FIXED Orchestrator initialized session {session_id}")
    
    async def _initialize_text_session(self, session_id: str, text: str, context: Optional[Dict], callback: Optional[Callable]):
        """Initialize lightweight session for text analysis"""
        self.active_sessions[session_id] = {
            'session_id': session_id,
            'input_text': text,
            'start_time': datetime.now(),
            'status': 'analyzing',
            'context': context or {},
            'callback': callback,
            'processing_type': 'text_only'
        }
        
        # Initialize minimal tracking
        self.session_analysis_history[session_id] = []
        self.accumulated_patterns[session_id] = {}
        self.processing_steps[session_id] = []
        self.agent_states[session_id] = {
            'scam_analysis_active': False,
            'policy_analysis_active': False,
            'decision_active': False,
            'case_management_active': False,
            'current_agent': '',
            'agents_completed': []
        }
    
    # ===== AUDIO CALLBACK HANDLING =====
    
    def _create_audio_callback(self, session_id: str, external_callback: Optional[Callable]) -> Callable:
        """Create callback for audio processing that orchestrates the pipeline"""
        async def orchestrator_audio_callback(message_data: Dict) -> None:
            try:
                # Forward to external callback if provided (e.g., WebSocket)
                if external_callback:
                    await external_callback(message_data)
                
                # Handle orchestration logic
                await self._handle_audio_message(session_id, message_data)
                
            except Exception as e:
                logger.error(f"❌ FIXED Orchestrator audio callback error: {e}")
        
        return orchestrator_audio_callback
    
    async def _handle_audio_message(self, session_id: str, message_data: Dict):
        """Handle messages from audio processor and orchestrate next steps"""
        message_type = message_data.get('type')
        data = message_data.get('data', {})
        
        if message_type == 'transcription_segment':
            speaker = data.get('speaker')
            text = data.get('text', '')
            
            if speaker == 'customer' and text.strip():
                await self._process_customer_speech(session_id, text, data)
        
        elif message_type == 'processing_complete':
            await self._handle_processing_complete(session_id)
    
    async def _process_customer_speech(self, session_id: str, customer_text: str, segment_data: Dict):
        """Process customer speech and trigger agent pipeline"""
        try:
            # Add to speech buffer
            if session_id not in self.customer_speech_buffer:
                self.customer_speech_buffer[session_id] = []
            
            self.customer_speech_buffer[session_id].append({
                'text': customer_text,
                'timestamp': segment_data.get('start', 0),
                'confidence': segment_data.get('confidence', 0.0),
                'segment_id': segment_data.get('segment_id')
            })
            
            # Trigger analysis pipeline
            accumulated_speech = " ".join([
                segment['text'] for segment in self.customer_speech_buffer[session_id]
            ])
            
            if len(accumulated_speech.strip()) >= 15:
                await self._run_agent_pipeline(session_id, accumulated_speech)
            
        except Exception as e:
            logger.error(f"❌ FIXED Orchestrator error processing customer speech: {e}")
    
    # ===== CORE AGENT PIPELINE =====
    
    async def _run_agent_pipeline(self, session_id: str, customer_text: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """FIXED core orchestration method with proper ADK runner implementation"""
        try:
            logger.info(f"🎭 FIXED Orchestrator running agent pipeline for session {session_id}")
            
            session_data = self.active_sessions[session_id]
            
            # STEP 1: Scam Detection Agent
            logger.info("🤖 FIXED Orchestrator: Running Scam Detection Agent")
            await self._notify_agent_start('scam_detection', session_id, callback)
            
            scam_analysis = await self._run_scam_detection_fixed(session_id, customer_text)
            parsed_analysis = await self._parse_and_accumulate_patterns(session_id, scam_analysis)
            
            await self._notify_agent_complete('scam_detection', session_id, parsed_analysis, callback)
            
            risk_score = parsed_analysis.get('risk_score', 0)
            
            # STEP 2: Policy Guidance Agent (if medium+ risk)
            if risk_score >= 40:
                logger.info("📚 FIXED Orchestrator: Running Policy Guidance Agent")
                await self._notify_agent_start('policy_guidance', session_id, callback)
                
                policy_result = await self._run_policy_guidance_fixed(session_id, parsed_analysis)
                session_data['policy_guidance'] = policy_result
                
                await self._notify_agent_complete('policy_guidance', session_id, policy_result, callback)
            
            # STEP 3: Decision Agent (if high risk)
            if risk_score >= 60:
                logger.info("🎯 FIXED Orchestrator: Running Decision Agent")
                await self._notify_agent_start('decision', session_id, callback)
                
                decision_result = await self._run_decision_agent_fixed(session_id, parsed_analysis)
                session_data['decision_result'] = decision_result
                
                await self._notify_agent_complete('decision', session_id, decision_result, callback)
            
            # STEP 4: Case Management (if required)
            if risk_score >= 50:
                logger.info("📋 FIXED Orchestrator: Running Case Management Agent")
                await self._run_case_management(session_id, parsed_analysis, callback)
            
            logger.info(f"✅ FIXED Orchestrator completed pipeline: {risk_score}% risk")
            
            return {
                'risk_score': risk_score,
                'scam_analysis': parsed_analysis,
                'policy_guidance': session_data.get('policy_guidance'),
                'decision_result': session_data.get('decision_result'),
                'case_info': session_data.get('case_info'),
                'orchestration_complete': True
            }
            
        except Exception as e:
            logger.error(f"❌ FIXED Orchestrator pipeline error: {e}")
            return {'error': str(e)}
    
    # ===== FIXED ADK AGENT EXECUTION METHODS =====
    
    async def _run_scam_detection_fixed(self, session_id: str, customer_text: str) -> str:
        """FIXED scam detection with proper ADK runner"""
        try:
            if not self.adk_agents_available:
                return self._fallback_scam_detection(customer_text)
            
            session_data = self.active_sessions[session_id]
            
            context = {
                'session_id': session_id,
                'scam_context': session_data.get('scam_type_context', 'unknown'),
                'customer_profile': session_data.get('customer_profile', {}),
                'call_duration': (datetime.now() - session_data['start_time']).total_seconds()
            }
            
            prompt = f"""
FRAUD ANALYSIS REQUEST:
CUSTOMER SPEECH: "{customer_text}"
CONTEXT: {json.dumps(context, indent=2)}

Please provide structured fraud analysis with:
- Risk Score (0-100%)
- Risk Level (MINIMAL/LOW/MEDIUM/HIGH/CRITICAL)
- Scam Type (romance_scam/investment_scam/impersonation_scam/authority_scam/unknown)
- Detected Patterns
- Confidence level
- Recommended Action
"""
            
            # Use proper ADK runner pattern from TradeSage
            with WarningSuppressionContext():
                from google.adk.runners import InMemoryRunner
                from google.adk.sessions import InMemorySessionService
                from google.genai import types
                
                # Create session service and runner
                session_service = InMemorySessionService()
                runner = InMemoryRunner(
                    agent=self.scam_detection_agent,
                    session_service=session_service
                )
                
                # Create session
                app_name = "fraud_detection"
                user_id = session_id
                adk_session_id = f"{session_id}_scam"
                
                await session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=adk_session_id
                )
                
                # Create message
                message = types.Content(
                    role='user',
                    parts=[types.Part(text=prompt)]
                )
                
                # Collect response
                response_text = ""
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=adk_session_id,
                    new_message=message
                ):
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts') and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text
                
                logger.info(f"✅ FIXED ADK scam detection completed for {session_id}")
                return response_text or self._fallback_scam_detection(customer_text)
                
        except Exception as e:
            logger.error(f"❌ FIXED ADK scam detection error: {e}")
            return self._fallback_scam_detection(customer_text)
    
    async def _run_policy_guidance_fixed(self, session_id: str, scam_analysis: Dict) -> Dict:
        """FIXED policy guidance with proper ADK runner"""
        try:
            if not self.adk_agents_available:
                return self._fallback_policy_guidance(scam_analysis)
            
            context = {
                'session_id': session_id,
                'customer_profile': self.active_sessions[session_id].get('customer_profile', {}),
                'scam_analysis': scam_analysis
            }
            
            prompt = f"""
POLICY GUIDANCE REQUEST:
SCAM ANALYSIS RESULTS: {json.dumps(scam_analysis, indent=2)}
CONTEXT: {json.dumps(context, indent=2)}

Please provide structured policy guidance with:
- Policy ID and Title
- Immediate Alerts
- Recommended Actions
- Key Questions to ask customer
- Customer Education points
- Escalation threshold
"""
            
            with WarningSuppressionContext():
                from google.adk.runners import InMemoryRunner
                from google.adk.sessions import InMemorySessionService
                from google.genai import types
                
                session_service = InMemorySessionService()
                runner = InMemoryRunner(
                    agent=self.policy_guidance_agent,
                    session_service=session_service
                )
                
                app_name = "fraud_detection"
                user_id = session_id
                adk_session_id = f"{session_id}_policy"
                
                await session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=adk_session_id
                )
                
                message = types.Content(
                    role='user',
                    parts=[types.Part(text=prompt)]
                )
                
                response_text = ""
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=adk_session_id,
                    new_message=message
                ):
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts') and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text
                
                # Parse response into structured format
                return self._parse_policy_response(response_text) or self._fallback_policy_guidance(scam_analysis)
                
        except Exception as e:
            logger.error(f"❌ FIXED ADK policy guidance error: {e}")
            return self._fallback_policy_guidance(scam_analysis)
    
    async def _run_decision_agent_fixed(self, session_id: str, scam_analysis: Dict) -> Dict:
        """FIXED decision agent with proper ADK runner"""
        try:
            if not self.adk_agents_available:
                return self._fallback_decision_agent(scam_analysis)
            
            session_data = self.active_sessions[session_id]
            
            consolidated_analysis = {
                'scam_analysis': scam_analysis,
                'policy_guidance': session_data.get('policy_guidance', {}),
                'accumulated_patterns': self.accumulated_patterns.get(session_id, {}),
                'session_context': {
                    'session_id': session_id,
                    'customer_profile': session_data.get('customer_profile', {}),
                    'call_duration': (datetime.now() - session_data['start_time']).total_seconds()
                }
            }
            
            prompt = f"""
DECISION REQUEST:
ANALYSIS DATA: {json.dumps(consolidated_analysis, indent=2)}

Please provide final decision with:
- DECISION (BLOCK_AND_ESCALATE/VERIFY_AND_MONITOR/MONITOR_AND_EDUCATE/CONTINUE_NORMAL)
- REASONING
- PRIORITY (CRITICAL/HIGH/MEDIUM/LOW)
- IMMEDIATE_ACTIONS
- CASE_CREATION_REQUIRED (YES/NO)
- ESCALATION_PATH
- CUSTOMER_IMPACT
"""
            
            with WarningSuppressionContext():
                from google.adk.runners import InMemoryRunner
                from google.adk.sessions import InMemorySessionService
                from google.genai import types
                
                session_service = InMemorySessionService()
                runner = InMemoryRunner(
                    agent=self.decision_agent,
                    session_service=session_service
                )
                
                app_name = "fraud_detection"
                user_id = session_id
                adk_session_id = f"{session_id}_decision"
                
                await session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=adk_session_id
                )
                
                message = types.Content(
                    role='user',
                    parts=[types.Part(text=prompt)]
                )
                
                response_text = ""
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=adk_session_id,
                    new_message=message
                ):
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts') and event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text
                
                return self._parse_decision_response(response_text, scam_analysis.get('risk_score', 0)) or self._fallback_decision_agent(scam_analysis)
                
        except Exception as e:
            logger.error(f"❌ FIXED ADK decision agent error: {e}")
            return self._fallback_decision_agent(scam_analysis)
    
    # ===== FALLBACK METHODS =====
    
    def _fallback_scam_detection(self, customer_text: str) -> str:
        """Fallback scam detection using pattern matching"""
        risk_score = 0
        detected_patterns = []
        scam_type = "unknown"
        
        text_lower = customer_text.lower()
        
        # Romance scam patterns
        if any(word in text_lower for word in ['boyfriend', 'girlfriend', 'online', 'never met', 'emergency', 'stuck', 'money']):
            risk_score += 30
            detected_patterns.append('romance_exploitation')
            scam_type = "romance_scam"
        
        # Investment scam patterns
        if any(word in text_lower for word in ['investment', 'guaranteed', 'returns', 'profit', 'trading']):
            risk_score += 35
            detected_patterns.append('investment_fraud')
            scam_type = "investment_scam"
        
        # Urgency patterns
        if any(word in text_lower for word in ['urgent', 'immediately', 'today', 'now', 'quickly']):
            risk_score += 20
            detected_patterns.append('urgency_pressure')
        
        risk_score = min(risk_score, 100)
        
        if risk_score >= 80:
            risk_level = "CRITICAL"
            action = "IMMEDIATE_ESCALATION"
        elif risk_score >= 60:
            risk_level = "HIGH"
            action = "ENHANCED_MONITORING"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
            action = "VERIFY_CUSTOMER_INTENT"
        else:
            risk_level = "LOW"
            action = "CONTINUE_NORMAL_PROCESSING"
        
        return f"""
Risk Score: {risk_score}%
Risk Level: {risk_level}
Scam Type: {scam_type}
Detected Patterns: {detected_patterns}
Key Evidence: {customer_text[:100]}...
Confidence: 75%
Recommended Action: {action}
"""
    
    def _fallback_policy_guidance(self, scam_analysis: Dict) -> Dict:
        """Fallback policy guidance"""
        risk_score = scam_analysis.get('risk_score', 0)
        
        if risk_score >= 80:
            return {
                'policy_id': 'FP-CRITICAL-001',
                'policy_title': 'Critical Fraud Response Policy',
                'immediate_alerts': ['STOP ALL TRANSACTIONS', 'ESCALATE IMMEDIATELY'],
                'recommended_actions': [
                    'Halt all pending transfers',
                    'Contact fraud prevention team',
                    'Document all evidence',
                    'Secure customer account'
                ],
                'key_questions': [
                    'Can you verify your identity?',
                    'Are you being coerced?',
                    'Have you been asked to keep this secret?'
                ],
                'customer_education': [
                    'This appears to be a fraud attempt',
                    'We are protecting your account',
                    'Fraudsters often create urgency'
                ],
                'escalation_threshold': 80
            }
        elif risk_score >= 40:
            return {
                'policy_id': 'FP-STANDARD-002',
                'policy_title': 'Enhanced Verification Policy',
                'immediate_alerts': ['Enhanced verification required'],
                'recommended_actions': [
                    'Verify customer identity',
                    'Ask additional security questions',
                    'Explain transaction risks'
                ],
                'key_questions': [
                    'Can you confirm your full name and address?',
                    'What is the purpose of this transaction?'
                ],
                'customer_education': [
                    'Always verify requests independently',
                    'Be cautious of unsolicited offers'
                ],
                'escalation_threshold': 60
            }
        else:
            return {
                'policy_id': 'FP-STANDARD-001',
                'policy_title': 'Standard Processing Policy',
                'immediate_alerts': [],
                'recommended_actions': ['Follow standard procedures'],
                'key_questions': [],
                'customer_education': [],
                'escalation_threshold': 40
            }
    
    def _fallback_decision_agent(self, scam_analysis: Dict) -> Dict:
        """Fallback decision agent"""
        risk_score = scam_analysis.get('risk_score', 0)
        
        if risk_score >= 80:
            decision = "BLOCK_AND_ESCALATE"
            priority = "CRITICAL"
            actions = ["Stop all transactions", "Escalate to fraud team", "Contact customer"]
        elif risk_score >= 60:
            decision = "VERIFY_AND_MONITOR"
            priority = "HIGH"
            actions = ["Enhanced verification", "Additional questions", "Monitor account"]
        elif risk_score >= 40:
            decision = "MONITOR_AND_EDUCATE"
            priority = "MEDIUM"
            actions = ["Customer education", "Document interaction", "Standard monitoring"]
        else:
            decision = "CONTINUE_NORMAL"
            priority = "LOW"
            actions = ["Standard processing"]
        
        return {
            'decision': decision,
            'reasoning': f"Risk score of {risk_score}% triggers {decision} protocol",
            'priority': priority,
            'immediate_actions': actions,
            'case_creation_required': risk_score >= 50,
            'escalation_path': 'Fraud Prevention Team' if risk_score >= 60 else 'Standard Processing',
            'customer_impact': 'Transaction may be delayed for verification' if risk_score >= 60 else 'No impact'
        }
    
    # ===== RESPONSE PARSING METHODS =====
    
    def _parse_policy_response(self, response_text: str) -> Dict:
        """Parse policy guidance response"""
        try:
            parsed = {}
            lines = response_text.split('\n')
            
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                
                if ':' in line and not line.startswith('-') and not line.startswith('•'):
                    # Save previous section
                    if current_section and current_content:
                        if current_section in ['policy_id', 'policy_title']:
                            parsed[current_section] = current_content[0] if current_content else ''
                        else:
                            parsed[current_section] = current_content
                        current_content = []
                    
                    # Parse new section
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    current_section = key
                    
                    if value:
                        current_content = [value]
                
                elif current_section and line:
                    current_content.append(line)
            
            # Save final section
            if current_section and current_content:
                if current_section in ['policy_id', 'policy_title']:
                    parsed[current_section] = current_content[0] if current_content else ''
                else:
                    parsed[current_section] = current_content
            
            # Ensure required fields exist
            parsed.setdefault('policy_id', 'FP-AUTO-001')
            parsed.setdefault('policy_title', 'Automated Policy Guidance')
            parsed.setdefault('immediate_alerts', [])
            parsed.setdefault('recommended_actions', [])
            parsed.setdefault('key_questions', [])
            parsed.setdefault('customer_education', [])
            parsed.setdefault('escalation_threshold', 60)
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Error parsing policy response: {e}")
            return {}
    
    def _parse_decision_response(self, response_text: str, risk_score: float) -> Dict:
        """Parse decision agent response"""
        try:
            parsed = {}
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('-'):
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    parsed[key] = value
            
            # Extract decision from text
            decision_text = response_text.upper()
            if 'BLOCK_AND_ESCALATE' in decision_text:
                decision = 'BLOCK_AND_ESCALATE'
                priority = 'CRITICAL'
            elif 'VERIFY_AND_MONITOR' in decision_text:
                decision = 'VERIFY_AND_MONITOR'
                priority = 'HIGH'
            elif 'MONITOR_AND_EDUCATE' in decision_text:
                decision = 'MONITOR_AND_EDUCATE'
                priority = 'MEDIUM'
            else:
                decision = 'CONTINUE_NORMAL'
                priority = 'LOW'
            
            return {
                'decision': parsed.get('decision', decision),
                'reasoning': parsed.get('reasoning', f'Risk score {risk_score}% analysis'),
                'priority': parsed.get('priority', priority),
                'immediate_actions': parsed.get('immediate_actions', f'{decision} protocol').split(','),
                'case_creation_required': risk_score >= 50,
                'escalation_path': parsed.get('escalation_path', 'Fraud Prevention Team'),
                'customer_impact': parsed.get('customer_impact', 'Standard processing')
            }
            
        except Exception as e:
            logger.error(f"❌ Error parsing decision response: {e}")
            return {}
    
    # ===== CASE MANAGEMENT =====
    
    async def _run_case_management(self, session_id: str, analysis: Dict, callback: Optional[Callable]):
        """Run case management agent"""
        try:
            await self._notify_agent_start('case_management', session_id, callback)
            
            session_data = self.active_sessions[session_id]
            
            incident_data = {
                'customer_name': session_data.get('customer_profile', {}).get('name', 'Unknown'),
                'customer_account': session_data.get('customer_profile', {}).get('account', 'Unknown'),
                'customer_phone': session_data.get('customer_profile', {}).get('phone', 'Unknown'),
                'risk_score': analysis.get('risk_score', 0),
                'scam_type': analysis.get('scam_type', 'unknown'),
                'session_id': session_id,
                'confidence_score': analysis.get('confidence', 0.0),
                'detected_patterns': self.accumulated_patterns.get(session_id, {}),
                'auto_created': True,
                'creation_method': 'fixed_orchestrator_auto_creation'
            }
            
            case_result = await self.case_management_agent.create_fraud_incident(
                incident_data=incident_data,
                context={'session_id': session_id}
            )
            
            session_data['case_info'] = case_result
            
            await self._notify_agent_complete('case_management', session_id, case_result, callback)
            
        except Exception as e:
            logger.error(f"❌ FIXED Orchestrator case management error: {e}")
    
    # ===== PATTERN PARSING AND ACCUMULATION =====
    
    async def _parse_and_accumulate_patterns(self, session_id: str, analysis_result: str) -> Dict[str, Any]:
        """Parse analysis and accumulate patterns"""
        try:
            parsed = self._parse_agent_result('scam_detection', analysis_result)
            
            # Extract patterns from result
            current_patterns = []
            if 'detected_patterns' in parsed:
                if isinstance(parsed['detected_patterns'], list):
                    current_patterns = parsed['detected_patterns']
                elif isinstance(parsed['detected_patterns'], str):
                    current_patterns = [p.strip() for p in parsed['detected_patterns'].split(',')]
            
            # Accumulate patterns
            session_patterns = self.accumulated_patterns.get(session_id, {})
            
            for pattern_name in current_patterns:
                if pattern_name and pattern_name != 'unknown':
                    if pattern_name in session_patterns:
                        session_patterns[pattern_name]['count'] += 1
                        session_patterns[pattern_name]['confidence'] = min(
                            session_patterns[pattern_name]['confidence'] + 0.1, 1.0
                        )
                    else:
                        session_patterns[pattern_name] = {
                            'pattern_name': pattern_name,
                            'count': 1,
                            'weight': 25,
                            'severity': 'medium',
                            'confidence': 0.7,
                            'matches': [pattern_name]
                        }
            
            self.accumulated_patterns[session_id] = session_patterns
            self.session_analysis_history[session_id].append(parsed)
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ FIXED Orchestrator pattern parsing error: {e}")
            return {'risk_score': 0.0, 'error': str(e)}
    
    def _parse_agent_result(self, agent_name: str, result_text: str) -> Dict:
        """Parse agent result text into structured data"""
        try:
            parsed = {}
            lines = result_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('-'):
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    # Handle specific fields
                    if 'risk_score' in key:
                        # Extract percentage
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)%?', value)
                        if match:
                            parsed['risk_score'] = float(match.group(1))
                    elif 'confidence' in key:
                        match = re.search(r'(\d+(?:\.\d+)?)%?', value)
                        if match:
                            parsed['confidence'] = float(match.group(1)) / 100
                    elif 'detected_patterns' in key:
                        # Parse patterns list
                        if '[' in value and ']' in value:
                            pattern_text = value.strip('[]')
                            patterns = [p.strip().strip("'\"") for p in pattern_text.split(',')]
                            parsed['detected_patterns'] = [p for p in patterns if p and p != 'unknown']
                        else:
                            parsed['detected_patterns'] = [value] if value and value != 'unknown' else []
                    else:
                        parsed[key] = value
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Error parsing {agent_name} result: {e}")
            return {'error': str(e), 'raw_text': result_text}
    
    # ===== NOTIFICATION METHODS =====
    
    async def _notify_agent_start(self, agent_name: str, session_id: str, callback: Optional[Callable]):
        """Notify about agent start"""
        if callback:
            await callback({
                'type': f'adk_{agent_name}_started',
                'data': {
                    'session_id': session_id,
                    'agent': f'FIXED ADK {agent_name.replace("_", " ").title()} Agent',
                    'orchestrator': 'FIXED_FraudDetectionOrchestrator',
                    'timestamp': get_current_timestamp()
                }
            })
    
    async def _notify_agent_complete(self, agent_name: str, session_id: str, result: Dict, callback: Optional[Callable]):
        """Notify about agent completion"""
        if callback:
            await callback({
                'type': f'{agent_name}_analysis_complete',
                'data': {
                    'session_id': session_id,
                    'agent_result': result,
                    'orchestrator': 'FIXED_FraudDetectionOrchestrator',
                    'timestamp': get_current_timestamp()
                }
            })
    
    # ===== UTILITY METHODS =====
    
    def _extract_scam_context_from_filename(self, filename: str) -> str:
        """Extract scam context from filename"""
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
        """Get customer profile for filename"""
        for key, profile in self.customer_profiles.items():
            if key in filename:
                return profile
        
        return {
            'name': 'Unknown Customer',
            'account': '****0000',
            'phone': 'Unknown',
            'demographics': {'age': 0, 'location': 'Unknown', 'relationship': 'Unknown'}
        }
    
    def _get_agents_involved(self, risk_score: float) -> List[str]:
        """Get list of agents involved based on risk score"""
        agents = ["scam_detection"]
        
        if risk_score >= 40:
            agents.append("policy_guidance")
        
        if risk_score >= 60:
            agents.append("decision")
        
        if risk_score >= 50:
            agents.append("case_management")
        
        return agents
    
    async def _cleanup_session(self, session_id: str):
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
            if session_id in self.processing_steps:
                del self.processing_steps[session_id]
            if session_id in self.agent_states:
                del self.agent_states[session_id]
        except Exception as e:
            logger.error(f"❌ FIXED Orchestrator cleanup error: {e}")
    
    async def _handle_processing_complete(self, session_id: str):
        """Handle processing completion"""
        try:
            analysis_history = self.session_analysis_history.get(session_id, [])
            if analysis_history:
                final_analysis = analysis_history[-1]
                final_risk_score = final_analysis.get('risk_score', 0)
                logger.info(f"🎭 FIXED Orchestrator processing complete: {session_id}, final risk: {final_risk_score}%")
        except Exception as e:
            logger.error(f"❌ FIXED Orchestrator completion handling error: {e}")
    
    # ===== STATUS AND MONITORING =====
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        return {
            'orchestrator_type': 'FIXED_FraudDetectionOrchestrator',
            'active_sessions': len(self.active_sessions),
            'agents_available': self.adk_agents_available,
            'agents_managed': [
                'AudioProcessorAgent',
                'FIXED_ScamDetectionAgent', 
                'FIXED_PolicyGuidanceAgent',
                'FIXED_DecisionAgent',
                'FIXED_SummarizationAgent',
                'FIXED_CaseManagementAgent'
            ],
            'capabilities': [
                'FIXED Multi-agent orchestration',
                'FIXED Session state management', 
                'FIXED Pipeline coordination',
                'FIXED Audio and text processing',
                'FIXED Real-time and batch analysis',
                'FIXED WebSocket and REST API support',
                'FIXED ADK runner integration',
                'FIXED Fallback mechanisms'
            ],
            'statistics': {
                'total_sessions': len(self.active_sessions),
                'total_analyses': sum(len(history) for history in self.session_analysis_history.values()),
                'total_patterns': sum(len(patterns) for patterns in self.accumulated_patterns.values())
            },
            'timestamp': get_current_timestamp()
        }
    
    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """Get complete session data"""
        if session_id not in self.active_sessions:
            return None
        
        return {
            'session_info': self.active_sessions[session_id],
            'speech_buffer': self.customer_speech_buffer.get(session_id, []),
            'analysis_history': self.session_analysis_history.get(session_id, []),
            'accumulated_patterns': self.accumulated_patterns.get(session_id, {}),
            'processing_steps': self.processing_steps.get(session_id, []),
            'agent_states': self.agent_states.get(session_id, {})
        }

# ===== WEBSOCKET COMPATIBILITY LAYER (UNCHANGED) =====

class FraudDetectionWebSocketHandler:
    """WebSocket Handler - Thin wrapper around FIXED FraudDetectionOrchestrator"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.orchestrator = FraudDetectionOrchestrator(connection_manager)
        logger.info("🔌 FIXED WebSocket Handler initialized with Orchestrator delegation")
    
    async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]) -> None:
        """Handle WebSocket messages and delegate to orchestrator"""
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
            logger.error(f"❌ FIXED WebSocket handler error: {e}")
            await self.send_error(websocket, client_id, f"Message processing error: {str(e)}")
    
    async def handle_audio_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Handle audio processing via FIXED orchestrator"""
        filename = data.get('filename')
        session_id = data.get('session_id')
        
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
        
        try:
            # Create WebSocket callback
            async def websocket_callback(message_data: Dict) -> None:
                await self.send_message(websocket, client_id, message_data)
            
            # Delegate to FIXED orchestrator
            result = await self.orchestrator.orchestrate_audio_processing(
                filename=filename,
                session_id=session_id,
                callback=websocket_callback
            )
            
            if not result.get('success'):
                await self.send_error(websocket, client_id, result.get('error', 'Processing failed'))
                return
            
            # Send confirmation
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'session_id': result['session_id'],
                    'filename': filename,
                    'orchestrator': 'FIXED_FraudDetectionOrchestrator',
                    'handler': 'FIXED_WebSocketHandler',
                    'pipeline_mode': 'complete_multi_agent',
                    'timestamp': get_current_timestamp()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ FIXED WebSocket audio processing error: {e}")
            await self.send_error(websocket, client_id, f"Failed to start processing: {str(e)}")
    
    async def handle_stop_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Handle stop processing request"""
        session_id = data.get('session_id')
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            # Stop audio processing
            stop_result = await audio_processor_agent.stop_streaming(session_id)
            
            # Clean up in orchestrator
            await self.orchestrator._cleanup_session(session_id)
            
            await self.send_message(websocket, client_id, {
                'type': 'processing_stopped',
                'data': {
                    'session_id': session_id,
                    'status': 'stopped',
                    'orchestrator': 'FIXED_FraudDetectionOrchestrator',
                    'timestamp': get_current_timestamp()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error stopping processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to stop: {str(e)}")
    
    async def handle_status_request(self, websocket: WebSocket, client_id: str):
        """Handle status request"""
        try:
            status = self.orchestrator.get_orchestrator_status()
            status['websocket_handler'] = 'FIXED_WebSocketHandler'
            status['connection_manager'] = True
            
            await self.send_message(websocket, client_id, {
                'type': 'status_response',
                'data': status
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling status request: {e}")
            await self.send_error(websocket, client_id, f"Status request failed: {str(e)}")
    
    async def handle_session_analysis_request(self, websocket: WebSocket, client_id: str, data: Dict):
        """Handle session analysis request"""
        session_id = data.get('session_id')
        if not session_id:
            await self.send_error(websocket, client_id, "Invalid session ID")
            return
        
        try:
            session_data = self.orchestrator.get_session_data(session_id)
            if not session_data:
                await self.send_error(websocket, client_id, "Session not found")
                return
            
            await self.send_message(websocket, client_id, {
                'type': 'session_analysis_response',
                'data': {
                    'session_id': session_id,
                    'orchestrator': 'FIXED_FraudDetectionOrchestrator',
                    **session_data
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling session analysis request: {e}")
            await self.send_error(websocket, client_id, f"Failed to get session analysis: {str(e)}")
    
    async def handle_manual_case_creation(self, websocket: WebSocket, client_id: str, data: Dict):
        """Handle manual case creation"""
        session_id = data.get('session_id')
        if not session_id:
            await self.send_error(websocket, client_id, "Invalid session ID")
            return
        
        try:
            session_data = self.orchestrator.get_session_data(session_id)
            if not session_data:
                await self.send_error(websocket, client_id, "Session not found")
                return
            
            # Get latest analysis
            analysis_history = session_data.get('analysis_history', [])
            latest_analysis = analysis_history[-1] if analysis_history else {}
            
            # Create case via orchestrator's case management agent
            case_data = {
                'customer_name': session_data['session_info'].get('customer_profile', {}).get('name', 'Unknown'),
                'customer_account': session_data['session_info'].get('customer_profile', {}).get('account', 'Unknown'),
                'risk_score': latest_analysis.get('risk_score', 0),
                'scam_type': latest_analysis.get('scam_type', 'unknown'),
                'session_id': session_id,
                'auto_created': False,
                'creation_method': 'manual_via_fixed_websocket'
            }
            
            case_result = await self.orchestrator.case_management_agent.create_fraud_incident(
                incident_data=case_data,
                context={'session_id': session_id, 'creation_type': 'manual'}
            )
            
            if case_result.get('success'):
                await self.send_message(websocket, client_id, {
                    'type': 'manual_case_created',
                    'data': {
                        'session_id': session_id,
                        'case_number': case_result.get('incident_number'),
                        'case_url': case_result.get('incident_url'),
                        'orchestrator': 'FIXED_FraudDetectionOrchestrator',
                        'timestamp': get_current_timestamp()
                    }
                })
            else:
                await self.send_error(websocket, client_id, f"Failed to create case: {case_result.get('error')}")
            
        except Exception as e:
            logger.error(f"❌ Error creating manual case: {e}")
            await self.send_error(websocket, client_id, f"Failed to create case: {str(e)}")
    
    def cleanup_client_sessions(self, client_id: str) -> None:
        """Clean up sessions for disconnected client"""
        sessions_to_cleanup = []
        for session_id, session in self.orchestrator.active_sessions.items():
            if session.get('client_id') == client_id:
                sessions_to_cleanup.append(session_id)
        
        for session_id in sessions_to_cleanup:
            asyncio.create_task(self.orchestrator._cleanup_session(session_id))
        
        if sessions_to_cleanup:
            logger.info(f"🧹 FIXED WebSocket handler cleaned up {len(sessions_to_cleanup)} sessions for {client_id}")
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.orchestrator.active_sessions)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        orchestrator_status = self.orchestrator.get_orchestrator_status()
        return {
            **orchestrator_status['statistics'],
            'websocket_handler': 'FIXED_WebSocketHandler',
            'orchestrator_delegation': 'FIXED_FraudDetectionOrchestrator'
        }
    
    # ===== WEBSOCKET COMMUNICATION METHODS =====
    
    async def send_message(self, websocket: WebSocket, client_id: str, message: Dict) -> bool:
        """Send message to WebSocket client"""
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
        """Send error message to WebSocket client"""
        await self.send_message(websocket, client_id, {
            'type': 'error',
            'data': create_error_response(error_message, "FIXED_WEBSOCKET_ERROR")
        })

# Create the handler class for export (maintains compatibility)
FraudDetectionWebSocketHandler = FraudDetectionWebSocketHandler
