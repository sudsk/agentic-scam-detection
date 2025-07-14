# backend/orchestrator/fraud_detection_orchestrator.py - CORRECTED TRADESAGE PATTERN
"""
Corrected Fraud Detection Orchestrator using the exact TradeSage pattern
"""

import asyncio
import json
import logging
import time
import warnings
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import uuid

# Suppress warnings 
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

class WarningSuppressionContext:
    """Context manager to suppress warnings"""
    
    def __init__(self):
        import sys
        from io import StringIO
        self.original_stderr = sys.stderr
        self.suppressed_stderr = StringIO()
        self.warning_patterns = [
            'Warning: there are non-text parts in the response',
            'non-text parts in the response',
            'returning concatenated text result from text parts',
            'Check the full candidates.content.parts accessor'
        ]
    
    def __enter__(self):
        import sys
        sys.stderr = self.suppressed_stderr
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        captured = self.suppressed_stderr.getvalue()
        sys.stderr = self.original_stderr
        
        # Only show lines that don't match warning patterns
        if captured:
            lines = captured.split('\n')
            filtered_lines = []
            for line in lines:
                if line.strip() and not any(pattern in line for pattern in self.warning_patterns):
                    filtered_lines.append(line)
            
            if filtered_lines:
                print('\n'.join(filtered_lines), file=sys.stderr)

class FraudDetectionOrchestrator:
    """Fraud Detection Orchestrator """
    
    def __init__(self, connection_manager=None):
        self.connection_manager = connection_manager
        
        # Session tracking
        self.active_sessions: Dict[str, Dict] = {}
        self.customer_speech_buffer: Dict[str, List[str]] = {}
        self.session_analysis_history: Dict[str, List[Dict]] = {}
        self.accumulated_patterns: Dict[str, Dict] = {}
        
        # Initialize settings with fallback
        try:
            from ..config.settings import get_settings
            self.settings = get_settings()
        except ImportError:
            logger.warning("Settings import failed, using defaults")
            self.settings = self._get_default_settings()
        
        # Initialize agents 
        self.agents = self._initialize_agents()
        
        # Initialize customer profiles
        self._initialize_customer_profiles()
        
        logger.info("🎭 Orchestrator initialized")
    
    def _get_default_settings(self):
        """Fallback settings if import fails"""
        class DefaultSettings:
            def __init__(self):
                self.risk_threshold_critical = 80
                self.risk_threshold_high = 60
                self.risk_threshold_medium = 40
                self.risk_threshold_low = 20
        return DefaultSettings()
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize agents """
        try:
            from google.adk.agents import Agent
            
            agents = {
                "scam_detection": self._create_agent(
                    "scam_detection",
                    "Analyzes customer speech for fraud patterns",
                    self._get_scam_detection_instruction()
                ),
                "policy_guidance": self._create_agent(
                    "policy_guidance", 
                    "Provides procedural guidance for fraud scenarios",
                    self._get_policy_guidance_instruction()
                ),
                "decision": self._create_agent(
                    "decision",
                    "Makes final fraud prevention decisions", 
                    self._get_decision_instruction()
                ),
                "summarization": self._create_agent(
                    "summarization",
                    "Creates professional incident summaries",
                    self._get_summarization_instruction()
                )
            }
            
            # Case management agent
            try:
                from ..agents.case_management.adk_agent import PureADKCaseManagementAgent
                self.case_management_agent = PureADKCaseManagementAgent()
            except ImportError:
                self.case_management_agent = None
            
            logger.info(f"✅ Initialized {len(agents)} agents ")
            return agents
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
            return {}
    
    def _create_agent(self, name: str, description: str, instruction: str):
        """Create agent """
        from google.adk.agents import Agent
        
        return Agent(
            name=name,
            model="gemini-1.5-pro",
            description=description,
            instruction=instruction,
            tools=[],
        )
    
    def _get_scam_detection_instruction(self) -> str:
        return """
You are the Scam Detection Agent for HSBC Fraud Prevention System.

Output STRUCTURED analysis in this EXACT format:

Risk Score: [0-100]%
Risk Level: [MINIMAL/LOW/MEDIUM/HIGH/CRITICAL]
Scam Type: [romance_scam/investment_scam/impersonation_scam/authority_scam/unknown]
Detected Patterns: [list of patterns]
Key Evidence: [exact phrases]
Confidence: [0-100]%
Recommended Action: [IMMEDIATE_ESCALATION/ENHANCED_MONITORING/VERIFY_CUSTOMER_INTENT/CONTINUE_NORMAL_PROCESSING]

Analyze customer speech for fraud indicators and provide immediate assessment.
"""
    
    def _get_policy_guidance_instruction(self) -> str:
        return """
You are the Policy Guidance Agent for HSBC Fraud Prevention.

Output STRUCTURED guidance in this EXACT format:

Policy ID: FP-[TYPE]-[NUMBER]
Policy Title: [Specific policy name]
Immediate Alerts: [Urgent warnings]
Recommended Actions: [Step-by-step procedures]
Key Questions: [Questions to ask customer]
Customer Education: [What to explain]
Escalation Threshold: [Risk score % when to escalate]

Provide immediate, actionable policy guidance.
"""
    
    def _get_decision_instruction(self) -> str:
        return """
You are the Decision Agent for HSBC Fraud Prevention.

Output IMMEDIATE decisions in this EXACT format:

DECISION: [BLOCK_AND_ESCALATE/VERIFY_AND_MONITOR/MONITOR_AND_EDUCATE/CONTINUE_NORMAL]
REASONING: [Why this decision was made]
PRIORITY: [CRITICAL/HIGH/MEDIUM/LOW]
IMMEDIATE_ACTIONS: [Specific steps]
CASE_CREATION_REQUIRED: [YES/NO]
ESCALATION_PATH: [Which team]
CUSTOMER_IMPACT: [How this affects customer]

Make final fraud prevention decisions immediately.
"""
    
    def _get_summarization_instruction(self) -> str:
        return """
You are the Summarization Agent for HSBC Fraud Prevention.

Create PROFESSIONAL incident summaries in this format:

EXECUTIVE SUMMARY
[2-3 sentences with risk level and findings]

CUSTOMER REQUEST ANALYSIS
[What customer wanted, red flags]

FRAUD INDICATORS DETECTED
[Specific patterns with evidence]

RECOMMENDED ACTIONS
[Immediate steps required]

Create complete incident summaries for compliance documentation.
"""
    
    def _initialize_customer_profiles(self):
        """Initialize customer profiles"""
        self.customer_profiles = {
            'romance_scam_1.wav': {
                'name': 'Mrs Patricia Williams',
                'account': '****3847',
                'phone': '+44 1202 555123'
            },
            'investment_scam_live_call.wav': {
                'name': 'Mr David Chen', 
                'account': '****5691',
                'phone': '+44 161 555456'
            },
            'impersonation_scam_live_call.wav': {
                'name': 'Ms Sarah Thompson',
                'account': '****7234', 
                'phone': '+44 121 555789'
            }
        }
    
    # ===== MAIN ORCHESTRATION METHODS =====
    
    async def orchestrate_audio_processing(
        self, 
        filename: str, 
        session_id: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Main audio processing orchestration"""
        session_id = session_id or f"orchestrator_session_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"🎭 Orchestrator starting: {filename}")
            
            # Initialize session
            await self._initialize_session(session_id, filename, callback)
            
            # Create callback
            audio_callback = self._create_audio_callback(session_id, callback)
            
            # Start audio processing
            try:
                from ..agents.audio_processor.agent import audio_processor_agent
                result = await audio_processor_agent.start_realtime_processing(
                    session_id=session_id,
                    audio_filename=filename,
                    websocket_callback=audio_callback
                )
            except ImportError as e:
                logger.error(f"Audio processor import failed: {e}")
                return {'success': False, 'error': 'Audio processor not available'}
            
            if 'error' in result:
                await self._cleanup_session(session_id)
                return {'success': False, 'error': result['error']}
            
            return {
                'success': True,
                'session_id': session_id,
                'orchestrator': 'Orchestrator',
                'agents_available': len(self.agents) > 0,
                'audio_duration': result.get('audio_duration', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}")
            await self._cleanup_session(session_id)
            return {'success': False, 'error': str(e)}
    
    async def orchestrate_text_analysis(
        self,
        customer_text: str,
        session_id: Optional[str] = None,
        context: Optional[Dict] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Text analysis orchestration"""
        session_id = session_id or f"text_analysis_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"🎭 Text analysis: {session_id}")
            
            # Initialize session
            await self._initialize_text_session(session_id, customer_text, context, callback)
            
            # Run the agent pipeline
            result = await self._run_agent_pipeline(session_id, customer_text, callback)
            
            return {
                'success': True,
                'session_id': session_id,
                'orchestrator': 'Orchestrator',
                'analysis_result': result,
                'agents_involved': self._get_agents_involved(result.get('risk_score', 0))
            }
            
        except Exception as e:
            logger.error(f"❌ Text analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    # ===== AGENT EXECUTION PATTERN =====
    
    async def _run_agent_pipeline_tradesage(self, session_id: str, customer_text: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run agent pipelin"""
        try:
            logger.info(f"🎭 Running agent pipeline: {session_id}")
            
            session_data = self.active_sessions[session_id]
            
            # STEP 1: Scam Detection 
            scam_analysis = await self._run_agent("scam_detection", {
                "customer_text": customer_text,
                "session_id": session_id
            })
            
            parsed_analysis = await self._parse_and_accumulate_patterns(session_id, scam_analysis)
            risk_score = parsed_analysis.get('risk_score', 0)
            
            # STEP 2: Policy Guidance (if medium+ risk)
            if risk_score >= 40:
                policy_result = await self._run_agent("policy_guidance", {
                    "scam_analysis": parsed_analysis,
                    "session_id": session_id
                })
                session_data['policy_guidance'] = self._parse_policy_response(policy_result)
            
            # STEP 3: Decision Agent (if high risk)
            if risk_score >= 60:
                decision_result = await self._run_agent("decision", {
                    "scam_analysis": parsed_analysis,
                    "policy_guidance": session_data.get('policy_guidance', {}),
                    "session_id": session_id
                })
                session_data['decision_result'] = self._parse_decision_response(decision_result, risk_score)
            
            # STEP 4: Case Management (if required)
            if risk_score >= 50 and self.case_management_agent:
                await self._run_case_management(session_id, parsed_analysis, callback)
            
            logger.info(f"✅ Pipeline completed: {risk_score}% risk")
            
            return {
                'risk_score': risk_score,
                'scam_analysis': parsed_analysis,
                'policy_guidance': session_data.get('policy_guidance'),
                'decision_result': session_data.get('decision_result'),
                'case_info': session_data.get('case_info'),
                'orchestration_complete': True
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            return {'error': str(e)}
    
    async def _run_agent(self, agent_name: str, input_data: Dict[str, Any]) -> str:
        """Run agent """
        try:
            if agent_name not in self.agents:
                return self._fallback_agent_response(agent_name, input_data)
            
            agent = self.agents[agent_name]
            
            # Format input 
            user_message = self._format_agent_input(agent_name, input_data)
            
            # Use warning suppression pattern
            with WarningSuppressionContext():
                from google.adk.runners import Runner
                from google.genai import types
                
                # Create runner 
                runner = Runner(
                    agent=agent,
                    app_name=f"fraud_detection_{agent_name}",
                    session_service=None 
                )
                
                # Create message 
                message = types.Content(
                    role='user',
                    parts=[types.Part(text=user_message)]
                )
                
                # Collect response 
                text_responses = []
                function_calls = []
                function_responses = []
                errors = []
                
                # Process events 
                async for event in runner.run_async(
                    user_id="fraud_detection_user",
                    session_id=f"session_{agent_name}_{id(input_data)}",
                    new_message=message
                ):
                    # Handle ALL part types to avoid warnings 
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts') and event.content.parts:
                            for part in event.content.parts:
                                # Handle text parts
                                if hasattr(part, 'text') and part.text:
                                    text_responses.append(part.text)
                                
                                # Handle function calls (prevents warnings)
                                elif hasattr(part, 'function_call') and part.function_call:
                                    function_calls.append({
                                        "name": part.function_call.name,
                                        "args": dict(part.function_call.args) if part.function_call.args else {}
                                    })
                                
                                # Handle function responses (prevents warnings)
                                elif hasattr(part, 'function_response') and part.function_response:
                                    function_responses.append({
                                        "name": part.function_response.name,
                                        "response": part.function_response.response
                                    })
                                
                                # Handle any other part types to prevent warnings
                                else:
                                    pass
                    
                    # Handle errors
                    if hasattr(event, 'error') and event.error:
                        errors.append(str(event.error))
                
                # Combine response 
                final_text = " ".join(text_responses) if text_responses else ""
                
                if function_calls and not final_text:
                    final_text = f"Completed {len(function_calls)} tool calls successfully."
                
                logger.info(f"✅ Agent {agent_name} completed")
                return final_text or self._fallback_agent_response(agent_name, input_data)
                
        except Exception as e:
            logger.error(f"❌ Agent {agent_name} error: {e}")
            return self._fallback_agent_response(agent_name, input_data)
    
    def _format_agent_input(self, agent_name: str, input_data: Dict[str, Any]) -> str:
        """Format input for agent """
        if agent_name == "scam_detection":
            customer_text = input_data.get('customer_text', '')
            session_id = input_data.get('session_id', '')
            
            return f"""
FRAUD ANALYSIS REQUEST:
CUSTOMER SPEECH: "{customer_text}"
SESSION: {session_id}

Please provide structured fraud analysis with risk score, scam type, detected patterns, and recommended action.
"""
        
        elif agent_name == "policy_guidance":
            scam_analysis = input_data.get('scam_analysis', {})
            
            return f"""
POLICY GUIDANCE REQUEST:
SCAM ANALYSIS: {json.dumps(scam_analysis, indent=2)}

Please provide structured policy guidance with immediate alerts, recommended actions, and key questions.
"""
        
        elif agent_name == "decision":
            scam_analysis = input_data.get('scam_analysis', {})
            policy_guidance = input_data.get('policy_guidance', {})
            
            return f"""
DECISION REQUEST:
SCAM ANALYSIS: {json.dumps(scam_analysis, indent=2)}
POLICY GUIDANCE: {json.dumps(policy_guidance, indent=2)}

Please provide final decision with reasoning, priority, and immediate actions.
"""
        
        return str(input_data)
    
    # ===== FALLBACK METHODS =====
    
    def _fallback_agent_response(self, agent_name: str, input_data: Dict[str, Any]) -> str:
        """Fallback response when agent fails"""
        if agent_name == "scam_detection":
            return self._fallback_scam_detection(input_data.get('customer_text', ''))
        elif agent_name == "policy_guidance":
            return "Policy ID: FP-FALLBACK-001\nPolicy Title: Standard Processing\nImmediate Alerts: None\nRecommended Actions: Follow standard procedures"
        elif agent_name == "decision":
            return "DECISION: CONTINUE_NORMAL\nREASONING: Fallback processing\nPRIORITY: LOW\nIMMEDIATE_ACTIONS: Standard processing"
        else:
            return f"Fallback response for {agent_name}"
    
    def _fallback_scam_detection(self, customer_text: str) -> str:
        """Fallback scam detection"""
        risk_score = 0
        detected_patterns = []
        scam_type = "unknown"
        
        text_lower = customer_text.lower()
        
        # Romance scam patterns
        if any(word in text_lower for word in ['boyfriend', 'girlfriend', 'online', 'emergency', 'money']):
            risk_score += 30
            detected_patterns.append('romance_exploitation')
            scam_type = "romance_scam"
        
        # Investment scam patterns
        if any(word in text_lower for word in ['investment', 'guaranteed', 'returns', 'profit']):
            risk_score += 35
            detected_patterns.append('investment_fraud')
            scam_type = "investment_scam"
        
        # Urgency patterns
        if any(word in text_lower for word in ['urgent', 'immediately', 'today', 'now']):
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
    
    # ===== PARSING METHODS =====
    
    async def _parse_and_accumulate_patterns(self, session_id: str, analysis_result: str) -> Dict[str, Any]:
        """Parse analysis and accumulate patterns"""
        try:
            parsed = self._parse_agent_result('scam_detection', analysis_result)
            
            # Extract patterns
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
            logger.error(f"❌ Pattern parsing error: {e}")
            return {'risk_score': 0.0, 'error': str(e)}
    
    def _parse_agent_result(self, agent_name: str, result_text: str) -> Dict:
        """Parse agent result text"""
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
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)%?', value)
                        if match:
                            parsed['risk_score'] = float(match.group(1))
                    elif 'confidence' in key:
                        match = re.search(r'(\d+(?:\.\d+)?)%?', value)
                        if match:
                            parsed['confidence'] = float(match.group(1)) / 100
                    elif 'detected_patterns' in key:
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
    
    def _parse_policy_response(self, response_text: str) -> Dict:
        """Parse policy guidance response"""
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
        """Parse decision response"""
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
            
            # Extract decision from text if not parsed
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
    
    # ===== SESSION AND UTILITY METHODS =====
    
    async def _initialize_session(self, session_id: str, filename: str, callback: Optional[Callable]):
        """Initialize session"""
        self.active_sessions[session_id] = {
            'session_id': session_id,
            'filename': filename,
            'start_time': datetime.now(),
            'status': 'processing',
            'callback': callback,
            'customer_profile': self._get_customer_profile(filename)
        }
        
        self.customer_speech_buffer[session_id] = []
        self.session_analysis_history[session_id] = []
        self.accumulated_patterns[session_id] = {}
    
    async def _initialize_text_session(self, session_id: str, text: str, context: Optional[Dict], callback: Optional[Callable]):
        """Initialize text session"""
        self.active_sessions[session_id] = {
            'session_id': session_id,
            'input_text': text,
            'start_time': datetime.now(),
            'status': 'analyzing',
            'context': context or {},
            'callback': callback,
            'processing_type': 'text_only'
        }
        
        self.session_analysis_history[session_id] = []
        self.accumulated_patterns[session_id] = {}
    
    def _create_audio_callback(self, session_id: str, external_callback: Optional[Callable]) -> Callable:
        """Create audio callback"""
        async def orchestrator_callback(message_data: Dict) -> None:
            try:
                if external_callback:
                    await external_callback(message_data)
                await self._handle_audio_message(session_id, message_data)
            except Exception as e:
                logger.error(f"❌ Callback error: {e}")
        return orchestrator_callback
    
    async def _handle_audio_message(self, session_id: str, message_data: Dict):
        """Handle audio messages"""
        message_type = message_data.get('type')
        data = message_data.get('data', {})
        
        if message_type == 'transcription_segment':
            speaker = data.get('speaker')
            text = data.get('text', '')
            
            if speaker == 'customer' and text.strip():
                await self._process_customer_speech(session_id, text, data)
    
    async def _process_customer_speech(self, session_id: str, customer_text: str, segment_data: Dict):
        """Process customer speech"""
        try:
            if session_id not in self.customer_speech_buffer:
                self.customer_speech_buffer[session_id] = []
            
            self.customer_speech_buffer[session_id].append({
                'text': customer_text,
                'timestamp': segment_data.get('start', 0),
                'confidence': segment_data.get('confidence', 0.0)
            })
            
            # Trigger analysis
            accumulated_speech = " ".join([
                segment['text'] for segment in self.customer_speech_buffer[session_id]
            ])
            
            if len(accumulated_speech.strip()) >= 15:
                await self._run_agent_pipeline(session_id, accumulated_speech)
                
        except Exception as e:
            logger.error(f"❌ Error processing customer speech: {e}")
    
    async def _run_case_management(self, session_id: str, analysis: Dict, callback: Optional[Callable]):
        """Run case management"""
        try:
            if not self.case_management_agent:
                logger.warning("Case management agent not available")
                return
            
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
                'creation_method': 'auto_creation'
            }
            
            case_result = await self.case_management_agent.create_fraud_incident(
                incident_data=incident_data,
                context={'session_id': session_id}
            )
            
            session_data['case_info'] = case_result
            
            if callback:
                await callback({
                    'type': 'case_management_complete',
                    'data': {
                        'session_id': session_id,
                        'case_result': case_result,
                        'orchestrator': 'Orchestrator'
                    }
                })
            
        except Exception as e:
            logger.error(f"❌ Case management error: {e}")
    
    def _get_customer_profile(self, filename: str) -> Dict:
        """Get customer profile for filename"""
        for key, profile in self.customer_profiles.items():
            if key in filename:
                return profile
        
        return {
            'name': 'Unknown Customer',
            'account': '****0000',
            'phone': 'Unknown'
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
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    # ===== STATUS AND MONITORING =====
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        try:
            from ..utils import get_current_timestamp
        except ImportError:
            def get_current_timestamp():
                return datetime.now().isoformat()
        
        return {
            'orchestrator_type': 'Orchestrator',
            'active_sessions': len(self.active_sessions),
            'agents_available': len(self.agents) > 0,
            'agents_managed': list(self.agents.keys()) + (['case_management'] if self.case_management_agent else []),
            'capabilities': [
                'Agent execution',
                'Multi-agent orchestration',
                'Session state management', 
                'Pipeline coordination',
                'Audio and text processing',
                'Real-time and batch analysis',
                'WebSocket and REST API support',
                'Robust fallback mechanisms'
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
        }

# ===== WEBSOCKET COMPATIBILITY LAYER =====

class FraudDetectionWebSocketHandler:
    """WebSocket Handler """
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.orchestrator = FraudDetectionOrchestrator(connection_manager)
        logger.info("🔌 WebSocket Handler initialized")
    
    async def handle_message(self, websocket, client_id: str, message: Dict[str, Any]) -> None:
        """Handle WebSocket messages"""
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
                    'data': {'timestamp': datetime.now().isoformat()}
                })
            else:
                await self.send_error(websocket, client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ WebSocket handler error: {e}")
            await self.send_error(websocket, client_id, f"Message processing error: {str(e)}")
    
    async def handle_audio_processing(self, websocket, client_id: str, data: Dict) -> None:
        """Handle audio processing"""
        filename = data.get('filename')
        session_id = data.get('session_id')
        
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
        
        try:
            # Create WebSocket callback
            async def websocket_callback(message_data: Dict) -> None:
                await self.send_message(websocket, client_id, message_data)
            
            # Delegate to orchestrator
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
                    'orchestrator': 'Orchestrator',
                    'handler': 'WebSocketHandler',
                    'pipeline_mode': 'multi_agent',
                    'timestamp': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ WebSocket audio processing error: {e}")
            await self.send_error(websocket, client_id, f"Failed to start processing: {str(e)}")
    
    async def handle_stop_processing(self, websocket, client_id: str, data: Dict) -> None:
        """Handle stop processing request"""
        session_id = data.get('session_id')
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            # Stop audio processing
            try:
                from ..agents.audio_processor.agent import audio_processor_agent
                await audio_processor_agent.stop_streaming(session_id)
            except ImportError:
                logger.warning("Audio processor not available for stop")
            
            # Clean up in orchestrator
            await self.orchestrator._cleanup_session(session_id)
            
            await self.send_message(websocket, client_id, {
                'type': 'processing_stopped',
                'data': {
                    'session_id': session_id,
                    'status': 'stopped',
                    'orchestrator': 'Orchestrator',
                    'timestamp': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error stopping processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to stop: {str(e)}")
    
    async def handle_status_request(self, websocket, client_id: str):
        """Handle status request"""
        try:
            status = self.orchestrator.get_orchestrator_status()
            status['websocket_handler'] = 'WebSocketHandler'
            status['connection_manager'] = True
            
            await self.send_message(websocket, client_id, {
                'type': 'status_response',
                'data': status
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling status request: {e}")
            await self.send_error(websocket, client_id, f"Status request failed: {str(e)}")
    
    async def handle_session_analysis_request(self, websocket, client_id: str, data: Dict):
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
                    'orchestrator': 'Orchestrator',
                    **session_data
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling session analysis request: {e}")
            await self.send_error(websocket, client_id, f"Failed to get session analysis: {str(e)}")
    
    async def handle_manual_case_creation(self, websocket, client_id: str, data: Dict):
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
            if self.orchestrator.case_management_agent:
                case_data = {
                    'customer_name': session_data['session_info'].get('customer_profile', {}).get('name', 'Unknown'),
                    'customer_account': session_data['session_info'].get('customer_profile', {}).get('account', 'Unknown'),
                    'risk_score': latest_analysis.get('risk_score', 0),
                    'scam_type': latest_analysis.get('scam_type', 'unknown'),
                    'session_id': session_id,
                    'auto_created': False,
                    'creation_method': 'manual_via_websocket'
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
                            'orchestrator': 'Orchestrator',
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                else:
                    await self.send_error(websocket, client_id, f"Failed to create case: {case_result.get('error')}")
            else:
                await self.send_error(websocket, client_id, "Case management agent not available")
            
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
            logger.info(f"🧹 WebSocket handler cleaned up {len(sessions_to_cleanup)} sessions for {client_id}")
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.orchestrator.active_sessions)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        orchestrator_status = self.orchestrator.get_orchestrator_status()
        return {
            **orchestrator_status['statistics'],
            'websocket_handler': 'WebSocketHandler',
            'orchestrator_delegation': 'Orchestrator'
        }
    
    # ===== WEBSOCKET COMMUNICATION METHODS =====
    
    async def send_message(self, websocket, client_id: str, message: Dict) -> bool:
        """Send message to WebSocket client"""
        try:
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send message to {client_id}: {e}")
            return False
    
    async def send_error(self, websocket, client_id: str, error_message: str) -> None:
        """Send error message to WebSocket client"""
        await self.send_message(websocket, client_id, {
            'type': 'error',
            'data': {
                'status': 'error',
                'error': error_message,
                'code': 'WEBSOCKET_ERROR',
                'timestamp': datetime.now().isoformat()
            }
        })

# Create the handler class for export (maintains compatibility)
FraudDetectionWebSocketHandler = FraudDetectionWebSocketHandler
