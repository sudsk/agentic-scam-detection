# backend/orchestrator/fraud_detection_orchestrator.py - FIXED ADK VERSION
"""
Fixed Fraud Detection Orchestrator with proper ADK session management
Based on Google ADK patterns and successful implementations
"""

import asyncio
import json
import logging
import time
import warnings
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import uuid
from io import StringIO

# Import demo functionality
from ..config.demo_scripts import getDemoScript, DEMO_CONFIG
from .demo_orchestrator import DemoOrchestrator, getDemoScript
from ..config.fraud_patterns import create_keyword_mapping, FRAUD_PATTERN_CONFIG

# Suppress warnings at the top
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

class WarningSuppressionContext:
    """Context manager to suppress ADK warnings"""
    
    def __init__(self):
        self.original_stderr = sys.stderr
        self.suppressed_stderr = StringIO()
        self.warning_patterns = [
            'Warning: there are non-text parts in the response',
            'non-text parts in the response',
            'returning concatenated text result from text parts',
            'Check the full candidates.content.parts accessor'
        ]

    def __enter__(self):
        sys.stderr = self.suppressed_stderr
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
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
    """Fraud Detection Orchestrator with proper ADK session management"""
    
    def __init__(self, connection_manager=None):
        self.connection_manager = connection_manager
        
        # Session tracking
        self.active_sessions: Dict[str, Dict] = {}
        self.customer_speech_buffer: Dict[str, List[str]] = {}
        self.session_analysis_history: Dict[str, List[Dict]] = {}
        self.accumulated_patterns: Dict[str, Dict] = {}
        
        # Initialize settings 
        from ..config.settings import get_settings
        self.settings = get_settings()
        
        # Initialize all attributes to prevent AttributeError
        self.session_service = None
        self.agents = {}
        self.runners = {}
        self.case_management_agent = None
        self.servicenow_service = None
        self.types = None
        
        # Initialize ADK system
        adk_success = self._initialize_adk_system()
        
        # Initialize ServiceNow service directly
        self._initialize_servicenow()
        
        # Initialize customer profiles
        self._initialize_customer_profiles()
        
        if adk_success:
            logger.info("🎭 Fraud Detection Orchestrator initialized successfully with ADK")
        else:
            logger.error("🎭 Fraud Detection Orchestrator failed to initialize - ADK required")
            raise RuntimeError("ADK initialization failed - cannot continue without ADK")
        
        logger.info(f"🎭 Fraud Detection Orchestrator ready - Agents: {len(self.agents)}, Runners: {len(self.runners)}")

        # SIMPLE: Just track if a question is currently shown
        self.active_questions: Dict[str, Optional[str]] = {}  # session_id -> current_question_id or None

        # Initialize demo orchestrator
        self.demo_orchestrator = DemoOrchestrator(self)
        self.demo_mode_enabled = getattr(self.settings, 'demo_mode', True)
        logger.info(f"🎭 Demo orchestrator initialized - enabled: {self.demo_mode_enabled}")

        # ADD THIS LINE: Track analyzed speech to prevent duplicates
        self.analyzed_speech_segments: Dict[str, List[str]] = {}
     
    def _initialize_servicenow(self):
        """Initialize ServiceNow service"""
        try:
            from ..services.servicenow_service import ServiceNowService
            
            if self.settings.servicenow_enabled:
                self.servicenow_service = ServiceNowService(
                    instance_url=self.settings.servicenow_instance_url,
                    username=self.settings.servicenow_username,
                    password=self.settings.servicenow_password,
                    api_key=self.settings.servicenow_api_key
                )
                logger.info("✅ ServiceNow service initialized")
            else:
                logger.warning("⚠️ ServiceNow integration disabled")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize ServiceNow service: {e}")
            self.servicenow_service = None
    
    def _get_default_settings(self):
        """Fallback settings if import fails"""
        class DefaultSettings:
            def __init__(self):
                self.risk_threshold_critical = 80
                self.risk_threshold_high = 60
                self.risk_threshold_medium = 40
                self.risk_threshold_low = 20
                self.servicenow_enabled = False
                self.servicenow_instance_url = ""
                self.servicenow_username = ""
                self.servicenow_password = ""
                self.servicenow_api_key = ""
        return DefaultSettings()
    
    def _initialize_adk_system(self):
        """Initialize ADK system using pre-defined agents - ADK required"""
        try:
            logger.info("🔧 Starting ADK system initialization with pre-defined agents...")
            
            # Import ADK components
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            from google.genai import types
            
            logger.info("✅ ADK imports successful")
            
            # Check environment variables
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION")
            use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
           
            # Create session service
            try:
                self.session_service = InMemorySessionService()
                logger.info("✅ InMemorySessionService created")
            except Exception as e:
                logger.error(f"❌ Failed to create session service: {e}")
                return False
            
            self.types = types
            
            # Import individual agents from their own folders
            try:
                from ..agents.scam_detection.agent import scam_detection_agent
                from ..agents.policy_guidance.agent import policy_guidance_agent
                from ..agents.decision.agent import decision_agent
                from ..agents.summarization.agent import summarization_agent
                
                # Store agents with their keys
                self.agents = {
                    "scam_detection": scam_detection_agent,
                    "policy_guidance": policy_guidance_agent,
                    "decision": decision_agent,
                    "summarization": summarization_agent
                }
                
                logger.info(f"✅ Imported {len(self.agents)} individual agents from their own folders")
                
            except ImportError as e:
                logger.error(f"❌ Failed to import individual agents: {e}")
                logger.error("❌ ADK agents are required - no fallback available")
                return False
            
            # Create runners for each agent
            self.runners = {}
            for agent_key, agent in self.agents.items():
                try:
                    runner = Runner(
                        agent=agent,
                        app_name=f"fraud_detection_{agent_key}",
                        session_service=self.session_service
                    )
                    self.runners[agent_key] = runner
                except Exception as runner_error:
                    logger.error(f"❌ Failed to create runner for {agent_key}: {runner_error}")
                    continue
            
            # Case management agent initialization
            try:
                from ..agents.case_management.agent import create_case_management_agent
                self.case_management_agent = create_case_management_agent()
            except Exception as e:
                logger.warning(f"⚠️ Case management agent not available: {e}")
                self.case_management_agent = None

            if not self.agents:
                logger.error("❌ NO ADK AGENTS LOADED!")
                return False
        
            success = len(self.agents) > 0 and len(self.runners) > 0
            if success:
                logger.info(f"✅ ADK system initialized with pre-defined agents: {len(self.agents)} agents, {len(self.runners)} runners")
            else:
                logger.error("❌ Failed to initialize with pre-defined agents")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ADK system: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def _initialize_customer_profiles(self):
        """Initialize customer profiles"""
        self.customer_profiles = {
            'romance_scam_1.wav': {
                'name': 'Mrs Patricia Williams',
                'account': '****3847',
                'phone': '+44 1202 555123'
            },
            'investment_scam_1.wav': {
                'name': 'Mr Robert Chen', 
                'account': '****5691',
                'phone': '+44 161 555456'
            },
            'impersonation_scam_1.wav': {
                'name': 'Mrs Sarah Foster',
                'account': '****7234', 
                'phone': '+44 121 555789'
            },
            'app_scam_1.wav': {
                'name': 'Mr Adam Henderson',
                'account': '****6390', 
                'phone': '+44 121 555789'
            },
            'legitimate_call.wav': {
                'name': 'Mr Michael Thompson',
                'account': '****9205', 
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

            # 🎭 HYBRID DEMO: Start demo overlay but continue with real processing
            demo_script = getDemoScript(filename)
            if demo_script and self.demo_mode_enabled:
                logger.info(f"🎭 Hybrid Demo Mode: Real processing + scripted enhancements for {filename}")
                
                # Start demo orchestrator in background (non-blocking)
                demo_task = asyncio.create_task(
                    self.demo_orchestrator.start_demo_overlay(
                        demo_script, 
                        callback,
                        session_id  # Pass session_id for coordination
                    )
                )
                
                # Store demo task for cleanup
                if not hasattr(self, 'demo_tasks'):
                    self.demo_tasks = {}
                self.demo_tasks[session_id] = demo_task
                
            # Create enhanced callback that coordinates with demo
            audio_callback = self._create_hybrid_audio_callback(session_id, callback)
            
            # 🎵 REAL AUDIO PROCESSING HAPPENS
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
                'orchestrator': 'FraudDetectionOrchestrator',
                'mode': 'hybrid_demo' if demo_script else 'production',
                'agents_available': len(self.agents) > 0,
                'adk_session_service': self.session_service is not None,
                'audio_duration': result.get('audio_duration', 0),
                'demo_script': demo_script.get('title', 'Demo Script') if demo_script else None
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}")
            await self._cleanup_session(session_id)
            return {'success': False, 'error': str(e)}
            
    def _create_hybrid_audio_callback(self, session_id: str, external_callback: Optional[Callable]) -> Callable:
        """Create audio callback that coordinates with demo orchestrator"""
        async def hybrid_callback(message_data: Dict) -> None:
            try:
                # Send to external callback first
                if external_callback:
                    await external_callback(message_data)
    
                # Handle processing_complete in orchestrator
                if message_data.get('type') == 'processing_complete':
                    logger.info(f"🔍 HYBRID: Real audio processing complete for {session_id}")
                    
                    # Stop demo overlay if running
                    if hasattr(self, 'demo_tasks') and session_id in self.demo_tasks:
                        demo_task = self.demo_tasks[session_id]
                        if not demo_task.done():
                            await self.demo_orchestrator.coordinate_completion(session_id)
                    
                    # Continue with call completion
                    await self.handle_call_completion(session_id, external_callback)
            
                # Coordinate with demo for customer speech
                await self._handle_hybrid_audio_message(session_id, message_data)
                
            except Exception as e:
                logger.error(f"❌ Hybrid callback error: {e}")
        
        return hybrid_callback

    async def _handle_hybrid_audio_message(self, session_id: str, message_data: Dict):
        """Handle audio messages with demo coordination"""
        message_type = message_data.get('type')
        data = message_data.get('data', {})
        
        if message_type == 'transcription_segment':
            speaker = data.get('speaker')
            text = data.get('text', '')
            
            if speaker == 'customer' and text.strip():
                # Notify demo orchestrator of real customer speech
                if hasattr(self, 'demo_tasks') and session_id in self.demo_tasks:
                    await self.demo_orchestrator.handle_real_customer_speech(
                        session_id, text, data
                    )
                
                # Continue with normal processing
                await self._process_customer_speech(session_id, text, data)
            
    async def orchestrate_text_analysis(
        self,
        customer_text: str,
        session_id: Optional[str] = None,
        context: Optional[Dict] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Text analysis orchestration with proper ADK execution"""
        session_id = session_id or f"text_analysis_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"🎭 Text analysis: {session_id}")
            
            # Initialize session
            await self._initialize_text_session(session_id, customer_text, context, callback)
            
            # Run the ADK agent pipeline
            result = await self._run_adk_agent_pipeline(session_id, customer_text, callback)
            
            return {
                'success': True,
                'session_id': session_id,
                'orchestrator': 'FraudDetectionOrchestrator',
                'analysis_result': result,
                'agents_involved': self._get_agents_involved(result.get('risk_score', 0)),
                'adk_execution': True
            }
            
        except Exception as e:
            logger.error(f"❌ Text analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_call_completion(self, session_id: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Handle call completion - run summarization and case management"""
        try:
            logger.info(f"🏁 Call completion handler started for session {session_id}")
            
            if session_id not in self.active_sessions:
                logger.warning(f"❌ Session {session_id} not found for call completion")
                return {'success': False, 'error': 'Session not found'}
            
            session_data = self.active_sessions[session_id]
            
            # Get accumulated customer speech
            accumulated_speech = ""
            if session_id in self.customer_speech_buffer:
                accumulated_speech = " ".join([
                    segment['text'] for segment in self.customer_speech_buffer[session_id]
                ])
            
            if not accumulated_speech.strip():
                logger.info(f"ℹ️ No customer speech to analyze for session {session_id}")
                return {'success': True, 'message': 'No customer speech to analyze'}
            
            # Send call completion started message
            if callback:
                await callback({
                    'type': 'call_completion_started',
                    'data': {
                        'session_id': session_id,
                        'accumulated_speech_length': len(accumulated_speech),
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
            # Get the latest analysis from session (from progressive updates)
            latest_analysis = await self._get_latest_analysis_from_session(session_id, accumulated_speech)
            risk_score = latest_analysis.get('risk_score', 0)
            
            # Run summarization and case management only if risk is high enough
            if risk_score >= 50:
                # STEP 4: Summarization Agent
                logger.info(f"📝 Running summarization for call completion: {session_id}")
                summarization_result = await self._run_adk_agent("summarization", {
                    "fraud_analysis": latest_analysis,
                    "policy_guidance": session_data.get('policy_guidance', {}),
                    "decision_result": session_data.get('decision_result', {}),
                    "session_id": session_id
                })
                session_data['incident_summary'] = self._parse_summarization_response(summarization_result)
                
                if callback:
                    await callback({
                        'type': 'summarization_complete',
                        'data': {
                            'session_id': session_id,
                            'summary_ready': True,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
                # STEP 5: ServiceNow Case Creation
                if self.servicenow_service:
                    logger.info(f"📋 Creating ServiceNow case for call completion: {session_id}")
                    
                    if callback:
                        await callback({
                            'type': 'case_creation_started',
                            'data': {
                                'session_id': session_id,
                                'risk_score': risk_score,
                                'timestamp': datetime.now().isoformat()
                            }
                        })
                    
                    case_result = await self._create_servicenow_case(session_id, latest_analysis, callback)
                    session_data['case_info'] = case_result
            
            # Mark session as completed
            session_data['status'] = 'completed'
            session_data['completion_time'] = datetime.now()
            
            # Send final completion message
            if callback:
                await callback({
                    'type': 'call_completion_finished',
                    'data': {
                        'session_id': session_id,
                        'final_risk_score': risk_score,
                        'case_created': risk_score >= 50 and self.servicenow_service is not None,
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
            logger.info(f"✅ Call completion finished: {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'final_risk_score': risk_score,
                'case_created': risk_score >= 50 and self.servicenow_service is not None,
                'case_info': session_data.get('case_info')
            }
            
        except Exception as e:
            logger.error(f"❌ Call completion error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _get_latest_analysis_from_session(self, session_id: str, accumulated_speech: str) -> Dict[str, Any]:
        """Get the latest analysis from session data or run fresh analysis"""
        try:
            # Run fresh scam detection with accumulated speech
            scam_analysis_raw = await self._run_adk_agent("scam_detection", {
                "customer_text": accumulated_speech,
                "session_id": session_id
            })
            
            return await self._parse_and_accumulate_patterns(session_id, scam_analysis_raw)
            
        except Exception as e:
            logger.error(f"❌ Error getting latest analysis: {e}")
            return {'risk_score': 0, 'risk_level': 'MINIMAL', 'scam_type': 'unknown'}

# ===== ADK AGENT EXECUTION PATTERN =====
    
    async def _run_adk_agent_pipeline(self, session_id: str, customer_text: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run ADK agent pipeline with proper session management and UI updates"""
        try:
            logger.info(f"🎭 Running ADK agent pipeline: {session_id}")
            
            session_data = self.active_sessions[session_id]
            
            # STEP 1: Scam Detection using proper ADK execution
            logger.info(f"🔍 Step 1: Running scam detection for session {session_id}")
            scam_analysis_raw = await self._run_adk_agent("scam_detection", {
                "customer_text": customer_text,
                "session_id": session_id
            })
            
            # Parse and send UI update
            parsed_analysis = await self._parse_and_accumulate_patterns(session_id, scam_analysis_raw)
            risk_score = parsed_analysis.get('risk_score', 0)
            
            # Send fraud analysis update to UI with detailed logging
            if callback:
                fraud_update_data = {
                    'session_id': session_id,
                    'risk_score': risk_score,
                    'risk_level': parsed_analysis.get('risk_level', 'MINIMAL'),
                    'scam_type': parsed_analysis.get('scam_type', 'unknown'),
                    'detected_patterns': self.accumulated_patterns.get(session_id, {}),
                    'confidence': parsed_analysis.get('confidence', 0.0),
                    'explanation': parsed_analysis.get('explanation', ''),
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"🔍 Sending fraud_analysis_update: risk_score={risk_score}, patterns={list(fraud_update_data['detected_patterns'].keys())}")
                
                await callback({
                    'type': 'fraud_analysis_update',
                    'data': fraud_update_data
                })
            
            # STEP 2: Policy Guidance (if medium+ risk)
            if risk_score >= 40:
                logger.info(f"📚 Step 2: Running policy guidance for session {session_id}")
                if callback:
                    await callback({
                        'type': 'policy_analysis_started',
                        'data': {
                            'session_id': session_id,
                            'risk_score': risk_score,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
                policy_result = await self._run_adk_agent("policy_guidance", {
                    "scam_analysis": parsed_analysis,
                    "session_id": session_id
                })
                session_data['policy_guidance'] = self._parse_policy_response(policy_result)
                
                # Send policy guidance to UI with detailed logging
                if callback:
                    policy_data = session_data['policy_guidance']
                    
                    # Debug the policy data structure
                    logger.info(f"📚 Policy guidance data structure:")
                    logger.info(f"  - policy_id: {policy_data.get('policy_id', 'None')}")
                    logger.info(f"  - immediate_alerts type: {type(policy_data.get('immediate_alerts', []))}")
                    logger.info(f"  - immediate_alerts length: {len(policy_data.get('immediate_alerts', []))}")
                    logger.info(f"  - recommended_actions type: {type(policy_data.get('recommended_actions', []))}")
                    logger.info(f"  - recommended_actions length: {len(policy_data.get('recommended_actions', []))}")
                    logger.info(f"  - key_questions type: {type(policy_data.get('key_questions', []))}")
                    logger.info(f"  - customer_education type: {type(policy_data.get('customer_education', []))}")
                    
                    await callback({
                        'type': 'policy_guidance_ready',
                        'data': {
                            'session_id': session_id,
                            'policy_guidance': policy_data,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
            
            # STEP 3: Decision Agent (if high risk)
            if risk_score >= 60:
                logger.info(f"🎯 Step 3: Running decision agent for session {session_id}")
                decision_result = await self._run_adk_agent("decision", {
                    "scam_analysis": parsed_analysis,
                    "policy_guidance": session_data.get('policy_guidance', {}),
                    "session_id": session_id
                })
                session_data['decision_result'] = self._parse_decision_response(decision_result, risk_score)
                
                # Send decision update to UI
                if callback:
                    await callback({
                        'type': 'decision_made',
                        'data': {
                            'session_id': session_id,
                            'decision': session_data['decision_result'].get('decision', 'CONTINUE_NORMAL'),
                            'priority': session_data['decision_result'].get('priority', 'LOW'),
                            'reasoning': session_data['decision_result'].get('reasoning', ''),
                            'immediate_actions': session_data['decision_result'].get('immediate_actions', []),
                            'case_creation_required': session_data['decision_result'].get('case_creation_required', False),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
            
            logger.info(f"✅ ADK pipeline completed: {risk_score}% risk")
            
            # Send final completion message
            if callback:
                await callback({
                    'type': 'analysis_complete',
                    'data': {
                        'session_id': session_id,
                        'final_risk_score': risk_score,
                        'total_steps_completed': 3 if risk_score >= 60 else 2 if risk_score >= 40 else 1,
                        'agents_involved': self._get_agents_involved(risk_score),
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
            return {
                'risk_score': risk_score,
                'scam_analysis': parsed_analysis,
                'policy_guidance': session_data.get('policy_guidance'),
                'decision_result': session_data.get('decision_result'),
                'case_info': session_data.get('case_info'),
                'orchestration_complete': True,
                'adk_execution': True
            }
            
        except Exception as e:
            logger.error(f"❌ ADK pipeline error: {e}")
            return {'error': str(e), 'adk_execution': False}
    
    async def _create_servicenow_case(self, session_id: str, analysis: Dict, callback: Optional[Callable]) -> Dict[str, Any]:
        """Create ServiceNow case with proper formatting"""
        try:
            session_data = self.active_sessions[session_id]
            customer_profile = session_data.get('customer_profile', {})
            
            # Get scam type for ServiceNow mapping
            scam_type = analysis.get('scam_type', 'unknown')
            risk_score = analysis.get('risk_score', 0)
            
            # Map to ServiceNow categories (only standard categories/subcategories)
            category = "fraud_detection"
            
            subcategory_mapping = {
                'app_scam': 'app_scam',                     # Maps to "APP Scam" label
                'crypto_scam': 'crypto_scam',               # Maps to "Crypto Scam" label
                'impersonation_scam': 'impersonation_scam', # Maps to "Impersonation Scam" label
                'investment_scam': 'investment_scam',       # Maps to "Investment Scam" label
                'romance_scam': 'romance_scam',             # Maps to "Romance Scam" label
                'unknown': 'app_scam'                       # Default fallback to APP Scam if unknown
            }
            subcategory = subcategory_mapping.get(scam_type, 'app_scam')
            
            # Determine priority based on risk score
            if risk_score >= 80:
                priority = "1"
                urgency = "1"
            elif risk_score >= 60:
                priority = "2"
                urgency = "2"
            elif risk_score >= 40:
                priority = "3"
                urgency = "3"
            else:
                priority = "4"
                urgency = "3"
            
            # Format transcript for ServiceNow description
            transcript_text = ""
            if session_id in self.customer_speech_buffer:
                transcript_text = "\n".join([
                    f"[{segment.get('timestamp', 0):.1f}s] {segment.get('text', '')}"
                    for segment in self.customer_speech_buffer[session_id]
                ])
            
            # Format detected patterns
            patterns_text = ""
            if session_id in self.accumulated_patterns:
                patterns_text = "\n".join([
                    f"• {pattern.replace('_', ' ').title()}: {data.get('count', 0)} occurrences"
                    for pattern, data in self.accumulated_patterns[session_id].items()
                ])
            
            # Get summarization if available
            incident_summary = session_data.get('incident_summary', {})
            summary_text = incident_summary.get('executive_summary', 'Automated fraud detection alert')
            
            # Format incident data with standard ServiceNow fields only
            incident_data = {
                "category": category,
                "subcategory": subcategory,
                "short_description": f"Fraud Alert: {scam_type.replace('_', ' ').title()} - {customer_profile.get('name', 'Unknown Customer')}",
                "description": f"""
=== AUTOMATED FRAUD DETECTION ALERT ===

EXECUTIVE SUMMARY:
{summary_text}

CUSTOMER INFORMATION:
• Name: {customer_profile.get('name', 'Unknown')}
• Account: {customer_profile.get('account', 'Unknown')}
• Phone: {customer_profile.get('phone', 'Unknown')}

FRAUD ANALYSIS:
• Risk Score: {risk_score}%
• Risk Level: {analysis.get('risk_level', 'UNKNOWN')}
• Scam Type: {scam_type.replace('_', ' ').title()}
• Confidence: {analysis.get('confidence', 0):.1%}
• Session ID: {session_id}

DETECTED PATTERNS:
{patterns_text or 'No specific patterns detected'}

POLICY GUIDANCE:
{session_data.get('policy_guidance', {}).get('policy_id', 'No policy guidance available')}

DECISION OUTCOME:
{session_data.get('decision_result', {}).get('decision', 'CONTINUE_NORMAL')}

CUSTOMER SPEECH TRANSCRIPT:
{transcript_text or 'No transcript available'}

RECOMMENDED ACTIONS:
{chr(10).join(f"• {action}" for action in session_data.get('policy_guidance', {}).get('recommended_actions', []))}

Generated: {datetime.now().isoformat()}
System: HSBC Fraud Detection Orchestrator
                """.strip(),
                "priority": priority,
                "urgency": urgency,
                "state": "2",
                "caller_id": "fraud_detection_api",
                "opened_by": "fraud_detection_api"
            }
            
            # Create incident in ServiceNow
            logger.info(f"🎯 Creating ServiceNow incident with category: {incident_data['category']}, subcategory: {incident_data['subcategory']}")
            result = await self.servicenow_service.create_incident(incident_data)
            
            if result["success"]:
                logger.info(f"✅ ServiceNow case created: {result['incident_number']}")
                
                # Send success to UI
                if callback:
                    await callback({
                        'type': 'case_created',
                        'data': {
                            'session_id': session_id,
                            'case_number': result['incident_number'],
                            'case_url': result['incident_url'],
                            'success': True,
                            'case_id': result['incident_sys_id'],
                            'priority': priority,
                            'category': incident_data['category'],
                            'subcategory': incident_data['subcategory'],
                            'risk_score': risk_score,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
                return {
                    'success': True,
                    'incident_number': result['incident_number'],
                    'incident_sys_id': result['incident_sys_id'],
                    'incident_url': result['incident_url'],
                    'priority': priority,
                    'category': incident_data['category'],
                    'subcategory': incident_data['subcategory']
                }
            else:
                logger.error(f"❌ ServiceNow case creation failed: {result['error']}")
                
                # Send error to UI
                if callback:
                    await callback({
                        'type': 'case_creation_error',
                        'data': {
                            'session_id': session_id,
                            'error': result['error'],
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
                return {
                    'success': False,
                    'error': result['error']
                }
                
        except Exception as e:
            logger.error(f"❌ ServiceNow case creation error: {e}")
            
            # Send error to UI
            if callback:
                await callback({
                    'type': 'case_creation_error',
                    'data': {
                        'session_id': session_id,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # Always close ServiceNow session
            if self.servicenow_service:
                await self.servicenow_service.close_session()
    
    def _parse_summarization_response(self, response_text: str) -> Dict[str, Any]:
        """Parse summarization agent response"""
        try:
            parsed = {}
            lines = response_text.split('\n')
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if line.upper().startswith('EXECUTIVE SUMMARY'):
                    current_section = 'executive_summary'
                    current_content = []
                elif line.upper().startswith('CUSTOMER REQUEST ANALYSIS'):
                    if current_section and current_content:
                        parsed[current_section] = '\n'.join(current_content).strip()
                    current_section = 'customer_request_analysis'
                    current_content = []
                elif line.upper().startswith('FRAUD INDICATORS DETECTED'):
                    if current_section and current_content:
                        parsed[current_section] = '\n'.join(current_content).strip()
                    current_section = 'fraud_indicators'
                    current_content = []
                elif line.upper().startswith('RECOMMENDED ACTIONS'):
                    if current_section and current_content:
                        parsed[current_section] = '\n'.join(current_content).strip()
                    current_section = 'recommended_actions'
                    current_content = []
                elif line.upper().startswith('INCIDENT CLASSIFICATION'):
                    if current_section and current_content:
                        parsed[current_section] = '\n'.join(current_content).strip()
                    current_section = 'incident_classification'
                    current_content = []
                else:
                    current_content.append(line)
            
            # Save final section
            if current_section and current_content:
                parsed[current_section] = '\n'.join(current_content).strip()
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Error parsing summarization response: {e}")
            return {
                'executive_summary': 'Error parsing incident summary',
                'error': str(e)
            }
    
    async def _run_adk_agent(self, agent_name: str, input_data: Dict[str, Any]) -> str:
        """Run ADK agent with proper session management and validation"""
        try:
            # Validate agent and runner exist
            if agent_name not in self.agents:
                logger.error(f"❌ Agent {agent_name} not found in self.agents")
                raise RuntimeError(f"Agent {agent_name} not available")
                
            if agent_name not in self.runners:
                logger.error(f"❌ Runner {agent_name} not found in self.runners")
                raise RuntimeError(f"Runner {agent_name} not available")
            
            agent = self.agents[agent_name]
            runner = self.runners[agent_name]
            
            # Validate agent is not None
            if agent is None:
                logger.error(f"❌ Agent {agent_name} is None")
                raise RuntimeError(f"Agent {agent_name} is None")
            
            # Create proper ADK session
            app_name = f"fraud_detection_{agent_name}"
            user_id = "fraud_detection_user"
            session_adk_id = f"session_{agent_name}_{id(input_data)}"
            
            # Create session using session service
            try:
                session = await self.session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_adk_id
                )
                logger.debug(f"✅ Created ADK session: {session_adk_id}")
            except Exception as e:
                logger.error(f"❌ Failed to create ADK session: {e}")
                raise RuntimeError(f"Failed to create ADK session: {e}")
            
            # Format input for agent
            user_message = self._format_agent_input(agent_name, input_data)
            
            # Create proper message
            message = self.types.Content(
                role='user',
                parts=[self.types.Part(text=user_message)]
            )
            
            # Use warning suppression and proper event handling
            with WarningSuppressionContext():
                # Collect ALL events and parts properly
                text_responses = []
                function_calls = []
                function_responses = []
                tool_results = {}
                errors = []
                
                # Process all events and handle ALL part types to avoid warnings
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_adk_id,
                    new_message=message
                ):
                    # Handle different event types - process ALL parts to avoid warnings
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts') and event.content.parts:
                            for part in event.content.parts:
                                # Handle ALL part types to avoid warnings
                                
                                # Handle text parts
                                if hasattr(part, 'text') and part.text:
                                    text_responses.append(part.text)
                                
                                # Handle function calls (prevents warning about non-text parts)
                                elif hasattr(part, 'function_call') and part.function_call:
                                    function_call = {
                                        "name": part.function_call.name,
                                        "args": dict(part.function_call.args) if part.function_call.args else {}
                                    }
                                    function_calls.append(function_call)
                                
                                # Handle function responses (prevents warning about non-text parts)
                                elif hasattr(part, 'function_response') and part.function_response:
                                    function_response = {
                                        "name": part.function_response.name,
                                        "response": part.function_response.response
                                    }
                                    function_responses.append(function_response)
                                    
                                    # Store tool results for easy access
                                    tool_results[part.function_response.name] = part.function_response.response
                                
                                # Handle any other part types to prevent warnings
                                else:
                                    # This catches any other part types and processes them silently
                                    pass
                    
                    # Handle errors
                    if hasattr(event, 'error') and event.error:
                        errors.append(str(event.error))
                
                # Combine all response parts properly
                final_text = " ".join(text_responses) if text_responses else ""
                
                # If we have function calls but no text response, create summary
                if function_calls and not final_text:
                    final_text = f"Completed {len(function_calls)} tool calls successfully."
                
                if errors:
                    logger.warning(f"⚠️ ADK agent {agent_name} reported {len(errors)} errors")
                    for error in errors:
                        logger.warning(f"   Error: {error}")
                
                if final_text:
                    logger.info(f"✅ ADK agent {agent_name} completed successfully")
                    return final_text
                else:
                    logger.error(f"❌ ADK agent {agent_name} returned empty response")
                    raise RuntimeError(f"ADK agent {agent_name} returned empty response")
                
        except Exception as e:
            logger.error(f"❌ ADK agent {agent_name} error: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise e
    
    def _format_agent_input(self, agent_name: str, input_data: Dict[str, Any]) -> str:
        """Format input for ADK agent"""
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
        
        elif agent_name == "summarization":
            fraud_analysis = input_data.get('fraud_analysis', {})
            policy_guidance = input_data.get('policy_guidance', {})
            decision_result = input_data.get('decision_result', {})
            
            return f"""
SUMMARIZATION REQUEST:
FRAUD ANALYSIS: {json.dumps(fraud_analysis, indent=2)}
POLICY GUIDANCE: {json.dumps(policy_guidance, indent=2)}
DECISION RESULT: {json.dumps(decision_result, indent=2)}

Please provide professional incident summary for ServiceNow case documentation.
"""
        
        return str(input_data)

    # ===== PARSING METHODS =====
    
    async def _parse_and_accumulate_patterns(self, session_id: str, analysis_result: str) -> Dict[str, Any]:
        """
        FIXED: Parse analysis with PROPER pattern matching - no more false matches
        """
        try:
            # Parse the raw ADK result
            parsed = self._parse_agent_result('scam_detection', analysis_result)
            
            # Get keyword mapping from config
            keyword_mapping = create_keyword_mapping()
            
            # Extract raw patterns from ADK response
            raw_patterns = []
            if 'detected_patterns' in parsed:
                if isinstance(parsed['detected_patterns'], list):
                    raw_patterns = parsed['detected_patterns']
                elif isinstance(parsed['detected_patterns'], str):
                    # Handle comma-separated or other formats
                    raw_patterns = [p.strip() for p in parsed['detected_patterns'].split(',')]
            
            # STEP 1: FIXED PATTERN MATCHING with priority order
            clean_patterns = {}
            
            for raw_pattern in raw_patterns:
                if not raw_pattern or raw_pattern == 'unknown':
                    continue
                    
                logger.info(f"🔍 Trying to match: '{raw_pattern}'")
                
                # Split comma-separated patterns first
                individual_patterns = [p.strip() for p in raw_pattern.split(',')]
                
                for individual_pattern in individual_patterns:
                    if not individual_pattern:
                        continue
                        
                    normalized_pattern = individual_pattern.lower().strip()
                    matched_config = None
                    
                    # METHOD 1: EXACT DIRECT NAME MATCH (highest priority)
                    for pattern_id, config in FRAUD_PATTERN_CONFIG.items():
                        if config['name'].lower() == individual_pattern.strip():
                            matched_config = {
                                'name': config['name'],
                                'weight': config['weight'],
                                'severity': config['severity']
                            }
                            logger.info(f"   ✅ EXACT NAME MATCH: '{individual_pattern}' → {matched_config['name']} ({matched_config['weight']}%)")
                            break
                    
                    # METHOD 2: EXACT KEYWORD MATCH (if no direct name match)
                    if not matched_config:
                        if normalized_pattern in keyword_mapping:
                            matched_config = keyword_mapping[normalized_pattern]
                            logger.info(f"   ✅ EXACT KEYWORD MATCH: '{individual_pattern}' → {matched_config['name']} ({matched_config['weight']}%)")
                    
                    # METHOD 3: SPECIFIC PATTERN RECOGNITION (most important)
                    if not matched_config:
                        matched_config = self._match_specific_patterns(individual_pattern)
                        if matched_config:
                            logger.info(f"   ✅ SPECIFIC PATTERN MATCH: '{individual_pattern}' → {matched_config['name']} ({matched_config['weight']}%)")
                    
                    # METHOD 4: KEYWORD CONTAINS MATCH (careful fuzzy matching)
                    if not matched_config:
                        matched_config = self._match_by_keywords(individual_pattern, keyword_mapping)
                        if matched_config:
                            logger.info(f"   ✅ KEYWORD CONTAINS MATCH: '{individual_pattern}' → {matched_config['name']} ({matched_config['weight']}%)")
                    
                    # Use matched config or create fallback
                    if matched_config:
                        clean_name = matched_config['name']
                        weight = matched_config['weight']
                        severity = matched_config['severity']
                    else:
                        # Fallback for unrecognized patterns
                        clean_name = individual_pattern.replace('_', ' ').title()
                        weight = 10  # Default weight
                        severity = 'low'
                        logger.info(f"   ❌ NO MATCH: '{individual_pattern}' → fallback: {clean_name} ({weight}%)")
                    
                    # STEP 2: Deduplicate and accumulate counts
                    if clean_name in clean_patterns:
                        # Increment count for existing pattern
                        clean_patterns[clean_name]['count'] += 1
                        clean_patterns[clean_name]['raw_matches'].append(individual_pattern)
                    else:
                        # Create new clean pattern entry
                        clean_patterns[clean_name] = {
                            'pattern_name': clean_name,
                            'count': 1,
                            'weight': weight,
                            'severity': severity,
                            'confidence': 0.8,
                            'raw_matches': [individual_pattern]
                        }
            
            # STEP 3: Store clean patterns in session accumulation
            if session_id not in self.accumulated_patterns:
                self.accumulated_patterns[session_id] = {}
            
            # Merge with existing session patterns
            session_patterns = self.accumulated_patterns[session_id]
            
            for clean_name, pattern_data in clean_patterns.items():
                if clean_name in session_patterns:
                    # Update existing pattern - increment count
                    session_patterns[clean_name]['count'] += pattern_data['count']
                    session_patterns[clean_name]['raw_matches'].extend(pattern_data['raw_matches'])
                    # Keep highest severity
                    if pattern_data['severity'] == 'critical' or session_patterns[clean_name]['severity'] != 'critical':
                        session_patterns[clean_name]['severity'] = pattern_data['severity']
                else:
                    # Add new pattern
                    session_patterns[clean_name] = pattern_data
            
            self.accumulated_patterns[session_id] = session_patterns
            
            # STEP 4: Update analysis history
            if session_id not in self.session_analysis_history:
                self.session_analysis_history[session_id] = []
            
            self.session_analysis_history[session_id].append(parsed)
            
            # STEP 5: Calculate total risk score
            total_pattern_risk = 0
            for pattern in session_patterns.values():
                base_weight = pattern['weight']
                extra_occurrences = max(0, pattern['count'] - 1)
                diminishing_weight = extra_occurrences * (base_weight * 0.5)
                pattern_total_risk = base_weight + diminishing_weight
                total_pattern_risk += pattern_total_risk
                # Debug log to verify math
                logger.info(f"   MATH CHECK: {pattern['pattern_name']} = {base_weight}% + ({extra_occurrences} × {base_weight * 0.5}%) = {pattern_total_risk}%")

            # Cap at maximum and add baseline
            baseline_risk = 5
            calculated_risk = min(baseline_risk + total_pattern_risk, 100)
            
            # Update parsed result
            parsed['risk_score'] = int(calculated_risk)
            parsed['pattern_count'] = len(session_patterns)
            parsed['total_pattern_weight'] = int(total_pattern_risk)
            parsed['baseline_risk'] = baseline_risk
            
            # Enhanced logging
            logger.info(f"🧹 CLEAN PATTERNS: {session_id} - {len(session_patterns)} unique patterns, {calculated_risk:.0f}% total risk")
            for name, data in session_patterns.items():
                base_weight = data['weight']
                extra_occurrences = max(0, data['count'] - 1)
                diminishing_weight = extra_occurrences * (base_weight * 0.5)
                total_weight = base_weight + diminishing_weight
                
                logger.info(f"   • {name}: {base_weight}% + ({extra_occurrences} × {base_weight * 0.5}%) = {total_weight:.0f}% [{data['severity']}]")
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Pattern parsing error: {e}")
            return {'risk_score': 0.0, 'error': str(e)}
    
    # ADD THESE HELPER METHODS:
    
    def _match_specific_patterns(self, pattern_text: str) -> Optional[Dict[str, Any]]:
        """Match specific common patterns that ADK returns"""
        pattern_lower = pattern_text.lower()
        
        # Specific pattern recognition
        if 'urgency' in pattern_lower and 'pressure' in pattern_lower:
            return {'name': 'Urgency Pressure', 'weight': 15, 'severity': 'medium'}
        
        elif 'international transfer' in pattern_lower or 'overseas transfer' in pattern_lower:
            return {'name': 'Overseas Transfer', 'weight': 12, 'severity': 'medium'}
        
        elif 'romance' in pattern_lower or 'boyfriend' in pattern_lower or 'girlfriend' in pattern_lower:
            return {'name': 'Romance Exploitation', 'weight': 30, 'severity': 'critical'}
        
        elif 'emergency' in pattern_lower and ('money' in pattern_lower or 'help' in pattern_lower):
            return {'name': 'Emergency Money Request', 'weight': 25, 'severity': 'critical'}
        
        elif 'never met' in pattern_lower or 'not met' in pattern_lower:
            return {'name': 'Never Met Beneficiary', 'weight': 20, 'severity': 'high'}
        
        elif 'secret' in pattern_lower or 'don\'t tell' in pattern_lower:
            return {'name': 'Secrecy Request', 'weight': 22, 'severity': 'high'}
        
        elif 'authority' in pattern_lower or 'police' in pattern_lower or 'government' in pattern_lower:
            return {'name': 'Authority Impersonation', 'weight': 35, 'severity': 'critical'}
        
        elif 'guaranteed' in pattern_lower and ('return' in pattern_lower or 'profit' in pattern_lower):
            return {'name': 'Guaranteed Investment Returns', 'weight': 28, 'severity': 'critical'}
        
        elif 'social engineering' in pattern_lower:
            return {'name': 'Urgency Pressure', 'weight': 15, 'severity': 'medium'}  # Map social engineering to urgency
        
        return None
    
    def _match_by_keywords(self, pattern_text: str, keyword_mapping: Dict) -> Optional[Dict[str, Any]]:
        """Careful keyword matching to avoid false positives"""
        pattern_lower = pattern_text.lower()
        
        # Only match if the keyword is a significant part of the pattern
        for keyword, config in keyword_mapping.items():
            if len(keyword) >= 4:  # Only match keywords that are 4+ characters
                if keyword in pattern_lower:
                    # Additional check: make sure it's not a substring of a larger word
                    words_in_pattern = pattern_lower.split()
                    words_in_keyword = keyword.split()
                    
                    # Check if all keyword words appear in pattern
                    if all(kw in ' '.join(words_in_pattern) for kw in words_in_keyword):
                        return config
        
        return None
    
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
        """Parse policy guidance response with proper formatting for UI"""
        try:
            parsed = {}
            lines = response_text.split('\n')
            
            # Initialize arrays for UI
            immediate_alerts = []
            recommended_actions = []
            key_questions = []
            customer_education = []
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for section headers
                if 'immediate alerts:' in line.lower():
                    current_section = 'immediate_alerts'
                    # Extract content after colon
                    content = line.split(':', 1)[1].strip() if ':' in line else ''
                    if content and content != 'None':
                        immediate_alerts.append(content.strip('"'))
                elif 'recommended actions:' in line.lower():
                    current_section = 'recommended_actions'
                    content = line.split(':', 1)[1].strip() if ':' in line else ''
                    if content and content != 'None':
                        recommended_actions.append(content.strip('"'))
                elif 'key questions:' in line.lower():
                    current_section = 'key_questions'
                    content = line.split(':', 1)[1].strip() if ':' in line else ''
                    if content and content != 'None':
                        key_questions.append(content.strip('"'))
                elif 'customer education:' in line.lower():
                    current_section = 'customer_education'
                    content = line.split(':', 1)[1].strip() if ':' in line else ''
                    if content and content != 'None':
                        customer_education.append(content.strip('"'))
                elif ':' in line and not line.startswith('-'):
                    # Handle other fields
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    parsed[key] = value
                    current_section = None
                elif line.startswith('-') or line.startswith('•') and current_section:
                    # Add to current section
                    content = line.lstrip('-•').strip()
                    if content:
                        if current_section == 'immediate_alerts':
                            immediate_alerts.append(content.strip('"'))
                        elif current_section == 'recommended_actions':
                            recommended_actions.append(content.strip('"'))
                        elif current_section == 'key_questions':
                            key_questions.append(content.strip('"'))
                        elif current_section == 'customer_education':
                            customer_education.append(content.strip('"'))
                elif current_section and line:
                    # Add line content to current section
                    if line.startswith('"') and line.endswith('"'):
                        line = line.strip('"')
                    if current_section == 'immediate_alerts':
                        immediate_alerts.append(line)
                    elif current_section == 'recommended_actions':
                        recommended_actions.append(line)
                    elif current_section == 'key_questions':
                        key_questions.append(line)
                    elif current_section == 'customer_education':
                        customer_education.append(line)
            
            # Ensure required fields exist with defaults for UI
            parsed.setdefault('policy_id', 'FP-AUTO-001')
            parsed.setdefault('policy_title', 'Automated Policy Guidance')
            parsed['immediate_alerts'] = immediate_alerts or ["Review customer request carefully"]
            parsed['recommended_actions'] = recommended_actions or ["Follow standard fraud prevention procedures"]
            parsed['key_questions'] = key_questions or ["Can you verify your identity?", "Is this request unusual for you?"]
            parsed['customer_education'] = customer_education or ["Be aware of common fraud tactics", "Take time to verify requests"]
            parsed.setdefault('escalation_threshold', 60)
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Error parsing policy response: {e}")
            # Return safe defaults for UI
            return {
                'policy_id': 'FP-ERROR-001',
                'policy_title': 'Error in Policy Retrieval',
                'immediate_alerts': ["Policy system error - use manual procedures"],
                'recommended_actions': ["Follow standard fraud prevention procedures", "Escalate to supervisor if needed"],
                'key_questions': ["Can you verify your identity?", "Is this request unusual for you?"],
                'customer_education': ["Be cautious with unusual requests", "Verify before proceeding"],
                'escalation_threshold': 60,
                'error': str(e)
            }
    
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

                # Handle processing_complete in orchestrator
                if message_data.get('type') == 'processing_complete':
                    logger.info(f"🔍 ORCHESTRATOR: Triggering call completion for {session_id}")
                    await self.handle_call_completion(session_id, external_callback)
            
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
        """Process customer speech and send analysis updates to UI"""
        try:
            if session_id not in self.customer_speech_buffer:
                self.customer_speech_buffer[session_id] = []

            # Add to speech buffer            
            self.customer_speech_buffer[session_id].append({
                'text': customer_text,
                'timestamp': segment_data.get('start', 0),
                'confidence': segment_data.get('confidence', 0.0)
            })

            # DEDUPLICATION: Check if this speech segment is new enough to analyze
            if session_id not in self.analyzed_speech_segments:
                self.analyzed_speech_segments[session_id] = []
                
            # Get accumulated speech
            accumulated_speech = " ".join([
                segment['text'] for segment in self.customer_speech_buffer[session_id]
            ])

            # DEDUPLICATION: Only analyze if we have significant new content
            should_analyze = False

            if len(self.analyzed_speech_segments[session_id]) == 0:
                # First analysis
                should_analyze = True
                logger.info(f"🔍 ANALYSIS TRIGGER: First customer speech")
            
            elif len(accumulated_speech.strip()) >= 15:
                # Check if we have enough new content since last analysis
                last_analyzed_length = len(" ".join(self.analyzed_speech_segments[session_id]))
                new_content_length = len(accumulated_speech) - last_analyzed_length

                if new_content_length >= 10:  # At least 10 new characters
                    should_analyze = True
                    logger.info(f"🔍 ANALYSIS TRIGGER: {new_content_length} new characters")
                else:
                    logger.info(f"🔍 ANALYSIS SKIPPED: Only {new_content_length} new characters")

            if should_analyze:
                # Store what we're analyzing to prevent re-analysis
                self.analyzed_speech_segments[session_id].append(customer_text)
            
                # Get callback for this session
                session_data = self.active_sessions.get(session_id, {})
                callback = session_data.get('callback')
                
                # Send analysis started message
                if callback:
                    await callback({
                        'type': 'fraud_analysis_started',
                        'data': {
                            'session_id': session_id,
                            'customer_text': customer_text,
                            'accumulated_length': len(accumulated_speech),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
                # Run analysis pipeline
                analysis_result = await self._run_adk_agent_pipeline(session_id, accumulated_speech, callback)
                
                # Send analysis update to UI
                if callback and analysis_result:
                    scam_analysis = analysis_result.get('scam_analysis', {})
                    risk_score = scam_analysis.get('risk_score', 0)

                    logger.info(f"🔍 WEBSOCKET SEND: detected_patterns type = {type(self.accumulated_patterns.get(session_id, {}))}")
                    logger.info(f"🔍 WEBSOCKET SEND: detected_patterns = {self.accumulated_patterns.get(session_id, {})}")
                    
                    await callback({
                        'type': 'fraud_analysis_update',
                        'data': {
                            'session_id': session_id,
                            'risk_score': risk_score,
                            'risk_level': scam_analysis.get('risk_level', 'MINIMAL'),
                            'scam_type': scam_analysis.get('scam_type', 'unknown'),
                            'detected_patterns': self.accumulated_patterns.get(session_id, {}),
                            'confidence': scam_analysis.get('confidence', 0.0),
                            'analysis_trigger': 'new_content',                            
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    
                    # Send policy guidance if available
                    policy_guidance = analysis_result.get('policy_guidance')
                    if policy_guidance:
                        await callback({
                            'type': 'policy_guidance_ready',
                            'data': {
                                'session_id': session_id,
                                'policy_guidance': policy_guidance,
                                'timestamp': datetime.now().isoformat()
                            }
                        })

                    # NEW: Trigger question prompt after analysis update
                    await self._trigger_question_prompt(
                        session_id, accumulated_speech, 
                        self.accumulated_patterns.get(session_id, {}),
                        risk_score, callback
                    )

            else:
                logger.info(f"🔍 SPEECH DEDUPLICATION: Skipped analysis for similar content")
         
        except Exception as e:
            logger.error(f"❌ Error processing customer speech: {e}")

    async def _trigger_question_prompt(self, session_id: str, customer_text: str, detected_patterns: Dict, risk_score: float, callback: Optional[Callable]):
        """Simple question triggering with debug logging"""
        try:
            # DEBUG: Check current state
            current_active = self.active_questions.get(session_id)
            print(f"🔍 QUESTION STATE DEBUG:")
            print(f"  - session_id: {session_id}")
            print(f"  - current_active_question: {current_active}")
            print(f"  - active_questions dict: {self.active_questions}")
            
            # LAYER 1: Only one question active per session
            if current_active is not None:
                print(f"❌ QUESTION BLOCKED: Active question exists: {current_active}")
                return  # Question already active, don't send another
            
            print(f"✅ QUESTION CHECK - No active question, proceeding")
            print(f"  - patterns: {list(detected_patterns.keys())}")
            print(f"  - risk_score: {risk_score}")
            
            # Get best question
            from ..config.question_triggers import select_best_question
            best_question = select_best_question(detected_patterns, risk_score, customer_text)
            
            if not best_question:
                print(f"❌ NO QUESTION FOUND")
                return
                
            if not callback:
                print(f"❌ NO CALLBACK AVAILABLE")
                return
            
            # Generate simple question ID
            question_id = f"q_{session_id}_{int(time.time())}"
            
            # Mark as active BEFORE sending
            self.active_questions[session_id] = question_id
            print(f"✅ MARKED ACTIVE: {question_id} for session {session_id}")
            print(f"✅ Updated active_questions: {self.active_questions}")
            
            print(f"✅ SENDING QUESTION: {best_question['question']}")
            
            # Send to UI
            await callback({
                'type': 'question_prompt_ready',
                'data': {
                    'session_id': session_id,
                    'question': best_question['question'],
                    'context': best_question['context'],
                    'urgency': best_question['urgency'],
                    'pattern': best_question['pattern'],
                    'risk_level': risk_score,
                    'question_id': question_id,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            print(f"✅ QUESTION SENT SUCCESSFULLY")
            
        except Exception as e:
            print(f"❌ QUESTION ERROR: {e}")
            # Clear active question on error
            if session_id in self.active_questions:
                self.active_questions[session_id] = None
                print(f"🧹 CLEARED ACTIVE QUESTION due to error")
    
    def mark_question_answered(self, session_id: str, question_id: str, action: str):
        """Clear active question when answered/skipped with debug logging"""
        print(f"🔄 MARK_QUESTION_ANSWERED called:")
        print(f"  - session_id: {session_id}")
        print(f"  - question_id: {question_id}")
        print(f"  - action: {action}")
        print(f"  - before: active_questions = {self.active_questions}")
        
        if session_id in self.active_questions:
            self.active_questions[session_id] = None
            print(f"✅ CLEARED active question for {session_id}")
        else:
            print(f"⚠️ Session {session_id} not found in active_questions")
        
        print(f"  - after: active_questions = {self.active_questions}")
        print(f"✅ QUESTION {action.upper()}: Ready for new questions in {session_id}")

    async def _run_case_management(self, session_id: str, analysis: Dict, callback: Optional[Callable]):
        """Run case management with proper UI updates"""
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
            
            # Send case creation update to UI
            if callback:
                await callback({
                    'type': 'case_created',
                    'data': {
                        'session_id': session_id,
                        'case_number': case_result.get('incident_number', 'Unknown'),
                        'case_url': case_result.get('incident_url', ''),
                        'success': case_result.get('success', False),
                        'case_id': case_result.get('incident_sys_id', ''),
                        'priority': case_result.get('priority', 'Medium'),
                        'risk_score': analysis.get('risk_score', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
        except Exception as e:
            logger.error(f"❌ Case management error: {e}")
            # Send error to UI
            if callback:
                await callback({
                    'type': 'case_creation_error',
                    'data': {
                        'session_id': session_id,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                })
    
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
            # Cleanup demo task
            if hasattr(self, 'demo_tasks') and session_id in self.demo_tasks:
                demo_task = self.demo_tasks[session_id]
                if not demo_task.done():
                    demo_task.cancel()
                del self.demo_tasks[session_id]
            
            # Stop demo orchestrator for this session
            if hasattr(self, 'demo_orchestrator'):
                await self.demo_orchestrator.stop_session_demo(session_id)      
                
            # Existing cleanup code
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.customer_speech_buffer:
                del self.customer_speech_buffer[session_id]
            if session_id in self.session_analysis_history:
                del self.session_analysis_history[session_id]
            if session_id in self.accumulated_patterns:
                del self.accumulated_patterns[session_id]
            if session_id in self.active_questions:
                del self.active_questions[session_id]       

            if session_id in self.analyzed_speech_segments:
                del self.analyzed_speech_segments[session_id]
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    # ===== STATUS AND MONITORING =====
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status with safe attribute access"""
        try:
            from ..utils import get_current_timestamp
        except ImportError:
            def get_current_timestamp():
                return datetime.now().isoformat()
        
        # Safe attribute access with defaults
        agents = getattr(self, 'agents', {})
        runners = getattr(self, 'runners', {})
        session_service = getattr(self, 'session_service', None)
        case_management_agent = getattr(self, 'case_management_agent', None)
        
        agents_managed = list(agents.keys())
        if case_management_agent is not None:
            agents_managed.append('case_management')
        
        return {
            'orchestrator_type': 'FraudDetectionOrchestrator',
            'system_status': 'operational' if session_service else 'degraded',
            'active_sessions': len(getattr(self, 'active_sessions', {})),
            'agents_available': len(agents) > 0,
            'adk_session_service': session_service is not None,
            'adk_runners': len(runners),
            'agents_managed': agents_managed,
            'capabilities': [
                'ADK Agent execution with proper session management',
                'Multi-agent orchestration',
                'Session state management', 
                'Pipeline coordination',
                'Audio and text processing',
                'Real-time and batch analysis',
                'WebSocket and REST API support',
                'Proper warning suppression'
            ],
            'statistics': {
                'total_sessions': len(getattr(self, 'active_sessions', {})),
                'total_analyses': sum(len(history) for history in getattr(self, 'session_analysis_history', {}).values()),
                'total_patterns': sum(len(patterns) for patterns in getattr(self, 'accumulated_patterns', {}).values())
            },
            'adk_info': {
                'session_service_active': session_service is not None,
                'agents_initialized': len(agents),
                'runners_created': len(runners),
                'warning_suppression': True,
                'case_management_available': case_management_agent is not None
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
            'adk_execution': True
        }

class FraudDetectionWebSocketHandler:
    """WebSocket Handler that delegates to the fixed ADK orchestrator"""
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
        self.orchestrator = FraudDetectionOrchestrator(connection_manager)
        logger.info("🔌 WebSocket Handler initialized with fixed ADK orchestrator")

    async def handle_question_answered(self, websocket, client_id: str, data: Dict) -> None:
        """Handle agent question answered/skipped"""
        session_id = data.get('session_id')
        question_id = data.get('question_id') 
        action = data.get('action')  # 'asked' or 'skipped'
        
        print(f"🔄 WEBSOCKET: handle_question_answered called")
        print(f"  - session_id: {session_id}")
        print(f"  - question_id: {question_id}")
        print(f"  - action: {action}")
        
        if not session_id or not question_id:
            print(f"❌ WEBSOCKET: Missing session_id or question_id")
            await self.send_error(websocket, client_id, "Missing session_id or question_id")
            return
        
        try:
            print(f"🔄 WEBSOCKET: Calling orchestrator.mark_question_answered")
            
            # THIS IS THE KEY CALL - Make sure it happens
            self.orchestrator.mark_question_answered(session_id, question_id, action)
            
            print(f"✅ WEBSOCKET: mark_question_answered completed")
            
            # Send confirmation back
            await self.send_message(websocket, client_id, {
                'type': 'question_answer_acknowledged',
                'data': {
                    'session_id': session_id,
                    'question_id': question_id,
                    'action': action,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            print(f"✅ WEBSOCKET: Confirmation sent")
            
        except Exception as e:
            print(f"❌ WEBSOCKET: Error handling question answer: {e}")
            await self.send_error(websocket, client_id, f"Failed to handle question answer: {str(e)}")
    
    async def handle_message(self, websocket, client_id: str, message: Dict[str, Any]) -> None:
        """Handle WebSocket messages"""
        message_type = message.get('type')
        data = message.get('data', {})

        # ADD THIS DEBUG LINE
        logger.info(f"🔍 WebSocket received: {message_type} for session: {data.get('session_id', 'unknown')}")
    
        try:
            if message_type in ['process_audio', 'start_realtime_session']:
                await self.handle_audio_processing(websocket, client_id, data)
            elif message_type == 'stop_processing':
                await self.handle_stop_processing(websocket, client_id, data)
            #elif message_type == 'create_manual_case':
            #    await self.handle_manual_case_creation(websocket, client_id, data)
            elif message_type == 'processing_complete':
                await self.handle_audio_completion(websocket, client_id, data)      
            elif message_type == 'agent_question_answered':  # MAKE SURE THIS IS HERE
                print(f"🔄 WEBSOCKET: Routing to handle_question_answered")
                await self.handle_question_answered(websocket, client_id, data)                
            else:
                await self.send_error(websocket, client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ WebSocket handler error: {e}")
            await self.send_error(websocket, client_id, f"Message processing error: {str(e)}")

    async def handle_audio_completion(self, websocket, client_id: str, data: Dict) -> None:
        """Handle audio processing completion and trigger call completion"""
        session_id = data.get('session_id')
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            logger.info(f"🏁 Audio processing completed, triggering call completion for {session_id}")
            
            # Create WebSocket callback
            async def websocket_callback(message_data: Dict) -> None:
                await self.send_message(websocket, client_id, message_data)
            
            # Trigger call completion
            result = await self.orchestrator.handle_call_completion(
                session_id=session_id,
                callback=websocket_callback
            )
            
            if not result.get('success'):
                await self.send_error(websocket, client_id, result.get('error', 'Call completion failed'))
                return
            
            # Send final confirmation
            await self.send_message(websocket, client_id, {
                'type': 'call_completion_result',
                'data': {
                    'session_id': session_id,
                    'success': True,
                    'final_risk_score': result.get('final_risk_score', 0),
                    'case_created': result.get('case_created', False),
                    'timestamp': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Audio completion error: {e}")
            await self.send_error(websocket, client_id, f"Audio completion failed: {str(e)}")
            
    async def handle_call_completion(self, websocket, client_id: str, data: Dict) -> None:
        """Handle call completion request"""
        session_id = data.get('session_id')
        if not session_id:
            await self.send_error(websocket, client_id, "Missing session_id")
            return
        
        try:
            # Create WebSocket callback
            async def websocket_callback(message_data: Dict) -> None:
                await self.send_message(websocket, client_id, message_data)
            
            # Delegate to orchestrator
            result = await self.orchestrator.handle_call_completion(
                session_id=session_id,
                callback=websocket_callback
            )
            
            if not result.get('success'):
                await self.send_error(websocket, client_id, result.get('error', 'Call completion failed'))
                return
            
            # Send final confirmation
            await self.send_message(websocket, client_id, {
                'type': 'call_completion_result',
                'data': {
                    'session_id': session_id,
                    'success': True,
                    'final_risk_score': result.get('final_risk_score', 0),
                    'case_created': result.get('case_created', False),
                    'timestamp': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Call completion error: {e}")
            await self.send_error(websocket, client_id, f"Call completion failed: {str(e)}")
    
    async def handle_audio_processing(self, websocket, client_id: str, data: Dict) -> None:
        """Handle audio processing with ADK orchestrator"""
        filename = data.get('filename')
        session_id = data.get('session_id')
        
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
        
        try:
            # Create WebSocket callback
            async def websocket_callback(message_data: Dict) -> None:
                await self.send_message(websocket, client_id, message_data)
            
            # Delegate to fixed ADK orchestrator
            result = await self.orchestrator.orchestrate_audio_processing(
                filename=filename,
                session_id=session_id,
                callback=websocket_callback
            )
            
            if not result.get('success'):
                await self.send_error(websocket, client_id, result.get('error', 'Processing failed'))
                return
            
            # Send confirmation with ADK info
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'session_id': result['session_id'],
                    'filename': filename,
                    'orchestrator': 'FraudDetectionOrchestrator',
                    'handler': 'WebSocketHandler',
                    'pipeline_mode': 'adk_multi_agent',
                    'adk_session_service': result.get('adk_session_service', False),
                    'agents_available': result.get('agents_available', False),
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
                    'orchestrator': 'FraudDetectionOrchestrator',
                    'adk_cleanup': True,
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
                    'orchestrator': 'FraudDetectionOrchestrator',
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
                            'orchestrator': 'FraudDetectionOrchestrator',
                            'adk_execution': True,
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
            'orchestrator_delegation': 'FraudDetectionOrchestrator',
            'adk_execution': True
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

# Export the fixed classes
__all__ = ['FraudDetectionOrchestrator', 'FraudDetectionWebSocketHandler']
