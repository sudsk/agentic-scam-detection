# backend/agents/fraud_detection_system.py 
"""
Main Fraud Detection System Coordinator 
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import the UNIFIED audio agent (now with real Google STT)
from .audio_processor.agent import audio_processor_agent  # ← Now has REAL Google STT
from .scam_detection.agent import fraud_detection_agent
from .policy_guidance.agent import policy_guidance_agent
from .case_management.agent import case_management_agent

logger = logging.getLogger(__name__)

class FraudDetectionSystem:
    """
    Main system orchestrating all fraud detection agents
    UNIFIED: Now uses single audio agent with REAL Google STT
    """
    
    def __init__(self):
        # Agent references - UNIFIED to use the real audio agent
        self.audio_processor = audio_processor_agent  # ← Now has REAL Google STT
        self.fraud_detector = fraud_detection_agent
        self.policy_guide = policy_guidance_agent
        self.case_manager = case_management_agent
        
        # System state
        self.session_data = {}
        self.processing_queue = {}
        
        logger.info("🚀 Unified Fraud Detection System initialized")
        logger.info(f"  - {self.audio_processor.agent_name} (REAL Google STT)")
        logger.info(f"  - {self.fraud_detector.agent_name}")
        logger.info(f"  - {self.policy_guide.agent_name}")
        logger.info(f"  - {self.case_manager.agent_name}")
    
    async def process_audio_file(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """
        Process audio file through complete multi-agent pipeline using UNIFIED agent
        
        Flow: UNIFIED Audio Processing → Fraud Detection → Policy Guidance → Case Management
        """
        try:
            logger.info(f"🔄 Starting UNIFIED pipeline with REAL Google STT for: {file_path}")
            
            # STEP 1: Audio Processing using UNIFIED agent with REAL Google STT
            logger.info("📝 Step 1: UNIFIED Audio Processing Agent (REAL Google STT)")
            
            # Create a simple callback for the unified agent
            transcription_segments = []
            
            async def simple_callback(message_data: Dict) -> None:
                """Simple callback to collect transcription segments"""
                if message_data.get('type') == 'transcription_segment':
                    segment_data = message_data.get('data', {})
                    transcription_segments.append({
                        'speaker': segment_data.get('speaker', 'unknown'),
                        'start': segment_data.get('start', 0),
                        'end': segment_data.get('end', 0),
                        'text': segment_data.get('text', ''),
                        'confidence': segment_data.get('confidence', 0.0)
                    })
                    logger.info(f"📝 Collected segment: {segment_data.get('speaker')} - {segment_data.get('text', '')[:50]}...")
            
            # Extract filename from path
            from pathlib import Path
            filename = Path(file_path).name
            
            # Use the UNIFIED agent with REAL Google STT
            logger.info(f"🎯 Using UNIFIED agent with REAL Google STT for: {filename}")
            processing_result = await self.audio_processor.start_realtime_processing(
                session_id=session_id,
                audio_filename=filename,
                websocket_callback=simple_callback
            )
            
            if 'error' in processing_result:
                logger.error(f"❌ UNIFIED Google STT failed: {processing_result['error']}")
                return {
                    "session_id": session_id,
                    "status": "error",
                    "error": processing_result['error'],
                    "pipeline_stage": "unified_audio_processing"
                }
            
            # Wait for processing to complete and collect segments
            logger.info("⏳ Waiting for UNIFIED Google STT processing to complete...")
            await asyncio.sleep(3.0)  # Give time for processing
            
            # Stop the processing
            await self.audio_processor.stop_realtime_processing(session_id)
            
            logger.info(f"✅ UNIFIED Google STT completed: {len(transcription_segments)} segments")
            
            # Extract customer speech segments
            customer_segments = [
                seg for seg in transcription_segments
                if seg["speaker"] == "customer"
            ]
            
            if not customer_segments:
                logger.warning("⚠️ No customer speech detected in UNIFIED transcription")
                return {
                    "session_id": session_id,
                    "status": "no_customer_speech", 
                    "message": "No customer speech detected by UNIFIED Google STT",
                    "pipeline_stage": "unified_audio_processing",
                    "transcription_segments": transcription_segments
                }
            
            # Combine customer text for analysis
            customer_text = " ".join(seg["text"] for seg in customer_segments)
            logger.info(f"📝 Customer text extracted: '{customer_text[:100]}...'")
            
            # STEP 2: Fraud Detection - Analyze for fraud patterns
            logger.info("🔍 Step 2: Fraud Detection Agent")
            fraud_analysis = self.fraud_detector.process(customer_text)
            
            # STEP 3: Policy Guidance - Get response procedures (if risk detected)
            policy_guidance = {}
            if fraud_analysis["risk_score"] >= 40:
                logger.info("📚 Step 3: Policy Guidance Agent")
                policy_guidance = self.policy_guide.process({
                    'scam_type': fraud_analysis["scam_type"],
                    'risk_score': fraud_analysis["risk_score"]
                })
            else:
                logger.info("📚 Step 3: Policy Guidance Agent - Skipped (low risk)")
            
            # STEP 4: Case Management - Create case (if high risk)
            case_info = None
            if fraud_analysis["risk_score"] >= 60:
                logger.info("📋 Step 4: Case Management Agent")
                case_info = self.case_manager.process({
                    "session_id": session_id,
                    "fraud_analysis": fraud_analysis,
                    "customer_id": None
                })
            else:
                logger.info("📋 Step 4: Case Management Agent - Skipped (medium/low risk)")
            
            # STEP 5: Make final decision
            logger.info("🎭 Step 5: Orchestrator Decision")
            orchestrator_decision = self._make_orchestrator_decision(
                fraud_analysis, policy_guidance, case_info
            )
            
            # Compile comprehensive results
            result = {
                "session_id": session_id,
                "pipeline_complete": True,
                "processing_mode": "unified_google_stt",
                "agents_involved": self._get_agents_involved(fraud_analysis["risk_score"]),
                "transcription": {
                    "speaker_segments": transcription_segments,
                    "total_duration": max(seg["end"] for seg in transcription_segments) if transcription_segments else 0,
                    "speakers_detected": list(set(seg["speaker"] for seg in transcription_segments)),
                    "transcription_source": "unified_google_stt_streaming",
                    "processing_agent": self.audio_processor.agent_id
                },
                "fraud_analysis": fraud_analysis,
                "policy_guidance": policy_guidance.get("agent_guidance", {}),
                "orchestrator_decision": orchestrator_decision,
                "case_info": case_info,
                "processing_summary": {
                    "total_agents": len(self._get_agents_involved(fraud_analysis["risk_score"])),
                    "risk_level": fraud_analysis["risk_level"],
                    "escalation_required": fraud_analysis["risk_score"] >= 80,
                    "case_created": case_info is not None,
                    "unified_transcription": True,
                    "google_stt_used": True
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Store session data
            self.session_data[session_id] = result
            
            logger.info(f"✅ UNIFIED Google STT pipeline complete: Risk {fraud_analysis['risk_score']}% - {orchestrator_decision['decision']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ UNIFIED Google STT pipeline failed: {e}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "pipeline_stage": "unknown",
                "processing_mode": "unified_google_stt_failed",
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_agents_involved(self, risk_score: float) -> list:
        """Get list of agents involved based on risk score"""
        agents = ["unified_audio_processing", "fraud_detection"]
        
        if risk_score >= 40:
            agents.append("policy_guidance")
        
        if risk_score >= 60:
            agents.append("case_management")
        
        agents.append("orchestrator")
        return agents
    
    def _make_orchestrator_decision(
        self, 
        fraud_analysis: Dict, 
        policy_guidance: Dict, 
        case_info: Optional[Dict]
    ) -> Dict[str, Any]:
        """Make final orchestrator decision based on all agent inputs"""
        risk_score = fraud_analysis.get("risk_score", 0)
        scam_type = fraud_analysis.get("scam_type", "unknown")
        
        # Decision matrix based on risk score
        if risk_score >= 80:
            decision = "BLOCK_AND_ESCALATE"
            reasoning = "Critical fraud risk detected by UNIFIED Google STT - immediate intervention required"
            actions = [
                "Block all pending transactions immediately",
                "Escalate to Financial Crime Team",
                "Initiate fraud investigation",
                "Document all evidence from UNIFIED transcription",
                "Contact customer via secure channel"
            ]
            priority = "CRITICAL"
            
        elif risk_score >= 60:
            decision = "VERIFY_AND_MONITOR"
            reasoning = "High fraud risk from UNIFIED transcription - enhanced verification required"
            actions = [
                "Require additional customer verification",
                "Monitor account for 48 hours",
                "Flag for supervisor review",
                "Provide fraud education to customer",
                "Hold suspicious transactions for review"
            ]
            priority = "HIGH"
            
        elif risk_score >= 40:
            decision = "MONITOR_AND_EDUCATE"
            reasoning = "Moderate risk from UNIFIED transcription - increased vigilance recommended"
            actions = [
                "Continue normal processing with caution",
                "Provide relevant fraud warnings",
                "Document conversation details",
                "Consider follow-up contact",
                "Monitor for pattern escalation"
            ]
            priority = "MEDIUM"
            
        else:
            decision = "CONTINUE_NORMAL"
            reasoning = "Low fraud risk from UNIFIED transcription - standard processing"
            actions = [
                "Continue normal customer service",
                "Standard documentation procedures",
                "No special monitoring required"
            ]
            priority = "LOW"
        
        # Enhance decision based on agent inputs
        if policy_guidance:
            escalation_advice = policy_guidance.get("escalation_guidance", {})
            if escalation_advice.get("level") == "immediate_escalation":
                priority = "CRITICAL"
        
        if case_info:
            actions.append(f"Case created: {case_info.get('case_id', 'Unknown')}")
        
        return {
            "decision": decision,
            "reasoning": reasoning,
            "actions": actions,
            "priority": priority,
            "risk_score": risk_score,
            "scam_type": scam_type,
            "confidence": fraud_analysis.get("confidence", 0.9),
            "case_created": case_info is not None,
            "agents_consulted": len(self._get_agents_involved(risk_score)),
            "transcription_source": "unified_google_stt_streaming",
            "unified_transcription": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all agents"""
        from .shared.base_agent import agent_registry
        
        # Get registry status
        registry_status = agent_registry.get_system_status()
        
        return {
            "system_status": "operational",
            "framework": "UNIFIED Multi-Agent Fraud Detection with Google STT",
            "agents_count": 4,
            "agents": {
                "unified_audio_processing": "active",  # UNIFIED processor
                "fraud_detection": "active",
                "policy_guidance": "active", 
                "case_management": "active"
            },
            "agent_details": {
                "unified_audio_processing": {
                    "status": "active",
                    "type": "AudioProcessorAgent",
                    "transcription_engine": "google_stt_streaming",
                    "google_client_available": self.audio_processor.google_client is not None,
                    "active_sessions": len(self.audio_processor.active_sessions)
                },
                "fraud_detection": self.fraud_detector.get_status() if hasattr(self.fraud_detector, 'get_status') else {"status": "active"},
                "policy_guidance": self.policy_guide.get_status() if hasattr(self.policy_guide, 'get_status') else {"status": "active"},
                "case_management": self.case_manager.get_status() if hasattr(self.case_manager, 'get_status') else {"status": "active"}
            },
            "registry_status": registry_status,
            "session_count": len(self.session_data),
            "case_statistics": self.case_manager.case_statistics if hasattr(self.case_manager, 'case_statistics') else {},
            "unified_transcription": True,
            "google_stt_enabled": True,
            "last_updated": datetime.now().isoformat()
        }
    
    # ... (rest of the methods remain the same)

# Initialize the global system with UNIFIED processor
fraud_detection_system = FraudDetectionSystem()

# Main coordination function for API compatibility using UNIFIED transcription
def coordinate_fraud_analysis(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinate fraud analysis workflow for API endpoints using UNIFIED Google STT
    """
    session_id = transcript_data.get('session_id')
    text = transcript_data.get('text', '')
    speaker = transcript_data.get('speaker', 'unknown')
    
    logger.info(f"🎭 Coordinating fraud analysis for session {session_id} (UNIFIED transcription mode)")
    
    # Skip analysis for agent speech
    if speaker != 'customer':
        return {
            'session_id': session_id,
            'analysis_skipped': True,
            'reason': 'Agent speech - no analysis needed',
            'transcription_mode': 'unified_google_stt',
            'timestamp': datetime.now().isoformat()
        }
    
    # Use the same fraud detection workflow
    fraud_analysis = fraud_detection_system.fraud_detector.process(text)
    
    policy_guidance = {}
    if fraud_analysis['risk_score'] >= 40:
        policy_guidance = fraud_detection_system.policy_guide.process({
            'scam_type': fraud_analysis['scam_type'],
            'risk_score': fraud_analysis['risk_score']
        })
    
    case_info = None
    if fraud_analysis['risk_score'] >= 60:
        case_info = fraud_detection_system.case_manager.process({
            "session_id": session_id,
            "fraud_analysis": fraud_analysis,
            "customer_id": None
        })
    
    orchestrator_decision = fraud_detection_system._make_orchestrator_decision(
        fraud_analysis, policy_guidance, case_info
    )
    
    result = {
        'session_id': session_id,
        'fraud_detection': fraud_analysis,
        'policy_guidance': policy_guidance.get('agent_guidance', {}),
        'orchestrator_decision': orchestrator_decision,
        'case_info': case_info,
        'requires_escalation': fraud_analysis['risk_score'] >= 80,
        'agents_involved': fraud_detection_system._get_agents_involved(fraud_analysis['risk_score']),
        'transcription_mode': 'unified_google_stt',
        'unified_transcription': True,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"✅ Coordination complete: {orchestrator_decision['decision']} (UNIFIED transcription)")
    return result
