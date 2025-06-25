# agents/complete_scam_detection_system.py
"""
Complete HSBC Scam Detection Multi-Agent System using Google ADK
This module integrates all agents and provides the main orchestration logic
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Google ADK imports
from google.adk.agents import Agent
from google.adk.models import LiteLLM
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# Import agent modules
from scam_detection.agent import analyze_fraud_patterns, live_scam_detection_agent
from rag_policy.agent import retrieve_hsbc_policies, rag_policy_agent
from orchestrator.agent import coordinate_fraud_analysis

logger = logging.getLogger(__name__)

# Audio Processing Agent
def process_audio_stream(audio_data: bytes, session_id: str) -> Dict[str, Any]:
    """
    Process audio stream and convert to text
    
    Args:
        audio_data: Raw audio bytes
        session_id: Session identifier
        
    Returns:
        Dict with transcription results
    """
    logger.info(f"🎵 Processing audio stream for session {session_id}")
    
    # In production, this would use Google Speech-to-Text
    # For demo, return mock transcription based on filename
    mock_transcriptions = {
        "investment_scam": "My investment advisor just called saying there's a margin call on my trading account. I need to transfer fifteen thousand pounds immediately or I'll lose all my investments. He's been guaranteeing thirty-five percent monthly returns.",
        "romance_scam": "I need to send four thousand pounds to Turkey. My partner Alex is stuck in Istanbul and needs money for accommodation. We've been together for seven months online.",
        "impersonation_scam": "Someone from HSBC security called saying my account has been compromised. They asked me to confirm my card details and PIN to verify my identity immediately.",
        "legitimate": "Hi, I'd like to check my account balance and see if my salary has been deposited yet. I'm also interested in opening a savings account."
    }
    
    # Mock implementation - in production, use actual speech-to-text
    transcription_text = mock_transcriptions.get("investment_scam", "Sample transcription")
    
    result = {
        "session_id": session_id,
        "transcription": {
            "text": transcription_text,
            "speaker": "customer",
            "confidence": 0.95,
            "language": "en-GB",
            "duration_seconds": 15.2,
            "word_timestamps": []
        },
        "audio_metadata": {
            "format": "wav",
            "sample_rate": 16000,
            "channels": 1,
            "duration": 15.2
        },
        "processing_time_ms": 250,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"✅ Audio processed: {len(transcription_text)} characters transcribed")
    return result

def transcribe_audio(audio_file_path: str) -> Dict[str, Any]:
    """
    Transcribe audio file with speaker diarization
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Dict with detailed transcription including speaker segments
    """
    logger.info(f"📝 Transcribing audio file: {audio_file_path}")
    
    # Mock transcription based on filename
    filename = Path(audio_file_path).name.lower()
    
    if "investment" in filename:
        segments = [
            {"speaker": "customer", "start": 0.0, "end": 5.0, 
             "text": "My investment advisor just called saying there's a margin call on my trading account."},
            {"speaker": "agent", "start": 5.5, "end": 7.0, 
             "text": "I see. Can you tell me more about this advisor?"},
            {"speaker": "customer", "start": 7.5, "end": 12.0, 
             "text": "He's been guaranteeing thirty-five percent monthly returns and says I need to transfer fifteen thousand pounds immediately."}
        ]
    elif "romance" in filename:
        segments = [
            {"speaker": "customer", "start": 0.0, "end": 4.0,
             "text": "I need to send four thousand pounds to Turkey urgently."},
            {"speaker": "agent", "start": 4.5, "end": 6.0,
             "text": "May I ask who this payment is for?"},
            {"speaker": "customer", "start": 6.5, "end": 10.0,
             "text": "My partner Alex is stuck in Istanbul. We've been together for seven months online."}
        ]
    else:
        segments = [
            {"speaker": "customer", "start": 0.0, "end": 3.0,
             "text": "Hi, I'd like to check my account balance please."}
        ]
    
    result = {
        "file_path": audio_file_path,
        "total_duration": sum(s["end"] - s["start"] for s in segments),
        "speaker_segments": segments,
        "speakers_detected": list(set(s["speaker"] for s in segments)),
        "primary_language": "en-GB",
        "confidence_score": 0.94,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"✅ Transcription complete: {len(segments)} segments")
    return result

# Create Audio Processing Agent
audio_processing_agent = Agent(
    name="audio_processing_agent",
    model=LiteLLM("gemini/gemini-2.0-flash"),
    description="Audio stream processing and transcription agent",
    instruction="""You are an audio processing specialist for HSBC call center.

Your responsibilities:
1. Process real-time audio streams from customer calls
2. Convert speech to text with high accuracy
3. Identify speakers (customer vs agent)
4. Handle multiple audio formats and quality levels

When you receive audio data:
- Use process_audio_stream for real-time processing
- Use transcribe_audio for file-based transcription
- Ensure speaker diarization is accurate
- Maintain high transcription quality even with background noise""",
    tools=[process_audio_stream, transcribe_audio]
)

# Case Management Agent
def create_fraud_case(
    session_id: str,
    fraud_analysis: Dict[str, Any],
    customer_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a fraud investigation case in Quantexa
    
    Args:
        session_id: Call session ID
        fraud_analysis: Fraud detection results
        customer_id: Customer identifier
        
    Returns:
        Dict with case details
    """
    logger.info(f"📋 Creating fraud case for session {session_id}")
    
    risk_score = fraud_analysis.get("risk_score", 0)
    scam_type = fraud_analysis.get("scam_type", "unknown")
    
    # Generate case ID
    case_id = f"HSBC-FD-{datetime.now().strftime('%Y%m%d')}-{session_id[:8].upper()}"
    
    # Determine priority based on risk score
    if risk_score >= 80:
        priority = "critical"
        assigned_team = "financial-crime-immediate"
    elif risk_score >= 60:
        priority = "high"
        assigned_team = "fraud-investigation-team"
    elif risk_score >= 40:
        priority = "medium"
        assigned_team = "fraud-prevention-team"
    else:
        priority = "low"
        assigned_team = "customer-service-enhanced"
    
    case_data = {
        "case_id": case_id,
        "session_id": session_id,
        "customer_id": customer_id or f"CUST-{session_id[:6]}",
        "fraud_type": scam_type,
        "risk_score": risk_score,
        "priority": priority,
        "status": "open",
        "assigned_team": assigned_team,
        "created_at": datetime.now().isoformat(),
        "evidence": {
            "transcript": fraud_analysis.get("transcript_text", ""),
            "detected_patterns": fraud_analysis.get("detected_patterns", {}),
            "risk_factors": fraud_analysis.get("explanation", "")
        },
        "quantexa_integration": {
            "entity_resolution": "pending",
            "network_analysis": "queued",
            "historical_patterns": "analyzing"
        },
        "next_actions": [
            "Review customer transaction history",
            "Check for similar cases in network",
            "Validate customer identity",
            "Monitor account for 48 hours"
        ]
    }
    
    # In production, this would integrate with Quantexa API
    logger.info(f"✅ Case created: {case_id} - Priority: {priority}")
    
    return case_data

# Create Case Management Agent
case_management_agent = Agent(
    name="case_management_agent",
    model=LiteLLM("gemini/gemini-2.0-flash"),
    description="Fraud case creation and management with Quantexa integration",
    instruction="""You are a case management specialist for HSBC fraud investigations.

Your responsibilities:
1. Create investigation cases for detected fraud
2. Integrate with Quantexa for entity resolution
3. Assign cases to appropriate teams
4. Track case progress and outcomes

When fraud is detected:
- Create detailed investigation cases
- Ensure all evidence is properly documented
- Assign to the right team based on severity
- Set up monitoring and follow-up actions""",
    tools=[create_fraud_case]
)

# Compliance Agent
def check_compliance_requirements(
    transcript: str,
    fraud_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check GDPR, FCA and compliance requirements
    
    Args:
        transcript: Call transcript
        fraud_analysis: Fraud detection results
        
    Returns:
        Dict with compliance checks and recommendations
    """
    logger.info("🔒 Checking compliance requirements")
    
    compliance_checks = {
        "gdpr_compliance": {
            "pii_detected": False,
            "pii_types": [],
            "redaction_required": False,
            "consent_verified": True
        },
        "fca_compliance": {
            "treating_customers_fairly": True,
            "vulnerability_detected": False,
            "appropriate_warnings_given": True,
            "documented_properly": True
        },
        "data_retention": {
            "retention_period_days": 2555,  # 7 years for financial records
            "deletion_scheduled": False,
            "audit_trail_created": True
        }
    }
    
    # Check for PII in transcript
    pii_patterns = {
        "card_number": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        "pin": r'\b(pin|PIN)\s*(number|code)?[\s:]*\d{4,6}\b',
        "password": r'\b(password|passcode)\s*[:=]\s*\S+',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b'
    }
    
    import re
    for pii_type, pattern in pii_patterns.items():
        if re.search(pattern, transcript, re.IGNORECASE):
            compliance_checks["gdpr_compliance"]["pii_detected"] = True
            compliance_checks["gdpr_compliance"]["pii_types"].append(pii_type)
            compliance_checks["gdpr_compliance"]["redaction_required"] = True
    
    # Check vulnerability indicators
    vulnerability_indicators = [
        "elderly", "confused", "don't understand", "medical condition",
        "mental health", "recently bereaved", "financial difficulty"
    ]
    
    transcript_lower = transcript.lower()
    for indicator in vulnerability_indicators:
        if indicator in transcript_lower:
            compliance_checks["fca_compliance"]["vulnerability_detected"] = True
            break
    
    # Generate recommendations
    recommendations = []
    
    if compliance_checks["gdpr_compliance"]["pii_detected"]:
        recommendations.append("Redact PII from transcript before storage")
        recommendations.append("Ensure customer consent for data processing")
    
    if compliance_checks["fca_compliance"]["vulnerability_detected"]:
        recommendations.append("Apply enhanced customer protection measures")
        recommendations.append("Consider specialist support team involvement")
    
    if fraud_analysis.get("risk_score", 0) >= 60:
        recommendations.append("File Suspicious Activity Report (SAR)")
        recommendations.append("Notify Money Laundering Reporting Officer (MLRO)")
    
    result = {
        "compliance_checks": compliance_checks,
        "recommendations": recommendations,
        "regulatory_requirements": {
            "sar_required": fraud_analysis.get("risk_score", 0) >= 60,
            "fca_notification": compliance_checks["fca_compliance"]["vulnerability_detected"],
            "gdpr_article_17": False,  # Right to erasure
            "audit_log_created": True
        },
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"✅ Compliance check complete: {len(recommendations)} recommendations")
    return result

# Create Compliance Agent
compliance_agent = Agent(
    name="compliance_agent",
    model=LiteLLM("gemini/gemini-2.0-flash"),
    description="Regulatory compliance and data protection agent",
    instruction="""You are a compliance specialist ensuring HSBC meets all regulatory requirements.

Your responsibilities:
1. Check GDPR compliance and PII handling
2. Ensure FCA treating customers fairly principles
3. Identify vulnerable customers
4. Generate regulatory reports

Always prioritize:
- Customer data protection
- Regulatory compliance
- Vulnerable customer identification
- Proper documentation for audits""",
    tools=[check_compliance_requirements]
)

# Master Orchestrator Agent (Enhanced)
master_orchestrator_agent = Agent(
    name="master_orchestrator_agent",
    model=LiteLLM("gemini/gemini-2.0-flash"),
    description="Master coordinator for all fraud detection agents",
    instruction="""You are the master orchestrator coordinating all HSBC fraud detection agents.

Your responsibilities:
1. Coordinate the complete fraud detection workflow
2. Route messages between specialized agents
3. Make final decisions based on all agent inputs
4. Ensure timely response to fraud threats

Workflow:
1. Receive transcribed audio from audio_processing_agent
2. Send to live_scam_detection_agent for fraud analysis
3. If risk detected, engage rag_policy_agent for guidance
4. For high-risk cases, activate case_management_agent
5. Ensure compliance_agent checks all interactions
6. Synthesize all inputs into final decision

Always prioritize customer protection while maintaining service quality.""",
    tools=[coordinate_fraud_analysis]
)

class HSBCFraudDetectionSystem:
    """Main system orchestrating all agents"""
    
    def __init__(self):
        self.agents = {
            "audio_processing": audio_processing_agent,
            "live_scam_detection": live_scam_detection_agent,
            "rag_policy": rag_policy_agent,
            "master_orchestrator": master_orchestrator_agent,
            "case_management": case_management_agent,
            "compliance": compliance_agent
        }
        
        self.session_service = InMemorySessionService()
        self.runners = {}
        
        logger.info("🚀 HSBC Fraud Detection System initialized with 6 agents")
    
    async def process_audio_file(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """Process audio file through complete pipeline"""
        try:
            logger.info(f"🔄 Processing audio file: {file_path}")
            
            # Step 1: Transcribe audio
            transcription_result = transcribe_audio(file_path)
            
            # Extract customer speech segments
            customer_segments = [
                seg for seg in transcription_result["speaker_segments"]
                if seg["speaker"] == "customer"
            ]
            
            if not customer_segments:
                return {
                    "session_id": session_id,
                    "status": "no_customer_speech",
                    "message": "No customer speech detected"
                }
            
            # Combine customer text
            customer_text = " ".join(seg["text"] for seg in customer_segments)
            
            # Step 2: Analyze for fraud
            fraud_analysis = analyze_fraud_patterns(customer_text)
            
            # Step 3: Get policy guidance if risk detected
            policy_guidance = {}
            if fraud_analysis["risk_score"] >= 40:
                policy_guidance = retrieve_hsbc_policies(
                    fraud_analysis["scam_type"],
                    fraud_analysis["risk_score"]
                )
            
            # Step 4: Orchestrate decision
            orchestration_result = coordinate_fraud_analysis({
                "session_id": session_id,
                "text": customer_text,
                "speaker": "customer"
            })
            
            # Step 5: Create case if needed
            case_info = None
            if fraud_analysis["risk_score"] >= 60:
                case_info = create_fraud_case(
                    session_id,
                    fraud_analysis,
                    customer_id=None
                )
            
            # Step 6: Check compliance
            compliance_check = check_compliance_requirements(
                customer_text,
                fraud_analysis
            )
            
            # Compile results
            result = {
                "session_id": session_id,
                "transcription": transcription_result,
                "fraud_analysis": fraud_analysis,
                "policy_guidance": policy_guidance.get("agent_guidance", {}),
                "orchestrator_decision": orchestration_result.get("orchestrator_decision", {}),
                "case_info": case_info,
                "compliance_check": compliance_check,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Pipeline complete: Risk {fraud_analysis['risk_score']}%")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing audio: {e}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_realtime_stream(self, audio_stream: bytes, session_id: str) -> Dict[str, Any]:
        """Process real-time audio stream"""
        try:
            # Process audio stream
            audio_result = process_audio_stream(audio_stream, session_id)
            
            # Extract transcription
            transcription_text = audio_result["transcription"]["text"]
            
            # Run through fraud detection pipeline
            return await self.process_text(transcription_text, session_id, "customer")
            
        except Exception as e:
            logger.error(f"❌ Error in real-time processing: {e}")
            return {"error": str(e)}
    
    async def process_text(self, text: str, session_id: str, speaker: str = "customer") -> Dict[str, Any]:
        """Process text through fraud detection pipeline"""
        if speaker != "customer":
            return {
                "session_id": session_id,
                "analysis_skipped": True,
                "reason": "Agent speech - no analysis needed"
            }
        
        # Run through orchestrator
        result = coordinate_fraud_analysis({
            "session_id": session_id,
            "text": text,
            "speaker": speaker
        })
        
        return result
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "system_status": "operational",
            "agents_count": len(self.agents),
            "agents": {
                name: "ready" for name in self.agents.keys()
            },
            "framework": "Google ADK 1.4.2",
            "last_updated": datetime.now().isoformat()
        }

# Initialize system
fraud_detection_system = HSBCFraudDetectionSystem()

# Example usage
async def demo_fraud_detection():
    """Demo the fraud detection system"""
    
    # Test with investment scam
    result = await fraud_detection_system.process_audio_file(
        "data/sample_audio/investment_scam_live_call.wav",
        "demo_session_001"
    )
    
    print(f"Risk Score: {result['fraud_analysis']['risk_score']}%")
    print(f"Scam Type: {result['fraud_analysis']['scam_type']}")
    print(f"Decision: {result['orchestrator_decision']['decision']}")
    
    if result.get('case_info'):
        print(f"Case Created: {result['case_info']['case_id']}")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_fraud_detection())
