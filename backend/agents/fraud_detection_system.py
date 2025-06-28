# agents/fraud_detection_system.py
"""
Main Fraud Detection System Coordinator
Imports and orchestrates all modular agents
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import all modular agents
from .audio_processor.agent import transcribe_audio, audio_processor_agent
from .scam_detection.agent import analyze_fraud_patterns, fraud_detection_agent
from .policy_guidance.agent import retrieve_fraud_policies, policy_guidance_agent
from .case_management.agent import create_fraud_case, case_management_agent

logger = logging.getLogger(__name__)

class FraudDetectionSystem:
    """
    Main system orchestrating all fraud detection agents
    Coordinates the flow between modular agents
    """
    
    def __init__(self):
        # Agent references
        self.audio_processor = audio_processor_agent
        self.fraud_detector = fraud_detection_agent
        self.policy_guide = policy_guidance_agent
        self.case_manager = case_management_agent
        
        # System state
        self.session_data = {}
        self.processing_queue = {}
        
        logger.info("🚀 Fraud Detection System initialized with 4 modular agents")
        logger.info(f"  - {self.audio_processor.agent_name}")
        logger.info(f"  - {self.fraud_detector.agent_name}")
        logger.info(f"  - {self.policy_guide.agent_name}")
        logger.info(f"  - {self.case_manager.agent_name}")
    
    async def process_audio_file(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """
        Process audio file through complete multi-agent pipeline
        
        Flow: Audio Processing → Fraud Detection → Policy Guidance → Case Management
        """
        try:
            logger.info(f"🔄 Starting multi-agent pipeline for: {file_path}")
            
            # AGENT 1: Audio Processing - Transcribe audio
            logger.info("📝 Step 1: Audio Processing Agent")
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
                    "message": "No customer speech detected",
                    "pipeline_stage": "audio_processing"
                }
            
            # Combine customer text for analysis
            customer_text = " ".join(seg["text"] for seg in customer_segments)
            
            # AGENT 2: Fraud Detection - Analyze for fraud patterns
            logger.info("🔍 Step 2: Fraud Detection Agent")
            fraud_analysis = analyze_fraud_patterns(customer_text)
            
            # AGENT 3: Policy Guidance - Get response procedures (if risk detected)
            policy_guidance = {}
            if fraud_analysis["risk_score"] >= 40:
                logger.info("📚 Step 3: Policy Guidance Agent")
                policy_guidance = retrieve_fraud_policies(
                    fraud_analysis["scam_type"],
                    fraud_analysis["risk_score"]
                )
            else:
                logger.info("📚 Step 3: Policy Guidance Agent - Skipped (low risk)")
            
            # AGENT 4: Case Management - Create case (if high risk)
            case_info = None
            if fraud_analysis["risk_score"] >= 60:
                logger.info("📋 Step 4: Case Management Agent")
                case_info = create_fraud_case(
                    session_id,
                    fraud_analysis,
                    customer_id=None
                )
            else:
                logger.info("📋 Step 4: Case Management Agent - Skipped (medium/low risk)")
            
            # ORCHESTRATOR: Make final decision
            logger.info("🎭 Step 5: Orchestrator Decision")
            orchestrator_decision = self._make_orchestrator_decision(
                fraud_analysis, policy_guidance, case_info
            )
            
            # Compile comprehensive results
            result = {
                "session_id": session_id,
                "pipeline_complete": True,
                "agents_involved": self._get_agents_involved(fraud_analysis["risk_score"]),
                "transcription": transcription_result,
                "fraud_analysis": fraud_analysis,
                "policy_guidance": policy_guidance.get("agent_guidance", {}),
                "orchestrator_decision": orchestrator_decision,
                "case_info": case_info,
                "processing_summary": {
                    "total_agents": len(self._get_agents_involved(fraud_analysis["risk_score"])),
                    "risk_level": fraud_analysis["risk_level"],
                    "escalation_required": fraud_analysis["risk_score"] >= 80,
                    "case_created": case_info is not None
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Store session data
            self.session_data[session_id] = result
            
            logger.info(f"✅ Multi-agent pipeline complete: Risk {fraud_analysis['risk_score']}% - {orchestrator_decision['decision']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Multi-agent pipeline failed: {e}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "pipeline_stage": "unknown",
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_agents_involved(self, risk_score: float) -> List[str]:
        """Get list of agents involved based on risk score"""
        agents = ["audio_processing", "fraud_detection"]
        
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
        """
        Make final orchestrator decision based on all agent inputs
        """
        risk_score = fraud_analysis.get("risk_score", 0)
        scam_type = fraud_analysis.get("scam_type", "unknown")
        
        # Decision matrix based on risk score
        if risk_score >= 80:
            decision = "BLOCK_AND_ESCALATE"
            reasoning = "Critical fraud risk detected - immediate intervention required"
            actions = [
                "Block all pending transactions immediately",
                "Escalate to Financial Crime Team",
                "Initiate fraud investigation",
                "Document all evidence",
                "Contact customer via secure channel"
            ]
            priority = "CRITICAL"
            
        elif risk_score >= 60:
            decision = "VERIFY_AND_MONITOR"
            reasoning = "High fraud risk - enhanced verification required"
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
            reasoning = "Moderate risk - increased vigilance recommended"
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
            reasoning = "Low fraud risk - standard processing"
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
            "timestamp": datetime.now().isoformat()
        }
    
    async def process_realtime_stream(self, audio_stream: bytes, session_id: str) -> Dict[str, Any]:
        """Process real-time audio stream through agents"""
        try:
            # Process through audio agent
            audio_result = self.audio_processor.process_realtime_stream(audio_stream, session_id)
            
            if audio_result.get("interim_text"):
                # Quick fraud check on interim text
                fraud_result = analyze_fraud_patterns(audio_result["interim_text"])
                
                return {
                    "session_id": session_id,
                    "realtime_result": True,
                    "interim_analysis": fraud_result,
                    "audio_processed": audio_result,
                    "timestamp": datetime.now().isoformat()
                }
            
            return audio_result
            
        except Exception as e:
            logger.error(f"❌ Real-time processing failed: {e}")
            return {"error": str(e)}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all agents"""
        from .shared.base_agent import agent_registry
        
        # Get registry status
        registry_status = agent_registry.get_system_status()
        
        return {
            "system_status": "operational",
            "framework": "Modular Multi-Agent Fraud Detection System",
            "agents_count": 4,
            "agents": {
                "audio_processing": self.audio_processor.status.value if hasattr(self.audio_processor, 'status') else "active",
                "fraud_detection": self.fraud_detector.status.value if hasattr(self.fraud_detector, 'status') else "active",
                "policy_guidance": self.policy_guide.status.value if hasattr(self.policy_guide, 'status') else "active",
                "case_management": self.case_manager.status.value if hasattr(self.case_manager, 'status') else "active"
            },
            "agent_details": {
                "audio_processing": self.audio_processor.get_status() if hasattr(self.audio_processor, 'get_status') else {"status": "active"},
                "fraud_detection": self.fraud_detector.get_status() if hasattr(self.fraud_detector, 'get_status') else {"status": "active"},
                "policy_guidance": self.policy_guide.get_status() if hasattr(self.policy_guide, 'get_status') else {"status": "active"},
                "case_management": self.case_manager.get_status() if hasattr(self.case_manager, 'get_status') else {"status": "active"}
            },
            "registry_status": registry_status,
            "session_count": len(self.session_data),
            "case_statistics": self.case_manager.get_case_statistics() if hasattr(self.case_manager, 'get_case_statistics') else {},
            "last_updated": datetime.now().isoformat()
        }
    
    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """Get data for a specific session"""
        return self.session_data.get(session_id)
    
    def get_all_sessions(self) -> Dict[str, Dict]:
        """Get all session data"""
        return self.session_data
    
    def get_agent_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from all agents"""
        return {
            "audio_processing": self.audio_processor.get_status(),
            "fraud_detection": self.fraud_detector.get_status(),
            "policy_guidance": self.policy_guide.get_status(),
            "case_management": self.case_manager.get_status(),
            "system_metrics": {
                "total_sessions": len(self.session_data),
                "active_cases": len(self.case_manager.active_cases),
                "average_risk_score": self._calculate_average_risk_score(),
                "pipeline_success_rate": self._calculate_pipeline_success_rate()
            }
        }
    
    def _calculate_average_risk_score(self) -> float:
        """Calculate average risk score across all sessions"""
        if not self.session_data:
            return 0.0
        
        risk_scores = [
            session.get("fraud_analysis", {}).get("risk_score", 0)
            for session in self.session_data.values()
        ]
        
        return sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    
    def _calculate_pipeline_success_rate(self) -> float:
        """Calculate pipeline success rate"""
        if not self.session_data:
            return 1.0
        
        successful = sum(
            1 for session in self.session_data.values()
            if session.get("pipeline_complete", False)
        )
        
        return successful / len(self.session_data)

# Initialize the global system
fraud_detection_system = FraudDetectionSystem()

# Main coordination function for API compatibility
def coordinate_fraud_analysis(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinate fraud analysis workflow for API endpoints
    
    Args:
        transcript_data: Customer speech data with text, speaker, session_id
        
    Returns:
        Dict containing comprehensive fraud analysis and recommendations
    """
    session_id = transcript_data.get('session_id')
    text = transcript_data.get('text', '')
    speaker = transcript_data.get('speaker', 'unknown')
    
    logger.info(f"🎭 Coordinating fraud analysis for session {session_id}")
    
    # Skip analysis for agent speech
    if speaker != 'customer':
        return {
            'session_id': session_id,
            'analysis_skipped': True,
            'reason': 'Agent speech - no analysis needed',
            'timestamp': datetime.now().isoformat()
        }
    
    # Step 1: Fraud Detection Analysis
    fraud_analysis = analyze_fraud_patterns(text)
    
    # Step 2: Policy Retrieval (if risk detected)
    policy_guidance = {}
    if fraud_analysis['risk_score'] >= 40:
        policy_guidance = retrieve_fraud_policies(
            fraud_analysis['scam_type'], 
            fraud_analysis['risk_score']
        )
    
    # Step 3: Case Creation (if high risk)
    case_info = None
    if fraud_analysis['risk_score'] >= 60:
        case_info = create_fraud_case(
            session_id,
            fraud_analysis,
            customer_id=None
        )
    
    # Step 4: Make Final Decision
    orchestrator_decision = fraud_detection_system._make_orchestrator_decision(
        fraud_analysis, policy_guidance, case_info
    )
    
    # Compile results
    result = {
        'session_id': session_id,
        'fraud_detection': fraud_analysis,
        'policy_guidance': policy_guidance.get('agent_guidance', {}),
        'orchestrator_decision': orchestrator_decision,
        'case_info': case_info,
        'requires_escalation': fraud_analysis['risk_score'] >= 80,
        'agents_involved': fraud_detection_system._get_agents_involved(fraud_analysis['risk_score']),
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"✅ Coordination complete: {orchestrator_decision['decision']}")
    return result

# Example usage and testing
async def demo_modular_system():
    """Demo the modular fraud detection system"""
    
    print("🚀 Fraud Detection System Demo - Modular Architecture")
    print("=" * 60)
    
    # Test with investment scam
    result = await fraud_detection_system.process_audio_file(
        "data/sample_audio/investment_scam_live_call.wav",
        "demo_session_001"
    )
    
    print(f"\n📊 Results Summary:")
    print(f"  Risk Score: {result['fraud_analysis']['risk_score']}%")
    print(f"  Scam Type: {result['fraud_analysis']['scam_type']}")
    print(f"  Decision: {result['orchestrator_decision']['decision']}")
    print(f"  Agents Involved: {', '.join(result['agents_involved'])}")
    
    if result.get('case_info'):
        print(f"  Case Created: {result['case_info']['case_id']}")
    
    # Show agent status
    status = fraud_detection_system.get_agent_status()
    print(f"\n🤖 Agent Status:")
    for agent_name, agent_status in status['agents'].items():
        print(f"  - {agent_name}: {agent_status}")

if __name__ == "__main__":
    asyncio.run(demo_modular_system())
