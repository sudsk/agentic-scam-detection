# agents/orchestrator/agent.py
"""
Master Orchestrator Agent for HSBC Scam Detection System
Coordinates all agents and makes final fraud detection decisions
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
from enum import Enum

from google.adk.agents import Agent
from google.adk.models import LiteLLM

from ..shared.base_agent import BaseAgent, AgentCapability, Message
from ..scam_detection.agent import analyze_fraud_patterns
from ..rag_policy.agent import retrieve_hsbc_policies

logger = logging.getLogger(__name__)

class Decision(Enum):
    """Orchestrator decision types"""
    BLOCK_AND_ESCALATE = "BLOCK_AND_ESCALATE"
    VERIFY_AND_MONITOR = "VERIFY_AND_MONITOR"
    MONITOR_AND_EDUCATE = "MONITOR_AND_EDUCATE"
    CONTINUE_NORMAL = "CONTINUE_NORMAL"
    REQUIRE_ADDITIONAL_INFO = "REQUIRE_ADDITIONAL_INFO"

class WorkflowStage(Enum):
    """Workflow stages"""
    AUDIO_PROCESSING = "audio_processing"
    TRANSCRIPTION = "transcription"
    FRAUD_DETECTION = "fraud_detection"
    POLICY_RETRIEVAL = "policy_retrieval"
    COMPLIANCE_CHECK = "compliance_check"
    CASE_CREATION = "case_creation"
    DECISION_MAKING = "decision_making"
    COMPLETE = "complete"

class MasterOrchestratorAgent(BaseAgent):
    """
    Master orchestrator coordinating all fraud detection agents
    """
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="workflow_coordination",
                description="Coordinate multi-agent fraud detection workflow",
                input_types=["transcript", "audio_file"],
                output_types=["fraud_decision"],
                processing_time_ms=3000,
                confidence_level=0.98
            ),
            AgentCapability(
                name="decision_making",
                description="Make final fraud detection decisions",
                input_types=["fraud_analysis", "policy_guidance"],
                output_types=["final_decision"],
                processing_time_ms=500,
                confidence_level=0.96
            ),
            AgentCapability(
                name="agent_routing",
                description="Route messages between specialized agents",
                input_types=["agent_message"],
                output_types=["routed_message"],
                processing_time_ms=100,
                confidence_level=0.99
            ),
            AgentCapability(
                name="workflow_monitoring",
                description="Monitor and optimize workflow performance",
                input_types=["workflow_metrics"],
                output_types=["optimization_recommendations"],
                processing_time_ms=1000,
                confidence_level=0.94
            )
        ]
        
        config = {
            "max_workflow_duration_ms": 10000,
            "decision_thresholds": {
                "block_and_escalate": 80,
                "verify_and_monitor": 60,
                "monitor_and_educate": 40
            },
            "retry_policy": {
                "max_retries": 3,
                "backoff_multiplier": 2
            },
            "agent_priorities": {
                "audio_processing": 1,
                "transcription": 2,
                "fraud_detection": 3,
                "policy_retrieval": 4,
                "compliance": 5,
                "case_management": 6
            }
        }
        
        super().__init__(
            agent_id="orchestrator_001",
            agent_name="Master Orchestrator Agent",
            capabilities=capabilities,
            config=config
        )
        
        self.registered_agents = {}
        self.active_workflows = {}
        self.workflow_history = []
        self.decision_cache = {}
    
    async def _initialize_agent(self):
        """Initialize orchestrator"""
        try:
            # Initialize workflow templates
            await self._load_workflow_templates()
            
            # Set up inter-agent communication
            await self._setup_agent_communication()
            
            # Load decision models
            await self._load_decision_models()
            
            logger.info("Master orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def _load_workflow_templates(self):
        """Load workflow templates"""
        self.workflow_templates = {
            "standard_fraud_check": {
                "stages": [
                    WorkflowStage.TRANSCRIPTION,
                    WorkflowStage.FRAUD_DETECTION,
                    WorkflowStage.POLICY_RETRIEVAL,
                    WorkflowStage.COMPLIANCE_CHECK,
                    WorkflowStage.DECISION_MAKING
                ],
                "timeout_ms": 8000
            },
            "high_risk_fraud_check": {
                "stages": [
                    WorkflowStage.TRANSCRIPTION,
                    WorkflowStage.FRAUD_DETECTION,
                    WorkflowStage.POLICY_RETRIEVAL,
                    WorkflowStage.COMPLIANCE_CHECK,
                    WorkflowStage.CASE_CREATION,
                    WorkflowStage.DECISION_MAKING
                ],
                "timeout_ms": 10000
            },
            "quick_check": {
                "stages": [
                    WorkflowStage.FRAUD_DETECTION,
                    WorkflowStage.DECISION_MAKING
                ],
                "timeout_ms": 3000
            }
        }
    
    async def _setup_agent_communication(self):
        """Set up communication channels with other agents"""
        self.communication_channels = {
            "audio_processing": asyncio.Queue(),
            "transcription": asyncio.Queue(),
            "fraud_detection": asyncio.Queue(),
            "policy_retrieval": asyncio.Queue(),
            "compliance": asyncio.Queue(),
            "case_management": asyncio.Queue()
        }
    
    async def _load_decision_models(self):
        """Load decision-making models"""
        # In production, load ML models for decision making
        self.decision_model_loaded = True
    
    async def register_agent(self, agent_name: str, agent_instance: BaseAgent):
        """Register an agent with the orchestrator"""
        self.registered_agents[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name}")
    
    async def _process_message(self, message: Message) -> Optional[Message]:
        """Process orchestrator messages"""
        try:
            if message.message_type == "new_transcript":
                return await self._handle_new_transcript(message)
            elif message.message_type == "audio_file":
                return await self._handle_audio_file(message)
            elif message.message_type == "workflow_complete":
                return await self._handle_workflow_complete(message)
            elif message.message_type == "agent_response":
                return await self._handle_agent_response(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            return self._create_error_response(message, str(e))
    
    async def _handle_new_transcript(self, message: Message) -> Message:
        """Handle new transcript and start fraud detection workflow"""
        session_id = message.session_id
        transcript_data = message.data
        
        # Skip if agent speech
        if transcript_data.get("speaker") != "customer":
            return Message(
                message_id=str(uuid.uuid4()),
                message_type="analysis_skipped",
                source_agent=self.agent_id,
                target_agent=message.source_agent,
                session_id=session_id,
                data={
                    "reason": "Agent speech - no analysis needed",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Start workflow
        workflow_id = str(uuid.uuid4())
        workflow = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "start_time": datetime.now(),
            "transcript_data": transcript_data,
            "current_stage": WorkflowStage.FRAUD_DETECTION,
            "results": {},
            "status": "active"
        }
        
        self.active_workflows[workflow_id] = workflow
        
        # Execute workflow stages
        try:
            # Stage 1: Fraud Detection
            fraud_result = await self._execute_fraud_detection(transcript_data)
            workflow["results"]["fraud_detection"] = fraud_result
            
            # Stage 2: Policy Retrieval (if needed)
            if fraud_result["risk_score"] >= 40:
                policy_result = await self._execute_policy_retrieval(fraud_result)
                workflow["results"]["policy_retrieval"] = policy_result
            
            # Stage 3: Compliance Check
            compliance_result = await self._execute_compliance_check(
                transcript_data.get("text", ""), 
                fraud_result
            )
            workflow["results"]["compliance_check"] = compliance_result
            
            # Stage 4: Case Creation (if high risk)
            if fraud_result["risk_score"] >= 60:
                case_result = await self._execute_case_creation(
                    session_id, 
                    fraud_result, 
                    transcript_data.get("customer_id")
                )
                workflow["results"]["case_creation"] = case_result
            
            # Stage 5: Final Decision
            final_decision = await self._make_final_decision(workflow["results"])
            workflow["results"]["final_decision"] = final_decision
            
            # Mark workflow complete
            workflow["status"] = "complete"
            workflow["end_time"] = datetime.now()
            workflow["duration_ms"] = (
                workflow["end_time"] - workflow["start_time"]
            ).total_seconds() * 1000
            
            # Store in history
            self.workflow_history.append(workflow)
            
            # Create comprehensive response
            return Message(
                message_id=str(uuid.uuid4()),
                message_type="comprehensive_fraud_analysis",
                source_agent=self.agent_id,
                target_agent="agent_interface",
                session_id=session_id,
                data={
                    "workflow_id": workflow_id,
                    "session_id": session_id,
                    "fraud_detection": fraud_result,
                    "policy_guidance": workflow["results"].get("policy_retrieval", {}).get("agent_guidance", {}),
                    "compliance_check": compliance_result,
                    "case_info": workflow["results"].get("case_creation"),
                    "orchestrator_decision": final_decision,
                    "requires_escalation": fraud_result["risk_score"] >= 80,
                    "processing_time_ms": workflow["duration_ms"],
                    "timestamp": datetime.now().isoformat()
                },
                priority=4 if fraud_result["risk_score"] >= 80 else 2
            )
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            
            return self._create_error_response(message, f"Workflow failed: {str(e)}")
    
    async def _execute_fraud_detection(self, transcript_data: Dict) -> Dict[str, Any]:
        """Execute fraud detection stage"""
        text = transcript_data.get("text", "")
        
        # Use the fraud detection function
        fraud_result = analyze_fraud_patterns(text)
        
        # Add metadata
        fraud_result["analysis_timestamp"] = datetime.now().isoformat()
        fraud_result["analyzer"] = "live_scam_detection_agent"
        
        return fraud_result
    
    async def _execute_policy_retrieval(self, fraud_result: Dict) -> Dict[str, Any]:
        """Execute policy retrieval stage"""
        scam_type = fraud_result.get("scam_type", "unknown")
        risk_score = fraud_result.get("risk_score", 0)
        
        # Use the policy retrieval function
        policy_result = retrieve_hsbc_policies(scam_type, risk_score)
        
        # Add metadata
        policy_result["retrieval_timestamp"] = datetime.now().isoformat()
        policy_result["retriever"] = "rag_policy_agent"
        
        return policy_result
    
    async def _execute_compliance_check(self, transcript: str, fraud_result: Dict) -> Dict[str, Any]:
        """Execute compliance check stage"""
        # In production, send to compliance agent
        # For now, return mock result
        compliance_result = {
            "compliant": True,
            "pii_detected": False,
            "vulnerable_customer": False,
            "sar_required": fraud_result.get("risk_score", 0) >= 60,
            "gdpr_compliant": True,
            "fca_compliant": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return compliance_result
    
    async def _execute_case_creation(self, session_id: str, fraud_result: Dict, customer_id: Optional[str]) -> Dict[str, Any]:
        """Execute case creation stage"""
        # In production, send to case management agent
        # For now, return mock result
        case_result = {
            "case_id": f"HSBC-FD-{datetime.now().strftime('%Y%m%d')}-{session_id[:8].upper()}",
            "created": True,
            "priority": "high" if fraud_result.get("risk_score", 0) >= 80 else "medium",
            "assigned_to": "fraud-investigation-team@hsbc.com",
            "timestamp": datetime.now().isoformat()
        }
        
        return case_result
    
    async def _make_final_decision(self, workflow_results: Dict) -> Dict[str, Any]:
        """Make final orchestrator decision based on all inputs"""
        fraud_result = workflow_results.get("fraud_detection", {})
        policy_result = workflow_results.get("policy_retrieval", {})
        compliance_result = workflow_results.get("compliance_check", {})
        case_result = workflow_results.get("case_creation", {})
        
        risk_score = fraud_result.get("risk_score", 0)
        scam_type = fraud_result.get("scam_type", "unknown")
        
        # Decision logic
        if risk_score >= self.config["decision_thresholds"]["block_and_escalate"]:
            decision = Decision.BLOCK_AND_ESCALATE
            reasoning = "Critical fraud risk detected - immediate intervention required"
            actions = [
                "Block all pending transactions",
                "Escalate to Financial Crime Team",
                "Initiate fraud investigation",
                "Document all evidence"
            ]
            priority = "CRITICAL"
            
        elif risk_score >= self.config["decision_thresholds"]["verify_and_monitor"]:
            decision = Decision.VERIFY_AND_MONITOR
            reasoning = "High fraud risk - enhanced verification required"
            actions = [
                "Require additional customer verification",
                "Monitor account for 48 hours",
                "Flag for supervisor review",
                "Provide fraud education to customer"
            ]
            priority = "HIGH"
            
        elif risk_score >= self.config["decision_thresholds"]["monitor_and_educate"]:
            decision = Decision.MONITOR_AND_EDUCATE
            reasoning = "Moderate risk - increased vigilance recommended"
            actions = [
                "Continue normal processing with caution",
                "Provide relevant fraud warnings",
                "Document conversation details",
                "Consider follow-up contact"
            ]
            priority = "MEDIUM"
            
        else:
            decision = Decision.CONTINUE_NORMAL
            reasoning = "Low fraud risk - standard processing"
            actions = [
                "Continue normal customer service",
                "No special actions required"
            ]
            priority = "LOW"
        
        # Adjust decision based on compliance
        if compliance_result.get("vulnerable_customer", False):
            actions.append("Apply vulnerable customer procedures")
            priority = "HIGH" if priority == "MEDIUM" else priority
        
        if compliance_result.get("sar_required", False):
            actions.append("File Suspicious Activity Report within 24 hours")
        
        decision_data = {
            "decision": decision.value,
            "reasoning": reasoning,
            "actions": actions,
            "priority": priority,
            "risk_score": risk_score,
            "scam_type": scam_type,
            "confidence": fraud_result.get("confidence", 0.9),
            "case_created": case_result.get("created", False),
            "case_id": case_result.get("case_id"),
            "timestamp": datetime.now().isoformat()
        }
        
        return decision_data
    
    async def _handle_audio_file(self, message: Message) -> Message:
        """Handle audio file processing request"""
        file_path = message.data.get("file_path")
        session_id = message.session_id
        
        # In production, coordinate with audio processing agent
        # For now, simulate workflow
        workflow_id = str(uuid.uuid4())
        
        logger.info(f"Starting audio file workflow: {workflow_id}")
        
        # Return acknowledgment
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="workflow_started",
            source_agent=self.agent_id,
            target_agent=message.source_agent,
            session_id=session_id,
            data={
                "workflow_id": workflow_id,
                "estimated_duration_ms": 5000
            }
        )
    
    async def _handle_workflow_complete(self, message: Message) -> Message:
        """Handle workflow completion notification"""
        workflow_id = message.data.get("workflow_id")
        
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow["status"] = "complete"
            workflow["end_time"] = datetime.now()
            
            logger.info(f"Workflow {workflow_id} completed")
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="workflow_acknowledged",
            source_agent=self.agent_id,
            target_agent=message.source_agent,
            session_id=message.session_id,
            data={"workflow_id": workflow_id}
        )
    
    async def _handle_agent_response(self, message: Message) -> Message:
        """Handle response from another agent"""
        source_agent = message.source_agent
        response_data = message.data
        
        logger.info(f"Received response from {source_agent}")
        
        # Route to appropriate handler based on source
        # In production, update workflow state
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="response_acknowledged",
            source_agent=self.agent_id,
            target_agent=source_agent,
            session_id=message.session_id,
            data={"acknowledged": True}
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        base_metrics = self.get_status()
        
        # Calculate workflow metrics
        total_workflows = len(self.workflow_history)
        successful_workflows = sum(
            1 for w in self.workflow_history 
            if w.get("status") == "complete"
        )
        
        avg_duration = 0
        if successful_workflows > 0:
            avg_duration = sum(
                w.get("duration_ms", 0) for w in self.workflow_history
                if w.get("status") == "complete"
            ) / successful_workflows
        
        orchestrator_metrics = {
            "active_workflows": len(self.active_workflows),
            "total_workflows_processed": total_workflows,
            "successful_workflows": successful_workflows,
            "failed_workflows": total_workflows - successful_workflows,
            "average_workflow_duration_ms": avg_duration,
            "registered_agents": len(self.registered_agents),
            "decision_cache_size": len(self.decision_cache),
            "workflow_success_rate": (
                successful_workflows / total_workflows if total_workflows > 0 else 0
            )
        }
        
        base_metrics.update(orchestrator_metrics)
        return base_metrics
    
    async def optimize_workflow(self, performance_data: Dict) -> Dict[str, Any]:
        """Optimize workflow based on performance data"""
        recommendations = []
        
        # Check average duration
        avg_duration = performance_data.get("average_workflow_duration_ms", 0)
        if avg_duration > self.config["max_workflow_duration_ms"]:
            recommendations.append({
                "type": "performance",
                "recommendation": "Consider parallelizing independent stages",
                "impact": "high"
            })
        
        # Check failure rate
        failure_rate = 1 - performance_data.get("workflow_success_rate", 1)
        if failure_rate > 0.1:
            recommendations.append({
                "type": "reliability",
                "recommendation": "Implement circuit breakers for failing agents",
                "impact": "medium"
            })
        
        return {
            "recommendations": recommendations,
            "current_performance": performance_data,
            "optimization_timestamp": datetime.now().isoformat()
        }

# Main orchestration function for ADK
def coordinate_fraud_analysis(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinate the complete fraud detection workflow
    
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
    if fraud_analysis['risk_score'] >= 40:  # Only retrieve policies for medium+ risk
        policy_guidance = retrieve_hsbc_policies(
            fraud_analysis['scam_type'], 
            fraud_analysis['risk_score']
        )
    
    # Step 3: Make Final Decision
    orchestrator_decision = {}
    risk_score = fraud_analysis['risk_score']
    
    if risk_score >= 80:
        orchestrator_decision = {
            'decision': 'BLOCK_AND_ESCALATE',
            'reasoning': 'Critical fraud risk detected - immediate intervention required',
            'actions': [
                'Block all pending transactions',
                'Escalate to Financial Crime Team',
                'Initiate fraud investigation',
                'Document all evidence'
            ],
            'priority': 'CRITICAL'
        }
    elif risk_score >= 60:
        orchestrator_decision = {
            'decision': 'VERIFY_AND_MONITOR',
            'reasoning': 'High fraud risk - enhanced verification required',
            'actions': [
                'Require additional customer verification',
                'Monitor account for 48 hours',
                'Flag for supervisor review',
                'Provide fraud education to customer'
            ],
            'priority': 'HIGH'
        }
    elif risk_score >= 40:
        orchestrator_decision = {
            'decision': 'MONITOR_AND_EDUCATE',
            'reasoning': 'Moderate risk - increased vigilance recommended',
            'actions': [
                'Continue normal processing with caution',
                'Provide relevant fraud warnings',
                'Document conversation details',
                'Consider follow-up contact'
            ],
            'priority': 'MEDIUM'
        }
    else:
        orchestrator_decision = {
            'decision': 'CONTINUE_NORMAL',
            'reasoning': 'Low fraud risk - standard processing',
            'actions': [
                'Continue normal customer service',
                'No special actions required'
            ],
            'priority': 'LOW'
        }
    
    # Compile results
    result = {
        'session_id': session_id,
        'fraud_detection': fraud_analysis,
        'policy_guidance': policy_guidance.get('agent_guidance', {}),
        'orchestrator_decision': orchestrator_decision,
        'requires_escalation': risk_score >= 80,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"✅ Orchestration complete: {orchestrator_decision['decision']}")
    return result

# Create the Master Orchestrator Agent using Google ADK
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

Decision thresholds:
- Risk >= 80%: BLOCK_AND_ESCALATE
- Risk >= 60%: VERIFY_AND_MONITOR
- Risk >= 40%: MONITOR_AND_EDUCATE
- Risk < 40%: CONTINUE_NORMAL

Always prioritize customer protection while maintaining service quality.""",
    tools=[coordinate_fraud_analysis]
)
