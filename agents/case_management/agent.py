# agents/case_management/agent.py
"""
Case Management Agent for HSBC Scam Detection System
Integrates with Quantexa and ServiceNow for fraud case management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid
from enum import Enum

from google.adk.agents import Agent
from google.adk.models import LiteLLM

from ..shared.base_agent import BaseAgent, AgentCapability, Message

logger = logging.getLogger(__name__)

class CaseStatus(Enum):
    """Case status enumeration"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FRAUDULENT = "fraudulent"
    FALSE_POSITIVE = "false_positive"

class CasePriority(Enum):
    """Case priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class CaseManagementAgent(BaseAgent):
    """
    Agent responsible for creating and managing fraud investigation cases
    """
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="case_creation",
                description="Create fraud investigation cases",
                input_types=["fraud_analysis"],
                output_types=["case_created"],
                processing_time_ms=500,
                confidence_level=0.99
            ),
            AgentCapability(
                name="quantexa_integration",
                description="Integrate with Quantexa for entity resolution",
                input_types=["case_data"],
                output_types=["entity_analysis"],
                processing_time_ms=2000,
                confidence_level=0.95
            ),
            AgentCapability(
                name="case_workflow",
                description="Manage case investigation workflow",
                input_types=["case_update"],
                output_types=["workflow_action"],
                processing_time_ms=300,
                confidence_level=0.98
            ),
            AgentCapability(
                name="reporting",
                description="Generate fraud investigation reports",
                input_types=["case_id"],
                output_types=["investigation_report"],
                processing_time_ms=1000,
                confidence_level=0.97
            )
        ]
        
        config = {
            "quantexa_api_url": "https://api.quantexa.hsbc.internal",
            "quantexa_api_key": "mock_key",
            "servicenow_url": "https://hsbc.service-now.com",
            "servicenow_api_key": "mock_key",
            "auto_escalation_threshold": 80,
            "case_retention_days": 2555,  # 7 years
            "enable_auto_assignment": True,
            "team_assignments": {
                "critical": "financial-crime-immediate@hsbc.com",
                "high": "fraud-investigation-team@hsbc.com",
                "medium": "fraud-prevention-team@hsbc.com",
                "low": "customer-service-enhanced@hsbc.com"
            }
        }
        
        super().__init__(
            agent_id="case_management_001",
            agent_name="Case Management Agent",
            capabilities=capabilities,
            config=config
        )
        
        self.active_cases = {}
        self.case_statistics = {
            "total_created": 0,
            "escalated": 0,
            "resolved": 0,
            "false_positives": 0
        }
    
    async def _initialize_agent(self):
        """Initialize case management connections"""
        try:
            # Initialize Quantexa connection
            await self._init_quantexa_connection()
            
            # Initialize ServiceNow connection
            await self._init_servicenow_connection()
            
            # Load case templates
            await self._load_case_templates()
            
            logger.info("Case management agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize case management: {e}")
            raise
    
    async def _init_quantexa_connection(self):
        """Initialize Quantexa API connection"""
        # In production, establish actual API connection
        self.quantexa_connected = True
        logger.info("Quantexa connection established")
    
    async def _init_servicenow_connection(self):
        """Initialize ServiceNow connection"""
        # In production, establish actual ServiceNow connection
        self.servicenow_connected = True
        logger.info("ServiceNow connection established")
    
    async def _load_case_templates(self):
        """Load case investigation templates"""
        self.case_templates = {
            "investment_scam": {
                "title": "Investment Fraud Investigation - {case_id}",
                "category": "Financial Crime",
                "subcategory": "Investment Fraud",
                "required_evidence": [
                    "Call transcript",
                    "Transaction history",
                    "Investment company details",
                    "Communication records"
                ],
                "investigation_steps": [
                    "Verify investment company registration",
                    "Check FCA warnings list",
                    "Review customer transaction history",
                    "Identify related cases",
                    "Contact customer for verification"
                ]
            },
            "romance_scam": {
                "title": "Romance Fraud Investigation - {case_id}",
                "category": "Financial Crime",
                "subcategory": "Romance Fraud",
                "required_evidence": [
                    "Call transcript",
                    "Online communication records",
                    "Transfer requests",
                    "Beneficiary details"
                ],
                "investigation_steps": [
                    "Document relationship timeline",
                    "Verify beneficiary identity",
                    "Check for similar cases",
                    "Assess customer vulnerability",
                    "Provide victim support resources"
                ]
            },
            "impersonation_scam": {
                "title": "Impersonation Fraud Investigation - {case_id}",
                "category": "Financial Crime",
                "subcategory": "Impersonation Fraud",
                "required_evidence": [
                    "Call transcript",
                    "Caller details",
                    "Authority claims",
                    "Threat details"
                ],
                "investigation_steps": [
                    "Verify caller identity claims",
                    "Report to relevant authorities",
                    "Check for data breach",
                    "Review account security",
                    "Implement additional safeguards"
                ]
            }
        }
    
    async def _process_message(self, message: Message) -> Optional[Message]:
        """Process case management requests"""
        try:
            if message.message_type == "create_case":
                return await self._handle_case_creation(message)
            elif message.message_type == "update_case":
                return await self._handle_case_update(message)
            elif message.message_type == "escalate_case":
                return await self._handle_case_escalation(message)
            elif message.message_type == "resolve_case":
                return await self._handle_case_resolution(message)
            elif message.message_type == "generate_report":
                return await self._handle_report_generation(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error in case management: {e}")
            return self._create_error_response(message, str(e))
    
    async def _handle_case_creation(self, message: Message) -> Message:
        """Create new fraud investigation case"""
        fraud_analysis = message.data.get("fraud_analysis", {})
        session_id = message.session_id
        customer_id = message.data.get("customer_id")
        
        # Generate case ID
        case_id = self._generate_case_id()
        
        # Determine priority
        risk_score = fraud_analysis.get("risk_score", 0)
        priority = self._determine_priority(risk_score)
        
        # Get case template
        scam_type = fraud_analysis.get("scam_type", "unknown")
        template = self.case_templates.get(scam_type, self.case_templates["investment_scam"])
        
        # Create case data
        case_data = {
            "case_id": case_id,
            "session_id": session_id,
            "customer_id": customer_id or f"CUST-{session_id[:8]}",
            "created_at": datetime.now().isoformat(),
            "status": CaseStatus.OPEN.value,
            "priority": priority.value,
            "fraud_type": scam_type,
            "risk_score": risk_score,
            "title": template["title"].format(case_id=case_id),
            "category": template["category"],
            "subcategory": template["subcategory"],
            "description": self._generate_case_description(fraud_analysis),
            "evidence": {
                "transcript": fraud_analysis.get("transcript_text", ""),
                "detected_patterns": fraud_analysis.get("detected_patterns", {}),
                "risk_factors": fraud_analysis.get("explanation", ""),
                "audio_file": message.data.get("audio_file_path", "")
            },
            "investigation_steps": template["investigation_steps"],
            "assigned_to": self.config["team_assignments"][priority.value],
            "timeline": [{
                "timestamp": datetime.now().isoformat(),
                "action": "Case created",
                "actor": "System",
                "details": f"Automated case creation from session {session_id}"
            }]
        }
        
        # Store case
        self.active_cases[case_id] = case_data
        self.case_statistics["total_created"] += 1
        
        # Create in external systems
        external_refs = await self._create_external_cases(case_data)
        case_data["external_references"] = external_refs
        
        # Perform initial Quantexa analysis
        quantexa_analysis = await self._quantexa_entity_resolution(case_data)
        case_data["quantexa_analysis"] = quantexa_analysis
        
        logger.info(f"📋 Case created: {case_id} - Priority: {priority.value}")
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="case_created",
            source_agent=self.agent_id,
            target_agent="orchestrator",
            session_id=session_id,
            data={
                "case_id": case_id,
                "case_data": case_data,
                "actions_required": self._get_immediate_actions(case_data),
                "notification_sent": True
            },
            priority=4 if priority == CasePriority.CRITICAL else 3
        )
    
    def _generate_case_id(self) -> str:
        """Generate unique case ID"""
        date_str = datetime.now().strftime("%Y%m%d")
        random_suffix = str(uuid.uuid4())[:8].upper()
        return f"HSBC-FD-{date_str}-{random_suffix}"
    
    def _determine_priority(self, risk_score: float) -> CasePriority:
        """Determine case priority based on risk score"""
        if risk_score >= 80:
            return CasePriority.CRITICAL
        elif risk_score >= 60:
            return CasePriority.HIGH
        elif risk_score >= 40:
            return CasePriority.MEDIUM
        else:
            return CasePriority.LOW
    
    def _generate_case_description(self, fraud_analysis: Dict) -> str:
        """Generate detailed case description"""
        risk_score = fraud_analysis.get("risk_score", 0)
        scam_type = fraud_analysis.get("scam_type", "unknown")
        patterns = fraud_analysis.get("detected_patterns", {})
        
        description = f"Potential {scam_type.replace('_', ' ')} detected with risk score of {risk_score}%. "
        
        if patterns:
            pattern_names = list(patterns.keys())
            description += f"Detected patterns include: {', '.join(pattern_names)}. "
        
        description += fraud_analysis.get("explanation", "")
        
        return description
    
    async def _create_external_cases(self, case_data: Dict) -> Dict[str, str]:
        """Create cases in external systems"""
        external_refs = {}
        
        # Create ServiceNow incident
        if self.servicenow_connected:
            snow_ref = await self._create_servicenow_incident(case_data)
            external_refs["servicenow"] = snow_ref
        
        # Create Quantexa investigation
        if self.quantexa_connected:
            quantexa_ref = await self._create_quantexa_investigation(case_data)
            external_refs["quantexa"] = quantexa_ref
        
        return external_refs
    
    async def _create_servicenow_incident(self, case_data: Dict) -> str:
        """Create ServiceNow incident"""
        # Mock implementation
        return f"INC{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}"
    
    async def _create_quantexa_investigation(self, case_data: Dict) -> str:
        """Create Quantexa investigation"""
        # Mock implementation
        return f"QTX-{case_data['case_id']}"
    
    async def _quantexa_entity_resolution(self, case_data: Dict) -> Dict[str, Any]:
        """Perform Quantexa entity resolution and network analysis"""
        # Mock Quantexa analysis
        customer_id = case_data["customer_id"]
        
        analysis = {
            "entity_id": f"ENT-{customer_id}",
            "risk_score": 0.75,
            "network_connections": 3,
            "related_entities": [
                {
                    "entity_id": "ENT-REL001",
                    "relationship": "frequent_transfers",
                    "risk_indicator": "high_velocity_transfers"
                }
            ],
            "historical_cases": 0,
            "suspicious_patterns": [
                "new_beneficiary_high_amount",
                "offshore_transfer_pattern"
            ],
            "recommendation": "enhanced_due_diligence"
        }
        
        return analysis
    
    def _get_immediate_actions(self, case_data: Dict) -> List[str]:
        """Get immediate actions required for case"""
        actions = []
        priority = CasePriority(case_data["priority"])
        
        if priority == CasePriority.CRITICAL:
            actions.extend([
                "Block all pending transactions immediately",
                "Contact customer via secure channel",
                "Notify Financial Crime team lead",
                "Initiate account freeze if confirmed fraud"
            ])
        elif priority == CasePriority.HIGH:
            actions.extend([
                "Review pending transactions",
                "Enhanced authentication for all transactions",
                "Schedule customer callback within 1 hour",
                "Flag account for monitoring"
            ])
        else:
            actions.extend([
                "Monitor account activity",
                "Review transaction history",
                "Prepare customer education materials"
            ])
        
        return actions
    
    async def _handle_case_update(self, message: Message) -> Message:
        """Update existing case"""
        case_id = message.data.get("case_id")
        updates = message.data.get("updates", {})
        
        if case_id not in self.active_cases:
            return self._create_error_response(message, f"Case {case_id} not found")
        
        case_data = self.active_cases[case_id]
        
        # Apply updates
        for key, value in updates.items():
            if key in case_data:
                case_data[key] = value
        
        # Add to timeline
        case_data["timeline"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "Case updated",
            "actor": message.data.get("actor", "System"),
            "details": f"Updated fields: {', '.join(updates.keys())}"
        })
        
        # Update external systems
        await self._update_external_systems(case_id, updates)
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="case_updated",
            source_agent=self.agent_id,
            target_agent=message.source_agent,
            session_id=message.session_id,
            data={
                "case_id": case_id,
                "updates_applied": list(updates.keys()),
                "current_status": case_data["status"]
            }
        )
    
    async def _handle_case_escalation(self, message: Message) -> Message:
        """Escalate case to higher priority"""
        case_id = message.data.get("case_id")
        reason = message.data.get("reason", "Risk score increase")
        
        if case_id not in self.active_cases:
            return self._create_error_response(message, f"Case {case_id} not found")
        
        case_data = self.active_cases[case_id]
        old_priority = case_data["priority"]
        
        # Escalate priority
        current_priority = CasePriority(old_priority)
        if current_priority == CasePriority.LOW:
            new_priority = CasePriority.MEDIUM
        elif current_priority == CasePriority.MEDIUM:
            new_priority = CasePriority.HIGH
        else:
            new_priority = CasePriority.CRITICAL
        
        case_data["priority"] = new_priority.value
        case_data["status"] = CaseStatus.ESCALATED.value
        
        # Update assignment
        case_data["assigned_to"] = self.config["team_assignments"][new_priority.value]
        
        # Add to timeline
        case_data["timeline"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "Case escalated",
            "actor": message.data.get("actor", "System"),
            "details": f"Escalated from {old_priority} to {new_priority.value}. Reason: {reason}"
        })
        
        self.case_statistics["escalated"] += 1
        
        # Notify relevant teams
        await self._send_escalation_notifications(case_data, reason)
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="case_escalated",
            source_agent=self.agent_id,
            target_agent="orchestrator",
            session_id=message.session_id,
            data={
                "case_id": case_id,
                "old_priority": old_priority,
                "new_priority": new_priority.value,
                "assigned_to": case_data["assigned_to"],
                "escalation_reason": reason
            },
            priority=4
        )
    
    async def _handle_case_resolution(self, message: Message) -> Message:
        """Resolve and close case"""
        case_id = message.data.get("case_id")
        resolution = message.data.get("resolution", {})
        
        if case_id not in self.active_cases:
            return self._create_error_response(message, f"Case {case_id} not found")
        
        case_data = self.active_cases[case_id]
        
        # Update case status
        case_data["status"] = CaseStatus.RESOLVED.value
        case_data["resolution"] = {
            "outcome": resolution.get("outcome", "investigation_complete"),
            "fraud_confirmed": resolution.get("fraud_confirmed", False),
            "amount_recovered": resolution.get("amount_recovered", 0),
            "customer_compensated": resolution.get("customer_compensated", False),
            "lessons_learned": resolution.get("lessons_learned", []),
            "resolved_at": datetime.now().isoformat(),
            "resolved_by": resolution.get("resolved_by", "System")
        }
        
        # Update statistics
        self.case_statistics["resolved"] += 1
        if not resolution.get("fraud_confirmed", False):
            self.case_statistics["false_positives"] += 1
        
        # Generate final report
        final_report = await self._generate_final_report(case_data)
        case_data["final_report"] = final_report
        
        # Archive case
        await self._archive_case(case_data)
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="case_resolved",
            source_agent=self.agent_id,
            target_agent="orchestrator",
            session_id=message.session_id,
            data={
                "case_id": case_id,
                "resolution": case_data["resolution"],
                "final_report_id": final_report.get("report_id"),
                "case_archived": True
            }
        )
    
    async def _handle_report_generation(self, message: Message) -> Message:
        """Generate investigation report"""
        case_id = message.data.get("case_id")
        report_type = message.data.get("report_type", "standard")
        
        if case_id not in self.active_cases:
            return self._create_error_response(message, f"Case {case_id} not found")
        
        case_data = self.active_cases[case_id]
        
        # Generate report based on type
        if report_type == "executive":
            report = await self._generate_executive_report(case_data)
        elif report_type == "technical":
            report = await self._generate_technical_report(case_data)
        else:
            report = await self._generate_standard_report(case_data)
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="report_generated",
            source_agent=self.agent_id,
            target_agent=message.source_agent,
            session_id=message.session_id,
            data={
                "case_id": case_id,
                "report": report,
                "report_type": report_type
            }
        )
    
    async def _generate_standard_report(self, case_data: Dict) -> Dict[str, Any]:
        """Generate standard investigation report"""
        return {
            "report_id": f"RPT-{case_data['case_id']}",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": self._generate_executive_summary(case_data),
            "investigation_details": {
                "timeline": case_data["timeline"],
                "evidence_collected": list(case_data["evidence"].keys()),
                "patterns_detected": case_data["evidence"]["detected_patterns"],
                "quantexa_findings": case_data.get("quantexa_analysis", {})
            },
            "recommendations": self._generate_recommendations(case_data),
            "next_steps": self._generate_next_steps(case_data)
        }
    
    def _generate_executive_summary(self, case_data: Dict) -> str:
        """Generate executive summary for report"""
        summary = f"Case {case_data['case_id']} investigated a potential {case_data['fraud_type'].replace('_', ' ')} "
        summary += f"with a risk score of {case_data['risk_score']}%. "
        
        if case_data["status"] == CaseStatus.RESOLVED.value:
            resolution = case_data.get("resolution", {})
            if resolution.get("fraud_confirmed"):
                summary += "The investigation confirmed fraudulent activity. "
            else:
                summary += "The investigation determined this was a false positive. "
        else:
            summary += f"The case is currently {case_data['status']}. "
        
        return summary
    
    def _generate_recommendations(self, case_data: Dict) -> List[str]:
        """Generate recommendations based on case"""
        recommendations = []
        
        if case_data["risk_score"] >= 80:
            recommendations.extend([
                "Implement additional authentication measures",
                "Review and update fraud detection rules",
                "Provide enhanced customer education"
            ])
        
        if case_data.get("quantexa_analysis", {}).get("network_connections", 0) > 2:
            recommendations.append("Investigate related entities in network")
        
        return recommendations
    
    def _generate_next_steps(self, case_data: Dict) -> List[str]:
        """Generate next steps for case"""
        next_steps = []
        
        if case_data["status"] != CaseStatus.RESOLVED.value:
            next_steps.extend(case_data.get("investigation_steps", []))
        else:
            next_steps.extend([
                "Monitor customer account for 30 days",
                "Update fraud detection models with case learnings",
                "Share findings with relevant teams"
            ])
        
        return next_steps
    
    async def _update_external_systems(self, case_id: str, updates: Dict):
        """Update case in external systems"""
        # Mock implementation
        logger.info(f"Updated external systems for case {case_id}")
    
    async def _send_escalation_notifications(self, case_data: Dict, reason: str):
        """Send escalation notifications"""
        # Mock implementation
        logger.info(f"Escalation notifications sent for case {case_data['case_id']}")
    
    async def _generate_final_report(self, case_data: Dict) -> Dict[str, Any]:
        """Generate final investigation report"""
        return await self._generate_standard_report(case_data)
    
    async def _archive_case(self, case_data: Dict):
        """Archive resolved case"""
        # In production, move to long-term storage
        case_data["archived_at"] = datetime.now().isoformat()
        logger.info(f"Case {case_data['case_id']} archived")
    
    async def _generate_executive_report(self, case_data: Dict) -> Dict[str, Any]:
        """Generate executive-level report"""
        report = await self._generate_standard_report(case_data)
        report["executive_focus"] = {
            "business_impact": "High" if case_data["risk_score"] >= 60 else "Medium",
            "regulatory_implications": self._assess_regulatory_impact(case_data),
            "strategic_recommendations": [
                "Enhance fraud detection capabilities",
                "Increase customer education initiatives"
            ]
        }
        return report
    
    async def _generate_technical_report(self, case_data: Dict) -> Dict[str, Any]:
        """Generate technical investigation report"""
        report = await self._generate_standard_report(case_data)
        report["technical_details"] = {
            "detection_patterns": case_data["evidence"]["detected_patterns"],
            "ml_model_performance": {
                "confidence_score": 0.92,
                "false_positive_rate": 0.05
            },
            "system_recommendations": [
                "Update pattern matching rules",
                "Retrain models with case data"
            ]
        }
        return report
    
    def _assess_regulatory_impact(self, case_data: Dict) -> str:
        """Assess regulatory impact of case"""
        if case_data["risk_score"] >= 80:
            return "SAR filing required under MLR 2017"
        elif case_data["risk_score"] >= 60:
            return "Enhanced due diligence recommended"
        else:
            return "Standard compliance procedures apply"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get case management metrics"""
        base_metrics = self.get_status()
        
        case_metrics = {
            "active_cases": len(self.active_cases),
            "cases_created_total": self.case_statistics["total_created"],
            "cases_escalated": self.case_statistics["escalated"],
            "cases_resolved": self.case_statistics["resolved"],
            "false_positive_rate": (
                self.case_statistics["false_positives"] / 
                max(self.case_statistics["resolved"], 1)
            ),
            "average_resolution_time_hours": 24.5,  # Mock value
            "quantexa_integration_status": "connected" if self.quantexa_connected else "disconnected",
            "servicenow_integration_status": "connected" if self.servicenow_connected else "disconnected"
        }
        
        base_metrics.update(case_metrics)
        return base_metrics

# Tool function for ADK
def create_fraud_case(
    session_id: str,
    fraud_analysis: Dict[str, Any],
    customer_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a fraud investigation case"""
    logger.info(f"📋 Creating fraud case for session {session_id}")
    
    risk_score = fraud_analysis.get("risk_score", 0)
    scam_type = fraud_analysis.get("scam_type", "unknown")
    
    # Generate case ID
    case_id = f"HSBC-FD-{datetime.now().strftime('%Y%m%d')}-{session_id[:8].upper()}"
    
    # Determine priority
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
    
    logger.info(f"✅ Case created: {case_id} - Priority: {priority}")
    
    return case_data

# Create the Case Management Agent using Google ADK
case_management_agent = Agent(
    name="case_management_agent",
    model=LiteLLM("gemini/gemini-2.0-flash"),
    description="Fraud case creation and management with Quantexa integration",
    instruction="""You are a case management specialist for HSBC fraud investigations.

Your responsibilities:
1. Create detailed investigation cases for detected fraud
2. Integrate with Quantexa for entity resolution and network analysis
3. Manage case workflow and assignments
4. Generate investigation reports

Key priorities:
- Ensure all evidence is properly documented
- Assign cases to appropriate teams based on severity
- Track investigation progress and outcomes
- Facilitate cross-system integration

Always maintain compliance with regulatory requirements and internal policies.""",
    tools=[create_fraud_case]
)
