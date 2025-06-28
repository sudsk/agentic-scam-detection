# agents/case_management/agent.py - UPDATED
"""
Case Management Agent - Creates and manages fraud investigation cases
UPDATED: Removed wrapper functions, uses centralized config
"""

import time
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

class CaseManagementAgent(BaseAgent):
    """Agent responsible for creating and managing fraud investigation cases"""
    
    def __init__(self):
        super().__init__(
            agent_type="case_management",
            agent_name="Case Management Agent"
        )
        
        # Initialize case storage and statistics
        self._initialize_case_system()
        
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"📋 {self.agent_name} ready for case management")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration from centralized settings"""
        settings = get_settings()
        return settings.get_agent_config("case_management")
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get case management agent capabilities"""
        return [
            AgentCapability.CASE_MANAGEMENT,
            AgentCapability.ESCALATION_MANAGEMENT
        ]
    
    def _initialize_case_system(self):
        """Initialize case management system"""
        # Active cases storage (in production, this would be a database)
        self.active_cases = {}
        self.case_statistics = {
            "total_cases": 0,
            "high_risk_cases": 0,
            "resolved_cases": 0,
            "false_positives": 0,
            "cases_by_type": {},
            "cases_by_priority": {},
            "average_resolution_time": 0.0
        }
        
        logger.info(f"✅ Case management system initialized")
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Main processing method - create or manage fraud cases"""
        if isinstance(input_data, dict):
            # Check if this is a case creation request
            if 'session_id' in input_data and 'fraud_analysis' in input_data:
                return self.create_fraud_case(
                    input_data['session_id'],
                    input_data['fraud_analysis'],
                    input_data.get('customer_id')
                )
            elif 'case_id' in input_data and 'action' in input_data:
                # Case management action
                return self._handle_case_action(input_data)
        
        return {
            "error": "Invalid input data for case management",
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def create_fraud_case(
        self, 
        session_id: str, 
        fraud_analysis: Dict[str, Any], 
        customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a fraud investigation case"""
        start_time = time.time()
        
        try:
            self.log_activity(f"Creating fraud case", {"session_id": session_id})
            
            risk_score = fraud_analysis.get("risk_score", 0)
            scam_type = fraud_analysis.get("scam_type", "unknown")
            
            # Generate case ID using centralized prefix
            case_id = self._generate_case_id()
            
            # Determine priority and assignment using centralized config
            priority = self._determine_priority(risk_score)
            assigned_team = self.config["team_assignments"][priority]
            
            # Create case data
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
                "updated_at": datetime.now().isoformat(),
                "evidence": {
                    "transcript": fraud_analysis.get("transcript_text", ""),
                    "detected_patterns": fraud_analysis.get("detected_patterns", {}),
                    "risk_factors": fraud_analysis.get("explanation", ""),
                    "confidence_score": fraud_analysis.get("confidence", 0.0)
                },
                "investigation_steps": self._get_investigation_steps(scam_type),
                "next_actions": self._get_next_actions(priority, scam_type),
                "timeline": [{
                    "timestamp": datetime.now().isoformat(),
                    "action": "Case created",
                    "actor": self.agent_id,
                    "details": f"Automated case creation from session {session_id}"
                }],
                "case_agent": self.agent_id
            }
            
            # Store case
            self.active_cases[case_id] = case_data
            
            # Update statistics
            self._update_statistics(case_data, "created")
            
            # Update metrics
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=True)
            
            self.log_activity(f"Case created successfully", {
                "case_id": case_id, 
                "priority": priority,
                "processing_time": processing_time
            })
            
            return case_data
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=False)
            self.handle_error(e, "create_fraud_case")
            
            return {
                "error": str(e),
                "session_id": session_id,
                "case_agent": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_case_id(self) -> str:
        """Generate unique case ID using centralized prefix"""
        date_str = datetime.now().strftime("%Y%m%d")
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"{self.config['case_id_prefix']}-{date_str}-{unique_id}"
    
    def _determine_priority(self, risk_score: float) -> str:
        """Determine case priority based on centralized risk thresholds"""
        thresholds = self.config["risk_thresholds"]
        
        if risk_score >= thresholds["critical"]:
            return "critical"
        elif risk_score >= thresholds["high"]:
            return "high"
        elif risk_score >= thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _get_investigation_steps(self, scam_type: str) -> List[str]:
        """Get investigation steps based on scam type"""
        investigation_templates = {
            "investment_scam": [
                "Verify investment company registration with regulatory authorities",
                "Check FCA/SEC warnings list for company name",
                "Review customer transaction history for similar patterns",
                "Identify related cases with same company/advisor",
                "Contact customer for additional verification"
            ],
            "romance_scam": [
                "Document relationship timeline and communication channels",
                "Verify beneficiary identity and location",
                "Check for similar cases with same beneficiary details",
                "Assess customer vulnerability factors",
                "Provide victim support resources"
            ],
            "impersonation_scam": [
                "Verify caller identity claims against official records",
                "Report to relevant authority fraud departments",
                "Check for potential data breach sources",
                "Review account security measures",
                "Implement additional customer safeguards"
            ],
            "authority_scam": [
                "Document claimed authority and threats made",
                "Verify with actual authority if contact was legitimate",
                "Check customer for similar previous contacts",
                "Provide education on authority contact procedures",
                "Report to appropriate law enforcement"
            ]
        }
        
        return investigation_templates.get(scam_type, [
            "Review case evidence and documentation",
            "Verify customer identity and account status",
            "Check for similar fraud patterns",
            "Document findings and recommendations"
        ])
    
    def _get_next_actions(self, priority: str, scam_type: str) -> List[str]:
        """Get immediate next actions based on priority and scam type"""
        if priority == "critical":
            return [
                "Block all pending transactions immediately",
                "Contact customer via secure channel within 15 minutes",
                "Notify Financial Crime team lead",
                "Initiate account monitoring",
                "Prepare emergency response procedures"
            ]
        elif priority == "high":
            return [
                "Review and hold pending transactions",
                "Enhanced authentication for all new transactions",
                "Schedule customer callback within 1 hour",
                "Flag account for monitoring",
                "Notify fraud prevention team"
            ]
        elif priority == "medium":
            return [
                "Monitor account activity for 24 hours",
                "Review recent transaction history",
                "Prepare customer education materials",
                "Schedule follow-up within 48 hours"
            ]
        else:
            return [
                "Document interaction details",
                "Standard account monitoring",
                "No immediate action required"
            ]
    
    def _handle_case_action(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle various case management actions"""
        case_id = input_data.get("case_id")
        action = input_data.get("action")
        
        if case_id not in self.active_cases:
            return {"error": f"Case {case_id} not found", "agent_id": self.agent_id}
        
        try:
            if action == "update":
                return self.update_case(case_id, input_data.get("updates", {}))
            elif action == "resolve":
                return self.resolve_case(case_id, input_data.get("resolution", {}))
            elif action == "escalate":
                return self.escalate_case(case_id, input_data.get("escalation_data", {}))
            elif action == "add_note":
                return self.add_case_note(case_id, input_data.get("note", ""))
            else:
                return {"error": f"Unknown action: {action}", "agent_id": self.agent_id}
                
        except Exception as e:
            self.handle_error(e, f"_handle_case_action_{action}")
            return {"error": str(e), "agent_id": self.agent_id}
    
    def update_case(self, case_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing case"""
        if case_id not in self.active_cases:
            raise ValueError(f"Case {case_id} not found")
        
        case_data = self.active_cases[case_id]
        
        # Apply updates
        for key, value in updates.items():
            if key in case_data:
                case_data[key] = value
        
        # Update timestamp
        case_data["updated_at"] = datetime.now().isoformat()
        
        # Add to timeline
        case_data["timeline"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "Case updated",
            "actor": updates.get("updated_by", self.agent_id),
            "details": f"Updated fields: {', '.join(updates.keys())}"
        })
        
        self.log_activity(f"Case updated", {"case_id": case_id, "updates": list(updates.keys())})
        return case_data
    
    def resolve_case(self, case_id: str, resolution: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve and close case"""
        if case_id not in self.active_cases:
            raise ValueError(f"Case {case_id} not found")
        
        case_data = self.active_cases[case_id]
        
        # Update case status
        case_data["status"] = "resolved"
        case_data["resolved_at"] = datetime.now().isoformat()
        case_data["resolution"] = {
            "outcome": resolution.get("outcome", "investigation_complete"),
            "fraud_confirmed": resolution.get("fraud_confirmed", False),
            "amount_involved": resolution.get("amount_involved", 0),
            "customer_protected": resolution.get("customer_protected", True),
            "lessons_learned": resolution.get("lessons_learned", []),
            "resolved_by": resolution.get("resolved_by", self.agent_id)
        }
        
        # Update statistics
        self._update_statistics(case_data, "resolved")
        
        # Add to timeline
        case_data["timeline"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "Case resolved",
            "actor": resolution.get("resolved_by", self.agent_id),
            "details": f"Outcome: {resolution.get('outcome', 'investigation_complete')}"
        })
        
        self.log_activity(f"Case resolved", {"case_id": case_id, "outcome": resolution.get('outcome')})
        return case_data
    
    def escalate_case(self, case_id: str, escalation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Escalate case to higher priority"""
        if case_id not in self.active_cases:
            raise ValueError(f"Case {case_id} not found")
        
        case_data = self.active_cases[case_id]
        escalated_to = escalation_data.get("escalated_to", self.config["team_assignments"]["critical"])
        reason = escalation_data.get("reason", "Risk level increased")
        
        # Update priority
        old_priority = case_data["priority"]
        if old_priority != "critical":
            case_data["priority"] = "high" if old_priority == "medium" else "critical"
        
        # Update assignment
        case_data["assigned_team"] = escalated_to
        case_data["updated_at"] = datetime.now().isoformat()
        
        # Add escalation to timeline
        escalation_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "Case escalated",
            "actor": self.agent_id,
            "details": f"Escalated from {old_priority} to {case_data['priority']} - Reason: {reason}",
            "escalated_to": escalated_to
        }
        
        case_data["timeline"].append(escalation_record)
        
        self.log_activity(f"Case escalated", {"case_id": case_id, "from": old_priority, "to": case_data['priority']})
        return case_data
    
    def add_case_note(self, case_id: str, note: str) -> Dict[str, Any]:
        """Add a note to a case"""
        if case_id not in self.active_cases:
            raise ValueError(f"Case {case_id} not found")
        
        case_data = self.active_cases[case_id]
        
        if "notes" not in case_data:
            case_data["notes"] = []
        
        new_note = {
            "note": note,
            "added_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
        
        case_data["notes"].append(new_note)
        case_data["updated_at"] = datetime.now().isoformat()
        
        self.log_activity(f"Note added to case", {"case_id": case_id})
        return {"note_added": True, "case_id": case_id}
    
    def _update_statistics(self, case_data: Dict[str, Any], action: str):
        """Update case statistics"""
        if action == "created":
            self.case_statistics["total_cases"] += 1
            
            # Update by type
            fraud_type = case_data.get("fraud_type", "unknown")
            if fraud_type not in self.case_statistics["cases_by_type"]:
                self.case_statistics["cases_by_type"][fraud_type] = 0
            self.case_statistics["cases_by_type"][fraud_type] += 1
            
            # Update by priority
            priority = case_data.get("priority", "low")
            if priority not in self.case_statistics["cases_by_priority"]:
                self.case_statistics["cases_by_priority"][priority] = 0
            self.case_statistics["cases_by_priority"][priority] += 1
            
            # High risk cases
            if case_data.get("priority") in ["critical", "high"]:
                self.case_statistics["high_risk_cases"] += 1
                
        elif action == "resolved":
            self.case_statistics["resolved_cases"] += 1
            
            # Calculate resolution time if available
            created_at = case_data.get("created_at")
            resolved_at = case_data.get("resolved_at")
            if created_at and resolved_at:
                try:
                    created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    resolved_time = datetime.fromisoformat(resolved_at.replace('Z', '+00:00'))
                    resolution_hours = (resolved_time - created_time).total_seconds() / 3600
                    
                    # Update average resolution time
                    current_avg = self.case_statistics["average_resolution_time"]
                    resolved_count = self.case_statistics["resolved_cases"]
                    self.case_statistics["average_resolution_time"] = (
                        (current_avg * (resolved_count - 1) + resolution_hours) / resolved_count
                    )
                except Exception as e:
                    logger.warning(f"Could not calculate resolution time: {e}")
            
            # Check for false positive
            resolution = case_data.get("resolution", {})
            if not resolution.get("fraud_confirmed", False):
                self.case_statistics["false_positives"] += 1

# Create global instance
case_management_agent = CaseManagementAgent()
