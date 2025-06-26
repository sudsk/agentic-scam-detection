# agents/case_management/agent.py
"""
Case Management Agent - Creates and manages fraud investigation cases
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

logger = logging.getLogger(__name__)

class CaseManagementAgent:
    """Agent responsible for creating and managing fraud investigation cases"""
    
    def __init__(self):
        self.agent_id = "case_management_001"
        self.agent_name = "Case Management Agent"
        self.status = "active"
        
        # Configuration
        self.config = {
            "case_id_prefix": "FD",
            "priority_thresholds": {
                "critical": 80,
                "high": 60,
                "medium": 40,
                "low": 0
            },
            "team_assignments": {
                "critical": "financial-crime-immediate@bank.com",
                "high": "fraud-investigation-team@bank.com",
                "medium": "fraud-prevention-team@bank.com",
                "low": "customer-service-enhanced@bank.com"
            },
            "auto_escalation_enabled": True,
            "case_retention_days": 2555  # 7 years
        }
        
        # Active cases storage (in production, this would be a database)
        self.active_cases = {}
        self.case_statistics = {
            "total_cases": 0,
            "high_risk_cases": 0,
            "resolved_cases": 0,
            "false_positives": 0
        }
        
        logger.info(f"✅ {self.agent_name} initialized")
    
    def create_fraud_case(
        self, 
        session_id: str, 
        fraud_analysis: Dict[str, Any], 
        customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a fraud investigation case
        
        Args:
            session_id: Call session ID
            fraud_analysis: Fraud detection results
            customer_id: Customer identifier
            
        Returns:
            Dict with case details
        """
        logger.info(f"📋 Creating fraud case for session {session_id}")
        
        try:
            risk_score = fraud_analysis.get("risk_score", 0)
            scam_type = fraud_analysis.get("scam_type", "unknown")
            
            # Generate case ID
            case_id = self._generate_case_id()
            
            # Determine priority and assignment
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
            self.case_statistics["total_cases"] += 1
            if priority in ["critical", "high"]:
                self.case_statistics["high_risk_cases"] += 1
            
            logger.info(f"✅ Case created: {case_id} - Priority: {priority}")
            
            return case_data
            
        except Exception as e:
            logger.error(f"❌ Case creation failed: {e}")
            raise
    
    def _generate_case_id(self) -> str:
        """Generate unique case ID"""
        date_str = datetime.now().strftime("%Y%m%d")
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"{self.config['case_id_prefix']}-{date_str}-{unique_id}"
    
    def _determine_priority(self, risk_score: float) -> str:
        """Determine case priority based on risk score"""
        thresholds = self.config["priority_thresholds"]
        
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
        
        logger.info(f"📝 Case updated: {case_id}")
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
        self.case_statistics["resolved_cases"] += 1
        if not resolution.get("fraud_confirmed", False):
            self.case_statistics["false_positives"] += 1
        
        # Add to timeline
        case_data["timeline"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "Case resolved",
            "actor": resolution.get("resolved_by", self.agent_id),
            "details": f"Outcome: {resolution.get('outcome', 'investigation_complete')}"
        })
        
        logger.info(f"✅ Case resolved: {case_id}")
        return case_data
    
    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get case by ID"""
        return self.active_cases.get(case_id)
    
    def list_cases(self, status: Optional[str] = None, priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """List cases with optional filtering"""
        cases = list(self.active_cases.values())
        
        if status:
            cases = [case for case in cases if case["status"] == status]
        
        if priority:
            cases = [case for case in cases if case["priority"] == priority]
        
        return cases
    
    def get_case_statistics(self) -> Dict[str, Any]:
        """Get case management statistics"""
        return {
            **self.case_statistics,
            "active_cases": len([c for c in self.active_cases.values() if c["status"] == "open"]),
            "false_positive_rate": (
                self.case_statistics["false_positives"] / 
                max(self.case_statistics["resolved_cases"], 1)
            ),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "capabilities": [
                "case_creation",
                "case_management",
                "investigation_coordination",
                "team_assignment"
            ],
            "config": self.config,
            "active_cases": len(self.active_cases),
            "statistics": self.case_statistics
        }

# Create global instance
case_management_agent = CaseManagementAgent()

# Export function for system integration
def create_fraud_case(
    session_id: str,
    fraud_analysis: Dict[str, Any],
    customer_id: Optional[str] = None
) -> Dict[str, Any]:
    """Main function for case creation"""
    return case_management_agent.create_fraud_case(session_id, fraud_analysis, customer_id)
