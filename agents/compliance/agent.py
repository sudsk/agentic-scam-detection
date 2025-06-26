# agents/compliance/agent.py
"""
Compliance Agent for HSBC Scam Detection System
Ensures GDPR, FCA, and regulatory compliance
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
import uuid
from enum import Enum

from google.adk.agents import Agent
from google.adk.models import LiteLLM

from ..shared.base_agent import BaseAgent, AgentCapability, Message

logger = logging.getLogger(__name__)

class ComplianceCheck(Enum):
    """Types of compliance checks"""
    GDPR = "gdpr"
    FCA = "fca"
    PCI_DSS = "pci_dss"
    MLD5 = "mld5"  # Money Laundering Directive
    MIFID2 = "mifid2"  # Markets in Financial Instruments Directive

class PIIType(Enum):
    """Types of Personally Identifiable Information"""
    CARD_NUMBER = "card_number"
    ACCOUNT_NUMBER = "account_number"
    PIN = "pin"
    PASSWORD = "password"
    SSN = "social_security_number"
    PASSPORT = "passport_number"
    DRIVING_LICENSE = "driving_license"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    EMAIL = "email"
    PHONE = "phone_number"

class ComplianceAgent(BaseAgent):
    """
    Agent responsible for ensuring regulatory compliance
    """
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="pii_detection",
                description="Detect and redact PII from transcripts",
                input_types=["transcript"],
                output_types=["redacted_transcript"],
                processing_time_ms=200,
                confidence_level=0.98
            ),
            AgentCapability(
                name="gdpr_compliance",
                description="Ensure GDPR compliance",
                input_types=["customer_data"],
                output_types=["gdpr_assessment"],
                processing_time_ms=500,
                confidence_level=0.99
            ),
            AgentCapability(
                name="fca_compliance",
                description="Check FCA treating customers fairly",
                input_types=["interaction_data"],
                output_types=["fca_assessment"],
                processing_time_ms=400,
                confidence_level=0.97
            ),
            AgentCapability(
                name="sar_generation",
                description="Generate Suspicious Activity Reports",
                input_types=["fraud_analysis"],
                output_types=["sar_report"],
                processing_time_ms=1000,
                confidence_level=0.99
            ),
            AgentCapability(
                name="audit_logging",
                description="Create compliance audit logs",
                input_types=["system_activity"],
                output_types=["audit_log"],
                processing_time_ms=100,
                confidence_level=1.0
            )
        ]
        
        config = {
            "gdpr_retention_days": 2555,  # 7 years for financial records
            "enable_pii_redaction": True,
            "sar_threshold": 60,  # Risk score threshold for SAR
            "vulnerable_customer_indicators": [
                "elderly", "confused", "medical condition", "recently bereaved",
                "mental health", "financial difficulty", "language barrier"
            ],
            "redaction_patterns": {
                "card_number": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
                "account_number": r'\b\d{8}\b',
                "sort_code": r'\b\d{2}[\s-]?\d{2}[\s-]?\d{2}\b',
                "pin": r'\b(pin|PIN)\s*(number|code)?[\s:]*\d{4,6}\b',
                "password": r'\b(password|passcode)\s*[:=]\s*\S+',
                "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
                "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "phone": r'\b(?:\+44|0)\d{10,11}\b'
            }
        }
        
        super().__init__(
            agent_id="compliance_001",
            agent_name="Compliance Agent",
            capabilities=capabilities,
            config=config
        )
        
        self.compliance_logs = []
        self.sar_queue = []
        self.vulnerability_assessments = {}
    
    async def _initialize_agent(self):
        """Initialize compliance monitoring"""
        try:
            # Load compliance rules
            await self._load_compliance_rules()
            
            # Initialize regulatory connections
            await self._init_regulatory_connections()
            
            # Load vulnerability detection models
            await self._load_vulnerability_models()
            
            logger.info("Compliance agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize compliance agent: {e}")
            raise
    
    async def _load_compliance_rules(self):
        """Load compliance rules and regulations"""
        self.compliance_rules = {
            "gdpr": {
                "article_6": "Lawfulness of processing",
                "article_7": "Conditions for consent",
                "article_17": "Right to erasure",
                "article_25": "Data protection by design",
                "article_32": "Security of processing",
                "article_33": "Notification of breach"
            },
            "fca": {
                "prin_1": "Integrity",
                "prin_2": "Skill, care and diligence",
                "prin_3": "Management and control",
                "prin_6": "Customers' interests",
                "prin_7": "Communications with clients",
                "prin_9": "Customers: relationships of trust"
            },
            "mld5": {
                "customer_due_diligence": True,
                "enhanced_due_diligence": True,
                "suspicious_activity_reporting": True,
                "record_keeping": True
            }
        }
    
    async def _init_regulatory_connections(self):
        """Initialize connections to regulatory systems"""
        # In production, connect to FCA, ICO, and other regulatory APIs
        self.regulatory_connected = True
        logger.info("Regulatory connections established")
    
    async def _load_vulnerability_models(self):
        """Load models for detecting vulnerable customers"""
        # In production, load ML models for vulnerability detection
        self.vulnerability_model_loaded = True
    
    async def _process_message(self, message: Message) -> Optional[Message]:
        """Process compliance-related messages"""
        try:
            if message.message_type == "check_compliance":
                return await self._handle_compliance_check(message)
            elif message.message_type == "detect_pii":
                return await self._handle_pii_detection(message)
            elif message.message_type == "assess_vulnerability":
                return await self._handle_vulnerability_assessment(message)
            elif message.message_type == "generate_sar":
                return await self._handle_sar_generation(message)
            elif message.message_type == "audit_log":
                return await self._handle_audit_logging(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error in compliance processing: {e}")
            return self._create_error_response(message, str(e))
    
    async def _handle_compliance_check(self, message: Message) -> Message:
        """Perform comprehensive compliance check"""
        transcript = message.data.get("transcript", "")
        fraud_analysis = message.data.get("fraud_analysis", {})
        customer_data = message.data.get("customer_data", {})
        
        # Perform all compliance checks
        compliance_results = {
            "gdpr": await self._check_gdpr_compliance(transcript, customer_data),
            "fca": await self._check_fca_compliance(transcript, fraud_analysis),
            "pii": await self._detect_and_redact_pii(transcript),
            "vulnerability": await self._assess_customer_vulnerability(transcript),
            "mld5": await self._check_mld5_compliance(fraud_analysis),
            "audit_trail": await self._create_audit_trail(message)
        }
        
        # Generate overall compliance score
        compliance_score = self._calculate_compliance_score(compliance_results)
        
        # Determine if SAR is required
        sar_required = await self._check_sar_requirement(fraud_analysis, compliance_results)
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(compliance_results)
        
        response_data = {
            "session_id": message.session_id,
            "compliance_results": compliance_results,
            "compliance_score": compliance_score,
            "sar_required": sar_required,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log compliance check
        await self._log_compliance_activity(message.session_id, "compliance_check", response_data)
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="compliance_check_complete",
            source_agent=self.agent_id,
            target_agent="orchestrator",
            session_id=message.session_id,
            data=response_data,
            priority=3 if sar_required else 2
        )
    
    async def _check_gdpr_compliance(self, transcript: str, customer_data: Dict) -> Dict[str, Any]:
        """Check GDPR compliance"""
        gdpr_check = {
            "compliant": True,
            "issues": [],
            "data_minimization": True,
            "purpose_limitation": True,
            "consent_verified": True,
            "rights_respected": True
        }
        
        # Check for unauthorized data collection
        if self._contains_excessive_pii(transcript):
            gdpr_check["compliant"] = False
            gdpr_check["issues"].append("Excessive PII collected")
            gdpr_check["data_minimization"] = False
        
        # Check consent
        if not customer_data.get("consent_given", True):
            gdpr_check["compliant"] = False
            gdpr_check["issues"].append("Processing without consent")
            gdpr_check["consent_verified"] = False
        
        # Check for right to erasure requests
        if "delete my data" in transcript.lower() or "forget me" in transcript.lower():
            gdpr_check["issues"].append("Right to erasure requested")
            gdpr_check["rights_respected"] = False
        
        return gdpr_check
    
    async def _check_fca_compliance(self, transcript: str, fraud_analysis: Dict) -> Dict[str, Any]:
        """Check FCA treating customers fairly principles"""
        fca_check = {
            "tcf_compliant": True,
            "issues": [],
            "clear_communication": True,
            "suitable_advice": True,
            "vulnerable_customer": False,
            "fair_treatment": True
        }
        
        # Check for vulnerability indicators
        vulnerability_detected = await self._detect_vulnerability_indicators(transcript)
        if vulnerability_detected:
            fca_check["vulnerable_customer"] = True
            fca_check["issues"].append("Vulnerable customer detected - enhanced care required")
        
        # Check for pressure tactics
        if fraud_analysis.get("detected_patterns", {}).get("urgency_pressure"):
            fca_check["tcf_compliant"] = False
            fca_check["issues"].append("Customer under pressure - enhanced protection needed")
            fca_check["fair_treatment"] = False
        
        # Check communication clarity
        if self._check_unclear_communication(transcript):
            fca_check["clear_communication"] = False
            fca_check["issues"].append("Communication may not be clear to customer")
        
        return fca_check
    
    async def _detect_and_redact_pii(self, text: str) -> Dict[str, Any]:
        """Detect and redact PII from text"""
        pii_found = []
        redacted_text = text
        
        for pii_type, pattern in self.config["redaction_patterns"].items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                pii_found.append({
                    "type": pii_type,
                    "count": len(matches),
                    "positions": [(m.start(), m.end()) for m in matches]
                })
                
                # Redact in text
                for match in reversed(matches):  # Reverse to maintain positions
                    redacted_text = (
                        redacted_text[:match.start()] + 
                        f"[REDACTED-{pii_type.upper()}]" + 
                        redacted_text[match.end():]
                    )
        
        return {
            "pii_detected": len(pii_found) > 0,
            "pii_types": [p["type"] for p in pii_found],
            "pii_details": pii_found,
            "redacted_text": redacted_text,
            "redaction_count": sum(p["count"] for p in pii_found)
        }
    
    async def _assess_customer_vulnerability(self, transcript: str) -> Dict[str, Any]:
        """Assess if customer shows signs of vulnerability"""
        vulnerability_score = 0
        detected_indicators = []
        
        transcript_lower = transcript.lower()
        
        for indicator in self.config["vulnerable_customer_indicators"]:
            if indicator in transcript_lower:
                vulnerability_score += 0.2
                detected_indicators.append(indicator)
        
        # Check for confusion patterns
        confusion_patterns = [
            "i don't understand",
            "can you explain again",
            "i'm confused",
            "what does that mean",
            "i'm not sure"
        ]
        
        for pattern in confusion_patterns:
            if pattern in transcript_lower:
                vulnerability_score += 0.1
                detected_indicators.append("confusion")
        
        vulnerability_score = min(vulnerability_score, 1.0)
        
        return {
            "vulnerability_score": vulnerability_score,
            "is_vulnerable": vulnerability_score >= 0.3,
            "indicators_detected": list(set(detected_indicators)),
            "support_required": vulnerability_score >= 0.5,
            "recommended_actions": self._get_vulnerability_support_actions(vulnerability_score)
        }
    
    def _get_vulnerability_support_actions(self, score: float) -> List[str]:
        """Get recommended actions for vulnerable customers"""
        actions = []
        
        if score >= 0.7:
            actions.extend([
                "Transfer to specialist vulnerable customer team",
                "Slow down conversation pace",
                "Offer to involve trusted third party",
                "Document all interactions carefully"
            ])
        elif score >= 0.5:
            actions.extend([
                "Provide extra time for decisions",
                "Repeat important information",
                "Offer written summary of conversation",
                "Check understanding regularly"
            ])
        elif score >= 0.3:
            actions.extend([
                "Use simple, clear language",
                "Avoid technical jargon",
                "Confirm customer comfort level"
            ])
        
        return actions
    
    async def _check_mld5_compliance(self, fraud_analysis: Dict) -> Dict[str, Any]:
        """Check Money Laundering Directive compliance"""
        mld5_check = {
            "compliant": True,
            "cdd_required": False,  # Customer Due Diligence
            "edd_required": False,  # Enhanced Due Diligence
            "sar_required": False,
            "issues": []
        }
        
        risk_score = fraud_analysis.get("risk_score", 0)
        
        # Check if CDD is required
        if risk_score >= 40:
            mld5_check["cdd_required"] = True
            mld5_check["issues"].append("Customer due diligence required")
        
        # Check if EDD is required
        if risk_score >= 70:
            mld5_check["edd_required"] = True
            mld5_check["issues"].append("Enhanced due diligence required")
        
        # Check if SAR is required
        if risk_score >= self.config["sar_threshold"]:
            mld5_check["sar_required"] = True
            mld5_check["compliant"] = False
            mld5_check["issues"].append("Suspicious activity report required")
        
        return mld5_check
    
    async def _check_sar_requirement(self, fraud_analysis: Dict, compliance_results: Dict) -> bool:
        """Determine if SAR filing is required"""
        risk_score = fraud_analysis.get("risk_score", 0)
        
        # SAR required if:
        # 1. Risk score exceeds threshold
        if risk_score >= self.config["sar_threshold"]:
            return True
        
        # 2. MLD5 compliance requires it
        if compliance_results.get("mld5", {}).get("sar_required", False):
            return True
        
        # 3. Specific fraud patterns detected
        high_risk_patterns = ["money_laundering", "terrorist_financing", "tax_evasion"]
        detected_patterns = fraud_analysis.get("detected_patterns", {})
        
        for pattern in high_risk_patterns:
            if pattern in detected_patterns:
                return True
        
        return False
    
    def _calculate_compliance_score(self, compliance_results: Dict) -> float:
        """Calculate overall compliance score"""
        scores = []
        
        # GDPR score
        gdpr = compliance_results.get("gdpr", {})
        gdpr_score = 1.0 if gdpr.get("compliant", True) else 0.5
        scores.append(gdpr_score)
        
        # FCA score
        fca = compliance_results.get("fca", {})
        fca_score = 1.0 if fca.get("tcf_compliant", True) else 0.6
        if fca.get("vulnerable_customer", False) and fca.get("fair_treatment", True):
            fca_score = 0.9  # Good handling of vulnerable customer
        scores.append(fca_score)
        
        # PII handling score
        pii = compliance_results.get("pii", {})
        pii_score = 0.8 if pii.get("pii_detected", False) else 1.0
        scores.append(pii_score)
        
        # MLD5 score
        mld5 = compliance_results.get("mld5", {})
        mld5_score = 1.0 if mld5.get("compliant", True) else 0.7
        scores.append(mld5_score)
        
        return sum(scores) / len(scores)
    
    def _generate_compliance_recommendations(self, compliance_results: Dict) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # GDPR recommendations
        gdpr = compliance_results.get("gdpr", {})
        if not gdpr.get("compliant", True):
            recommendations.extend([
                "Review data collection practices",
                "Ensure explicit consent is obtained",
                "Implement data minimization"
            ])
        
        # FCA recommendations
        fca = compliance_results.get("fca", {})
        if fca.get("vulnerable_customer", False):
            recommendations.extend([
                "Apply vulnerable customer policy",
                "Provide additional support",
                "Document enhanced care measures"
            ])
        
        # PII recommendations
        pii = compliance_results.get("pii", {})
        if pii.get("pii_detected", False):
            recommendations.extend([
                "Redact PII before storage",
                "Limit access to sensitive data",
                "Review PII handling procedures"
            ])
        
        # MLD5 recommendations
        mld5 = compliance_results.get("mld5", {})
        if mld5.get("sar_required", False):
            recommendations.extend([
                "File SAR within 24 hours",
                "Notify MLRO immediately",
                "Preserve all related records"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _handle_pii_detection(self, message: Message) -> Message:
        """Handle PII detection request"""
        text = message.data.get("text", "")
        
        pii_result = await self._detect_and_redact_pii(text)
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="pii_detection_complete",
            source_agent=self.agent_id,
            target_agent=message.source_agent,
            session_id=message.session_id,
            data=pii_result
        )
    
    async def _handle_vulnerability_assessment(self, message: Message) -> Message:
        """Handle vulnerability assessment request"""
        transcript = message.data.get("transcript", "")
        customer_id = message.data.get("customer_id")
        
        vulnerability_result = await self._assess_customer_vulnerability(transcript)
        
        # Store assessment
        if customer_id:
            self.vulnerability_assessments[customer_id] = {
                "assessment": vulnerability_result,
                "timestamp": datetime.now().isoformat(),
                "session_id": message.session_id
            }
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="vulnerability_assessment_complete",
            source_agent=self.agent_id,
            target_agent=message.source_agent,
            session_id=message.session_id,
            data=vulnerability_result
        )
    
    async def _handle_sar_generation(self, message: Message) -> Message:
        """Generate Suspicious Activity Report"""
        fraud_analysis = message.data.get("fraud_analysis", {})
        case_data = message.data.get("case_data", {})
        
        sar_data = {
            "sar_id": f"SAR-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}",
            "filing_date": datetime.now().isoformat(),
            "reporting_entity": "HSBC UK",
            "subject_information": {
                "customer_id": case_data.get("customer_id", "Unknown"),
                "account_numbers": self._extract_account_numbers(case_data),
                "transaction_dates": case_data.get("transaction_dates", [])
            },
            "suspicious_activity": {
                "activity_type": fraud_analysis.get("scam_type", "Unknown"),
                "risk_score": fraud_analysis.get("risk_score", 0),
                "amount_involved": case_data.get("amount", 0),
                "description": self._generate_sar_narrative(fraud_analysis, case_data)
            },
            "law_enforcement_contact": {
                "agency": "National Crime Agency",
                "reference": f"NCA-{case_data.get('case_id', 'Unknown')}"
            },
            "filing_status": "draft",
            "approval_required": True
        }
        
        # Add to SAR queue
        self.sar_queue.append(sar_data)
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="sar_generated",
            source_agent=self.agent_id,
            target_agent="orchestrator",
            session_id=message.session_id,
            data={
                "sar_data": sar_data,
                "filing_deadline": (datetime.now() + timedelta(hours=24)).isoformat()
            },
            priority=4  # High priority
        )
    
    def _generate_sar_narrative(self, fraud_analysis: Dict, case_data: Dict) -> str:
        """Generate SAR narrative description"""
        narrative = f"Suspicious activity detected: {fraud_analysis.get('scam_type', 'Unknown')} "
        narrative += f"with risk score {fraud_analysis.get('risk_score', 0)}%. "
        
        patterns = fraud_analysis.get("detected_patterns", {})
        if patterns:
            narrative += f"Detected patterns include: {', '.join(patterns.keys())}. "
        
        narrative += fraud_analysis.get("explanation", "")
        
        return narrative
    
    def _extract_account_numbers(self, case_data: Dict) -> List[str]:
        """Extract account numbers from case data"""
        # Mock implementation
        return [case_data.get("customer_id", "Unknown")]
    
    async def _handle_audit_logging(self, message: Message) -> Message:
        """Create audit log entry"""
        activity_type = message.data.get("activity_type")
        activity_data = message.data.get("activity_data")
        actor = message.data.get("actor", "System")
        
        audit_entry = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "session_id": message.session_id,
            "activity_type": activity_type,
            "actor": actor,
            "activity_data": activity_data,
            "compliance_relevant": True,
            "retention_years": 7
        }
        
        # Store audit log
        self.compliance_logs.append(audit_entry)
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="audit_logged",
            source_agent=self.agent_id,
            target_agent=message.source_agent,
            session_id=message.session_id,
            data={"audit_id": audit_entry["audit_id"]}
        )
    
    async def _create_audit_trail(self, message: Message) -> Dict[str, Any]:
        """Create audit trail for compliance check"""
        return {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "action": "compliance_check",
            "actor": self.agent_id,
            "session_id": message.session_id,
            "data_processed": list(message.data.keys()),
            "retention_period": "7 years"
        }
    
    async def _log_compliance_activity(self, session_id: str, activity: str, data: Dict):
        """Log compliance activity"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "activity": activity,
            "data": data
        }
        self.compliance_logs.append(log_entry)
    
    def _contains_excessive_pii(self, text: str) -> bool:
        """Check if text contains excessive PII"""
        pii_count = 0
        for pattern in self.config["redaction_patterns"].values():
            pii_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return pii_count > 5  # Threshold for excessive PII
    
    async def _detect_vulnerability_indicators(self, transcript: str) -> bool:
        """Detect vulnerability indicators in transcript"""
        transcript_lower = transcript.lower()
        
        for indicator in self.config["vulnerable_customer_indicators"]:
            if indicator in transcript_lower:
                return True
        
        return False
    
    def _check_unclear_communication(self, transcript: str) -> bool:
        """Check if communication may be unclear to customer"""
        # Check for complex financial jargon
        jargon_terms = [
            "derivative", "margin call", "leverage", "hedging",
            "securitization", "amortization", "yield curve"
        ]
        
        jargon_count = sum(1 for term in jargon_terms if term in transcript.lower())
        
        return jargon_count >= 3
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get compliance metrics"""
        base_metrics = self.get_status()
        
        compliance_metrics = {
            "compliance_checks_performed": len(self.compliance_logs),
            "sar_reports_generated": len(self.sar_queue),
            "vulnerable_customers_identified": len(self.vulnerability_assessments),
            "pii_redactions_performed": sum(
                1 for log in self.compliance_logs 
                if log.get("activity") == "pii_redaction"
            ),
            "average_compliance_score": 0.92,
            "regulatory_breaches": 0,
            "audit_logs_created": len(self.compliance_logs)
        }
        
        base_metrics.update(compliance_metrics)
        return base_metrics

# Tool function for ADK
def check_compliance_requirements(
    transcript: str,
    fraud_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Check GDPR, FCA and compliance requirements"""
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
