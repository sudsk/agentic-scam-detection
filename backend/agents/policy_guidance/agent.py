# agents/policy_guidance/agent.py - UPDATED
"""
Policy Guidance Agent - Provides procedural guidance and escalation recommendations
UPDATED: Removed wrapper functions, uses centralized config
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings

logger = logging.getLogger(__name__)

class PolicyGuidanceAgent(BaseAgent):
    """Agent that provides fraud response policies and guidance"""
    
    def __init__(self):
        super().__init__(
            agent_type="policy_guidance",
            agent_name="Policy Guidance Agent"
        )
        
        # Load policy database
        self._load_policy_database()
        
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"📚 {self.agent_name} ready with {len(self.policy_database)} policies")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration from centralized settings"""
        settings = get_settings()
        return settings.get_agent_config("policy_guidance")
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get policy guidance agent capabilities"""
        return [
            AgentCapability.POLICY_GUIDANCE,
            AgentCapability.ESCALATION_MANAGEMENT
        ]
    
    def _load_policy_database(self):
        """Load comprehensive policy database"""
        settings = get_settings()
        
        # Use centralized escalation thresholds
        escalation_thresholds = {
            "investment_scam": settings.risk_threshold_critical,
            "romance_scam": settings.risk_threshold_high + 15,  # 75
            "impersonation_scam": settings.risk_threshold_critical + 5,  # 85
            "authority_scam": settings.risk_threshold_critical + 5  # 85
        }
        
        self.policy_database = {
            'investment_scam_response': {
                'policy_id': 'FP-INV-001',
                'title': 'Investment Scam Detection and Response',
                'version': self.config["policy_version"],
                'last_updated': '2025-01-15',
                'escalation_threshold': escalation_thresholds["investment_scam"],
                'procedures': [
                    'Immediately halt any investment-related transfers showing guaranteed return promises',
                    'Ask customer: "Have you been able to withdraw any profits from this investment?"',
                    'Verify investment company against regulatory register using official website',
                    'Explain: All legitimate investments carry risk - guarantees indicate fraud',
                    'Document company name, contact details, and promised returns for investigation',
                    'Transfer to Financial Crime Team if multiple red flags present'
                ],
                'key_questions': [
                    'Have you been able to withdraw any money from this investment?',
                    'Did they guarantee specific percentage returns?',
                    'How did this investment company first contact you?',
                    'Can you check if they are regulated by financial authorities?',
                    'Have you researched this company independently?'
                ],
                'customer_education': [
                    'All legitimate investments carry risk - guaranteed returns are impossible',
                    'Real investment firms are registered and regulated by financial authorities',
                    'High-pressure sales tactics are warning signs of fraud',
                    'Take time to research - legitimate opportunities don\'t require immediate action',
                    'Be wary of unsolicited investment opportunities via phone or email'
                ],
                'regulatory_requirements': [
                    'Document all interaction details for potential SAR filing',
                    'Report to Financial Conduct Authority if fraud confirmed',
                    'Maintain evidence chain for potential law enforcement referral'
                ]
            },
            
            'romance_scam_response': {
                'policy_id': 'FP-ROM-001',
                'title': 'Romance Scam Detection Procedures',
                'version': '2.0.3',
                'last_updated': '2024-12-20',
                'escalation_threshold': escalation_thresholds["romance_scam"],
                'procedures': [
                    'Stop transfers to individuals customer has never met in person',
                    'Ask: "Have you met this person face-to-face?" and "Have they asked for money before?"',
                    'Explain romance scam patterns: relationship building followed by emergency requests',
                    'Check customer\'s transfer history for previous payments to same recipient',
                    'Advise customer to discuss with family/friends before proceeding',
                    'Escalate if customer insists on transfer despite warnings'
                ],
                'key_questions': [
                    'Have you met this person in person?',
                    'Have they asked for money before?',
                    'Why can\'t they get help from family or their employer?',
                    'How long have you known them?',
                    'What platform did you meet them on?'
                ],
                'customer_education': [
                    'Romance scammers build relationships over months to gain trust',
                    'They often claim to be overseas, in military, or traveling for work',
                    'Real partners don\'t repeatedly ask for money, especially for emergencies',
                    'Video calls can be faked - only meeting in person confirms identity',
                    'Trust your instincts - if something feels wrong, it probably is'
                ],
                'vulnerability_considerations': [
                    'Assess customer emotional state and potential vulnerability',
                    'Provide additional support resources if needed',
                    'Consider follow-up contact after initial conversation'
                ]
            },
            
            'impersonation_scam_response': {
                'policy_id': 'FP-IMP-001',
                'title': 'Authority Impersonation Scam Procedures',
                'version': '2.2.0',
                'last_updated': '2025-01-10',
                'escalation_threshold': escalation_thresholds["impersonation_scam"],
                'procedures': [
                    'Immediately inform customer: Banks never ask for PINs or passwords over phone',
                    'Instruct customer to hang up and call official number independently',
                    'Explain: Legitimate authorities send official letters before taking action',
                    'Ask: "What number did they call from?" and verify against official numbers',
                    'Document all impersonation details including claimed authority and threats',
                    'Report incident to relevant authority fraud departments'
                ],
                'key_questions': [
                    'What number did they call you from?',
                    'Did they ask for your PIN or online banking password?',
                    'Have you received any official letters about this matter?',
                    'Why do they claim payment is needed immediately?',
                    'What authority or organization are they claiming to represent?'
                ],
                'customer_education': [
                    'Banks will never ask for your PIN, password, or full card details over the phone',
                    'Legitimate authorities send official letters before taking any action',
                    'Government agencies don\'t accept payments via bank transfer or gift cards',
                    'When in doubt, hang up and call the organization\'s official number',
                    'Scammers use fear tactics to pressure immediate action'
                ],
                'immediate_actions': [
                    'Block any pending transactions immediately',
                    'Change online banking passwords if potentially compromised',
                    'Report to Action Fraud and relevant authority'
                ]
            }
        }
        
        # Map scam types to policies
        self.scam_policy_map = {
            'investment_scam': 'investment_scam_response',
            'romance_scam': 'romance_scam_response',
            'impersonation_scam': 'impersonation_scam_response',
            'authority_scam': 'impersonation_scam_response',
            'app_fraud': 'impersonation_scam_response',
            'crypto_scam': 'investment_scam_response'
        }
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Main processing method - retrieve policies and guidance"""
        if isinstance(input_data, dict):
            scam_type = input_data.get('scam_type', 'unknown')
            risk_score = input_data.get('risk_score', 0)
        else:
            scam_type = 'unknown'
            risk_score = 0
        
        return self.retrieve_fraud_policies(scam_type, risk_score)
    
    def retrieve_fraud_policies(self, scam_type: str, risk_score: float) -> Dict[str, Any]:
        """Retrieve relevant fraud policies and procedures"""
        start_time = time.time()
        
        try:
            self.log_activity(f"Retrieving policies", {"scam_type": scam_type, "risk_score": risk_score})
            
            # Get the appropriate policy
            policy_key = self.scam_policy_map.get(scam_type, 'investment_scam_response')
            policy = self.policy_database.get(policy_key, {})
            
            if not policy:
                raise ValueError(f"No policy found for scam type: {scam_type}")
            
            # Generate escalation guidance using centralized team assignments
            escalation_guidance = self._generate_escalation_guidance(risk_score)
            
            # Generate agent guidance
            agent_guidance = self._generate_agent_guidance(scam_type, risk_score, policy)
            
            # Check for regulatory requirements
            regulatory_requirements = policy.get('regulatory_requirements', [])
            
            result = {
                'primary_policy': policy,
                'escalation_guidance': escalation_guidance,
                'agent_guidance': agent_guidance,
                'regulatory_requirements': regulatory_requirements,
                'risk_assessment': {
                    'score': risk_score,
                    'requires_escalation': risk_score >= policy.get('escalation_threshold', 80),
                    'escalation_threshold': policy.get('escalation_threshold', 80)
                },
                'policy_metadata': {
                    'policy_id': policy.get('policy_id', 'unknown'),
                    'version': policy.get('version', '1.0.0'),
                    'last_updated': policy.get('last_updated', 'unknown')
                },
                'guidance_agent': self.agent_id,
                'timestamp': datetime.now().isoformat()
            }
            
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=True)
            
            self.log_activity(f"Policy retrieval complete", {
                "policy_id": policy.get('policy_id'),
                "escalation_required": result['risk_assessment']['requires_escalation'],
                "processing_time": processing_time
            })
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=False)
            self.handle_error(e, "retrieve_fraud_policies")
            
            return {
                'error': str(e),
                'scam_type': scam_type,
                'risk_score': risk_score,
                'guidance_agent': self.agent_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_escalation_guidance(self, risk_score: float) -> Dict[str, Any]:
        """Generate escalation guidance using centralized team assignments"""
        thresholds = self.config["risk_thresholds"]
        teams = self.config["team_assignments"]
        
        if risk_score >= thresholds["critical"]:
            return {
                'level': 'immediate_escalation',
                'team': teams["critical"],
                'timeframe': 'immediately',
                'priority': 'CRITICAL',
                'procedures': [
                    'Stop all transactions immediately',
                    'Transfer call to senior fraud specialist',
                    'Document all evidence and customer statements',
                    'Initiate fraud investigation case',
                    'Consider account security measures'
                ]
            }
        elif risk_score >= thresholds["high"]:
            return {
                'level': 'enhanced_monitoring',
                'team': teams["high"],
                'timeframe': 'within 1 hour',
                'priority': 'HIGH',
                'procedures': [
                    'Flag account for enhanced monitoring',
                    'Verify customer identity thoroughly',
                    'Document suspicious activity',
                    'Prepare case file for review',
                    'Schedule follow-up contact'
                ]
            }
        else:
            return {
                'level': 'standard_review',
                'team': teams["medium"],
                'timeframe': 'within 24 hours',
                'priority': 'MEDIUM',
                'procedures': [
                    'Document conversation details',
                    'Monitor for pattern changes',
                    'Provide customer education',
                    'Standard account monitoring'
                ]
            }
    
    def _generate_agent_guidance(self, scam_type: str, risk_score: float, policy: Dict) -> Dict[str, Any]:
        """Generate comprehensive agent guidance"""
        agent_guidance = {
            'immediate_alerts': [],
            'recommended_actions': policy.get('procedures', []),
            'key_questions': policy.get('key_questions', []),
            'escalation_advice': []
        }
        
        # Include customer education if enabled
        if self.config["include_customer_education"]:
            agent_guidance['customer_education'] = policy.get('customer_education', [])
        
        # Add immediate alerts based on risk level using centralized thresholds
        thresholds = self.config["risk_thresholds"]
        
        if risk_score >= thresholds["critical"]:
            agent_guidance['immediate_alerts'].extend([
                f"🚨 CRITICAL: {scam_type.replace('_', ' ').title()} detected",
                "🛑 DO NOT PROCESS ANY PAYMENTS - STOP TRANSACTION",
                "📞 ESCALATE TO FRAUD TEAM IMMEDIATELY",
                "🔒 CONSIDER ACCOUNT SECURITY MEASURES"
            ])
        elif risk_score >= thresholds["high"]:
            agent_guidance['immediate_alerts'].extend([
                f"⚠️ HIGH RISK: {scam_type.replace('_', ' ').title()} suspected",
                "🔍 ENHANCED VERIFICATION REQUIRED",
                "📋 DOCUMENT ALL DETAILS THOROUGHLY"
            ])
        elif risk_score >= thresholds["medium"]:
            agent_guidance['immediate_alerts'].extend([
                f"⚡ MEDIUM RISK: {scam_type.replace('_', ' ').title()} possible",
                "📋 FOLLOW VERIFICATION PROCEDURES",
                "🎓 PROVIDE CUSTOMER EDUCATION"
            ])
        
        # Add special considerations
        if 'vulnerability_considerations' in policy:
            agent_guidance['vulnerability_considerations'] = policy['vulnerability_considerations']
        
        if 'immediate_actions' in policy and risk_score >= 70:
            agent_guidance['immediate_actions'] = policy['immediate_actions']
        
        return agent_guidance

# Create global instance
policy_guidance_agent = PolicyGuidanceAgent()
