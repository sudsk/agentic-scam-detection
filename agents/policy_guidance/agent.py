# agents/policy_guidance/agent.py
"""
Policy Guidance Agent - Provides procedural guidance and escalation recommendations
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PolicyGuidanceAgent:
    """Agent that provides fraud response policies and guidance"""
    
    def __init__(self):
        self.agent_id = "policy_guidance_001"
        self.agent_name = "Policy Guidance Agent"
        self.status = "active"
        
        # Configuration
        self.config = {
            "escalation_thresholds": {
                "investment_scam": 80,
                "romance_scam": 75,
                "impersonation_scam": 85,
                "authority_scam": 85
            },
            "response_teams": {
                "immediate_escalation": "Financial Crime Team",
                "enhanced_monitoring": "Fraud Prevention Team",
                "standard_review": "Customer Service Team"
            }
        }
        
        # Financial Institution Policy Database
        self.policy_database = {
            'investment_scam_response': {
                'policy_id': 'FP-INV-001',
                'title': 'Investment Scam Detection and Response',
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
                    'Can you check if they are regulated by financial authorities?'
                ],
                'customer_education': [
                    'All legitimate investments carry risk - guaranteed returns are impossible',
                    'Real investment firms are regulated and listed on official registers', 
                    'High-pressure sales tactics are warning signs of fraud',
                    'Take time to research - legitimate opportunities don\'t require immediate action'
                ],
                'escalation_threshold': 80
            },
            
            'romance_scam_response': {
                'policy_id': 'FP-ROM-001',
                'title': 'Romance Scam Detection Procedures',
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
                    'How long have you known them?'
                ],
                'customer_education': [
                    'Romance scammers build relationships over months to gain trust',
                    'They often claim to be overseas, in military, or traveling for work',
                    'Real partners don\'t repeatedly ask for money, especially for emergencies',
                    'Video calls can be faked - only meeting in person confirms identity'
                ],
                'escalation_threshold': 75
            },
            
            'impersonation_scam_response': {
                'policy_id': 'FP-IMP-001',
                'title': 'Authority Impersonation Scam Procedures',
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
                    'Have you received any official letters about this?',
                    'Why do they need payment immediately?'
                ],
                'customer_education': [
                    'Banks will never ask for your PIN, password, or full card details over the phone',
                    'Legitimate authorities send official letters before taking any action',
                    'Government agencies don\'t accept payments via bank transfer',
                    'When in doubt, hang up and call the organization\'s official number'
                ],
                'escalation_threshold': 85
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
        
        logger.info(f"✅ {self.agent_name} initialized with {len(self.policy_database)} policies")
    
    def retrieve_fraud_policies(self, scam_type: str, risk_score: float) -> Dict[str, Any]:
        """
        Retrieve relevant fraud policies and procedures
        
        Args:
            scam_type: Type of scam detected (investment_scam, romance_scam, etc.)
            risk_score: Risk score from 0-100
            
        Returns:
            Dict containing relevant policies, procedures, and guidance
        """
        logger.info(f"📚 Retrieving policies for {scam_type} with risk score {risk_score}")
        
        try:
            # Get the appropriate policy
            policy_key = self.scam_policy_map.get(scam_type, 'investment_scam_response')
            policy = self.policy_database.get(policy_key, {})
            
            # Generate escalation guidance
            escalation_guidance = self._generate_escalation_guidance(risk_score)
            
            # Generate agent guidance
            agent_guidance = self._generate_agent_guidance(scam_type, risk_score, policy)
            
            result = {
                'primary_policy': policy,
                'escalation_guidance': escalation_guidance,
                'agent_guidance': agent_guidance,
                'risk_assessment': {
                    'score': risk_score,
                    'requires_escalation': risk_score >= policy.get('escalation_threshold', 80)
                },
                'guidance_agent': self.agent_id,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Retrieved policies for {scam_type}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Policy retrieval failed: {e}")
            raise
    
    def _generate_escalation_guidance(self, risk_score: float) -> Dict[str, Any]:
        """Generate escalation guidance based on risk score"""
        if risk_score >= 80:
            return {
                'level': 'immediate_escalation',
                'team': self.config["response_teams"]["immediate_escalation"],
                'timeframe': 'immediately',
                'procedures': [
                    'Stop all transactions immediately',
                    'Transfer call to senior fraud specialist',
                    'Document all evidence and customer statements',
                    'Initiate fraud investigation case'
                ]
            }
        elif risk_score >= 60:
            return {
                'level': 'enhanced_monitoring',
                'team': self.config["response_teams"]["enhanced_monitoring"],
                'timeframe': 'within 1 hour',
                'procedures': [
                    'Flag account for enhanced monitoring',
                    'Verify customer identity thoroughly',
                    'Document suspicious activity',
                    'Prepare case file for review'
                ]
            }
        else:
            return {
                'level': 'standard_review',
                'team': self.config["response_teams"]["standard_review"],
                'timeframe': 'within 24 hours',
                'procedures': [
                    'Document conversation details',
                    'Monitor for pattern changes',
                    'Provide customer education'
                ]
            }
    
    def _generate_agent_guidance(self, scam_type: str, risk_score: float, policy: Dict) -> Dict[str, Any]:
        """Generate comprehensive agent guidance"""
        agent_guidance = {
            'immediate_alerts': [],
            'recommended_actions': policy.get('procedures', []),
            'key_questions': policy.get('key_questions', []),
            'customer_education': policy.get('customer_education', []),
            'escalation_advice': []
        }
        
        # Add immediate alerts based on risk level
        if risk_score >= 80:
            agent_guidance['immediate_alerts'].extend([
                f"🚨 CRITICAL: {scam_type.replace('_', ' ').title()} detected",
                "🛑 DO NOT PROCESS ANY PAYMENTS - STOP TRANSACTION",
                "📞 ESCALATE TO FRAUD TEAM IMMEDIATELY"
            ])
        elif risk_score >= 60:
            agent_guidance['immediate_alerts'].extend([
                f"⚠️ HIGH RISK: {scam_type.replace('_', ' ').title()} suspected",
                "🔍 ENHANCED VERIFICATION REQUIRED"
            ])
        elif risk_score >= 40:
            agent_guidance['immediate_alerts'].extend([
                f"⚡ MEDIUM RISK: {scam_type.replace('_', ' ').title()} possible",
                "📋 FOLLOW VERIFICATION PROCEDURES"
            ])
        
        return agent_guidance
    
    def get_available_policies(self) -> List[Dict[str, str]]:
        """Get list of available policies"""
        return [
            {
                "policy_id": policy["policy_id"],
                "title": policy["title"],
                "scam_type": scam_type
            }
            for scam_type, policy_key in self.scam_policy_map.items()
            for policy in [self.policy_database.get(policy_key, {})]
            if policy
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "capabilities": [
                "policy_retrieval",
                "escalation_guidance",
                "agent_instruction_generation"
            ],
            "config": self.config,
            "policies_loaded": len(self.policy_database)
        }

# Create global instance
policy_guidance_agent = PolicyGuidanceAgent()

# Export function for system integration
def retrieve_fraud_policies(scam_type: str, risk_score: float) -> Dict[str, Any]:
    """Main function for policy retrieval"""
    return policy_guidance_agent.retrieve_fraud_policies(scam_type, risk_score)
