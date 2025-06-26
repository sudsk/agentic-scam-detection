# agents/scam_detection/agent.py
"""
Fraud Detection Agent - Analyzes text for fraud patterns and risk scoring
"""

import re
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class FraudDetectionAgent:
    """Agent specialized in detecting fraud patterns and calculating risk scores"""
    
    def __init__(self):
        self.agent_id = "fraud_detection_001"
        self.agent_name = "Fraud Detection Agent"
        self.status = "active"
        
        # Configuration
        self.config = {
            "risk_thresholds": {
                "critical": 80,
                "high": 60,
                "medium": 40,
                "low": 20
            },
            "pattern_weights": {
                "authority_impersonation": 30,
                "credential_requests": 35,
                "investment_fraud": 30,
                "romance_exploitation": 25,
                "urgency_pressure": 20,
                "third_party_instructions": 25
            }
        }
        
        # Enhanced fraud detection patterns
        self.scam_patterns = {
            'third_party_instructions': {
                'patterns': [
                    r'(my|the) (advisor|manager|broker) (said|told|asked)',
                    r'(someone|caller|person) (told|said|asked) me to',
                    r'(they|he|she) (said|told|asked) me to (transfer|send|pay)',
                    r'I received a call (saying|telling|asking)',
                    r'(company|platform|service) (contacted|called) me'
                ],
                'weight': 25,
                'severity': 'high'
            },
            
            'urgency_pressure': {
                'patterns': [
                    r'(urgent|immediately|right now|today only)',
                    r'within (\d+) (hour|minute)s?',
                    r'(deadline|expires|closes) (today|soon|tonight)',
                    r'(limited time|act fast|don\'t wait|hurry)',
                    r'before (it\'s too late|account frozen|arrested)'
                ],
                'weight': 20,
                'severity': 'high'
            },
            
            'authority_impersonation': {
                'patterns': [
                    r'(police|court|tax office|government) (called|contacted)',
                    r'(investigation|warrant|legal action|prosecution)',
                    r'(bank|security|fraud) (team|department) called',
                    r'(officer|investigator|agent|official) said',
                    r'(arrest|legal action|court case|prosecution)'
                ],
                'weight': 30,
                'severity': 'critical'
            },
            
            'financial_pressure': {
                'patterns': [
                    r'(freeze|block|close|suspend) (my|the|your) account',
                    r'(arrest|legal action|prosecution|court)',
                    r'(lose|miss) (money|opportunity|investment)',
                    r'penalty of £(\d+)',
                    r'(fine|fee|charge) of £(\d+)'
                ],
                'weight': 25,
                'severity': 'high'
            },
            
            'credential_requests': {
                'patterns': [
                    r'(verify|confirm) (your|my) (identity|details)',
                    r'need (your|my) (card|banking|login) (details|information)',
                    r'(PIN|password|security) (code|number)',
                    r'log in to (confirm|verify|check)',
                    r'(full|complete) card number'
                ],
                'weight': 35,
                'severity': 'critical'
            },
            
            'investment_fraud': {
                'patterns': [
                    r'guaranteed (\d+%|profit|return)',
                    r'(risk-free|no risk|100% safe)',
                    r'(margin call|trading|investment) (account|platform)',
                    r'(forex|crypto|bitcoin|stock) (trading|investment)',
                    r'(profit|return) of (\d+)%'
                ],
                'weight': 30,
                'severity': 'high'
            },
            
            'romance_exploitation': {
                'patterns': [
                    r'(met online|dating site|social media)',
                    r'(boyfriend|girlfriend|partner) (overseas|abroad)',
                    r'(stuck|stranded|trapped) (overseas|abroad)',
                    r'(military|deployed|peacekeeping|doctor)',
                    r'(emergency|medical|visa|travel) (funds|money)'
                ],
                'weight': 25,
                'severity': 'high'
            }
        }
        
        # Scam type classifiers
        self.scam_classifiers = {
            'investment_scam': [
                'investment', 'trading', 'forex', 'crypto', 'bitcoin', 'profit',
                'returns', 'margin call', 'platform', 'advisor', 'broker'
            ],
            'romance_scam': [
                'online', 'dating', 'relationship', 'boyfriend', 'girlfriend',
                'military', 'overseas', 'emergency', 'medical', 'visa'
            ],
            'impersonation_scam': [
                'bank', 'security', 'fraud', 'police', 'tax',
                'government', 'investigation', 'verification'
            ],
            'authority_scam': [
                'police', 'court', 'tax', 'fine', 'penalty',
                'arrest', 'legal action', 'warrant', 'prosecution'
            ]
        }
        
        logger.info(f"✅ {self.agent_name} initialized")
    
    def analyze_fraud_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for fraud patterns and return risk assessment
        
        Args:
            text: The customer speech text to analyze
            
        Returns:
            Dict containing risk analysis with score, patterns, and recommendations
        """
        logger.info(f"🔍 Analyzing text for fraud patterns: {text[:100]}...")
        
        try:
            text_lower = text.lower()
            risk_score = 0
            detected_patterns = {}
            
            # Analyze each pattern category
            for category, config in self.scam_patterns.items():
                matches = []
                for pattern in config['patterns']:
                    pattern_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                    if pattern_matches:
                        matches.extend([match.group() for match in pattern_matches])
                
                if matches:
                    detected_patterns[category] = {
                        'matches': matches,
                        'count': len(matches),
                        'weight': config['weight'],
                        'severity': config['severity']
                    }
                    risk_score += len(matches) * config['weight']
            
            # Classify scam type
            scam_type = self._classify_scam_type(text_lower)
            
            # Cap risk score at 100
            risk_score = min(risk_score, 100)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Generate explanation
            explanation = f"{risk_level} risk detected."
            if detected_patterns:
                pattern_names = list(detected_patterns.keys())
                explanation += f" Detected patterns: {', '.join(pattern_names)}."
            
            # Recommended action
            recommended_action = self._get_recommended_action(risk_score)
            
            result = {
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'scam_type': scam_type,
                'confidence': min(len(detected_patterns) * 0.2 + 0.3, 0.95),
                'detected_patterns': detected_patterns,
                'explanation': explanation,
                'recommended_action': recommended_action,
                'requires_escalation': risk_score >= self.config["risk_thresholds"]["critical"],
                'analysis_agent': self.agent_id,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Analysis complete: Risk {risk_score}% - {scam_type}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Fraud analysis failed: {e}")
            raise
    
    def _classify_scam_type(self, text_lower: str) -> str:
        """Classify the type of scam based on keywords"""
        scam_scores = {}
        
        for scam_type_key, keywords in self.scam_classifiers.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scam_scores[scam_type_key] = score / len(keywords)
        
        if scam_scores:
            best_match = max(scam_scores.items(), key=lambda x: x[1])
            return best_match[0]
        
        return "unknown"
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        thresholds = self.config["risk_thresholds"]
        
        if risk_score >= thresholds["critical"]:
            return "CRITICAL"
        elif risk_score >= thresholds["high"]:
            return "HIGH"
        elif risk_score >= thresholds["medium"]:
            return "MEDIUM"
        elif risk_score >= thresholds["low"]:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_recommended_action(self, risk_score: float) -> str:
        """Get recommended action based on risk score"""
        thresholds = self.config["risk_thresholds"]
        
        if risk_score >= thresholds["critical"]:
            return "IMMEDIATE_ESCALATION"
        elif risk_score >= thresholds["high"]:
            return "ENHANCED_MONITORING"
        elif risk_score >= thresholds["medium"]:
            return "VERIFY_CUSTOMER_INTENT"
        else:
            return "CONTINUE_NORMAL_PROCESSING"
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "capabilities": [
                "pattern_recognition",
                "risk_scoring",
                "scam_classification"
            ],
            "config": self.config,
            "patterns_loaded": len(self.scam_patterns)
        }

# Create global instance
fraud_detection_agent = FraudDetectionAgent()

# Export function for system integration
def analyze_fraud_patterns(text: str) -> Dict[str, Any]:
    """Main function for fraud pattern analysis"""
    return fraud_detection_agent.analyze_fraud_patterns(text)
