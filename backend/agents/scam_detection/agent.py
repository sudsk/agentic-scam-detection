# backend/agents/scam_detection/agent.py - FIXED OUTPUT FORMAT
"""
Fraud Detection Agent - Analyzes text for fraud patterns and risk scoring
FIXED: Returns proper DetectedPattern objects for API compatibility
"""

import re
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings

logger = logging.getLogger(__name__)

class FraudDetectionAgent(BaseAgent):
    """Agent specialized in detecting fraud patterns and calculating risk scores"""
    
    def __init__(self):
        super().__init__(
            agent_type="fraud_detection",
            agent_name="Fraud Detection Agent"
        )
        
        # Load fraud patterns
        self._load_fraud_patterns()
        
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"🔍 {self.agent_name} ready with {len(self.scam_patterns)} pattern categories")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration from centralized settings"""
        settings = get_settings()
        return settings.get_agent_config("fraud_detection")
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get fraud detection agent capabilities"""
        return [
            AgentCapability.FRAUD_DETECTION,
            AgentCapability.PATTERN_RECOGNITION,
            AgentCapability.RISK_SCORING
        ]
    
    def _load_fraud_patterns(self):
        """Load fraud detection patterns from centralized config"""
        settings = get_settings()
        
        self.scam_patterns = {
            'third_party_instructions': {
                'patterns': [
                    r'(my|the) (advisor|manager|broker) (said|told|asked)',
                    r'(someone|caller|person) (told|said|asked) me to',
                    r'(they|he|she) (said|told|asked) me to (transfer|send|pay)',
                    r'I received a call (saying|telling|asking)',
                    r'(company|platform|service) (contacted|called) me'
                ],
                'weight': settings.pattern_weights["third_party_instructions"],
                'severity': 'high'
            },
            'urgency_pressure': {
                'patterns': [
                    r'(urgent|immediately|right now|today only)',
                    r'within (\d+) (hour|minute)s?',
                    r'(deadline|expires|closes) (today|soon|tonight)',
                    r'(limited time|act fast|don\'t wait|hurry)',
                    r'before (it\'s too late|account frozen|arrested)',
                    r'(lose|miss) (this|the) (opportunity|chance)',
                    r'must (transfer|send|pay|act) .* (immediately|now|today)'
                ],
                'weight': settings.pattern_weights["urgency_pressure"],
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
                'weight': settings.pattern_weights["authority_impersonation"],
                'severity': 'critical'
            },
            'credential_requests': {
                'patterns': [
                    r'(verify|confirm) (your|my) (identity|details)',
                    r'need (your|my) (card|banking|login) (details|information)',
                    r'(PIN|password|security) (code|number)',
                    r'log in to (confirm|verify|check)',
                    r'(full|complete) card number'
                ],
                'weight': settings.pattern_weights["credential_requests"],
                'severity': 'critical'
            },
            'investment_fraud': {
                'patterns': [
                    r'guaranteed? (\d+%|profit|return)',
                    r'guarantees? (\d+%|\d+\s*percent)',
                    r'(risk-free|no risk|100% safe)',
                    r'(margin call|trading|investment) (account|platform)',
                    r'(forex|crypto|bitcoin|stock) (trading|investment)',
                    r'(profit|return) of (\d+)%',
                    r'(\d+)% (monthly|daily|weekly) returns?',
                    r'guarantees? .* returns?'
                ],
                'weight': settings.pattern_weights["investment_fraud"],
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
                'weight': settings.pattern_weights["romance_exploitation"],
                'severity': 'high'
            }
        }
        
        # Scam type classifiers from centralized config
        self.scam_classifiers = settings.scam_type_keywords
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Main processing method - analyze text for fraud patterns"""
        if isinstance(input_data, dict):
            text = input_data.get('text', '')
        else:
            text = str(input_data)
        
        return self.analyze_fraud_patterns(text)
    
    def analyze_fraud_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text for fraud patterns and return risk assessment"""
        start_time = time.time()
        
        try:
            self.log_activity(f"Analyzing fraud patterns", {"text_length": len(text)})
            
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
                    # Create proper DetectedPattern object format
                    detected_patterns[category] = {
                        'pattern_name': category,
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
            
            # Determine risk level using centralized thresholds
            risk_level = self._determine_risk_level(risk_score)
            
            # Generate explanation
            explanation = f"{risk_level} risk detected."
            if detected_patterns:
                pattern_names = list(detected_patterns.keys())
                explanation += f" Detected patterns: {', '.join(pattern_names)}."
            
            # Recommended action
            recommended_action = self._get_recommended_action(risk_score)
            
            # Calculate confidence
            confidence = min(len(detected_patterns) * 0.2 + 0.3, 0.95)
            
            result = {
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'scam_type': scam_type,
                'confidence': confidence,
                'detected_patterns': detected_patterns,
                'explanation': explanation,
                'recommended_action': recommended_action,
                'requires_escalation': risk_score >= self.config["risk_thresholds"]["critical"],
                'analysis_agent': self.agent_id,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=True)
            
            self.log_activity(f"Analysis complete", {
                "risk_score": risk_score, 
                "scam_type": scam_type,
                "processing_time": processing_time
            })
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=False)
            self.handle_error(e, "analyze_fraud_patterns")
            
            return {
                'error': str(e),
                'risk_score': 0,
                'risk_level': 'UNKNOWN',
                'scam_type': 'unknown',
                'analysis_agent': self.agent_id,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
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
        """Determine risk level based on centralized thresholds"""
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

# Create global instance
fraud_detection_agent = FraudDetectionAgent()
