import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from google.adk.agents import Agent
from google.adk.models import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

logger = logging.getLogger(__name__)

def analyze_fraud_patterns(text: str) -> Dict[str, Any]:
    """
    Analyze text for fraud patterns and return risk assessment
    
    Args:
        text: The customer speech text to analyze
        
    Returns:
        Dict containing risk analysis with score, patterns, and recommendations
    """
    logger.info(f"🔍 Analyzing text for fraud patterns: {text[:100]}...")
    
    text_lower = text.lower()
    risk_score = 0
    detected_patterns = {}
    scam_type = "unknown"
    
    # Enhanced fraud detection patterns
    scam_patterns = {
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
                r'(police|court|HMRC|government|tax office) (called|contacted)',
                r'(investigation|warrant|legal action|prosecution)',
                r'(HSBC|bank) (security|fraud) (team|department) called',
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
    
    # Analyze each pattern category
    for category, config in scam_patterns.items():
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
    scam_classifiers = {
        'investment_scam': [
            'investment', 'trading', 'forex', 'crypto', 'bitcoin', 'profit',
            'returns', 'margin call', 'platform', 'advisor', 'broker'
        ],
        'romance_scam': [
            'online', 'dating', 'relationship', 'boyfriend', 'girlfriend',
            'military', 'overseas', 'emergency', 'medical', 'visa'
        ],
        'impersonation_scam': [
            'HSBC', 'bank', 'security', 'fraud', 'police', 'HMRC',
            'government', 'investigation', 'verification'
        ],
        'authority_scam': [
            'police', 'court', 'HMRC', 'tax', 'fine', 'penalty',
            'arrest', 'legal action', 'warrant', 'prosecution'
        ]
    }
    
    scam_scores = {}
    for scam_type_key, keywords in scam_classifiers.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            scam_scores[scam_type_key] = score / len(keywords)
    
    if scam_scores:
        best_match = max(scam_scores.items(), key=lambda x: x[1])
        scam_type = best_match[0]
    
    # Cap risk score at 100
    risk_score = min(risk_score, 100)
    
    # Determine risk level
    if risk_score >= 80:
        risk_level = "CRITICAL"
    elif risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    elif risk_score >= 20:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"
    
    # Generate explanation
    explanation = f"{risk_level} risk detected."
    if detected_patterns:
        pattern_names = list(detected_patterns.keys())
        explanation += f" Detected patterns: {', '.join(pattern_names)}."
    
    # Recommended action
    if risk_score >= 80:
        recommended_action = "IMMEDIATE_ESCALATION"
    elif risk_score >= 60:
        recommended_action = "ENHANCED_MONITORING"
    elif risk_score >= 40:
        recommended_action = "VERIFY_CUSTOMER_INTENT"
    else:
        recommended_action = "CONTINUE_NORMAL_PROCESSING"
    
    result = {
        'risk_score': float(risk_score),
        'risk_level': risk_level,
        'scam_type': scam_type,
        'confidence': min(len(detected_patterns) * 0.2 + 0.3, 0.95),
        'detected_patterns': detected_patterns,
        'explanation': explanation,
        'recommended_action': recommended_action,
        'requires_escalation': risk_score >= 80,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"✅ Analysis complete: Risk {risk_score}% - {scam_type}")
    return result

# Create the Live Scam Detection Agent using Google ADK 1.4.2
live_scam_detection_agent = Agent(
    name="live_scam_detection_agent",
    model=LiteLlm("gemini/gemini-2.0-flash"),  # Using Gemini 2.0 Flash
    description="Real-time fraud detection agent that analyzes customer speech for scam patterns",
    instruction="""You are a specialized fraud detection agent for HSBC call center operations.

Your primary responsibilities:
1. Analyze customer speech in real-time for fraud indicators
2. Identify scam patterns including investment fraud, romance scams, impersonation, etc.
3. Calculate risk scores from 0-100% based on detected patterns
4. Provide immediate recommendations for agent actions

When you receive customer speech text, you should:
- Use the analyze_fraud_patterns tool to detect fraud indicators
- Focus only on customer speech (ignore agent speech)
- Provide clear risk assessments and actionable recommendations
- Flag high-risk scenarios (80%+) for immediate escalation

Always respond with structured analysis including risk score, scam type, detected patterns, and recommended actions.""",
    tools=[analyze_fraud_patterns]
)

