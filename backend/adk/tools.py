# backend/adk/tools.py - Google ADK Tools for Fraud Detection Agents

import re
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..config.settings import get_settings
from ..utils import get_current_timestamp

logger = logging.getLogger(__name__)

# ===== SCAM DETECTION TOOLS =====

def fraud_pattern_analyzer(customer_text: str, context: Dict = None) -> Dict[str, Any]:
    """Analyzes text for known fraud patterns and social engineering tactics"""
    
    settings = get_settings()
    detected_patterns = {}
    
    # Advanced pattern detection
    pattern_definitions = {
        'romance_exploitation': {
            'patterns': [
                r'(met online|dating site|social media)',
                r'(boyfriend|girlfriend|partner) (overseas|abroad)',
                r'(stuck|stranded|trapped) (overseas|abroad)',
                r'(never met|not met) (in person|face to face)',
                r'(camera broken|video call|can\'t video)',
                r'(emergency|medical|visa|travel) (funds|money|help)'
            ],
            'severity': 'critical',
            'weight': 35
        },
        'investment_fraud': {
            'patterns': [
                r'guaranteed? (\d+%|profit|return)',
                r'(risk-free|no risk|100% safe)',
                r'(\d+)% (monthly|daily|weekly) returns?',
                r'(margin call|trading|investment) (account|platform)',
                r'(forex|crypto|bitcoin) (trading|investment)',
                r'(limited time|act fast|exclusive) (offer|opportunity)'
            ],
            'severity': 'high',
            'weight': 40
        },
        'urgency_pressure': {
            'patterns': [
                r'(urgent|immediately|right now|today only)',
                r'within (\d+) (hour|minute)s?',
                r'(deadline|expires|closes) (today|soon|tonight)',
                r'(lose|miss) (this|the) (opportunity|chance)',
                r'before (it\'s too late|account frozen)',
                r'must (transfer|send|pay) .* (immediately|now|today)'
            ],
            'severity': 'high', 
            'weight': 30
        },
        'authority_impersonation': {
            'patterns': [
                r'(police|court|tax office|government) (called|contacted)',
                r'(investigation|warrant|legal action|prosecution)',
                r'(bank|security|fraud) (team|department) called',
                r'(officer|investigator|agent|official) said',
                r'(arrest|legal action|court case)'
            ],
            'severity': 'critical',
            'weight': 45
        },
        'credential_requests': {
            'patterns': [
                r'(verify|confirm) (your|my) (identity|details)',
                r'need (your|my) (card|banking|login) (details|information)',
                r'(PIN|password|security) (code|number)',
                r'log in to (confirm|verify|check)',
                r'(full|complete) card number'
            ],
            'severity': 'critical',
            'weight': 50
        }
    }
    
    text_lower = customer_text.lower()
    
    # Analyze each pattern category
    for pattern_name, pattern_config in pattern_definitions.items():
        matches = []
        for pattern in pattern_config['patterns']:
            pattern_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if pattern_matches:
                matches.extend([match.group() for match in pattern_matches])
        
        if matches:
            detected_patterns[pattern_name] = {
                'matches': matches,
                'count': len(matches),
                'severity': pattern_config['severity'],
                'weight': pattern_config['weight'],
                'confidence': min(len(matches) * 0.3 + 0.6, 1.0)
            }
    
    return {
        'detected_patterns': detected_patterns,
        'total_patterns': len(detected_patterns),
        'analysis_timestamp': get_current_timestamp()
    }

def risk_calculator(patterns: Dict, customer_profile: Dict = None, transaction_details: Dict = None) -> Dict[str, Any]:
    """Calculates fraud risk score based on detected patterns and customer behavior"""
    
    total_score = 0.0
    risk_factors = []
    
    # Calculate base risk from patterns
    for pattern_name, pattern_data in patterns.items():
        weight = pattern_data.get('weight', 20)
        count = pattern_data.get('count', 1)
        severity_multiplier = {
            'critical': 2.0,
            'high': 1.5,
            'medium': 1.0
        }.get(pattern_data.get('severity', 'medium'), 1.0)
        
        pattern_score = weight * count * severity_multiplier
        total_score += pattern_score
        
        risk_factors.append(f"{pattern_name}: {pattern_score:.1f} points")
    
    # Adjust for customer profile factors
    if customer_profile:
        age = customer_profile.get('age', 50)
        if age > 65:  # Higher vulnerability for seniors
            total_score *= 1.2
            risk_factors.append("Senior customer vulnerability: +20%")
        
        relationship_status = customer_profile.get('relationship_status', '')
        if relationship_status in ['single', 'divorced', 'widowed']:
            total_score *= 1.1
            risk_factors.append("Relationship vulnerability: +10%")
    
    # Adjust for transaction factors
    if transaction_details:
        amount = transaction_details.get('amount', 0)
        if amount > 5000:  # Large transactions increase risk
            total_score *= 1.3
            risk_factors.append("Large transaction amount: +30%")
        
        destination = transaction_details.get('destination_country', '')
        high_risk_countries = ['turkey', 'nigeria', 'ghana', 'malaysia', 'philippines']
        if destination.lower() in high_risk_countries:
            total_score *= 1.4
            risk_factors.append("High-risk destination country: +40%")
    
    # Cap at 100 and determine risk level
    final_score = min(total_score, 100)
    
    risk_level = "MINIMAL"
    if final_score >= 80:
        risk_level = "CRITICAL"
    elif final_score >= 60:
        risk_level = "HIGH"
    elif final_score >= 40:
        risk_level = "MEDIUM"
    elif final_score >= 20:
        risk_level = "LOW"
    
    # Determine recommended action
    if final_score >= 80:
        recommended_action = "IMMEDIATE_ESCALATION"
    elif final_score >= 60:
        recommended_action = "ENHANCED_MONITORING"
    elif final_score >= 40:
        recommended_action = "VERIFY_CUSTOMER_INTENT"
    else:
        recommended_action = "CONTINUE_NORMAL_PROCESSING"
    
    return {
        'risk_score': round(final_score, 1),
        'risk_level': risk_level,
        'recommended_action': recommended_action,
        'risk_factors': risk_factors,
        'confidence': min(len(patterns) * 0.2 + 0.5, 0.95),
        'calculation_timestamp': get_current_timestamp()
    }

# ===== POLICY GUIDANCE TOOLS =====

def policy_database_search(scam_type: str, risk_level: str, customer_segment: str = None) -> Dict[str, Any]:
    """Searches HSBC fraud prevention policies for specific scenarios"""
    
    policy_database = {
        'romance_scam': {
            'policy_id': 'FP-ROM-001',
            'title': 'Romance Scam Detection and Response',
            'version': '2.1.0',
            'procedures': [
                'Stop transfers to individuals customer has never met in person',
                'Ask: "Have you met this person face-to-face?" and "Have they asked for money before?"',
                'Explain romance scam patterns: relationship building followed by emergency requests',
                'Check customer transfer history for previous payments to same recipient',
                'Advise customer to discuss with family/friends before proceeding'
            ],
            'key_questions': [
                'Have you met this person in person?',
                'Have they asked for money before?', 
                'Why can\'t they get help from family or employer?',
                'How long have you known them?',
                'What platform did you meet them on?'
            ],
            'customer_education': [
                'Romance scammers build relationships over months to gain trust',
                'They often claim to be overseas, in military, or traveling for work',
                'Real partners don\'t repeatedly ask for money, especially for emergencies',
                'Video calls can be faked - only meeting in person confirms identity'
            ]
        },
        'investment_scam': {
            'policy_id': 'FP-INV-001',
            'title': 'Investment Scam Detection and Response',
            'version': '2.0.5',
            'procedures': [
                'Immediately halt investment-related transfers showing guaranteed returns',
                'Ask: "Have you been able to withdraw any profits from this investment?"',
                'Verify investment company against FCA register using official website',
                'Explain: All legitimate investments carry risk - guarantees indicate fraud',
                'Document company name, contact details, and promised returns'
            ],
            'key_questions': [
                'Have you been able to withdraw any money from this investment?',
                'Did they guarantee specific percentage returns?',
                'How did this investment company first contact you?',
                'Can you check if they are regulated by financial authorities?'
            ],
            'customer_education': [
                'All legitimate investments carry risk - guaranteed returns are impossible',
                'Real investment firms are registered and regulated by financial authorities',
                'High-pressure sales tactics are warning signs of fraud',
                'Take time to research - legitimate opportunities don\'t require immediate action'
            ]
        },
        'impersonation_scam': {
            'policy_id': 'FP-IMP-001',
            'title': 'Authority Impersonation Scam Procedures',
            'version': '2.2.0',
            'procedures': [
                'Inform customer: Banks never ask for PINs or passwords over phone',
                'Instruct customer to hang up and call official number independently',
                'Explain: Legitimate authorities send official letters before taking action',
                'Ask: "What number did they call from?" and verify against official numbers',
                'Document all impersonation details including claimed authority'
            ],
            'key_questions': [
                'What number did they call you from?',
                'Did they ask for your PIN or online banking password?',
                'Have you received any official letters about this matter?',
                'Why do they claim payment is needed immediately?'
            ],
            'customer_education': [
                'Banks never ask for PINs, passwords, or full card details over phone',
                'Legitimate authorities send official letters before taking action',
                'Government agencies don\'t accept payments via bank transfer',
                'When in doubt, hang up and call the organization\'s official number'
            ]
        }
    }
    
    policy = policy_database.get(scam_type, policy_database.get('impersonation_scam'))
    
    return {
        'policy': policy,
        'applicable': True,
        'search_timestamp': get_current_timestamp()
    }

def escalation_procedures(risk_score: float, scam_type: str, urgency_level: str = None) -> Dict[str, Any]:
    """Retrieves escalation procedures and contact information for fraud cases"""
    
    escalation_matrix = {
        'critical': {
            'threshold': 80,
            'team': 'Financial Crime Team',
            'contact': 'fraud.urgent@hsbc.com',
            'phone': '+44 20 7991 8888',
            'sla': '15 minutes',
            'actions': [
                'Stop all transactions immediately',
                'Escalate to Financial Crime Team lead',
                'Contact customer via secure channel within 15 minutes',
                'Initiate account monitoring',
                'Prepare emergency response procedures'
            ]
        },
        'high': {
            'threshold': 60,
            'team': 'Fraud Prevention Team',
            'contact': 'fraud.prevention@hsbc.com', 
            'phone': '+44 20 7991 8777',
            'sla': '1 hour',
            'actions': [
                'Review and hold pending transactions',
                'Enhanced authentication for all new transactions',
                'Schedule customer callback within 1 hour',
                'Flag account for monitoring',
                'Notify fraud prevention team'
            ]
        },
        'medium': {
            'threshold': 40,
            'team': 'Customer Protection Team',
            'contact': 'customer.protection@hsbc.com',
            'phone': '+44 20 7991 8666',
            'sla': '24 hours',
            'actions': [
                'Monitor account activity for 24 hours',
                'Review recent transaction history',
                'Prepare customer education materials',
                'Schedule follow-up within 48 hours'
            ]
        },
        'low': {
            'threshold': 20,
            'team': 'Customer Service',
            'contact': 'customer.service@hsbc.com',
            'sla': 'Standard',
            'actions': [
                'Document interaction details',
                'Standard account monitoring',
                'No immediate action required'
            ]
        }
    }
    
    # Determine escalation level
    escalation_level = 'low'
    if risk_score >= 80:
        escalation_level = 'critical'
    elif risk_score >= 60:
        escalation_level = 'high'
    elif risk_score >= 40:
        escalation_level = 'medium'
    
    escalation_info = escalation_matrix[escalation_level]
    
    return {
        'escalation_level': escalation_level,
        'escalation_required': risk_score >= 40,
        'escalation_info': escalation_info,
        'urgency_override': urgency_level == 'emergency',
        'timestamp': get_current_timestamp()
    }

# ===== SUMMARIZATION TOOLS =====

def transcript_analyzer(transcript_segments: List[Dict], speaker_identification: Dict = None) -> Dict[str, Any]:
    """Analyzes call transcripts for key information and evidence"""
    
    customer_segments = []
    agent_segments = []
    key_quotes = []
    timeline = []
    
    for segment in transcript_segments:
        speaker = segment.get('speaker', 'unknown')
        text = segment.get('text', '')
        timestamp = segment.get('start', 0)
        
        if speaker == 'customer':
            customer_segments.append(text)
            # Extract key quotes that might be evidence
            if any(keyword in text.lower() for keyword in ['never met', 'online', 'urgent', 'emergency', 'guaranteed', 'stuck']):
                key_quotes.append({
                    'text': text,
                    'timestamp': timestamp,
                    'significance': 'fraud_indicator'
                })
        else:
            agent_segments.append(text)
        
        timeline.append({
            'timestamp': timestamp,
            'speaker': speaker,
            'text': text[:100] + '...' if len(text) > 100 else text
        })
    
    # Analyze customer behavior patterns
    customer_text = ' '.join(customer_segments)
    behavior_indicators = []
    
    if 'never met' in customer_text.lower():
        behavior_indicators.append('Never met relationship partner in person')
    if any(word in customer_text.lower() for word in ['urgent', 'immediately', 'today']):
        behavior_indicators.append('Expressing urgency and time pressure')
    if any(word in customer_text.lower() for word in ['stuck', 'stranded', 'emergency']):
        behavior_indicators.append('Describing emergency situation')
    if 'guarantee' in customer_text.lower():
        behavior_indicators.append('Attracted to guaranteed returns')
    
    return {
        'customer_speech_segments': len(customer_segments),
        'agent_speech_segments': len(agent_segments),
        'key_quotes': key_quotes,
        'behavior_indicators': behavior_indicators,
        'full_customer_text': customer_text,
        'timeline': timeline,
        'analysis_timestamp': get_current_timestamp()
    }

def incident_formatter(fraud_analysis: Dict, policy_guidance: Dict, customer_info: Dict) -> Dict[str, Any]:
    """Formats incident data for ServiceNow case creation"""
    
    risk_score = fraud_analysis.get('risk_score', 0)
    scam_type = fraud_analysis.get('scam_type', 'unknown')
    detected_patterns = fraud_analysis.get('detected_patterns', {})
    
    # Format executive summary
    executive_summary = f"{risk_score}% risk {scam_type.replace('_', ' ')} detected involving {customer_info.get('name', 'customer')} (Account: {customer_info.get('account', 'unknown')})."
    
    if risk_score >= 80:
        executive_summary += " CRITICAL intervention required."
    elif risk_score >= 60:
        executive_summary += " HIGH risk scenario requiring enhanced monitoring."
    
    # Format detected patterns
    patterns_text = ""
    if detected_patterns:
        patterns_text = "\n".join([
            f"• {pattern.replace('_', ' ').title()}: {data.get('severity', 'medium').upper()} ({data.get('count', 1)} instances)"
            for pattern, data in detected_patterns.items()
        ])
    
    # Format recommended actions
    policy_actions = policy_guidance.get('policy', {}).get('procedures', [])
    escalation_actions = policy_guidance.get('escalation_info', {}).get('actions', [])
    all_actions = policy_actions + escalation_actions
    
    actions_text = "\n".join([f"• {action}" for action in all_actions[:6]])  # Limit to 6 actions
    
    # Determine priority
    if risk_score >= 80:
        priority = "1"  # Critical
        urgency = "1"
    elif risk_score >= 60:
        priority = "2"  # High
        urgency = "2"
    else:
        priority = "3"  # Medium
        urgency = "3"
    
    formatted_incident = {
        'short_description': f"Fraud Alert: {scam_type.replace('_', ' ').title()} - {customer_info.get('name', 'Customer')}",
        'description': f"""=== AUTOMATED FRAUD DETECTION ALERT ===

EXECUTIVE SUMMARY:
{executive_summary}

CUSTOMER INFORMATION:
• Name: {customer_info.get('name', 'Unknown')}
• Account: {customer_info.get('account', 'Unknown')}
• Phone: {customer_info.get('phone', 'Unknown')}

FRAUD ANALYSIS:
• Risk Score: {risk_score}%
• Scam Type: {scam_type.replace('_', ' ').title()}
• Confidence: {fraud_analysis.get('confidence', 0):.1%}

DETECTED PATTERNS:
{patterns_text or 'No specific patterns detected'}

RECOMMENDED ACTIONS:
{actions_text}

Generated: {get_current_timestamp()}
System: HSBC ADK Fraud Detection System
        """,
        'priority': priority,
        'urgency': urgency,
        'category': 'security',
        'state': '2',  # In Progress
        'caller_id': customer_info.get('name', 'Unknown'),
        'u_risk_score': risk_score,
        'u_scam_type': scam_type,
        'u_confidence': fraud_analysis.get('confidence', 0)
    }
    
    return {
        'formatted_incident': formatted_incident,
        'servicenow_ready': True,
        'priority_level': priority,
        'formatting_timestamp': get_current_timestamp()
    }

# ===== ORCHESTRATOR TOOLS =====

def agent_coordinator(scam_analysis: Dict, policy_guidance: Dict, incident_summary: Dict) -> Dict[str, Any]:
    """Coordinates inputs from multiple agents for decision-making"""
    
    # Extract key metrics from each agent
    risk_score = scam_analysis.get('risk_score', 0)
    risk_level = scam_analysis.get('risk_level', 'LOW')
    scam_type = scam_analysis.get('scam_type', 'unknown')
    detected_patterns = scam_analysis.get('detected_patterns', {})
    
    escalation_required = policy_guidance.get('escalation_required', False)
    escalation_level = policy_guidance.get('escalation_level', 'low')
    
    incident_ready = incident_summary.get('servicenow_ready', False)
    
    # Calculate coordination score
    coordination_factors = []
    
    if risk_score >= 80:
        coordination_factors.append('Critical risk level detected')
    if len(detected_patterns) >= 3:
        coordination_factors.append('Multiple fraud patterns confirmed')
    if escalation_required:
        coordination_factors.append('Policy escalation threshold exceeded')
    if scam_type in ['romance_scam', 'investment_scam']:
        coordination_factors.append('High-impact scam type identified')
    
    # Determine consensus recommendation
    if risk_score >= 80 or escalation_level == 'critical':
        consensus_action = 'BLOCK_AND_ESCALATE'
        consensus_priority = 'CRITICAL'
    elif risk_score >= 60 or escalation_level == 'high':
        consensus_action = 'VERIFY_AND_MONITOR'
        consensus_priority = 'HIGH'
    elif risk_score >= 40 or escalation_level == 'medium':
        consensus_action = 'MONITOR_AND_EDUCATE'
        consensus_priority = 'MEDIUM'
    else:
        consensus_action = 'CONTINUE_NORMAL'
        consensus_priority = 'LOW'
    
    return {
        'agent_consensus': {
            'scam_detection_confidence': scam_analysis.get('confidence', 0),
            'policy_guidance_applicable': policy_guidance.get('applicable', False),
            'incident_documentation_ready': incident_ready
        },
        'coordination_factors': coordination_factors,
        'consensus_action': consensus_action,
        'consensus_priority': consensus_priority,
        'all_agents_aligned': len(coordination_factors) >= 2,
        'coordination_timestamp': get_current_timestamp()
    }

def decision_engine(risk_assessment: Dict, policy_requirements: Dict, customer_impact: Dict) -> Dict[str, Any]:
    """Makes final fraud prevention decisions based on all available evidence"""
    
    risk_score = risk_assessment.get('risk_score', 0)
    risk_level = risk_assessment.get('risk_level', 'LOW')
    
    # Decision matrix
    if risk_score >= 80:
        decision = 'BLOCK_AND_ESCALATE'
        reasoning = 'Critical fraud risk detected - immediate intervention required'
        priority = 'CRITICAL'
        immediate_actions = [
            'Block all pending transactions immediately',
            'Escalate to Financial Crime Team',
            'Contact customer via secure channel within 15 minutes',
            'Initiate fraud investigation',
            'Document all evidence'
        ]
        escalation_path = 'Financial Crime Team (immediate)'
        customer_impact = 'Transaction blocked for protection, fraud team will contact customer'
        case_creation = True
        
    elif risk_score >= 60:
        decision = 'VERIFY_AND_MONITOR'
        reasoning = 'High fraud risk - enhanced verification required'
        priority = 'HIGH'
        immediate_actions = [
            'Require additional customer verification',
            'Monitor account for 48 hours',
            'Flag for supervisor review',
            'Provide fraud education to customer',
            'Hold suspicious transactions for review'
        ]
        escalation_path = 'Fraud Prevention Team (1 hour SLA)'
        customer_impact = 'Enhanced verification required, may experience brief delays'
        case_creation = True
        
    elif risk_score >= 40:
        decision = 'MONITOR_AND_EDUCATE'
        reasoning = 'Moderate risk - increased vigilance recommended'
        priority = 'MEDIUM'
        immediate_actions = [
            'Continue processing with caution',
            'Provide relevant fraud warnings',
            'Document conversation details',
            'Consider follow-up contact',
            'Monitor for pattern escalation'
        ]
        escalation_path = 'Customer Protection Team (24 hour SLA)'
        customer_impact = 'Standard processing with additional fraud education'
        case_creation = True  # Auto-create for medium+ risk
        
    else:
        decision = 'CONTINUE_NORMAL'
        reasoning = 'Low fraud risk - standard processing'
        priority = 'LOW'
        immediate_actions = [
            'Continue normal customer service',
            'Standard documentation procedures',
            'No special monitoring required'
        ]
        escalation_path = 'No escalation required'
        customer_impact = 'Normal service experience'
        case_creation = False
    
    return {
        'final_decision': decision,
        'reasoning': reasoning,
        'priority': priority,
        'immediate_actions': immediate_actions,
        'escalation_path': escalation_path,
        'customer_impact': customer_impact,
        'case_creation_required': case_creation,
        'confidence': min(risk_assessment.get('confidence', 0.5) + 0.2, 1.0),
        'decision_timestamp': get_current_timestamp()
    }
