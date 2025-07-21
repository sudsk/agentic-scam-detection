# backend/config/demo_scripts.py
"""
Complete Demo Scripts Configuration for Banking Fraud Detection
Contains all demo scenarios with banking context and compliance notes
"""

from typing import Dict, Any, List, Optional

# ===== DEMO SCRIPT INTERFACES =====

class DemoScriptEvent:
    def __init__(self, **kwargs):
        self.triggerAtSeconds = kwargs.get('triggerAtSeconds', 0)
        self.customerText = kwargs.get('customerText', '')
        self.detectedPatterns = kwargs.get('detectedPatterns', {})
        self.riskScore = kwargs.get('riskScore', 0)
        self.riskFactors = kwargs.get('riskFactors', [])
        self.suggestedQuestion = kwargs.get('suggestedQuestion', {})
        self.simulateAskedAfterMs = kwargs.get('simulateAskedAfterMs', 3000)
        self.agentAction = kwargs.get('agentAction', '')
        self.complianceNote = kwargs.get('complianceNote', '')
        self.bankingContext = kwargs.get('bankingContext', {})

class DemoScript:
    def __init__(self, **kwargs):
        self.title = kwargs.get('title', '')
        self.description = kwargs.get('description', '')
        self.customerProfile = kwargs.get('customerProfile', '')
        self.initialRisk = kwargs.get('initialRisk', 5)
        self.timeline = kwargs.get('timeline', [])
        self.duration = kwargs.get('duration', 60)
        self.scamType = kwargs.get('scamType', 'unknown')
        self.expectedOutcome = kwargs.get('expectedOutcome', '')
        self.complianceLevel = kwargs.get('complianceLevel', 'standard')

# ===== COMPLETE DEMO SCRIPTS =====

DEMO_SCRIPTS = {
    'romance_scam_1.wav': DemoScript(
        title='Romance Scam Investigation Demo',
        description='Progressive detection of romance scam with emotional manipulation tactics',
        customerProfile='Mrs Patricia Williams - Premier Banking Customer',
        initialRisk=5,
        duration=65,
        scamType='romance_scam',
        expectedOutcome='Transaction blocked, case escalated to Financial Crime Team',
        complianceLevel='critical',
        timeline=[
            DemoScriptEvent(
                triggerAtSeconds=12,
                customerText="I need to make an urgent international transfer to Turkey",
                detectedPatterns={
                    'Urgency Pressure': {
                        'pattern_name': 'Urgency Pressure',
                        'count': 1,
                        'weight': 15,
                        'severity': 'medium',
                        'matches': ['urgent', 'international transfer']
                    }
                },
                riskScore=20,
                riskFactors=['urgency_pressure', 'overseas_beneficiary'],
                suggestedQuestion={
                    'question': "What specifically makes this transaction urgent?",
                    'context': "Urgency pressure detected in customer request",
                    'urgency': 'medium',
                    'explanation': "AI detected urgency keywords 'urgent' and 'international' - common fraud indicators",
                    'pattern': "Urgency Pressure"
                },
                simulateAskedAfterMs=3000,
                agentAction="Agent investigates the claimed urgency",
                complianceNote="Following FCA guidelines for APP fraud prevention",
                bankingContext={
                    'transactionType': "International Transfer",
                    'amount': "£3,000",
                    'beneficiary': "Unknown - Turkey",
                    'riskFlags': ["Urgency", "International", "High-risk country"]
                }
            ),
            
            DemoScriptEvent(
                triggerAtSeconds=28,
                customerText="My boyfriend is stuck overseas and needs money for hospital bills",
                detectedPatterns={
                    'Urgency Pressure': {
                        'pattern_name': 'Urgency Pressure',
                        'count': 2,
                        'weight': 15,
                        'severity': 'medium',
                        'matches': ['urgent', 'stuck', 'needs money']
                    },
                    'Romance Exploitation': {
                        'pattern_name': 'Romance Exploitation',
                        'count': 1,
                        'weight': 25,
                        'severity': 'high',
                        'matches': ['boyfriend', 'overseas', 'money']
                    },
                    'Emergency Money Request': {
                        'pattern_name': 'Emergency Money Request',
                        'count': 1,
                        'weight': 18,
                        'severity': 'high',
                        'matches': ['hospital bills', 'needs money']
                    }
                },
                riskScore=45,
                riskFactors=['urgency_pressure', 'romance_exploitation', 'emergency_money_request', 'overseas_beneficiary'],
                suggestedQuestion={
                    'question': "How did you first meet this person?",
                    'context': "Romance scam indicators detected - investigating relationship origin",
                    'urgency': 'high',
                    'explanation': "AI detected romance scam pattern: boyfriend + overseas + emergency money request",
                    'pattern': "Romance Exploitation"
                },
                simulateAskedAfterMs=3500,
                agentAction="Agent probes relationship background and authenticity",
                complianceNote="Enhanced due diligence required for relationship-based transfers",
                bankingContext={
                    'transactionType': "Emergency Transfer",
                    'amount': "£3,000",
                    'beneficiary': "Romantic partner - Turkey",
                    'riskFlags': ["Romance scam", "Never met in person", "Emergency request", "Medical emergency"]
                }
            ),
            
            DemoScriptEvent(
                triggerAtSeconds=45,
                customerText="We met online through a dating app about 3 weeks ago. He's very sweet and caring",
                detectedPatterns={
                    'Romance Exploitation': {
                        'pattern_name': 'Romance Exploitation',
                        'count': 2,
                        'weight': 25,
                        'severity': 'high',
                        'matches': ['met online', 'dating app', 'sweet and caring']
                    },
                    'Online Relationship': {
                        'pattern_name': 'Online Relationship',
                        'count': 1,
                        'weight': 12,
                        'severity': 'medium',
                        'matches': ['met online', 'dating app']
                    },
                    'Short Relationship Duration': {
                        'pattern_name': 'Short Relationship Duration',
                        'count': 1,
                        'weight': 8,
                        'severity': 'medium',
                        'matches': ['3 weeks ago']
                    }
                },
                riskScore=68,
                riskFactors=['urgency_pressure', 'romance_exploitation', 'emergency_money_request', 'overseas_beneficiary', 'online_relationship', 'short_relationship'],
                suggestedQuestion={
                    'question': "Have you met this person in person?",
                    'context': "Short online relationship with emergency money request - high fraud risk",
                    'urgency': 'high',
                    'explanation': "Critical red flag: Online relationship + 3 weeks duration + emergency money = classic romance scam",
                    'pattern': "Online Relationship"
                },
                simulateAskedAfterMs=3000,
                agentAction="Agent investigates physical meeting to confirm authenticity",
                complianceNote="APP fraud prevention - verifying genuine relationship",
                bankingContext={
                    'transactionType': "High-Risk Transfer",
                    'amount': "£3,000",
                    'beneficiary': "Online dating contact - Never met",
                    'riskFlags': ["Online only", "Short duration", "Dating app", "No verification"]
                }
            ),
            
            DemoScriptEvent(
                triggerAtSeconds=62,
                customerText="No, we haven't met in person yet, but he says he loves me and needs my help. He asked me not to tell anyone about this",
                detectedPatterns={
                    'Romance Exploitation': {
                        'pattern_name': 'Romance Exploitation',
                        'count': 3,
                        'weight': 25,
                        'severity': 'critical',
                        'matches': ['loves me', 'needs my help', 'never met']
                    },
                    'Never Met Person': {
                        'pattern_name': 'Never Met Beneficiary',
                        'count': 1,
                        'weight': 20,
                        'severity': 'high',
                        'matches': ['haven\'t met in person']
                    },
                    'Secrecy Request': {
                        'pattern_name': 'Secrecy Request',
                        'count': 1,
                        'weight': 22,
                        'severity': 'critical',
                        'matches': ['not to tell anyone']
                    }
                },
                riskScore=95,
                riskFactors=['urgency_pressure', 'romance_exploitation', 'emergency_money_request', 'overseas_beneficiary', 'online_relationship', 'short_relationship', 'never_met_person', 'secrecy_request'],
                suggestedQuestion={
                    'question': "Why would someone who loves you ask you to keep financial help secret from family and friends?",
                    'context': "CRITICAL: All major romance scam indicators present",
                    'urgency': 'critical',
                    'explanation': "FRAUD ALERT: Never met + love claims + secrecy + emergency money = 95% fraud probability",
                    'pattern': "Secrecy Request"
                },
                simulateAskedAfterMs=4000,
                agentAction="Agent challenges the secrecy and emotional manipulation tactics",
                complianceNote="MANDATORY INTERVENTION: All APP fraud indicators present - transaction must be blocked",
                bankingContext={
                    'transactionType': "BLOCKED TRANSACTION",
                    'amount': "£3,000",
                    'beneficiary': "SUSPECTED ROMANCE SCAMMER",
                    'riskFlags': ["All fraud indicators", "Mandatory block", "FCA compliance", "Customer protection"]
                }
            )
        ]
    ),

    'investment_scam_1.wav': DemoScript(
        title='Investment Scam Detection Demo',
        description='Progressive detection of high-yield investment fraud with pressure tactics',
        customerProfile='Mr Robert Chen - Business Banking Customer',
        initialRisk=8,
        duration=58,
        scamType='investment_scam',
        expectedOutcome='Investment blocked, regulatory warnings provided',
        complianceLevel='enhanced',
        timeline=[
            DemoScriptEvent(
                triggerAtSeconds=15,
                customerText="I want to invest £25,000 in a cryptocurrency opportunity with guaranteed 15% monthly returns",
                detectedPatterns={
                    'Investment Fraud': {
                        'pattern_name': 'Investment Fraud',
                        'count': 1,
                        'weight': 20,
                        'severity': 'high',
                        'matches': ['cryptocurrency', 'guaranteed', '15% monthly']
                    },
                    'Unrealistic Returns': {
                        'pattern_name': 'Unrealistic Returns',
                        'count': 1,
                        'weight': 25,
                        'severity': 'high',
                        'matches': ['guaranteed 15% monthly']
                    }
                },
                riskScore=45,
                riskFactors=['investment_fraud', 'unrealistic_returns', 'large_amount'],
                suggestedQuestion={
                    'question': "What regulatory body oversees this investment opportunity?",
                    'context': "Unrealistic returns claimed - potential investment scam",
                    'urgency': 'high',
                    'explanation': "AI detected guaranteed high returns claim - major red flag for investment fraud",
                    'pattern': "Investment Fraud"
                },
                simulateAskedAfterMs=3500,
                agentAction="Agent investigates regulatory compliance and legitimacy",
                complianceNote="FCA requires verification of investment firms and realistic return expectations",
                bankingContext={
                    'transactionType': "Investment Transfer",
                    'amount': "£25,000",
                    'beneficiary': "Cryptocurrency platform",
                    'riskFlags': ["Guaranteed returns", "Unregulated", "High yield", "Crypto investment"]
                }
            ),

            DemoScriptEvent(
                triggerAtSeconds=32,
                customerText="They said this is a limited time offer and I need to invest today or I'll miss out. No regulatory oversight needed because it's blockchain-based",
                detectedPatterns={
                    'Investment Fraud': {
                        'pattern_name': 'Investment Fraud',
                        'count': 2,
                        'weight': 20,
                        'severity': 'high',
                        'matches': ['limited time', 'no regulatory oversight', 'blockchain-based']
                    },
                    'Urgency Pressure': {
                        'pattern_name': 'Urgency Pressure',
                        'count': 1,
                        'weight': 15,
                        'severity': 'medium',
                        'matches': ['today', 'miss out', 'limited time']
                    },
                    'Regulatory Avoidance': {
                        'pattern_name': 'Regulatory Avoidance',
                        'count': 1,
                        'weight': 18,
                        'severity': 'high',
                        'matches': ['no regulatory oversight']
                    }
                },
                riskScore=75,
                riskFactors=['investment_fraud', 'unrealistic_returns', 'urgency_pressure', 'regulatory_avoidance', 'large_amount'],
                suggestedQuestion={
                    'question': "Why would a legitimate investment opportunity claim to avoid regulatory oversight?",
                    'context': "Regulatory avoidance and time pressure - classic investment scam tactics",
                    'urgency': 'high',
                    'explanation': "Multiple red flags: time pressure + regulatory avoidance + unrealistic returns",
                    'pattern': "Regulatory Avoidance"
                },
                simulateAskedAfterMs=4000,
                agentAction="Agent explains regulatory requirements and warns about scam tactics",
                complianceNote="All legitimate investments require FCA authorization - explain regulatory protection",
                bankingContext={
                    'transactionType': "SUSPICIOUS INVESTMENT",
                    'amount': "£25,000",
                    'beneficiary': "Unregulated crypto scheme",
                    'riskFlags': ["No regulation", "Time pressure", "Blockchain excuse", "Classic scam"]
                }
            )
        ]
    ),

    'impersonation_scam_1.wav': DemoScript(
        title='Bank Impersonation Scam Demo',
        description='Detection of criminals impersonating HSBC security team',
        customerProfile='Mrs Sarah Foster - Personal Banking Customer',
        initialRisk=10,
        duration=42,
        scamType='impersonation_scam',
        expectedOutcome='Impersonation detected, security measures activated',
        complianceLevel='enhanced',
        timeline=[
            DemoScriptEvent(
                triggerAtSeconds=18,
                customerText="I received a call from HSBC security saying my account was compromised and I need to verify my details immediately",
                detectedPatterns={
                    'Impersonation Scam': {
                        'pattern_name': 'Impersonation Scam',
                        'count': 1,
                        'weight': 22,
                        'severity': 'high',
                        'matches': ['HSBC security', 'account compromised', 'verify details']
                    },
                    'Urgency Pressure': {
                        'pattern_name': 'Urgency Pressure',
                        'count': 1,
                        'weight': 15,
                        'severity': 'medium',
                        'matches': ['immediately']
                    }
                },
                riskScore=42,
                riskFactors=['impersonation_scam', 'urgency_pressure', 'credential_request'],
                suggestedQuestion={
                    'question': "What specific department did they claim to be calling from, and did they provide a reference number?",
                    'context': "Potential bank impersonation - verifying authenticity",
                    'urgency': 'high',
                    'explanation': "AI detected bank impersonation pattern - HSBC never requests verification over phone",
                    'pattern': "Impersonation Scam"
                },
                simulateAskedAfterMs=3200,
                agentAction="Agent verifies call legitimacy and explains security protocols",
                complianceNote="Bank impersonation scams require immediate customer education",
                bankingContext={
                    'transactionType': "Security Verification",
                    'amount': "N/A",
                    'beneficiary': "Fraudulent caller",
                    'riskFlags': ["Bank impersonation", "Credential harvesting", "Phone fraud"]
                }
            )
        ]
    ),

    'app_scam_1.wav': DemoScript(
        title='Authorized Push Payment Scam Demo',
        description='Detection of APP fraud with invoice manipulation',
        customerProfile='Mr Adam Henderson - Business Banking Customer',
        initialRisk=12,
        duration=48,
        scamType='app_scam',
        expectedOutcome='APP fraud detected, payment verification required',
        complianceLevel='enhanced',
        timeline=[
            DemoScriptEvent(
                triggerAtSeconds=20,
                customerText="I need to pay an urgent invoice that just came in. The supplier changed their bank details and needs payment today",
                detectedPatterns={
                    'APP Fraud': {
                        'pattern_name': 'APP Fraud',
                        'count': 1,
                        'weight': 18,
                        'severity': 'medium',
                        'matches': ['urgent invoice', 'changed bank details']
                    },
                    'Urgency Pressure': {
                        'pattern_name': 'Urgency Pressure',
                        'count': 1,
                        'weight': 15,
                        'severity': 'medium',
                        'matches': ['urgent', 'today']
                    },
                    'Bank Detail Change': {
                        'pattern_name': 'Bank Detail Change',
                        'count': 1,
                        'weight': 20,
                        'severity': 'high',
                        'matches': ['changed their bank details']
                    }
                },
                riskScore=38,
                riskFactors=['app_fraud', 'urgency_pressure', 'bank_detail_change'],
                suggestedQuestion={
                    'question': "How did you receive notification of the bank detail change?",
                    'context': "Bank detail changes are common in APP fraud - verification needed",
                    'urgency': 'medium',
                    'explanation': "AI detected APP fraud pattern: urgent payment + changed bank details",
                    'pattern': "Bank Detail Change"
                },
                simulateAskedAfterMs=3400,
                agentAction="Agent investigates communication method and authenticity",
                complianceNote="APP fraud prevention requires verification of bank detail changes",
                bankingContext={
                    'transactionType': "Business Payment",
                    'amount': "£8,500",
                    'beneficiary': "Supplier with changed details",
                    'riskFlags': ["Changed bank details", "Urgent payment", "APP fraud risk"]
                }
            )
        ]
    ),

    'legitimate_call_1.wav': DemoScript(
        title='Legitimate Banking Transaction Demo',
        description='Example of normal banking conversation with low fraud risk',
        customerProfile='Mr Michael Thompson - Personal Banking Customer',
        initialRisk=2,
        duration=35,
        scamType='none',
        expectedOutcome='Standard processing approved',
        complianceLevel='standard',
        timeline=[
            DemoScriptEvent(
                triggerAtSeconds=25,
                customerText="I'd like to transfer £500 to my daughter's account for her university expenses",
                detectedPatterns={},
                riskScore=8,
                riskFactors=['family_transfer'],
                suggestedQuestion={
                    'question': "Can you confirm your daughter's account details for verification?",
                    'context': "Standard family transfer - routine verification",
                    'urgency': 'low',
                    'explanation': "Low risk family transfer - standard verification procedures apply",
                    'pattern': "Family Transfer"
                },
                simulateAskedAfterMs=4000,
                agentAction="Agent performs standard identity and account verification",
                complianceNote="Standard KYC procedures for family transfers",
                bankingContext={
                    'transactionType': "Family Transfer",
                    'amount': "£500",
                    'beneficiary': "Family member",
                    'riskFlags': ["Standard verification", "Low risk", "Legitimate transaction"]
                }
            )
        ]
    )
}

# ===== DEMO CONFIGURATION SETTINGS =====

DEMO_CONFIG = {
    # Timing settings
    'DEFAULT_SIMULATION_DELAY': 3000,
    'MIN_SIMULATION_DELAY': 1500,
    'MAX_SIMULATION_DELAY': 5000,
    
    # UI settings
    'SHOW_EXPLANATIONS_DEFAULT': True,
    'AUTO_SIMULATE_DEFAULT': True,
    'SHOW_RISK_BREAKDOWN_DEFAULT': True,
    
    # Demo control settings
    'ALLOW_MANUAL_OVERRIDE': True,
    'SHOW_DEMO_CONTROLS': True,
    'SHOW_PROGRESS_TRACKING': True,
    
    # Explanation display settings
    'EXPLANATION_DURATION': 4000,
    'RISK_UPDATE_ANIMATION_DURATION': 800,
    'QUESTION_HIGHLIGHT_DURATION': 6000,
    
    # Risk thresholds for demo annotations
    'RISK_THRESHOLDS': {
        'LOW': 25,
        'MEDIUM': 45,
        'HIGH': 70,
        'CRITICAL': 85
    }
}

# ===== DEMO UTILITIES =====

def getDemoScript(filename: str) -> Optional[DemoScript]:
    """Get demo script for filename"""
    return DEMO_SCRIPTS.get(filename)

def getAllDemoScripts() -> Dict[str, DemoScript]:
    """Get all demo scripts"""
    return DEMO_SCRIPTS

def getDemoScriptList() -> List[Dict[str, Any]]:
    """Get list of demo scripts with basic info"""
    return [
        {
            'id': script_id,
            'title': script.title,
            'scamType': script.scamType,
            'duration': script.duration,
            'complianceLevel': script.complianceLevel
        }
        for script_id, script in DEMO_SCRIPTS.items()
    ]

def getRiskLevelFromScore(score: int) -> str:
    """Get risk level from score"""
    thresholds = DEMO_CONFIG['RISK_THRESHOLDS']
    if score >= thresholds['CRITICAL']:
        return 'CRITICAL'
    elif score >= thresholds['HIGH']:
        return 'HIGH'
    elif score >= thresholds['MEDIUM']:
        return 'MEDIUM'
    elif score >= thresholds['LOW']:
        return 'LOW'
    else:
        return 'MINIMAL'

def getDemoExplanation(riskScore: int, patterns: List[str]) -> str:
    """Get demo explanation for risk score and patterns"""
    riskLevel = getRiskLevelFromScore(riskScore)
    patternText = ', '.join(patterns) if patterns else 'No patterns'
    
    return f"Risk Level: {riskLevel} ({riskScore}%) - Detected: {patternText}"

# Export main functions
__all__ = [
    'DEMO_SCRIPTS',
    'DEMO_CONFIG', 
    'DemoScript',
    'DemoScriptEvent',
    'getDemoScript',
    'getAllDemoScripts',
    'getDemoScriptList',
    'getRiskLevelFromScore',
    'getDemoExplanation'
]
