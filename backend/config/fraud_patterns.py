# backend/config/fraud_patterns.py
"""
Fraud Pattern Configuration - Centralized pattern definitions
Shared across all scam types with realistic weights and keywords
"""

from typing import Dict, List, Any

# ===== UNIVERSAL FRAUD PATTERNS =====
# Patterns that can appear across multiple scam types

FRAUD_PATTERN_CONFIG = {
    # ===== CRITICAL RISK PATTERNS (25-35%) =====
    
    'authority_impersonation': {
        'name': 'Authority Impersonation',
        'weight': 35,
        'severity': 'critical',
        'scam_types': ['impersonation_scam', 'authority_scam'],
        'keywords': [
            'police', 'hmrc', 'tax office', 'bank security', 'fraud team',
            'government', 'official', 'investigation', 'court', 'legal action',
            'arrest warrant', 'prosecution', 'fine', 'penalty'
        ],
        'description': 'Impersonating government authorities, police, or bank officials'
    },
    
    'romance_exploitation': {
        'name': 'Romance Exploitation',
        'weight': 30,
        'severity': 'critical',
        'scam_types': ['romance_scam', 'app_scam'],
        'keywords': [
            'boyfriend', 'girlfriend', 'partner', 'love', 'relationship',
            'online dating', 'dating app', 'met online', 'internet dating',
            'soulmate', 'marriage', 'engaged', 'loves me', 'trust me'
        ],
        'description': 'Exploiting romantic relationships to request money'
    },
    
    'guaranteed_returns': {
        'name': 'Guaranteed Investment Returns',
        'weight': 28,
        'severity': 'critical',
        'scam_types': ['investment_scam'],
        'keywords': [
            'guaranteed returns', 'no risk', 'certain profit', '100% return',
            'guaranteed profit', 'risk-free', 'guaranteed income', 'sure thing',
            'can\'t lose', 'guaranteed success', 'monthly returns guaranteed'
        ],
        'description': 'Promising unrealistic guaranteed investment returns'
    },
    
    'emergency_money_request': {
        'name': 'Emergency Money Request',
        'weight': 25,
        'severity': 'critical',
        'scam_types': ['romance_scam', 'app_scam', 'impersonation_scam'],
        'keywords': [
            'emergency', 'hospital', 'medical emergency', 'stuck', 'stranded',
            'urgent help', 'crisis', 'accident', 'jail', 'arrested',
            'bail money', 'surgery', 'treatment', 'life threatening'
        ],
        'description': 'Claiming emergency situations requiring immediate financial help'
    },
    
    # ===== HIGH RISK PATTERNS (20-24%) =====
    
    'secrecy_request': {
        'name': 'Secrecy Request',
        'weight': 22,
        'severity': 'high',
        'scam_types': ['romance_scam', 'app_scam', 'authority_scam', 'investment_scam'],
        'keywords': [
            'keep secret', 'don\'t tell anyone', 'private', 'confidential',
            'between us', 'secret', 'don\'t mention', 'keep quiet',
            'family doesn\'t need to know', 'private matter'
        ],
        'description': 'Requesting victim keep transaction secret from family/friends'
    },
    
    'never_met_beneficiary': {
        'name': 'Never Met Beneficiary',
        'weight': 20,
        'severity': 'high',
        'scam_types': ['romance_scam', 'app_scam'],
        'keywords': [
            'never met', 'not met in person', 'online only', 'haven\'t met',
            'never seen', 'video call only', 'phone only', 'messages only',
            'will meet soon', 'planning to meet'
        ],
        'description': 'Sending money to someone never met in person'
    },
    
    # ===== MEDIUM RISK PATTERNS (15-19%) =====
    
    'investment_pressure': {
        'name': 'Investment Pressure Tactics',
        'weight': 18,
        'severity': 'medium',
        'scam_types': ['investment_scam'],
        'keywords': [
            'limited time offer', 'exclusive opportunity', 'insider information',
            'hot tip', 'act now', 'limited spaces', 'special deal',
            'friends and family rate', 'one time only'
        ],
        'description': 'High-pressure sales tactics for investment opportunities'
    },
    
    'social_engineering': {
        'name': 'Social Engineering',
        'weight': 15,
        'severity': 'medium', 
        'scam_types': ['impersonation_scam', 'authority_scam'],
        'keywords': ['social engineering'],
        'description': 'General social engineering tactics'
    },
    
    'urgency_pressure': {
        'name': 'Urgency Pressure',
        'weight': 15,
        'severity': 'medium',
        'scam_types': ['romance_scam', 'investment_scam', 'app_scam', 'authority_scam'],
        'keywords': [
            'urgent', 'immediately', 'right away', 'asap', 'today only',
            'deadline', 'time sensitive', 'can\'t wait', 'now or never',
            'expires today', 'last chance', 'urgent action required'
        ],
        'description': 'Creating artificial time pressure to rush decisions'
    },
    
    # ===== LOWER RISK PATTERNS (10-14%) =====
    
    'overseas_transfer': {
        'name': 'Overseas Transfer',
        'weight': 12,
        'severity': 'medium',
        'scam_types': ['romance_scam', 'app_scam', 'investment_scam'],
        'keywords': [
            'overseas', 'international', 'abroad', 'foreign country',
            'turkey', 'nigeria', 'ghana', 'philippines', 'malaysia',
            'sending abroad', 'international transfer'
        ],
        'description': 'International money transfers to high-risk jurisdictions'
    },
    
    'emotional_manipulation': {
        'name': 'Emotional Manipulation',
        'weight': 10,
        'severity': 'medium',
        'scam_types': ['romance_scam', 'app_scam'],
        'keywords': [
            'trust me', 'you\'re my only hope', 'i have no one else',
            'please help me', 'you\'re special', 'we\'re meant to be',
            'prove your love', 'if you love me', 'disappointing me'
        ],
        'description': 'Using emotional manipulation to influence victim decisions'
    },
    'unauthorised_investment_platform': {
        'name': 'Unauthorised Investment Platform',
        'weight': 10,  
        'severity': 'medium',
        'scam_types': ['investment_scam'],
        'keywords': [
            'not regulated by FCA', 'unregulated by FCA', 'not approved by FCA',
            'not authorized', 'no regulation', 'bypasses regulation',
            'unregulated platform', 'no oversight'
        ],
        'description': 'Investment platform lacks proper FCA regulation'
    },    

    'credential_requests': {
        'name': 'Credential Requests',
        'weight': 40,  # Very high risk
        'severity': 'critical',
        'scam_types': ['impersonation_scam', 'authority_scam'],
        'keywords': [
            'pin number', 'password', 'security code', 'sort code',
            'account number', 'card details', 'verification code',
            'online banking details', 'mother\'s maiden name'
        ],
        'description': 'Requesting banking credentials or personal security information'
    }
}

# ===== SCAM TYPE DEFINITIONS =====
# Which patterns are relevant for each scam type

SCAM_TYPE_PATTERNS = {
    'romance_scam': [
        'romance_exploitation',
        'emergency_money_request', 
        'secrecy_request',
        'never_met_beneficiary',
        'urgency_pressure',
        'overseas_transfer',
        'emotional_manipulation'
    ],
    
    'investment_scam': [
        'guaranteed_returns',
        'investment_pressure',
        'urgency_pressure',
        'secrecy_request',
        'overseas_transfer'
    ],
    
    'app_scam': [
        'emergency_money_request',
        'secrecy_request', 
        'urgency_pressure',
        'never_met_beneficiary',
        'overseas_transfer',
        'emotional_manipulation'
    ],
    
    'impersonation_scam': [
        'authority_impersonation',
        'credential_requests',
        'urgency_pressure',
        'secrecy_request'
    ],
    
    'authority_scam': [
        'authority_impersonation',
        'urgency_pressure',
        'secrecy_request',
        'credential_requests'
    ]
}

# ===== UTILITY FUNCTIONS =====

def get_pattern_config(pattern_id: str) -> Dict[str, Any]:
    """Get configuration for a specific pattern"""
    return FRAUD_PATTERN_CONFIG.get(pattern_id, {})

def get_pattern_weight(pattern_id: str) -> int:
    """Get weight for a specific pattern"""
    return FRAUD_PATTERN_CONFIG.get(pattern_id, {}).get('weight', 10)

def get_patterns_for_scam_type(scam_type: str) -> List[str]:
    """Get all applicable patterns for a scam type"""
    return SCAM_TYPE_PATTERNS.get(scam_type, [])

def create_keyword_mapping() -> Dict[str, Dict[str, Any]]:
    """
    Create mapping from keywords to pattern configs
    Used for pattern matching in the orchestrator
    """
    mapping = {}
    
    for pattern_id, config in FRAUD_PATTERN_CONFIG.items():
        # Map the pattern ID itself
        mapping[pattern_id] = {
            'name': config['name'],
            'weight': config['weight'],
            'severity': config['severity']
        }
        
        # Map all keywords to the same pattern
        for keyword in config['keywords']:
            mapping[keyword.lower()] = {
                'name': config['name'],
                'weight': config['weight'], 
                'severity': config['severity']
            }
    
    return mapping

def get_max_risk_score() -> int:
    """Calculate theoretical maximum risk score"""
    return sum(config['weight'] for config in FRAUD_PATTERN_CONFIG.values())

def get_risk_thresholds() -> Dict[str, int]:
    """Get risk score thresholds for decision making"""
    return {
        'minimal': 0,
        'low': 20,
        'medium': 40, 
        'high': 60,
        'critical': 80
    }

# ===== EXPORTS =====

__all__ = [
    'FRAUD_PATTERN_CONFIG',
    'SCAM_TYPE_PATTERNS', 
    'get_pattern_config',
    'get_pattern_weight',
    'get_patterns_for_scam_type',
    'create_keyword_mapping',
    'get_max_risk_score',
    'get_risk_thresholds'
]
