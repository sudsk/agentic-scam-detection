# Common patterns used across multiple scam types
FRAUD_PATTERN_CONFIG = {
    # High-Risk Universal Patterns (25-30%)
    'romance_exploitation': {
        'name': 'Romance Exploitation',
        'weight': 30,
        'severity': 'critical',
        'scam_types': ['romance_scam', 'app_scam'],
        'keywords': ['boyfriend', 'girlfriend', 'online relationship', 'dating app', 'never met']
    },
    
    'emergency_money_request': {
        'name': 'Emergency Money Request', 
        'weight': 25,
        'severity': 'critical',
        'scam_types': ['romance_scam', 'app_scam', 'impersonation_scam'],
        'keywords': ['emergency', 'hospital', 'stuck', 'stranded', 'urgent help']
    },
    
    'secrecy_request': {
        'name': 'Secrecy Request',
        'weight': 22,
        'severity': 'critical', 
        'scam_types': ['romance_scam', 'app_scam', 'authority_scam'],
        'keywords': ['keep secret', 'don\'t tell', 'private', 'confidential']
    },
    
    # Medium-Risk Universal Patterns (15-20%)
    'urgency_pressure': {
        'name': 'Urgency Pressure',
        'weight': 15,
        'severity': 'medium',
        'scam_types': ['romance_scam', 'investment_scam', 'app_scam', 'authority_scam'],
        'keywords': ['urgent', 'immediately', 'right away', 'today only', 'deadline']
    },
    
    'authority_impersonation': {
        'name': 'Authority Impersonation',
        'weight': 35,
        'severity': 'critical',
        'scam_types': ['impersonation_scam', 'authority_scam'],
        'keywords': ['police', 'HMRC', 'bank security', 'government', 'official']
    },
    
    # Scam-Specific Patterns
    'guaranteed_returns': {
        'name': 'Guaranteed Returns',
        'weight': 28,
        'severity': 'critical',
        'scam_types': ['investment_scam'],
        'keywords': ['guaranteed', 'no risk', 'certain profit', '100% return']
    },
    
    'never_met_beneficiary': {
        'name': 'Never Met Beneficiary',
        'weight': 20,
        'severity': 'high',
        'scam_types': ['romance_scam', 'app_scam'],
        'keywords': ['never met', 'not met in person', 'online only']
    },
    
    'overseas_transfer': {
        'name': 'Overseas Transfer',
        'weight': 12,
        'severity': 'medium',
        'scam_types': ['romance_scam', 'app_scam', 'investment_scam'],
        'keywords': ['overseas', 'international', 'abroad', 'foreign country']
    }
}

# Scam type definitions with applicable patterns
SCAM_TYPE_PATTERNS = {
    'romance_scam': [
        'romance_exploitation', 'emergency_money_request', 'secrecy_request',
        'urgency_pressure', 'never_met_beneficiary', 'overseas_transfer'
    ],
    'investment_scam': [
        'guaranteed_returns', 'urgency_pressure', 'authority_impersonation', 
        'overseas_transfer'
    ],
    'app_scam': [
        'emergency_money_request', 'secrecy_request', 'urgency_pressure',
        'never_met_beneficiary', 'overseas_transfer'
    ],
    'impersonation_scam': [
        'authority_impersonation', 'urgency_pressure', 'secrecy_request'
    ],
    'authority_scam': [
        'authority_impersonation', 'urgency_pressure', 'secrecy_request'
    ]
}
