def retrieve_hsbc_policies(scam_type: str, risk_score: float) -> Dict[str, Any]:
    """
    Retrieve relevant HSBC fraud policies and procedures
    
    Args:
        scam_type: Type of scam detected (investment_scam, romance_scam, etc.)
        risk_score: Risk score from 0-100
        
    Returns:
        Dict containing relevant policies, procedures, and guidance
    """
    logger.info(f"📚 Retrieving policies for {scam_type} with risk score {risk_score}")
    
    # HSBC Policy Database
    policy_database = {
        'investment_scam_response': {
            'policy_id': 'FP-INV-001',
            'title': 'Investment Scam Detection and Response',
            'procedures': [
                'Immediately halt any investment-related transfers showing guaranteed return promises',
                'Ask customer: "Have you been able to withdraw any profits from this investment?"',
                'Verify investment company against FCA register using official website',
                'Explain: All legitimate investments carry risk - guarantees indicate fraud',
                'Document company name, contact details, and promised returns for investigation',
                'Transfer to Financial Crime Team if multiple red flags present'
            ],
            'key_questions': [
                'Have you been able to withdraw any money from this investment?',
                'Did they guarantee specific percentage returns?',
                'How did this investment company first contact you?',
                'Can you check if they are FCA regulated?'
            ],
            'customer_education': [
                'All legitimate investments carry risk - guaranteed returns are impossible',
                'Real investment firms are FCA regulated and listed on official register', 
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
                'Immediately inform customer: HSBC never asks for PINs or passwords over phone',
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
                'HSBC will never ask for your PIN, password, or full card details over the phone',
                'Legitimate authorities send official letters before taking any action',
                'Government agencies don\'t accept payments via bank transfer',
                'When in doubt, hang up and call the organization\'s official number'
            ],
            'escalation_threshold': 85
        }
    }
    
    # Map scam types to policies
    scam_policy_map = {
        'investment_scam': 'investment_scam_response',
        'romance_scam': 'romance_scam_response', 
        'impersonation_scam': 'impersonation_scam_response',
        'authority_scam': 'impersonation_scam_response',
        'app_fraud': 'impersonation_scam_response',
        'crypto_scam': 'investment_scam_response'
    }
    
    policy_key = scam_policy_map.get(scam_type, 'investment_scam_response')
    policy = policy_database.get(policy_key, {})
    
    # Generate escalation guidance
    escalation_guidance = {}
    if risk_score >= 80:
        escalation_guidance = {
            'level': 'immediate_escalation',
            'team': 'Financial Crime Team',
            'timeframe': 'immediately',
            'procedures': [
                'Stop all transactions immediately',
                'Transfer call to senior fraud specialist',
                'Document all evidence and customer statements',
                'Initiate fraud investigation case'
            ]
        }
    elif risk_score >= 60:
        escalation_guidance = {
            'level': 'enhanced_monitoring',
            'team': 'Fraud Prevention Team',
            'timeframe': 'within 1 hour',
            'procedures': [
                'Flag account for enhanced monitoring',
                'Verify customer identity thoroughly',
                'Document suspicious activity',
                'Prepare case file for review'
            ]
        }
    
    # Generate agent guidance
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
    
    result = {
        'primary_policy': policy,
        'escalation_guidance': escalation_guidance,
        'agent_guidance': agent_guidance,
        'risk_assessment': {
            'score': risk_score,
            'requires_escalation': risk_score >= policy.get('escalation_threshold', 80)
        }
    }
    
    logger.info(f"✅ Retrieved policies for {scam_type}")
    return result

# Create the RAG Policy Agent using Google ADK 1.4.2
rag_policy_agent = Agent(
    name="rag_policy_agent",
    model=LiteLlm("gemini/gemini-2.0-flash"),
    description="HSBC policy retrieval agent that provides fraud response guidance",
    instruction="""You are an HSBC policy and procedure specialist.

Your responsibilities:
1. Retrieve relevant HSBC fraud policies based on detected scam types
2. Provide specific procedural guidance for agents
3. Generate escalation recommendations based on risk scores
4. Deliver actionable agent guidance with key questions and customer education

When you receive fraud analysis results, you should:
- Use retrieve_hsbc_policies tool to get relevant procedures
- Provide immediate alerts for high-risk scenarios
- Include specific questions agents should ask customers
- Explain customer education points
- Give clear escalation procedures when needed

Always provide comprehensive, actionable guidance that helps agents respond appropriately to potential fraud.""",
    tools=[retrieve_hsbc_policies]
)
