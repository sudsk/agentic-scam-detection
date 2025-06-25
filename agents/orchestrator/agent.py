def coordinate_fraud_analysis(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinate the complete fraud detection workflow
    
    Args:
        transcript_data: Customer speech data with text, speaker, session_id
        
    Returns:
        Dict containing comprehensive fraud analysis and recommendations
    """
    session_id = transcript_data.get('session_id')
    text = transcript_data.get('text', '')
    speaker = transcript_data.get('speaker', 'unknown')
    
    logger.info(f"🎭 Coordinating fraud analysis for session {session_id}")
    
    # Skip analysis for agent speech
    if speaker != 'customer':
        return {
            'session_id': session_id,
            'analysis_skipped': True,
            'reason': 'Agent speech - no analysis needed',
            'timestamp': datetime.now().isoformat()
        }
    
    # Step 1: Fraud Detection Analysis
    fraud_analysis = analyze_fraud_patterns(text)
    
    # Step 2: Policy Retrieval (if risk detected)
    policy_guidance = {}
    if fraud_analysis['risk_score'] >= 40:  # Only retrieve policies for medium+ risk
        policy_guidance = retrieve_hsbc_policies(
            fraud_analysis['scam_type'], 
            fraud_analysis['risk_score']
        )
    
    # Step 3: Make Final Decision
    orchestrator_decision = {}
    risk_score = fraud_analysis['risk_score']
    
    if risk_score >= 80:
        orchestrator_decision = {
            'decision': 'BLOCK_AND_ESCALATE',
            'reasoning': 'Critical fraud risk detected - immediate intervention required',
            'actions': [
                'Block all pending transactions',
                'Escalate to Financial Crime Team',
                'Initiate fraud investigation',
                'Document all evidence'
            ],
            'priority': 'CRITICAL'
        }
    elif risk_score >= 60:
        orchestrator_decision = {
            'decision': 'VERIFY_AND_MONITOR',
            'reasoning': 'High fraud risk - enhanced verification required',
            'actions': [
                'Require additional customer verification',
                'Monitor account for 48 hours',
                'Flag for supervisor review',
                'Provide fraud education to customer'
            ],
            'priority': 'HIGH'
        }
    elif risk_score >= 40:
        orchestrator_decision = {
            'decision': 'MONITOR_AND_EDUCATE',
            'reasoning': 'Moderate risk - increased vigilance recommended',
            'actions': [
                'Continue normal processing with caution',
                'Provide relevant fraud warnings',
                'Document conversation details',
                'Consider follow-up contact'
            ],
            'priority': 'MEDIUM'
        }
    else:
        orchestrator_decision = {
            'decision': 'CONTINUE_NORMAL',
            'reasoning': 'Low fraud risk - standard processing',
            'actions': [
                'Continue normal customer service',
                
