# backend/api/routes/fraud.py - CONSOLIDATED
"""
Fraud detection API routes - UPDATED to use all consolidated patterns
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from backend.middleware.error_handling import handle_api_errors, validate_input, require_session_id
from backend.services.mock_database import mock_db, session_service
from backend.api.models import (
    FraudAnalysisRequest, FraudAnalysisResult, RiskAssessmentRequest, 
    RiskAssessmentResponse, ScamPattern, FraudAlert, SuccessResponse
)
from backend.utils import (
    get_current_timestamp, generate_session_id, create_success_response,
    create_error_response, get_risk_level_from_score, calculate_percentage
)
from backend.config.settings import get_settings
from agents.scam_detection.agent import fraud_detection_agent
from agents.orchestrator.agent import coordinate_fraud_analysis

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyze", response_model=FraudAnalysisResult)
@handle_api_errors("fraud_analysis")
@validate_input(
    text={"required": True, "type": str, "min_length": 3, "max_length": 10000},
    session_id={"required": False, "type": str}
)
async def analyze_text_for_fraud(
    request: FraudAnalysisRequest,
    settings = Depends(get_settings)
):
    """
    Analyze text for fraud patterns using consolidated detection agent
    """
    
    # Generate session ID using centralized utility
    session_id = request.session_id or generate_session_id("fraud_analysis")
    
    # Skip analysis for agent speech in demo mode
    if request.speaker == "agent" and settings.demo_mode:
        return _create_minimal_response(session_id)
    
    # Use consolidated fraud detection agent directly (no wrapper function)
    analysis_result = fraud_detection_agent.process({
        'text': request.text,
        'session_id': session_id,
        'speaker': request.speaker,
        'context': request.context
    })
    
    # Handle errors from agent processing
    if 'error' in analysis_result:
        raise ValueError(f"Fraud analysis failed: {analysis_result['error']}")
    
    # Create comprehensive response using consolidated models
    response = FraudAnalysisResult(
        session_id=session_id,
        risk_score=analysis_result["risk_score"],
        risk_level=analysis_result["risk_level"],
        scam_type=analysis_result["scam_type"],
        confidence=analysis_result["confidence"],
        detected_patterns=analysis_result["detected_patterns"],
        explanation=analysis_result["explanation"],
        recommended_action=analysis_result["recommended_action"],
        requires_escalation=analysis_result["requires_escalation"],
        analysis_timestamp=analysis_result["analysis_timestamp"]
    )
    
    # Store session data using consolidated service
    session_data = {
        "session_id": session_id,
        "text_analyzed": request.text[:200] + "..." if len(request.text) > 200 else request.text,
        "risk_score": response.risk_score,
        "scam_type": response.scam_type,
        "speaker": request.speaker,
        "status": "analyzed"
    }
    session_service.create_session(session_data)
    
    logger.info(f"🔍 Fraud analysis completed: Risk {response.risk_score}% - {response.scam_type}")
    
    return response

@router.post("/comprehensive-analysis")
@handle_api_errors("comprehensive_analysis")
@require_session_id
async def comprehensive_fraud_analysis(
    transcript_data: Dict,
    settings = Depends(get_settings)
):
    """
    Perform comprehensive fraud analysis using orchestrator coordination
    """
    
    # Use consolidated orchestrator (no wrapper function)
    comprehensive_result = coordinate_fraud_analysis(transcript_data)
    
    # Store comprehensive results using consolidated database
    if comprehensive_result.get('session_id'):
        session_data = {
            "session_id": comprehensive_result['session_id'],
            "comprehensive_analysis": True,
            "fraud_detection": comprehensive_result.get('fraud_detection', {}),
            "policy_guidance": comprehensive_result.get('policy_guidance', {}),
            "orchestrator_decision": comprehensive_result.get('orchestrator_decision', {}),
            "requires_escalation": comprehensive_result.get('requires_escalation', False),
            "status": "comprehensive_complete"
        }
        session_service.update_session(
            comprehensive_result['session_id'], 
            session_data
        )
    
    return create_success_response(
        data=comprehensive_result,
        message="Comprehensive fraud analysis completed using consolidated orchestrator"
    )

@router.post("/risk-assessment", response_model=RiskAssessmentResponse)
@handle_api_errors("risk_assessment")
async def assess_customer_risk(
    request: RiskAssessmentRequest,
    settings = Depends(get_settings)
):
    """
    Perform comprehensive risk assessment using consolidated analysis
    """
    
    # Analyze conversation history using consolidated agent
    overall_risk_scores = []
    risk_factors = []
    
    for conversation in request.conversation_history:
        if conversation.get('speaker') == 'customer' and conversation.get('text'):
            analysis = fraud_detection_agent.process(conversation['text'])
            if 'risk_score' in analysis:
                overall_risk_scores.append(analysis['risk_score'])
                
                if analysis['risk_score'] > 40:  # Using centralized threshold
                    risk_factors.append({
                        "factor": f"{analysis['scam_type']}_patterns",
                        "score": analysis['risk_score'],
                        "weight": 0.4,
                        "detected_patterns": list(analysis.get('detected_patterns', {}).keys())
                    })
    
    # Calculate overall risk using centralized utilities
    overall_risk = calculate_percentage(
        sum(overall_risk_scores), 
        len(overall_risk_scores) * 100
    ) if overall_risk_scores else 0
    
    # Add customer profile risk factors
    if request.customer_profile:
        profile_risk = _assess_customer_profile_risk(request.customer_profile, settings)
        risk_factors.append(profile_risk)
        overall_risk = (overall_risk * 0.7) + (profile_risk['score'] * 0.3)
    
    # Generate recommendations using centralized risk levels
    recommendations = _generate_risk_recommendations(overall_risk, settings)
    
    # Determine monitoring and escalation using centralized thresholds
    monitoring_required = overall_risk >= settings.risk_threshold_medium
    escalation_priority = _determine_escalation_priority(overall_risk, settings)
    
    return RiskAssessmentResponse(
        overall_risk_score=overall_risk,
        risk_factors=risk_factors,
        recommendations=recommendations,
        monitoring_required=monitoring_required,
        escalation_priority=escalation_priority
    )

@router.get("/patterns", response_model=List[ScamPattern])
@handle_api_errors("get_patterns")
async def get_scam_patterns(settings = Depends(get_settings)):
    """
    Get available scam patterns from centralized configuration
    """
    
    patterns = []
    
    # Get patterns from centralized fraud detection agent
    agent_patterns = fraud_detection_agent.scam_patterns
    pattern_weights = settings.pattern_weights
    
    for pattern_id, pattern_data in agent_patterns.items():
        pattern = ScamPattern(
            pattern_id=pattern_id,
            pattern_name=pattern_id.replace('_', ' ').title(),
            pattern_type="regex",
            description=f"Detects {pattern_id.replace('_', ' ')} patterns in customer speech",
            weight=pattern_weights.get(pattern_id, pattern_data.get('weight', 20)),
            severity=pattern_data.get('severity', 'medium'),
            examples=[p[:50] + "..." if len(p) > 50 else p for p in pattern_data.get('patterns', [])[:3]]
        )
        patterns.append(pattern)
    
    return patterns

@router.get("/alerts/{session_id}", response_model=List[FraudAlert])
@handle_api_errors("get_session_alerts")
async def get_session_alerts(session_id: str):
    """
    Get fraud alerts for a specific session using consolidated database
    """
    
    # Get session data using consolidated service
    session_data = session_service.get_session(session_id)
    
    if not session_data:
        return []
    
    alerts = []
    
    # Generate alerts based on session risk score
    risk_score = session_data.get('risk_score', 0)
    scam_type = session_data.get('scam_type', 'unknown')
    
    if risk_score >= 80:
        alerts.append(FraudAlert(
            alert_id=f"alert_{session_id}_critical",
            session_id=session_id,
            alert_type="critical_risk_detected",
            risk_score=risk_score,
            scam_type=scam_type,
            message=f"Critical fraud risk detected: {scam_type.replace('_', ' ').title()}",
            priority="critical",
            timestamp=get_current_timestamp(),
            acknowledged=False
        ))
    elif risk_score >= 60:
        alerts.append(FraudAlert(
            alert_id=f"alert_{session_id}_high",
            session_id=session_id,
            alert_type="high_risk_detected",
            risk_score=risk_score,
            scam_type=scam_type,
            message=f"High fraud risk detected: {scam_type.replace('_', ' ').title()}",
            priority="high",
            timestamp=get_current_timestamp(),
            acknowledged=False
        ))
    
    return alerts

@router.post("/alerts/{alert_id}/acknowledge")
@handle_api_errors("acknowledge_alert")
async def acknowledge_alert(alert_id: str):
    """
    Acknowledge a fraud alert using consolidated database
    """
    
    # In production, this would update the alert in the database
    # For demo, we'll use the consolidated database to track acknowledgments
    alert_data = {
        "alert_id": alert_id,
        "acknowledged": True,
        "acknowledged_at": get_current_timestamp(),
        "acknowledged_by": "system"  # In production, use authenticated user
    }
    
    # Store acknowledgment using consolidated database
    mock_db.create("alert_acknowledgments", alert_id, alert_data)
    
    return create_success_response(
        data={"alert_id": alert_id, "acknowledged": True},
        message="Alert acknowledged using consolidated system"
    )

@router.get("/statistics")
@handle_api_errors("fraud_statistics")
async def get_fraud_statistics(
    time_period: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    session_id: Optional[str] = Query(None)
):
    """
    Get fraud detection statistics using consolidated database
    """
    
    # Get statistics from consolidated database service
    session_stats = mock_db.get_statistics("fraud_sessions")
    
    # Calculate fraud detection metrics
    sessions, total_count = mock_db.list_records("fraud_sessions", page=1, page_size=1000)
    sessions_data = [record.data for record in sessions]
    
    # Filter by session_id if provided
    if session_id:
        sessions_data = [s for s in sessions_data if s.get('session_id') == session_id]
    
    # Calculate statistics using centralized utilities
    total_analyses = len(sessions_data)
    fraud_detected = len([s for s in sessions_data if s.get('risk_score', 0) >= 40])
    high_risk_cases = len([s for s in sessions_data if s.get('risk_score', 0) >= 80])
    
    # Get scam type breakdown
    scam_types = {}
    for session in sessions_data:
        scam_type = session.get('scam_type', 'unknown')
        if scam_type != 'none' and scam_type != 'unknown':
            scam_types[scam_type] = scam_types.get(scam_type, 0) + 1
    
    # Calculate rates using centralized utilities
    fraud_detection_rate = calculate_percentage(fraud_detected, total_analyses)
    high_risk_rate = calculate_percentage(high_risk_cases, total_analyses)
    
    # Get average risk score
    risk_scores = [s.get('risk_score', 0) for s in sessions_data if s.get('risk_score')]
    average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
    
    statistics = {
        "time_period": time_period,
        "total_analyses": total_analyses,
        "fraud_detected": fraud_detected,
        "high_risk_cases": high_risk_cases,
        "fraud_detection_rate": fraud_detection_rate,
        "high_risk_rate": high_risk_rate,
        "average_risk_score": round(average_risk_score, 2),
        "scam_types": scam_types,
        "escalations": high_risk_cases,  # Simplified for demo
        "database_stats": session_stats,
        "consolidation_info": {
            "data_source": "Consolidated MockDatabase service",
            "utilities_used": "Centralized calculation functions",
            "configuration": "Centralized risk thresholds from settings.py"
        }
    }
    
    return create_success_response(
        data=statistics,
        message="Fraud statistics retrieved using consolidated services"
    )

@router.get("/agent-performance")
@handle_api_errors("agent_performance")
async def get_fraud_agent_performance():
    """
    Get fraud detection agent performance metrics
    """
    
    # Get performance from consolidated agent (no wrapper function)
    agent_status = fraud_detection_agent.get_status()
    
    # Get additional statistics using consolidated database
    pattern_stats = {}
    sessions, _ = mock_db.list_records("fraud_sessions", page=1, page_size=100)
    
    for session in sessions:
        session_data = session.data
        if 'detected_patterns' in session_data:
            for pattern_name in session_data['detected_patterns'].keys():
                pattern_stats[pattern_name] = pattern_stats.get(pattern_name, 0) + 1
    
    performance_data = {
        "agent_metrics": agent_status,
        "pattern_usage_statistics": pattern_stats,
        "consolidation_benefits": {
            "direct_agent_access": "No wrapper functions - direct agent.get_status() call",
            "centralized_patterns": "Patterns loaded from centralized configuration",
            "unified_metrics": "Consistent metrics across all agents",
            "shared_utilities": "Common calculation functions"
        }
    }
    
    return create_success_response(
        data=performance_data,
        message="Agent performance retrieved using consolidated system"
    )

# ===== HELPER FUNCTIONS =====

def _create_minimal_response(session_id: str) -> FraudAnalysisResult:
    """Create minimal response for agent speech using consolidated models"""
    return FraudAnalysisResult(
        session_id=session_id,
        risk_score=0.0,
        risk_level="MINIMAL",
        scam_type="none",
        confidence=0.1,
        detected_patterns={},
        explanation="Agent speech - no analysis performed",
        recommended_action="CONTINUE_NORMAL_PROCESSING",
        requires_escalation=False,
        analysis_timestamp=get_current_timestamp()
    )

def _assess_customer_profile_risk(profile: Dict, settings) -> Dict:
    """Assess customer profile risk using centralized logic"""
    profile_risk = 0
    risk_factors = []
    
    # Age-based risk (elderly customers more vulnerable)
    age = profile.get('age', 0)
    if age >= 70:
        profile_risk += 20
        risk_factors.append("elderly_customer")
    elif age <= 25:
        profile_risk += 10
        risk_factors.append("young_customer")
    
    # Recent account activity
    if profile.get('recent_large_transactions', False):
        profile_risk += 15
        risk_factors.append("recent_large_transactions")
    
    # Account tenure
    tenure_years = profile.get('account_tenure_years', 0)
    if tenure_years < 1:
        profile_risk += 10
        risk_factors.append("new_customer")
    
    return {
        "factor": "customer_profile",
        "score": min(profile_risk, 100),
        "weight": 0.3,
        "risk_factors": risk_factors
    }

def _generate_risk_recommendations(risk_score: float, settings) -> List[str]:
    """Generate risk recommendations using centralized thresholds"""
    recommendations = []
    
    if risk_score >= settings.risk_threshold_critical:
        recommendations.extend([
            "Immediate escalation to Financial Crime Team required",
            "Block all pending transactions",
            "Implement enhanced customer verification",
            "Document all evidence for investigation"
        ])
    elif risk_score >= settings.risk_threshold_high:
        recommendations.extend([
            "Enhanced monitoring required for 48 hours",
            "Additional customer verification recommended",
            "Flag account for supervisor review",
            "Consider customer education on fraud risks"
        ])
    elif risk_score >= settings.risk_threshold_medium:
        recommendations.extend([
            "Monitor customer interactions for patterns",
            "Provide fraud awareness education",
            "Document conversation details",
            "Consider follow-up contact within 24 hours"
        ])
    else:
        recommendations.append("Continue normal customer service procedures")
    
    return recommendations

def _determine_escalation_priority(risk_score: float, settings) -> str:
    """Determine escalation priority using centralized thresholds"""
    if risk_score >= settings.risk_threshold_critical:
        return "critical"
    elif risk_score >= settings.risk_threshold_high:
        return "high"
    elif risk_score >= settings.risk_threshold_medium:
        return "medium"
    else:
        return "low"
