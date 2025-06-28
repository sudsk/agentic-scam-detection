# backend/api/routes/fraud.py - ALL IMPORTS FIXED
"""
Fraud detection API routes - CLEAN VERSION with FIXED IMPORTS
Only uses models that are actually needed and exist
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query

# FIXED: Correct relative import paths from api/routes/ to backend/
from ...middleware.error_handling import handle_api_errors, validate_input, require_session_id
from ...services.mock_database import mock_db, session_service
from ..models import (
    FraudAnalysisRequest, 
    FraudAnalysisResult, 
    SuccessResponse
)
from ...utils import (
    get_current_timestamp, generate_session_id, create_success_response,
    create_error_response, calculate_percentage
)
from ...config.settings import get_settings
from ...agents.scam_detection.agent import fraud_detection_agent

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
    """Analyze text for fraud patterns using consolidated detection agent"""
    
    # Generate session ID using centralized utility
    session_id = request.session_id or generate_session_id("fraud_analysis")
    
    # Skip analysis for agent speech in demo mode
    if request.speaker == "agent" and settings.demo_mode:
        return _create_minimal_response(session_id)
    
    # Use consolidated fraud detection agent directly
    analysis_result = fraud_detection_agent.process({
        'text': request.text,
        'session_id': session_id,
        'speaker': request.speaker,
        'context': request.context
    })
    
    # Handle errors from agent processing
    if 'error' in analysis_result:
        raise ValueError(f"Fraud analysis failed: {analysis_result['error']}")
    
    # Create response using existing models
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

@router.get("/patterns")
@handle_api_errors("get_patterns")
async def get_scam_patterns(settings = Depends(get_settings)):
    """Get available scam patterns from centralized configuration"""
    
    patterns = []
    
    # Get patterns from centralized fraud detection agent
    agent_patterns = fraud_detection_agent.scam_patterns
    pattern_weights = settings.pattern_weights
    
    for pattern_id, pattern_data in agent_patterns.items():
        pattern = {
            "pattern_id": pattern_id,
            "pattern_name": pattern_id.replace('_', ' ').title(),
            "description": f"Detects {pattern_id.replace('_', ' ')} patterns in customer speech",
            "weight": pattern_weights.get(pattern_id, pattern_data.get('weight', 20)),
            "severity": pattern_data.get('severity', 'medium'),
            "examples": [p[:50] + "..." if len(p) > 50 else p for p in pattern_data.get('patterns', [])[:3]]
        }
        patterns.append(pattern)
    
    return create_success_response(
        data=patterns,
        message="Scam patterns retrieved from centralized configuration"
    )

@router.get("/statistics")
@handle_api_errors("fraud_statistics")
async def get_fraud_statistics(
    time_period: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    session_id: Optional[str] = Query(None)
):
    """Get fraud detection statistics using consolidated database"""
    
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

# ===== HELPER FUNCTIONS =====

def _create_minimal_response(session_id: str) -> FraudAnalysisResult:
    """Create minimal response for agent speech using existing models"""
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
