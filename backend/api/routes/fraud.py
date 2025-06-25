"""
Fraud detection API routes for HSBC Scam Detection Agent
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, validator

from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()

class FraudAnalysisRequest(BaseModel):
    """Request model for fraud analysis"""
    text: str
    session_id: Optional[str] = None
    speaker: str = "customer"  # customer or agent
    timestamp: Optional[str] = None
    context: Optional[Dict] = None

    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Text must be at least 3 characters long")
        if len(v) > 10000:
            raise ValueError("Text too long (max 10000 characters)")
        return v.strip()

class FraudAnalysisResponse(BaseModel):
    """Response model for fraud analysis"""
    session_id: str
    analysis_id: str
    risk_score: float
    risk_level: str
    scam_type: str
    confidence: float
    detected_patterns: Dict
    explanation: str
    recommended_action: str
    requires_escalation: bool
    processing_time_ms: int
    timestamp: str

class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment"""
    customer_id: Optional[str] = None
    conversation_history: List[Dict] = []
    customer_profile: Optional[Dict] = None
    transaction_context: Optional[Dict] = None

class RiskAssessmentResponse(BaseModel):
    """Response for risk assessment"""
    overall_risk_score: float
    risk_factors: List[Dict]
    recommendations: List[str]
    monitoring_required: bool
    escalation_priority: str

class ScamPattern(BaseModel):
    """Model for scam patterns"""
    pattern_id: str
    pattern_name: str
    pattern_type: str
    description: str
    weight: float
    severity: str
    examples: List[str]

class FraudAlert(BaseModel):
    """Model for fraud alerts"""
    alert_id: str
    session_id: str
    alert_type: str
    risk_score: float
    scam_type: str
    message: str
    priority: str
    timestamp: str
    acknowledged: bool = False

@router.post("/analyze", response_model=FraudAnalysisResponse)
async def analyze_text_for_fraud(
    request: FraudAnalysisRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Analyze text for fraud patterns and generate risk assessment
    """
    try:
        start_time = datetime.now()
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{int(start_time.timestamp())}"
        
        # Skip analysis for agent speech in demo mode
        if request.speaker == "agent" and settings.demo_mode:
            return _create_minimal_response(session_id, start_time)
        
        # Perform fraud analysis using mock detection for demo
        analysis_result = await _perform_fraud_analysis(request.text, settings)
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Create response
        response = FraudAnalysisResponse(
            session_id=session_id,
            analysis_id=f"analysis_{int(start_time.timestamp())}",
            risk_score=analysis_result["risk_score"],
            risk_level=analysis_result["risk_level"],
            scam_type=analysis_result["scam_type"],
            confidence=analysis_result["confidence"],
            detected_patterns=analysis_result["detected_patterns"],
            explanation=analysis_result["explanation"],
            recommended_action=analysis_result["recommended_action"],
            requires_escalation=analysis_result["requires_escalation"],
            processing_time_ms=processing_time,
            timestamp=start_time.isoformat()
        )
        
        # Log analysis
        logger.info(f"🔍 Fraud analysis completed: Risk {response.risk_score}% - {response.scam_type}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in fraud analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fraud analysis failed: {str(e)}"
        )

@router.post("/risk-assessment", response_model=RiskAssessmentResponse)
async def assess_customer_risk(
    request: RiskAssessmentRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Perform comprehensive risk assessment for a customer
    """
    try:
        # Perform risk assessment
        assessment = await _perform_risk_assessment(request, settings)
        
        return RiskAssessmentResponse(**assessment)
        
    except Exception as e:
        logger.error(f"Error in risk assessment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk assessment failed: {str(e)}"
        )

@router.get("/patterns", response_model=List[ScamPattern])
async def get_scam_patterns():
    """
    Get available scam patterns for reference
    """
    try:
        patterns = await _get_scam_patterns()
        return patterns
        
    except Exception as e:
        logger.error(f"Error getting scam patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve scam patterns"
        )

@router.get("/alerts/{session_id}", response_model=List[FraudAlert])
async def get_session_alerts(session_id: str):
    """
    Get fraud alerts for a specific session
    """
    try:
        alerts = await _get_session_alerts(session_id)
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting alerts for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts"
        )

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """
    Acknowledge a fraud alert
    """
    try:
        result = await _acknowledge_alert(alert_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        return {"alert_id": alert_id, "status": "acknowledged", "timestamp": datetime.now().isoformat()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert"
        )

@router.get("/statistics")
async def get_fraud_statistics(
    time_period: str = "24h",  # 1h, 24h, 7d, 30d
    session_id: Optional[str] = None
):
    """
    Get fraud detection statistics
    """
    try:
        stats = await _get_fraud_statistics(time_period, session_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting fraud statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )

# Helper functions

async def _perform_fraud_analysis(text: str, settings: Settings) -> Dict:
    """Perform fraud analysis on text (mock implementation for demo)"""
    
    # Simple keyword-based detection for demo
    text_lower = text.lower()
    risk_score = 0
    detected_patterns = {}
    scam_type = "unknown"
    
    # Investment scam patterns
    investment_keywords = [
        "guaranteed", "profit", "return", "investment", "trading", "forex",
        "margin call", "broker", "advisor", "platform"
    ]
    
    # Romance scam patterns
    romance_keywords = [
        "online", "dating", "relationship", "boyfriend", "girlfriend",
        "overseas", "military", "emergency", "stuck", "stranded"
    ]
    
    # Impersonation patterns
    impersonation_keywords = [
        "hsbc", "bank", "security", "fraud", "police", "hmrc",
        "government", "account compromised", "verify", "pin", "password"
    ]
    
    # Urgency patterns
    urgency_keywords = [
        "urgent", "immediately", "right now", "today only", "deadline",
        "expires", "limited time", "act fast", "hurry"
    ]
    
    # Check for patterns
    investment_matches = sum(1 for keyword in investment_keywords if keyword in text_lower)
    romance_matches = sum(1 for keyword in romance_keywords if keyword in text_lower)
    impersonation_matches = sum(1 for keyword in impersonation_keywords if keyword in text_lower)
    urgency_matches = sum(1 for keyword in urgency_keywords if keyword in text_lower)
    
    # Calculate risk score
    if investment_matches >= 3:
        risk_score += 40 + (investment_matches * 10)
        scam_type = "investment_scam"
        detected_patterns["investment_fraud"] = {
            "matches": investment_matches,
            "weight": 40,
            "severity": "high"
        }
    
    if romance_matches >= 2:
        risk_score += 35 + (romance_matches * 8)
        scam_type = "romance_scam"
        detected_patterns["romance_exploitation"] = {
            "matches": romance_matches,
            "weight": 35,
            "severity": "high"
        }
    
    if impersonation_matches >= 2:
        risk_score += 45 + (impersonation_matches * 12)
        scam_type = "impersonation_scam"
        detected_patterns["authority_impersonation"] = {
            "matches": impersonation_matches,
            "weight": 45,
            "severity": "critical"
        }
    
    if urgency_matches >= 1:
        risk_score += 20 + (urgency_matches * 5)
        detected_patterns["urgency_pressure"] = {
            "matches": urgency_matches,
            "weight": 20,
            "severity": "medium"
        }
    
    # Cap at 100
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
    
    return {
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "scam_type": scam_type,
        "confidence": min(len(detected_patterns) * 0.2 + 0.3, 0.95),
        "detected_patterns": detected_patterns,
        "explanation": explanation,
        "recommended_action": recommended_action,
        "requires_escalation": risk_score >= 80
    }

def _create_minimal_response(session_id: str, start_time: datetime) -> FraudAnalysisResponse:
    """Create minimal response for agent speech"""
    return FraudAnalysisResponse(
        session_id=session_id,
        analysis_id=f"analysis_{int(start_time.timestamp())}",
        risk_score=0.0,
        risk_level="MINIMAL",
        scam_type="none",
        confidence=0.1,
        detected_patterns={},
        explanation="Agent speech - no analysis performed",
        recommended_action="CONTINUE_NORMAL_PROCESSING",
        requires_escalation=False,
        processing_time_ms=1,
        timestamp=start_time.isoformat()
    )

async def _perform_risk_assessment(request: RiskAssessmentRequest, settings: Settings) -> Dict:
    """Perform comprehensive risk assessment (mock implementation)"""
    return {
        "overall_risk_score": 65.0,
        "risk_factors": [
            {"factor": "conversation_patterns", "score": 70, "weight": 0.4},
            {"factor": "customer_history", "score": 45, "weight": 0.3},
            {"factor": "transaction_context", "score": 80, "weight": 0.3}
        ],
        "recommendations": [
            "Enhanced customer verification required",
            "Monitor for additional fraud indicators",
            "Consider escalation if risk increases"
        ],
        "monitoring_required": True,
        "escalation_priority": "medium"
    }

async def _get_scam_patterns() -> List[ScamPattern]:
    """Get available scam patterns (mock implementation)"""
    return [
        ScamPattern(
            pattern_id="investment_fraud",
            pattern_name="Investment Fraud",
            pattern_type="keyword",
            description="Patterns indicating investment scams",
            weight=40.0,
            severity="high",
            examples=["guaranteed returns", "risk-free profit", "margin call"]
        ),
        ScamPattern(
            pattern_id="romance_scam",
            pattern_name="Romance Scam",
            pattern_type="keyword",
            description="Patterns indicating romance scams",
            weight=35.0,
            severity="high",
            examples=["met online", "overseas emergency", "military deployment"]
        )
    ]

async def _get_session_alerts(session_id: str) -> List[FraudAlert]:
    """Get alerts for a session (mock implementation)"""
    return [
        FraudAlert(
            alert_id=f"alert_{session_id}_1",
            session_id=session_id,
            alert_type="high_risk_detected",
            risk_score=85.0,
            scam_type="investment_scam",
            message="High-risk investment scam patterns detected",
            priority="high",
            timestamp=datetime.now().isoformat(),
            acknowledged=False
        )
    ]

async def _acknowledge_alert(alert_id: str) -> bool:
    """Acknowledge an alert (mock implementation)"""
    logger.info(f"Alert {alert_id} acknowledged")
    return True

async def _get_fraud_statistics(time_period: str, session_id: Optional[str]) -> Dict:
    """Get fraud statistics (mock implementation)"""
    return {
        "time_period": time_period,
        "total_analyses": 1547,
        "fraud_detected": 23,
        "false_positives": 3,
        "accuracy_rate": 0.94,
        "average_risk_score": 15.2,
        "scam_types": {
            "investment_scam": 12,
            "romance_scam": 6,
            "impersonation_scam": 5
        },
        "escalations": 18,
        "cases_created": 15
    }
