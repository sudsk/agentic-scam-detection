# backend/api/models/__init__.py
"""
Shared API models for HSBC Fraud Detection System
Consolidates common Pydantic models to eliminate redundancy
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator

# ===== BASE MODELS =====

class BaseResponse(BaseModel):
    """Base response model with common fields"""
    status: str = Field(..., description="Response status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    message: Optional[str] = Field(None, description="Optional message")

class SuccessResponse(BaseResponse):
    """Standard success response"""
    status: str = Field(default="success")
    data: Optional[Any] = Field(None, description="Response data")

class ErrorResponse(BaseResponse):
    """Standard error response"""
    status: str = Field(default="error")
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict] = Field(default_factory=dict)

class ProcessingResponse(BaseResponse):
    """Processing status response"""
    status: str = Field(default="processing")
    session_id: str = Field(..., description="Session identifier")
    progress_percentage: Optional[int] = Field(None, ge=0, le=100)
    estimated_completion: Optional[str] = Field(None)

# ===== ENUMS =====

class RiskLevel(str, Enum):
    """Risk level enumeration"""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class Priority(str, Enum):
    """Priority level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CaseStatus(str, Enum):
    """Case status enumeration"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    PENDING = "pending"
    RESOLVED = "resolved"
    CLOSED = "closed"

class ScamType(str, Enum):
    """Scam type enumeration"""
    INVESTMENT_SCAM = "investment_scam"
    ROMANCE_SCAM = "romance_scam"
    IMPERSONATION_SCAM = "impersonation_scam"
    AUTHORITY_SCAM = "authority_scam"
    APP_FRAUD = "app_fraud"
    CRYPTO_SCAM = "crypto_scam"
    UNKNOWN = "unknown"

class ProcessingStage(str, Enum):
    """Processing stage enumeration"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"

# ===== COMMON REQUEST MODELS =====

class SessionRequest(BaseModel):
    """Base request with session information"""
    session_id: Optional[str] = Field(None, description="Session identifier")
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    agent_id: Optional[str] = Field(None, description="Agent identifier")

class PaginationRequest(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")

class FilterRequest(BaseModel):
    """Common filtering parameters"""
    status: Optional[str] = Field(None, description="Filter by status")
    priority: Optional[Priority] = Field(None, description="Filter by priority")
    date_from: Optional[str] = Field(None, description="Filter from date (ISO format)")
    date_to: Optional[str] = Field(None, description="Filter to date (ISO format)")

# ===== AUDIO MODELS =====

class AudioMetadata(BaseModel):
    """Audio file metadata"""
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    duration_seconds: Optional[float] = Field(None, ge=0, description="Duration in seconds")
    format: str = Field(..., description="Audio format")
    sample_rate: Optional[int] = Field(None, ge=0, description="Sample rate in Hz")
    channels: Optional[int] = Field(None, ge=1, description="Number of audio channels")
    upload_timestamp: str = Field(..., description="Upload timestamp")

class AudioUploadRequest(SessionRequest):
    """Audio upload request"""
    pass  # File will be handled by FastAPI File parameter

class AudioUploadResponse(SuccessResponse):
    """Audio upload response"""
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Audio format")

# ===== TRANSCRIPTION MODELS =====

class SpeakerSegment(BaseModel):
    """Speaker segment in transcription"""
    speaker: str = Field(..., description="Speaker identifier")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")

class TranscriptionResult(BaseModel):
    """Transcription results"""
    file_id: str = Field(..., description="File identifier")
    transcript_text: str = Field(..., description="Full transcript")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence")
    speaker_segments: List[SpeakerSegment] = Field(default_factory=list)
    total_duration: float = Field(..., ge=0, description="Total duration")
    speakers_detected: List[str] = Field(default_factory=list)
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")

# ===== FRAUD DETECTION MODELS =====

class DetectedPattern(BaseModel):
    """Detected fraud pattern"""
    pattern_name: str = Field(..., description="Pattern identifier")
    matches: List[str] = Field(default_factory=list, description="Matched text segments")
    count: int = Field(..., ge=0, description="Number of matches")
    weight: float = Field(..., ge=0, description="Pattern weight")
    severity: str = Field(..., description="Pattern severity level")

class FraudAnalysisRequest(BaseModel):
    """Fraud analysis request"""
    text: str = Field(..., min_length=3, max_length=10000, description="Text to analyze")
    session_id: Optional[str] = Field(None, description="Session identifier")
    speaker: str = Field(default="customer", description="Speaker type")
    context: Optional[Dict] = Field(default_factory=dict, description="Additional context")

    @validator('text')
    def validate_text_content(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Text must be at least 3 characters long")
        return v.strip()

class FraudAnalysisResult(BaseModel):
    """Fraud analysis results"""
    session_id: str = Field(..., description="Session identifier")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    risk_level: RiskLevel = Field(..., description="Risk level")
    scam_type: ScamType = Field(..., description="Detected scam type")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    detected_patterns: Dict[str, DetectedPattern] = Field(default_factory=dict)
    explanation: str = Field(..., description="Analysis explanation")
    recommended_action: str = Field(..., description="Recommended action")
    requires_escalation: bool = Field(..., description="Whether escalation is required")
    analysis_timestamp: str = Field(..., description="Analysis timestamp")

# ===== POLICY GUIDANCE MODELS =====

class PolicyGuidance(BaseModel):
    """Policy guidance information"""
    policy_id: str = Field(..., description="Policy identifier")
    policy_title: str = Field(..., description="Policy title")
    immediate_alerts: List[str] = Field(default_factory=list, description="Immediate alerts")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    key_questions: List[str] = Field(default_factory=list, description="Key questions to ask")
    customer_education: List[str] = Field(default_factory=list, description="Customer education points")
    escalation_threshold: float = Field(..., ge=0, le=100, description="Escalation threshold")

class PolicyRequest(BaseModel):
    """Policy guidance request"""
    scam_type: ScamType = Field(..., description="Scam type")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score")

class PolicyResponse(SuccessResponse):
    """Policy guidance response"""
    guidance: PolicyGuidance = Field(..., description="Policy guidance")
    escalation_required: bool = Field(..., description="Whether escalation is required")

# ===== CASE MANAGEMENT MODELS =====

class CaseEvidence(BaseModel):
    """Case evidence"""
    transcript: str = Field(default="", description="Call transcript")
    detected_patterns: Dict = Field(default_factory=dict, description="Detected fraud patterns")
    risk_factors: str = Field(default="", description="Risk factors")
    confidence_score: float = Field(default=0.0, ge=0, le=1, description="Confidence score")

class CaseTimeline(BaseModel):
    """Case timeline entry"""
    timestamp: str = Field(..., description="Event timestamp")
    action: str = Field(..., description="Action taken")
    actor: str = Field(..., description="Who performed the action")
    details: str = Field(default="", description="Additional details")

class CaseCreateRequest(SessionRequest):
    """Case creation request"""
    fraud_type: ScamType = Field(..., description="Type of fraud detected")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score")
    description: str = Field(..., description="Case description")
    evidence: Optional[Dict] = Field(default_factory=dict, description="Evidence data")
    priority: Priority = Field(default=Priority.MEDIUM, description="Case priority")
    assigned_to: Optional[str] = Field(None, description="Assigned team/person")

class CaseUpdateRequest(BaseModel):
    """Case update request"""
    status: Optional[CaseStatus] = Field(None, description="New status")
    assigned_to: Optional[str] = Field(None, description="Reassign case")
    notes: Optional[str] = Field(None, description="Additional notes")
    resolution: Optional[str] = Field(None, description="Resolution details")
    tags: Optional[List[str]] = Field(None, description="Case tags")

class CaseInfo(BaseModel):
    """Case information"""
    case_id: str = Field(..., description="Unique case identifier")
    session_id: str = Field(..., description="Associated session")
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    fraud_type: ScamType = Field(..., description="Type of fraud")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score")
    description: str = Field(..., description="Case description")
    status: CaseStatus = Field(..., description="Current status")
    priority: Priority = Field(..., description="Case priority")
    assigned_to: Optional[str] = Field(None, description="Assigned to")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    resolved_at: Optional[str] = Field(None, description="Resolution timestamp")
    evidence: CaseEvidence = Field(default_factory=CaseEvidence, description="Case evidence")
    timeline: List[CaseTimeline] = Field(default_factory=list, description="Case timeline")
    tags: List[str] = Field(default_factory=list, description="Case tags")

# ===== ORCHESTRATOR MODELS =====

class OrchestratorDecision(BaseModel):
    """Orchestrator decision"""
    decision: str = Field(..., description="Final decision")
    reasoning: str = Field(..., description="Decision reasoning")
    actions: List[str] = Field(default_factory=list, description="Required actions")
    priority: Priority = Field(..., description="Priority level")
    confidence: float = Field(..., ge=0, le=1, description="Decision confidence")
    case_created: bool = Field(default=False, description="Whether case was created")
    escalation_required: bool = Field(default=False, description="Whether escalation is required")

class ComprehensiveAnalysis(BaseModel):
    """Comprehensive fraud analysis result"""
    session_id: str = Field(..., description="Session identifier")
    transcription: Optional[TranscriptionResult] = Field(None, description="Transcription results")
    fraud_analysis: FraudAnalysisResult = Field(..., description="Fraud analysis results")
    policy_guidance: Optional[PolicyGuidance] = Field(None, description="Policy guidance")
    case_info: Optional[CaseInfo] = Field(None, description="Case information if created")
    orchestrator_decision: OrchestratorDecision = Field(..., description="Final decision")
    agents_involved: List[str] = Field(default_factory=list, description="Agents that processed this")
    processing_time_ms: int = Field(..., ge=0, description="Total processing time")
    pipeline_complete: bool = Field(default=True, description="Whether pipeline completed successfully")

# ===== STATISTICS MODELS =====

class AgentMetrics(BaseModel):
    """Agent performance metrics"""
    total_requests: int = Field(default=0, ge=0)
    successful_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    average_processing_time: float = Field(default=0.0, ge=0)
    last_processing_time: float = Field(default=0.0, ge=0)

class SystemStatistics(BaseModel):
    """System-wide statistics"""
    total_sessions: int = Field(default=0, ge=0, description="Total sessions processed")
    active_cases: int = Field(default=0, ge=0, description="Currently active cases")
    average_risk_score: float = Field(default=0.0, ge=0, le=100, description="Average risk score")
    fraud_detection_rate: float = Field(default=0.0, ge=0, le=100, description="Fraud detection rate")
    false_positive_rate: float = Field(default=0.0, ge=0, le=100, description="False positive rate")
    agent_performance: Dict[str, AgentMetrics] = Field(default_factory=dict, description="Per-agent metrics")

# ===== HEALTH CHECK MODELS =====

class AgentHealth(BaseModel):
    """Individual agent health"""
    agent_id: str = Field(..., description="Agent identifier")
    agent_name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Agent status")
    healthy: bool = Field(..., description="Whether agent is healthy")
    last_activity: str = Field(..., description="Last activity timestamp")
    error_rate: float = Field(..., ge=0, le=1, description="Error rate")

class SystemHealth(BaseModel):
    """Overall system health"""
    system_status: str = Field(..., description="Overall system status")
    healthy: bool = Field(..., description="Whether system is healthy")
    agents_count: int = Field(..., ge=0, description="Number of agents")
    healthy_agents: int = Field(..., ge=0, description="Number of healthy agents")
    agents: Dict[str, str] = Field(default_factory=dict, description="Agent statuses")
    timestamp: str = Field(..., description="Health check timestamp")

# ===== EXPORT ALL MODELS =====

__all__ = [
    # Base models
    'BaseResponse', 'SuccessResponse', 'ErrorResponse', 'ProcessingResponse',
    
    # Enums
    'RiskLevel', 'Priority', 'CaseStatus', 'ScamType', 'ProcessingStage',
    
    # Common requests
    'SessionRequest', 'PaginationRequest', 'FilterRequest',
    
    # Audio models
    'AudioMetadata', 'AudioUploadRequest', 'AudioUploadResponse',
    
    # Transcription models
    'SpeakerSegment', 'TranscriptionResult',
    
    # Fraud detection models
    'DetectedPattern', 'FraudAnalysisRequest', 'FraudAnalysisResult',
    
    # Policy models
    'PolicyGuidance', 'PolicyRequest', 'PolicyResponse',
    
    # Case management models
    'CaseEvidence', 'CaseTimeline', 'CaseCreateRequest', 'CaseUpdateRequest', 'CaseInfo',
    
    # Orchestrator models
    'OrchestratorDecision', 'ComprehensiveAnalysis',
    
    # Statistics models
    'AgentMetrics', 'SystemStatistics',
    
    # Health models
    'AgentHealth', 'SystemHealth'
]
