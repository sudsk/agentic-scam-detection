"""
Case management API routes for HSBC Scam Detection Agent
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, validator

from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()

class CaseCreateRequest(BaseModel):
    """Request model for creating a new case"""
    session_id: str
    customer_id: Optional[str] = None
    fraud_type: str
    risk_score: float
    description: str
    evidence: List[Dict] = []
    priority: str = "medium"  # low, medium, high, critical
    assigned_to: Optional[str] = None

    @validator('risk_score')
    def validate_risk_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Risk score must be between 0 and 100")
        return v

    @validator('priority')
    def validate_priority(cls, v):
        if v not in ["low", "medium", "high", "critical"]:
            raise ValueError("Priority must be one of: low, medium, high, critical")
        return v

class CaseUpdateRequest(BaseModel):
    """Request model for updating a case"""
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None
    resolution: Optional[str] = None
    tags: Optional[List[str]] = None

class CaseResponse(BaseModel):
    """Response model for case data"""
    case_id: str
    session_id: str
    customer_id: Optional[str]
    fraud_type: str
    risk_score: float
    description: str
    status: str
    priority: str
    assigned_to: Optional[str]
    created_at: str
    updated_at: str
    resolved_at: Optional[str]
    evidence: List[Dict]
    notes: List[Dict]
    tags: List[str]
    escalation_history: List[Dict]

class CaseListResponse(BaseModel):
    """Response model for case listing"""
    cases: List[CaseResponse]
    total_count: int
    page: int
    page_size: int
    has_next: bool

class CaseStatistics(BaseModel):
    """Model for case statistics"""
    total_cases: int
    open_cases: int
    resolved_cases: int
    high_priority_cases: int
    average_resolution_time_hours: float
    cases_by_fraud_type: Dict[str, int]
    cases_by_status: Dict[str, int]
    resolution_rate_percentage: float

@router.post("/create", response_model=CaseResponse)
async def create_case(
    request: CaseCreateRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Create a new fraud investigation case
    """
    try:
        # Generate case ID
        case_id = f"HSBC-FD-{datetime.now().strftime('%Y%m%d')}-{str(uuid4())[:8].upper()}"
        
        # Create case data
        case_data = {
            "case_id": case_id,
            "session_id": request.session_id,
            "customer_id": request.customer_id,
            "fraud_type": request.fraud_type,
            "risk_score": request.risk_score,
            "description": request.description,
            "status": "open",
            "priority": request.priority,
            "assigned_to": request.assigned_to,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "resolved_at": None,
            "evidence": request.evidence,
            "notes": [],
            "tags": [],
            "escalation_history": []
        }
        
        # Store case (in production, save to Firestore/Quantexa)
        await _store_case(case_data)
        
        # Log case creation
        logger.info(f"📋 Case created: {case_id} - {request.fraud_type} (Risk: {request.risk_score}%)")
        
        return CaseResponse(**case_data)
        
    except Exception as e:
        logger.error(f"Error creating case: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create case: {str(e)}"
        )

@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case_id: str):
    """
    Get a specific case by ID
    """
    try:
        case_data = await _get_case_by_id(case_id)
        
        if not case_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found"
            )
        
        return CaseResponse(**case_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving case {case_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve case"
        )

@router.put("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str,
    request: CaseUpdateRequest
):
    """
    Update an existing case
    """
    try:
        # Get existing case
        case_data = await _get_case_by_id(case_id)
        
        if not case_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found"
            )
        
        # Update fields
        updates = {}
        if request.status is not None:
            updates["status"] = request.status
            if request.status == "resolved":
                updates["resolved_at"] = datetime.now().isoformat()
        
        if request.assigned_to is not None:
            updates["assigned_to"] = request.assigned_to
        
        if request.notes is not None:
            case_data["notes"].append({
                "note": request.notes,
                "added_by": "system",  # In production, use authenticated user
                "timestamp": datetime.now().isoformat()
            })
        
        if request.resolution is not None:
            updates["resolution"] = request.resolution
        
        if request.tags is not None:
            updates["tags"] = request.tags
        
        # Apply updates
        case_data.update(updates)
        case_data["updated_at"] = datetime.now().isoformat()
        
        # Store updated case
        await _update_case(case_id, case_data)
        
        logger.info(f"📝 Case updated: {case_id}")
        
        return CaseResponse(**case_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating case {case_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update case"
        )

@router.get("/", response_model=CaseListResponse)
async def list_cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    fraud_type: Optional[str] = Query(None),
    assigned_to: Optional[str] = Query(None),
    customer_id: Optional[str] = Query(None)
):
    """
    List cases with optional filtering and pagination
    """
    try:
        # Build filters
        filters = {}
        if status:
            filters["status"] = status
        if priority:
            filters["priority"] = priority
        if fraud_type:
            filters["fraud_type"] = fraud_type
        if assigned_to:
            filters["assigned_to"] = assigned_to
        if customer_id:
            filters["customer_id"] = customer_id
        
        # Get cases
        cases_data, total_count = await _list_cases(page, page_size, filters)
        
        # Convert to response models
        cases = [CaseResponse(**case) for case in cases_data]
        
        return CaseListResponse(
            cases=cases,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=page * page_size < total_count
        )
        
    except Exception as e:
        logger.error(f"Error listing cases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list cases"
        )

@router.post("/{case_id}/escalate")
async def escalate_case(
    case_id: str,
    escalation_reason: str,
    escalated_to: str
):
    """
    Escalate a case to higher priority or different team
    """
    try:
        case_data = await _get_case_by_id(case_id)
        
        if not case_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found"
            )
        
        # Add escalation to history
        escalation = {
            "escalated_by": "system",  # In production, use authenticated user
            "escalated_to": escalated_to,
            "reason": escalation_reason,
            "timestamp": datetime.now().isoformat(),
            "previous_priority": case_data["priority"]
        }
        
        case_data["escalation_history"].append(escalation)
        case_data["priority"] = "high" if case_data["priority"] != "critical" else "critical"
        case_data["assigned_to"] = escalated_to
        case_data["updated_at"] = datetime.now().isoformat()
        
        await _update_case(case_id, case_data)
        
        logger.info(f"🚨 Case escalated: {case_id} to {escalated_to}")
        
        return {
            "case_id": case_id,
            "status": "escalated",
            "escalated_to": escalated_to,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error escalating case {case_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to escalate case"
        )

@router.post("/{case_id}/notes")
async def add_case_note(
    case_id: str,
    note: str,
    note_type: str = "general"
):
    """
    Add a note to a case
    """
    try:
        case_data = await _get_case_by_id(case_id)
        
        if not case_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found"
            )
        
        new_note = {
            "note": note,
            "note_type": note_type,
            "added_by": "system",  # In production, use authenticated user
            "timestamp": datetime.now().isoformat()
        }
        
        case_data["notes"].append(new_note)
        case_data["updated_at"] = datetime.now().isoformat()
        
        await _update_case(case_id, case_data)
        
        return {
            "case_id": case_id,
            "note_added": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding note to case {case_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add note"
        )

@router.get("/statistics", response_model=CaseStatistics)
async def get_case_statistics(
    time_period: str = Query("30d", regex="^(1d|7d|30d|90d)$")
):
    """
    Get case management statistics
    """
    try:
        stats = await _get_case_statistics(time_period)
        return CaseStatistics(**stats)
        
    except Exception as e:
        logger.error(f"Error getting case statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )

@router.get("/session/{session_id}")
async def get_cases_by_session(session_id: str):
    """
    Get all cases associated with a session
    """
    try:
        cases = await _get_cases_by_session(session_id)
        return {"session_id": session_id, "cases": cases}
        
    except Exception as e:
        logger.error(f"Error getting cases for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session cases"
        )

# Helper functions (mock implementations for demo)

async def _store_case(case_data: Dict):
    """Store case data (mock implementation)"""
    # In production, store in Firestore and sync with Quantexa
    logger.debug(f"Stored case: {case_data['case_id']}")

async def _get_case_by_id(case_id: str) -> Optional[Dict]:
    """Get case by ID (mock implementation)"""
    # Mock case data for demo
    return {
        "case_id": case_id,
        "session_id": "demo-session-123",
        "customer_id": "CUST001234",
        "fraud_type": "investment_scam",
        "risk_score": 87.0,
        "description": "High-risk investment scam detected with guaranteed returns promise",
        "status": "open",
        "priority": "high",
        "assigned_to": "fraud.team@hsbc.com",
        "created_at": "2024-01-15T14:30:00Z",
        "updated_at": "2024-01-15T14:30:00Z",
        "resolved_at": None,
        "evidence": [
            {
                "type": "transcript",
                "content": "Customer mentioned guaranteed 35% monthly returns",
                "timestamp": "2024-01-15T14:25:00Z"
            }
        ],
        "notes": [],
        "tags": ["investment_fraud", "high_risk"],
        "escalation_history": []
    }

async def _update_case(case_id: str, case_data: Dict):
    """Update case data (mock implementation)"""
    logger.debug(f"Updated case: {case_id}")

async def _list_cases(page: int, page_size: int, filters: Dict) -> tuple:
    """List cases with pagination (mock implementation)"""
    # Mock case list for demo
    mock_cases = [
        {
            "case_id": "HSBC-FD-20240115-ABC123",
            "session_id": "session-001",
            "customer_id": "CUST001234",
            "fraud_type": "investment_scam",
            "risk_score": 87.0,
            "description": "Investment scam with guaranteed returns",
            "status": "open",
            "priority": "high",
            "assigned_to": "fraud.team@hsbc.com",
            "created_at": "2024-01-15T14:30:00Z",
            "updated_at": "2024-01-15T14:30:00Z",
            "resolved_at": None,
            "evidence": [],
            "notes": [],
            "tags": ["investment_fraud"],
            "escalation_history": []
        },
        {
            "case_id": "HSBC-FD-20240115-DEF456",
            "session_id": "session-002",
            "customer_id": "CUST005678",
            "fraud_type": "romance_scam",
            "risk_score": 72.0,
            "description": "Romance scam with overseas emergency",
            "status": "investigating",
            "priority": "medium",
            "assigned_to": "fraud.specialist@hsbc.com",
            "created_at": "2024-01-15T13:15:00Z",
            "updated_at": "2024-01-15T15:45:00Z",
            "resolved_at": None,
            "evidence": [],
            "notes": [],
            "tags": ["romance_scam"],
            "escalation_history": []
        }
    ]
    
    # Apply filters (simplified for demo)
    filtered_cases = mock_cases
    if filters.get("status"):
        filtered_cases = [c for c in filtered_cases if c["status"] == filters["status"]]
    
    # Apply pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_cases = filtered_cases[start_idx:end_idx]
    
    return paginated_cases, len(filtered_cases)

async def _get_case_statistics(time_period: str) -> Dict:
    """Get case statistics (mock implementation)"""
    return {
        "total_cases": 156,
        "open_cases": 23,
        "resolved_cases": 133,
        "high_priority_cases": 8,
        "average_resolution_time_hours": 24.5,
        "cases_by_fraud_type": {
            "investment_scam": 67,
            "romance_scam": 34,
            "impersonation_scam": 28,
            "app_fraud": 15,
            "other": 12
        },
        "cases_by_status": {
            "open": 23,
            "investigating": 15,
            "resolved": 118,
            "closed": 0
        },
        "resolution_rate_percentage": 85.3
    }

async def _get_cases_by_session(session_id: str) -> List[Dict]:
    """Get cases by session ID (mock implementation)"""
    return [
        {
            "case_id": "HSBC-FD-20240115-ABC123",
            "fraud_type": "investment_scam",
            "risk_score": 87.0,
            "status": "open",
            "created_at": "2024-01-15T14:30:00Z"
        }
    ]
