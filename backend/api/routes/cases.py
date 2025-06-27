# backend/api/routes/cases.py - CONSOLIDATED
"""
Case management API routes - UPDATED to use all consolidated patterns
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from backend.middleware.error_handling import handle_api_errors, validate_input, require_session_id
from backend.services.mock_database import mock_db, case_service, session_service
from backend.api.models import (
    CaseCreateRequest, CaseUpdateRequest, CaseInfo, CaseStatus, Priority,
    ScamType, SuccessResponse, PaginationRequest, FilterRequest
)
from backend.utils import (
    get_current_timestamp, generate_case_id, create_success_response,
    create_error_response, calculate_percentage, get_priority_from_risk_score
)
from backend.config.settings import get_settings
from agents.case_management.agent import case_management_agent

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/create", response_model=CaseInfo)
@handle_api_errors("case_creation")
@validate_input(
    fraud_type={"required": True, "type": str},
    risk_score={"required": True, "type": float, "min": 0, "max": 100},
    description={"required": True, "type": str, "min_length": 10}
)
async def create_case(
    request: CaseCreateRequest,
    settings = Depends(get_settings)
):
    """
    Create a new fraud investigation case using consolidated agent
    """
    
    # Create fraud analysis data for agent processing
    fraud_analysis = {
        "risk_score": request.risk_score,
        "scam_type": request.fraud_type,
        "explanation": request.description,
        "detected_patterns": request.evidence.get("detected_patterns", {}),
        "confidence": request.evidence.get("confidence_score", 0.9)
    }
    
    # Use consolidated case management agent directly (no wrapper function)
    case_data = case_management_agent.process({
        "session_id": request.session_id,
        "fraud_analysis": fraud_analysis,
        "customer_id": request.customer_id
    })
    
    # Handle errors from agent processing
    if 'error' in case_data:
        raise ValueError(f"Case creation failed: {case_data['error']}")
    
    # Store additional case data using consolidated database
    enhanced_case_data = {
        **case_data,
        "description": request.description,
        "manual_priority": request.priority,
        "manual_assigned_to": request.assigned_to,
        "evidence": request.evidence or {}
    }
    
    # Update case with manual overrides if provided
    if request.assigned_to and request.assigned_to != case_data.get("assigned_team"):
        enhanced_case_data["assigned_team"] = request.assigned_to
    
    if request.priority != Priority.MEDIUM:
        enhanced_case_data["priority"] = request.priority
    
    # Store in consolidated database
    case_service.create_case(enhanced_case_data)
    
    # Create response using consolidated model
    case_response = CaseInfo(**enhanced_case_data)
    
    logger.info(f"📋 Case created using consolidated agent: {case_response.case_id}")
    
    return case_response

@router.get("/{case_id}", response_model=CaseInfo)
@handle_api_errors("get_case")
async def get_case(case_id: str):
    """
    Get a specific case by ID using consolidated service
    """
    
    # Use consolidated case service
    case_data = case_service.get_case(case_id)
    
    if not case_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case {case_id} not found"
        )
    
    return CaseInfo(**case_data)

@router.put("/{case_id}", response_model=CaseInfo)
@handle_api_errors("update_case")
async def update_case(
    case_id: str,
    request: CaseUpdateRequest
):
    """
    Update an existing case using consolidated agent and database
    """
    
    # Get existing case using consolidated service
    existing_case = case_service.get_case(case_id)
    
    if not existing_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case {case_id} not found"
        )
    
    # Prepare updates for consolidated agent
    updates = {}
    if request.status is not None:
        updates["status"] = request.status.value
        if request.status == CaseStatus.RESOLVED:
            updates["resolved_at"] = get_current_timestamp()
    
    if request.assigned_to is not None:
        updates["assigned_to"] = request.assigned_to
    
    if request.resolution is not None:
        updates["resolution"] = request.resolution
    
    if request.tags is not None:
        updates["tags"] = request.tags
    
    # Use consolidated case management agent for updates
    if request.notes:
        # Add note using agent
        note_result = case_management_agent.process({
            "case_id": case_id,
            "action": "add_note",
            "note": request.notes
        })
        
        if 'error' in note_result:
            raise ValueError(f"Failed to add note: {note_result['error']}")
    
    # Update case using agent
    if updates:
        update_result = case_management_agent.process({
            "case_id": case_id,
            "action": "update",
            "updates": updates
        })
        
        if 'error' in update_result:
            raise ValueError(f"Failed to update case: {update_result['error']}")
    
    # Get updated case data using consolidated service
    updated_case = case_service.get_case(case_id)
    
    logger.info(f"📝 Case updated using consolidated system: {case_id}")
    
    return CaseInfo(**updated_case)

@router.get("/", response_model=List[CaseInfo])
@handle_api_errors("list_cases")
async def list_cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[CaseStatus] = Query(None),
    priority: Optional[Priority] = Query(None),
    fraud_type: Optional[ScamType] = Query(None),
    assigned_to: Optional[str] = Query(None),
    customer_id: Optional[str] = Query(None),
    settings = Depends(get_settings)
):
    """
    List cases with filtering and pagination using consolidated database
    """
    
    # Build filters for consolidated database
    filters = {}
    if status:
        filters["status"] = status.value
    if priority:
        filters["priority"] = priority.value
    if fraud_type:
        filters["fraud_type"] = fraud_type.value
    if assigned_to:
        filters["assigned_to"] = assigned_to
    if customer_id:
        filters["customer_id"] = customer_id
    
    # Use consolidated database service for listing
    cases, total_count = mock_db.list_records(
        "fraud_cases",
        filters=filters,
        page=page,
        page_size=page_size,
        sort_by="created_at",
        sort_desc=True
    )
    
    # Convert to response models
    case_responses = []
    for case_record in cases:
        try:
            case_info = CaseInfo(**case_record.data)
            case_responses.append(case_info)
        except Exception as e:
            logger.warning(f"Error converting case data for {case_record.id}: {e}")
            continue
    
    # Add pagination info to response headers (FastAPI will handle this)
    logger.info(f"📋 Listed {len(case_responses)}/{total_count} cases using consolidated database")
    
    return case_responses

@router.post("/{case_id}/escalate")
@handle_api_errors("escalate_case")
async def escalate_case(
    case_id: str,
    escalation_reason: str,
    escalated_to: str,
    settings = Depends(get_settings)
):
    """
    Escalate a case using consolidated agent
    """
    
    # Use consolidated case management agent for escalation
    escalation_result = case_management_agent.process({
        "case_id": case_id,
        "action": "escalate",
        "escalation_data": {
            "escalated_to": escalated_to,
            "reason": escalation_reason
        }
    })
    
    if 'error' in escalation_result:
        raise ValueError(f"Escalation failed: {escalation_result['error']}")
    
    logger.info(f"🚨 Case escalated using consolidated agent: {case_id} to {escalated_to}")
    
    return create_success_response(
        data={
            "case_id": case_id,
            "escalated_to": escalated_to,
            "reason": escalation_reason,
            "escalated_at": get_current_timestamp()
        },
        message="Case escalated successfully using consolidated system"
    )

@router.post("/{case_id}/resolve")
@handle_api_errors("resolve_case")
async def resolve_case(
    case_id: str,
    outcome: str,
    fraud_confirmed: bool = False,
    amount_involved: float = 0,
    customer_protected: bool = True,
    lessons_learned: List[str] = None
):
    """
    Resolve a case using consolidated agent
    """
    
    resolution_data = {
        "outcome": outcome,
        "fraud_confirmed": fraud_confirmed,
        "amount_involved": amount_involved,
        "customer_protected": customer_protected,
        "lessons_learned": lessons_learned or [],
        "resolved_by": "api_user"  # In production, use authenticated user
    }
    
    # Use consolidated case management agent for resolution
    resolution_result = case_management_agent.process({
        "case_id": case_id,
        "action": "resolve",
        "resolution": resolution_data
    })
    
    if 'error' in resolution_result:
        raise ValueError(f"Case resolution failed: {resolution_result['error']}")
    
    logger.info(f"✅ Case resolved using consolidated agent: {case_id}")
    
    return create_success_response(
        data={
            "case_id": case_id,
            "outcome": outcome,
            "fraud_confirmed": fraud_confirmed,
            "resolved_at": get_current_timestamp()
        },
        message="Case resolved successfully using consolidated system"
    )

@router.post("/{case_id}/notes")
@handle_api_errors("add_case_note")
async def add_case_note(
    case_id: str,
    note: str,
    note_type: str = "general"
):
    """
    Add a note to a case using consolidated agent
    """
    
    # Use consolidated case management agent
    note_result = case_management_agent.process({
        "case_id": case_id,
        "action": "add_note",
        "note": note,
        "note_type": note_type
    })
    
    if 'error' in note_result:
        raise ValueError(f"Failed to add note: {note_result['error']}")
    
    return create_success_response(
        data={
            "case_id": case_id,
            "note_added": True,
            "note_type": note_type,
            "added_at": get_current_timestamp()
        },
        message="Note added successfully using consolidated agent"
    )

@router.get("/statistics/overview")
@handle_api_errors("case_statistics")
async def get_case_statistics(
    time_period: str = Query("30d", regex="^(1d|7d|30d|90d)$"),
    settings = Depends(get_settings)
):
    """
    Get case management statistics using consolidated database and agent
    """
    
    # Get statistics from consolidated case management agent
    agent_stats = case_management_agent.get_case_statistics()
    
    # Get additional statistics from consolidated database
    db_stats = mock_db.get_statistics("fraud_cases")
    
    # Calculate time-based statistics using consolidated database
    cases, total_count = mock_db.list_records("fraud_cases", page=1, page_size=1000)
    cases_data = [record.data for record in cases]
    
    # Filter by time period
    cutoff_date = datetime.now()
    if time_period == "1d":
        cutoff_date -= timedelta(days=1)
    elif time_period == "7d":
        cutoff_date -= timedelta(days=7)
    elif time_period == "30d":
        cutoff_date -= timedelta(days=30)
    elif time_period == "90d":
        cutoff_date -= timedelta(days=90)
    
    recent_cases = []
    for case in cases_data:
        try:
            case_date = datetime.fromisoformat(case.get("created_at", "").replace('Z', '+00:00'))
            if case_date >= cutoff_date:
                recent_cases.append(case)
        except:
            continue
    
    # Calculate consolidated statistics using centralized utilities
    total_recent = len(recent_cases)
    open_cases = len([c for c in recent_cases if c.get("status") == "open"])
    resolved_cases = len([c for c in recent_cases if c.get("status") == "resolved"])
    high_priority = len([c for c in recent_cases if c.get("priority") in ["high", "critical"]])
    
    # Calculate resolution rate using centralized utility
    resolution_rate = calculate_percentage(resolved_cases, total_recent)
    
    # Get case distribution by fraud type
    fraud_type_distribution = {}
    for case in recent_cases:
        fraud_type = case.get("fraud_type", "unknown")
        fraud_type_distribution[fraud_type] = fraud_type_distribution.get(fraud_type, 0) + 1
    
    # Calculate average resolution time
    resolved_with_times = []
    for case in recent_cases:
        if case.get("status") == "resolved" and case.get("resolved_at") and case.get("created_at"):
            try:
                created = datetime.fromisoformat(case["created_at"].replace('Z', '+00:00'))
                resolved = datetime.fromisoformat(case["resolved_at"].replace('Z', '+00:00'))
                resolution_hours = (resolved - created).total_seconds() / 3600
                resolved_with_times.append(resolution_hours)
            except:
                continue
    
    avg_resolution_time = sum(resolved_with_times) / len(resolved_with_times) if resolved_with_times else 0
    
    comprehensive_stats = {
        "time_period": time_period,
        "overview": {
            "total_cases": total_recent,
            "open_cases": open_cases,
            "resolved_cases": resolved_cases,
            "high_priority_cases": high_priority,
            "resolution_rate_percentage": resolution_rate,
            "average_resolution_time_hours": round(avg_resolution_time, 2)
        },
        "distribution": {
            "by_fraud_type": fraud_type_distribution,
            "by_status": db_stats.get("by_status", {}),
            "by_priority": db_stats.get("by_priority", {})
        },
        "agent_performance": agent_stats,
        "database_metrics": db_stats,
        "consolidation_benefits": {
            "data_sources": [
                "Consolidated case management agent",
                "Centralized MockDatabase service",
                "Shared calculation utilities"
            ],
            "unified_metrics": "Consistent calculation methods across system",
            "centralized_config": "Risk thresholds from settings.py"
        }
    }
    
    return create_success_response(
        data=comprehensive_stats,
        message="Case statistics retrieved using consolidated services"
    )

@router.get("/session/{session_id}")
@handle_api_errors("get_session_cases")
async def get_cases_by_session(session_id: str):
    """
    Get all cases associated with a session using consolidated services
    """
    
    # Use consolidated case service
    cases = case_service.get_cases_by_session(session_id)
    
    # Also get session data using consolidated session service
    session_data = session_service.get_session(session_id)
    
    response_data = {
        "session_id": session_id,
        "session_info": session_data,
        "cases": cases,
        "total_cases": len(cases),
        "consolidation_note": "Retrieved using consolidated case and session services"
    }
    
    return create_success_response(
        data=response_data,
        message=f"Cases for session {session_id} retrieved using consolidated services"
    )

@router.get("/customer/{customer_id}")
@handle_api_errors("get_customer_cases")
async def get_cases_by_customer(customer_id: str):
    """
    Get all cases for a specific customer using consolidated services
    """
    
    # Use consolidated case service
    cases = case_service.get_cases_by_customer(customer_id)
    
    # Calculate customer risk profile using consolidated data
    if cases:
        risk_scores = [case.get("risk_score", 0) for case in cases if case.get("risk_score")]
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        # Count fraud types using consolidated utilities
        fraud_types = {}
        for case in cases:
            fraud_type = case.get("fraud_type", "unknown")
            fraud_types[fraud_type] = fraud_types.get(fraud_type, 0) + 1
        
        customer_profile = {
            "customer_id": customer_id,
            "total_cases": len(cases),
            "average_risk_score": round(avg_risk, 2),
            "fraud_types_encountered": fraud_types,
            "high_risk_cases": len([c for c in cases if c.get("risk_score", 0) >= 80]),
            "open_cases": len([c for c in cases if c.get("status") == "open"])
        }
    else:
        customer_profile = {
            "customer_id": customer_id,
            "total_cases": 0,
            "note": "No cases found for this customer"
        }
    
    return create_success_response(
        data={
            "customer_profile": customer_profile,
            "cases": cases
        },
        message=f"Customer cases retrieved using consolidated services"
    )

@router.get("/priority/{priority}")
@handle_api_errors("get_priority_cases")
async def get_cases_by_priority(
    priority: Priority,
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get cases by priority level using consolidated database
    """
    
    # Use consolidated case service
    cases = case_service.get_cases_by_status(priority.value)  # Using status method for priority
    
    # Actually filter by priority using consolidated database
    priority_cases, total_count = mock_db.list_records(
        "fraud_cases",
        filters={"priority": priority.value},
        page=1,
        page_size=limit,
        sort_by="created_at",
        sort_desc=True
    )
    
    cases_data = [record.data for record in priority_cases]
    
    # Add priority-specific insights using centralized configuration
    settings = get_settings()
    threshold_mapping = {
        "critical": settings.risk_threshold_critical,
        "high": settings.risk_threshold_high,
        "medium": settings.risk_threshold_medium,
        "low": settings.risk_threshold_low
    }
    
    priority_insights = {
        "priority_level": priority.value,
        "threshold_used": threshold_mapping.get(priority.value, 0),
        "total_cases": len(cases_data),
        "avg_risk_score": sum(c.get("risk_score", 0) for c in cases_data) / len(cases_data) if cases_data else 0,
        "team_assignment": settings.team_assignments.get(priority.value, "unknown"),
        "consolidation_note": "Priority thresholds from centralized configuration"
    }
    
    return create_success_response(
        data={
            "priority_insights": priority_insights,
            "cases": cases_data[:limit]
        },
        message=f"Priority {priority.value} cases retrieved using consolidated system"
    )

@router.post("/bulk-update")
@handle_api_errors("bulk_update_cases")
async def bulk_update_cases(
    case_ids: List[str],
    updates: Dict,
    settings = Depends(get_settings)
):
    """
    Bulk update multiple cases using consolidated database service
    """
    
    if len(case_ids) > 100:
        raise ValueError("Maximum 100 cases can be updated at once")
    
    # Prepare bulk updates for consolidated database
    bulk_updates = {}
    for case_id in case_ids:
        bulk_updates[case_id] = {
            **updates,
            "updated_at": get_current_timestamp(),
            "bulk_updated": True
        }
    
    # Use consolidated database bulk update
    updated_records = mock_db.bulk_update("fraud_cases", bulk_updates)
    
    successful_updates = len(updated_records)
    failed_updates = len(case_ids) - successful_updates
    
    # Log bulk operation using centralized logging
    logger.info(f"📦 Bulk update completed: {successful_updates} successful, {failed_updates} failed")
    
    return create_success_response(
        data={
            "requested_updates": len(case_ids),
            "successful_updates": successful_updates,
            "failed_updates": failed_updates,
            "updated_case_ids": [record.id for record in updated_records],
            "consolidation_info": {
                "method": "Consolidated database bulk_update service",
                "utilities": "Centralized timestamp generation",
                "logging": "Unified logging system"
            }
        },
        message="Bulk update completed using consolidated database service"
    )

@router.get("/trends/analysis")
@handle_api_errors("case_trends")
async def get_case_trends(
    days: int = Query(30, ge=1, le=365),
    settings = Depends(get_settings)
):
    """
    Analyze case trends using consolidated database and utilities
    """
    
    # Get cases from consolidated database
    cases, _ = mock_db.list_records("fraud_cases", page=1, page_size=2000)
    cases_data = [record.data for record in cases]
    
    # Filter by time period
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_cases = []
    
    for case in cases_data:
        try:
            case_date = datetime.fromisoformat(case.get("created_at", "").replace('Z', '+00:00'))
            if case_date >= cutoff_date:
                recent_cases.append(case)
        except:
            continue
    
    # Analyze trends using consolidated utilities
    daily_cases = {}
    fraud_type_trends = {}
    risk_score_trends = []
    
    for case in recent_cases:
        # Daily case count
        case_date = case.get("created_at", "")[:10]  # Get date part
        daily_cases[case_date] = daily_cases.get(case_date, 0) + 1
        
        # Fraud type trends
        fraud_type = case.get("fraud_type", "unknown")
        fraud_type_trends[fraud_type] = fraud_type_trends.get(fraud_type, 0) + 1
        
        # Risk score trends
        risk_score = case.get("risk_score", 0)
        if risk_score > 0:
            risk_score_trends.append(risk_score)
    
    # Calculate trend statistics using centralized utilities
    avg_daily_cases = sum(daily_cases.values()) / len(daily_cases) if daily_cases else 0
    avg_risk_score = sum(risk_score_trends) / len(risk_score_trends) if risk_score_trends else 0
    
    # High risk trend
    high_risk_cases = len([r for r in risk_score_trends if r >= settings.risk_threshold_high])
    high_risk_percentage = calculate_percentage(high_risk_cases, len(risk_score_trends))
    
    trends_analysis = {
        "analysis_period_days": days,
        "total_cases_analyzed": len(recent_cases),
        "daily_trends": {
            "average_daily_cases": round(avg_daily_cases, 2),
            "daily_case_counts": daily_cases
        },
        "fraud_type_trends": fraud_type_trends,
        "risk_trends": {
            "average_risk_score": round(avg_risk_score, 2),
            "high_risk_percentage": high_risk_percentage,
            "high_risk_threshold": settings.risk_threshold_high,
            "risk_score_distribution": {
                "critical": len([r for r in risk_score_trends if r >= settings.risk_threshold_critical]),
                "high": len([r for r in risk_score_trends if settings.risk_threshold_high <= r < settings.risk_threshold_critical]),
                "medium": len([r for r in risk_score_trends if settings.risk_threshold_medium <= r < settings.risk_threshold_high]),
                "low": len([r for r in risk_score_trends if r < settings.risk_threshold_medium])
            }
        },
        "consolidation_benefits": {
            "data_source": "Centralized MockDatabase service",
            "calculations": "Shared utility functions",
            "thresholds": "Centralized configuration from settings.py",
            "analysis_tools": "Unified date handling and statistics"
        }
    }
    
    return create_success_response(
        data=trends_analysis,
        message="Case trends analysis completed using consolidated services"
    )

@router.delete("/{case_id}")
@handle_api_errors("delete_case")
async def delete_case(case_id: str, confirm: bool = Query(False)):
    """
    Delete a case using consolidated database service
    """
    
    if not confirm:
        raise ValueError("Case deletion requires confirmation parameter (confirm=true)")
    
    # Check if case exists using consolidated service
    existing_case = case_service.get_case(case_id)
    if not existing_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case {case_id} not found"
        )
    
    # Delete using consolidated database
    deletion_success = mock_db.delete("fraud_cases", case_id)
    
    if not deletion_success:
        raise ValueError(f"Failed to delete case {case_id}")
    
    logger.warning(f"🗑️ Case deleted using consolidated database: {case_id}")
    
    return create_success_response(
        data={
            "case_id": case_id,
            "deleted": True,
            "deleted_at": get_current_timestamp()
        },
        message="Case deleted successfully using consolidated database service"
    )

# ===== SPECIALIZED ENDPOINTS =====

@router.get("/export/csv")
@handle_api_errors("export_cases_csv")
async def export_cases_csv(
    status: Optional[CaseStatus] = Query(None),
    priority: Optional[Priority] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None)
):
    """
    Export cases to CSV format using consolidated database
    """
    
    # Build filters
    filters = {}
    if status:
        filters["status"] = status.value
    if priority:
        filters["priority"] = priority.value
    
    # Get filtered cases using consolidated database
    cases, total_count = mock_db.list_records(
        "fraud_cases",
        filters=filters,
        page=1,
        page_size=10000  # Large export
    )
    
    # Convert to CSV format (simplified for demo)
    csv_headers = [
        "case_id", "fraud_type", "risk_score", "priority", "status",
        "created_at", "assigned_team", "customer_id"
    ]
    
    csv_rows = []
    for case in cases:
        case_data = case.data
        row = [
            case_data.get("case_id", ""),
            case_data.get("fraud_type", ""),
            str(case_data.get("risk_score", 0)),
            case_data.get("priority", ""),
            case_data.get("status", ""),
            case_data.get("created_at", ""),
            case_data.get("assigned_team", ""),
            case_data.get("customer_id", "")
        ]
        csv_rows.append(row)
    
    export_data = {
        "format": "csv",
        "headers": csv_headers,
        "rows": csv_rows,
        "total_exported": len(csv_rows),
        "export_timestamp": get_current_timestamp(),
        "filters_applied": filters,
        "consolidation_note": "Exported using consolidated database service"
    }
    
    return create_success_response(
        data=export_data,
        message=f"Exported {len(csv_rows)} cases using consolidated system"
    )
