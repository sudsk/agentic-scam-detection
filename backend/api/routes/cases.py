# backend/api/routes/cases.py - UPDATED to use Basic Auth ServiceNow Service
"""
ServiceNow Case Management API - Uses Basic Authentication
All case management happens directly in ServiceNow via Basic Auth
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from backend.middleware.error_handling import handle_api_errors, validate_input
from backend.utils import (
    get_current_timestamp, create_success_response, create_error_response
)
from backend.config.settings import get_settings
from backend.services.servicenow_service import ServiceNowService

logger = logging.getLogger(__name__)
router = APIRouter() 

# ===== HELPER FUNCTION TO CREATE SERVICENOW SERVICE =====

def get_servicenow_service(settings) -> ServiceNowService:
    """Create ServiceNow service instance with proper authentication"""
    if not settings.servicenow_enabled:
        raise HTTPException(status_code=400, detail="ServiceNow integration not enabled")
    
    if not settings.servicenow_username or not settings.servicenow_password:
        raise HTTPException(status_code=500, detail="ServiceNow credentials not configured")
    
    return ServiceNowService(
        instance_url=settings.servicenow_instance_url,
        username=settings.servicenow_username,
        password=settings.servicenow_password,
        api_key=settings.servicenow_api_key  # Optional
    )

# ===== API ENDPOINTS =====

@router.post("/create")
@handle_api_errors("create_fraud_case")
async def create_fraud_case(case_data: Dict[str, Any], settings = Depends(get_settings)):
    """Create fraud case in ServiceNow using Basic Auth"""
    
    # Create ServiceNow service
    servicenow = get_servicenow_service(settings)
    
    try:
        # Prepare incident data for ServiceNow
        risk_score = case_data.get('risk_score', 0)
        scam_type = case_data.get('scam_type', 'unknown')
        customer_name = case_data.get('customer_name', 'Unknown Customer')
        
        # Determine priority based on risk score
        if risk_score >= 80:
            priority = "1"  # Critical
            urgency = "1"
        elif risk_score >= 60:
            priority = "2"  # High
            urgency = "2"
        elif risk_score >= 40:
            priority = "3"  # Medium
            urgency = "3"
        else:
            priority = "4"  # Low
            urgency = "3"
        
        # Format transcript for ServiceNow
        transcript_text = ""
        if case_data.get('transcript_segments'):
            transcript_text = "\n".join([
                f"[{segment.get('start', 0):.1f}s] {segment.get('speaker', 'Unknown').upper()}: {segment.get('text', '')}"
                for segment in case_data.get('transcript_segments', [])
            ])
        
        # Format detected patterns
        patterns_text = ""
        if case_data.get('detected_patterns'):
            patterns_text = "\n".join([
                f"• {pattern.replace('_', ' ').title()}: {', '.join(data.get('matches', [])[:3])}"
                for pattern, data in case_data.get('detected_patterns', {}).items()
            ])
        
        incident_data = {
            "category": "security",  # Use standard category
            "short_description": f"Fraud Alert: {scam_type.replace('_', ' ').title()} - {customer_name}",
            "description": f"""
=== AUTOMATED FRAUD DETECTION ALERT ===

Customer Information:
• Name: {customer_name}
• Account: {case_data.get('customer_account', 'Unknown')}
• Phone: {case_data.get('customer_phone', 'Unknown')}

Fraud Analysis:
• Risk Score: {risk_score}%
• Scam Type: {scam_type.replace('_', ' ').title()}
• Confidence: {case_data.get('confidence_score', 0):.1%}
• Session ID: {case_data.get('session_id', 'N/A')}

Detected Patterns:
{patterns_text or 'No specific patterns detected'}

Recommended Actions:
{chr(10).join(f"• {action}" for action in case_data.get('recommended_actions', []))}

Call Transcript:
{transcript_text or 'No transcript available'}

Generated: {get_current_timestamp()}
System: HSBC Fraud Detection Agent
            """.strip(),
            "priority": priority,
            "urgency": urgency,
            "state": "2",  # In Progress
            "caller_id": settings.servicenow_username,
            "opened_by": settings.servicenow_username,
            "assigned_to": settings.servicenow_username
        }
        
        # Add custom fields if they exist
        if risk_score:
            incident_data["u_risk_score"] = risk_score
        if scam_type:
            incident_data["u_scam_type"] = scam_type
        if case_data.get('session_id'):
            incident_data["u_session_id"] = case_data.get('session_id')
        
        # Create incident in ServiceNow
        result = await servicenow.create_incident(incident_data)
        
        if result["success"]:
            return create_success_response(
                data={
                    "case_id": result["incident_sys_id"],
                    "case_number": result["incident_number"],
                    "case_url": result["incident_url"],
                    "priority": priority,
                    "state": result.get("state"),
                    "created_at": result.get("created_at"),
                    "system": "servicenow",
                    "auth_method": "basic_auth"
                },
                message="Fraud case created successfully in ServiceNow"
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    finally:
        await servicenow.close_session()

@router.get("/{case_identifier}")
@handle_api_errors("get_case")
async def get_case(case_identifier: str, settings = Depends(get_settings)):
    """Get case details from ServiceNow (by incident number or sys_id)"""
    
    servicenow = get_servicenow_service(settings)
    
    try:
        # Try to get by incident number first
        result = await servicenow.get_incident(case_identifier)
        
        if result["success"]:
            incident = result["incident"]
            return create_success_response(
                data={
                    "case_id": incident["sys_id"],
                    "case_number": incident["number"],
                    "case_url": f"{settings.servicenow_instance_url}/nav_to.do?uri=incident.do?sys_id={incident['sys_id']}",
                    "short_description": incident["short_description"],
                    "description": incident["description"],
                    "state": incident["state"],
                    "priority": incident["priority"],
                    "created_at": incident["sys_created_on"],
                    "updated_at": incident["sys_updated_on"],
                    "risk_score": incident.get("u_risk_score", 0),
                    "scam_type": incident.get("u_scam_type", "unknown"),
                    "session_id": incident.get("u_session_id", ""),
                    "system": "servicenow",
                    "auth_method": "basic_auth"
                },
                message="Case retrieved successfully"
            )
        else:
            raise HTTPException(status_code=404, detail=result["error"])
            
    finally:
        await servicenow.close_session()

@router.put("/{case_sys_id}")
@handle_api_errors("update_case")
async def update_case(
    case_sys_id: str,
    update_data: Dict[str, Any],
    settings = Depends(get_settings)
):
    """Update case in ServiceNow"""
    
    servicenow = get_servicenow_service(settings)
    
    try:
        # Prepare update data for ServiceNow
        servicenow_update = {}
        
        if update_data.get("state"):
            servicenow_update["state"] = update_data["state"]
        if update_data.get("priority"):
            servicenow_update["priority"] = update_data["priority"]
        if update_data.get("notes"):
            servicenow_update["work_notes"] = f"[{get_current_timestamp()}] {update_data['notes']}"
        if update_data.get("assigned_to"):
            servicenow_update["assigned_to"] = update_data["assigned_to"]
        
        result = await servicenow.update_incident(case_sys_id, servicenow_update)
        
        if result["success"]:
            return create_success_response(
                data={
                    "case_id": case_sys_id,
                    "updated_fields": list(servicenow_update.keys()),
                    "updated_at": get_current_timestamp(),
                    "system": "servicenow",
                    "auth_method": "basic_auth"
                },
                message="Case updated successfully"
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    finally:
        await servicenow.close_session()

@router.get("/")
@handle_api_errors("list_cases")
async def list_cases(
    limit: int = Query(50, ge=1, le=200),
    category: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    settings = Depends(get_settings)
):
    """List cases from ServiceNow"""
    
    servicenow = get_servicenow_service(settings)
    
    try:
        # Build filters
        filters = {
            "opened_by": settings.servicenow_username  # Only show incidents created by fraud system
        }
        
        if category:
            filters["category"] = category
        if state:
            filters["state"] = state
        if priority:
            filters["priority"] = priority
        
        result = await servicenow.get_incidents(limit=limit, filters=filters)
        
        if result["success"]:
            cases = []
            for incident in result["incidents"]:
                cases.append({
                    "case_id": incident["sys_id"],
                    "case_number": incident["number"],
                    "case_url": f"{settings.servicenow_instance_url}/nav_to.do?uri=incident.do?sys_id={incident['sys_id']}",
                    "short_description": incident["short_description"],
                    "state": incident["state"],
                    "priority": incident["priority"],
                    "created_at": incident["opened_at"],
                    "risk_score": incident.get("u_risk_score", 0),
                    "scam_type": incident.get("u_scam_type", "unknown"),
                    "session_id": incident.get("u_session_id", ""),
                    "system": "servicenow",
                    "auth_method": "basic_auth"
                })
            
            return create_success_response(
                data={
                    "cases": cases,
                    "total_count": result["total_count"],
                    "filters_applied": filters,
                    "system": "servicenow",
                    "auth_method": "basic_auth"
                },
                message="Cases retrieved successfully"
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    finally:
        await servicenow.close_session()

@router.get("/statistics/overview")
@handle_api_errors("case_statistics")
async def get_case_statistics(settings = Depends(get_settings)):
    """Get case statistics from ServiceNow"""
    
    servicenow = get_servicenow_service(settings)
    
    try:
        # Get all fraud incidents
        result = await servicenow.get_incidents(limit=1000, filters={"opened_by": settings.servicenow_username})
        
        if result["success"]:
            incidents = result["incidents"]
            
            # Calculate statistics
            total_cases = len(incidents)
            open_cases = len([i for i in incidents if i.get("state") in ["1", "2"]])  # New, In Progress
            resolved_cases = len([i for i in incidents if i.get("state") in ["6", "7"]])  # Resolved, Closed
            critical_cases = len([i for i in incidents if i.get("priority") == "1"])
            high_cases = len([i for i in incidents if i.get("priority") == "2"])
            
            # Risk score statistics
            risk_scores = [int(i.get("u_risk_score", 0)) for i in incidents if i.get("u_risk_score")]
            avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
            
            # Scam type distribution
            scam_types = {}
            for incident in incidents:
                scam_type = incident.get("u_scam_type", "unknown")
                scam_types[scam_type] = scam_types.get(scam_type, 0) + 1
            
            return create_success_response(
                data={
                    "total_cases": total_cases,
                    "open_cases": open_cases,
                    "resolved_cases": resolved_cases,
                    "critical_cases": critical_cases,
                    "high_cases": high_cases,
                    "average_risk_score": round(avg_risk_score, 1),
                    "scam_type_distribution": scam_types,
                    "system": "servicenow",
                    "auth_method": "basic_auth",
                    "instance_url": settings.servicenow_instance_url
                },
                message="Case statistics retrieved successfully"
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    finally:
        await servicenow.close_session()

@router.get("/test-connection")
@handle_api_errors("test_connection")
async def test_connection(settings = Depends(get_settings)):
    """Test ServiceNow connection using Basic Auth"""
    
    servicenow = get_servicenow_service(settings)
    
    try:
        result = await servicenow.test_connection()
        
        if result["success"]:
            return create_success_response(
                data={
                    "connected": True,
                    "instance_url": settings.servicenow_instance_url,
                    "username": settings.servicenow_username,
                    "auth_method": "basic_auth",
                    "status_code": result.get("status_code"),
                    "message": result.get("message")
                },
                message="ServiceNow connection successful"
            )
        else:
            return create_error_response(
                error=result.get("error", "Unknown connection error"),
                code="SERVICENOW_CONNECTION_ERROR",
                details={
                    "status_code": result.get("status_code"),
                    "auth_method": "basic_auth"
                }
            )
            
    finally:
        await servicenow.close_session()
