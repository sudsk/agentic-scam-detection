# backend/api/routes/cases.py - SIMPLIFIED ServiceNow-Only Version
"""
ServiceNow Case Management API - No Internal Database
All case management happens directly in ServiceNow
"""

import logging
import aiohttp
import json
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from backend.middleware.error_handling import handle_api_errors, validate_input
from backend.utils import (
    get_current_timestamp, create_success_response, create_error_response
)
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ===== SERVICENOW SERVICE =====

class ServiceNowService:
    """ServiceNow integration service"""
    
    def __init__(self, instance_url: str, username: str, password: str):
        self.instance_url = instance_url
        self.username = username
        self.password = password
        self.session = None
    
    async def _get_session(self):
        if self.session is None or self.session.closed:
            credentials = f"{self.username}:{self.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Basic {encoded_credentials}'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def create_incident(self, incident_data: Dict) -> Dict:
        """Create incident in ServiceNow"""
        try:
            session = await self._get_session()
            url = f"{self.instance_url}/api/now/table/incident"
            
            async with session.post(url, json=incident_data) as response:
                if response.status == 201:
                    result = await response.json()
                    return {
                        "success": True,
                        "incident_number": result["result"]["number"],
                        "incident_sys_id": result["result"]["sys_id"],
                        "incident_url": f"{self.instance_url}/nav_to.do?uri=incident.do?sys_id={result['result']['sys_id']}",
                        "state": result["result"]["state"],
                        "priority": result["result"]["priority"],
                        "created_at": result["result"]["sys_created_on"]
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"ServiceNow API error: {response.status} - {error_text}"
                    }
        except Exception as e:
            logger.error(f"Error creating ServiceNow incident: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_incident(self, incident_number: str) -> Dict:
        """Get incident details from ServiceNow"""
        try:
            session = await self._get_session()
            url = f"{self.instance_url}/api/now/table/incident"
            params = {"sysparm_query": f"number={incident_number}"}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("result"):
                        incident = result["result"][0]
                        return {
                            "success": True,
                            "incident": incident
                        }
                    else:
                        return {
                            "success": False,
                            "error": "Incident not found"
                        }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"ServiceNow API error: {response.status} - {error_text}"
                    }
        except Exception as e:
            logger.error(f"Error getting ServiceNow incident: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_incident(self, incident_sys_id: str, update_data: Dict) -> Dict:
        """Update incident in ServiceNow"""
        try:
            session = await self._get_session()
            url = f"{self.instance_url}/api/now/table/incident/{incident_sys_id}"
            
            async with session.patch(url, json=update_data) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "incident": result["result"]
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"ServiceNow API error: {response.status} - {error_text}"
                    }
        except Exception as e:
            logger.error(f"Error updating ServiceNow incident: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_incidents(self, limit: int = 50, filters: Dict = None) -> Dict:
        """Get multiple incidents from ServiceNow"""
        try:
            session = await self._get_session()
            url = f"{self.instance_url}/api/now/table/incident"
            
            # Build query parameters
            params = {
                "sysparm_limit": str(limit),
                "sysparm_fields": "number,short_description,state,priority,opened_at,sys_id,category,u_risk_score,u_scam_type,u_session_id",
                "sysparm_order": "-opened_at"
            }
            
            # Add filters if provided
            query_parts = []
            if filters:
                if filters.get("category"):
                    query_parts.append(f"category={filters['category']}")
                if filters.get("state"):
                    query_parts.append(f"state={filters['state']}")
                if filters.get("priority"):
                    query_parts.append(f"priority={filters['priority']}")
                if filters.get("opened_by"):
                    query_parts.append(f"opened_by={filters['opened_by']}")
            
            if query_parts:
                params["sysparm_query"] = "^".join(query_parts)
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "incidents": result.get("result", []),
                        "total_count": len(result.get("result", []))
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"ServiceNow API error: {response.status} - {error_text}"
                    }
        except Exception as e:
            logger.error(f"Error getting ServiceNow incidents: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close_session(self):
        if self.session:
            await self.session.close()

# ===== API ENDPOINTS =====

@router.post("/create")
@handle_api_errors("create_fraud_case")
async def create_fraud_case(case_data: Dict[str, Any], settings = Depends(get_settings)):
    """Create fraud case in ServiceNow"""
    
    if not settings.servicenow_enabled:
        raise HTTPException(status_code=400, detail="ServiceNow integration not enabled")
    
    if not settings.servicenow_username or not settings.servicenow_password:
        raise HTTPException(status_code=500, detail="ServiceNow credentials not configured")
    
    # Create ServiceNow service
    servicenow = ServiceNowService(
        instance_url=settings.servicenow_instance_url,
        username=settings.servicenow_username,
        password=settings.servicenow_password
    )
    
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
                    "state": result["state"],
                    "created_at": result["created_at"],
                    "system": "servicenow"
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
    
    if not settings.servicenow_enabled:
        raise HTTPException(status_code=400, detail="ServiceNow integration not enabled")
    
    servicenow = ServiceNowService(
        instance_url=settings.servicenow_instance_url,
        username=settings.servicenow_username,
        password=settings.servicenow_password
    )
    
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
                    "system": "servicenow"
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
    
    if not settings.servicenow_enabled:
        raise HTTPException(status_code=400, detail="ServiceNow integration not enabled")
    
    servicenow = ServiceNowService(
        instance_url=settings.servicenow_instance_url,
        username=settings.servicenow_username,
        password=settings.servicenow_password
    )
    
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
                    "system": "servicenow"
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
    
    if not settings.servicenow_enabled:
        raise HTTPException(status_code=400, detail="ServiceNow integration not enabled")
    
    servicenow = ServiceNowService(
        instance_url=settings.servicenow_instance_url,
        username=settings.servicenow_username,
        password=settings.servicenow_password
    )
    
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
                    "system": "servicenow"
                })
            
            return create_success_response(
                data={
                    "cases": cases,
                    "total_count": result["total_count"],
                    "filters_applied": filters,
                    "system": "servicenow"
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
    
    if not settings.servicenow_enabled:
        raise HTTPException(status_code=400, detail="ServiceNow integration not enabled")
    
    servicenow = ServiceNowService(
        instance_url=settings.servicenow_instance_url,
        username=settings.servicenow_username,
        password=settings.servicenow_password
    )
    
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
    """Test ServiceNow connection"""
    
    if not settings.servicenow_username or not settings.servicenow_password:
        raise HTTPException(status_code=500, detail="ServiceNow credentials not configured")
    
    servicenow = ServiceNowService(
        instance_url=settings.servicenow_instance_url,
        username=settings.servicenow_username,
        password=settings.servicenow_password
    )
    
    try:
        session = await servicenow._get_session()
        url = f"{settings.servicenow_instance_url}/api/now/table/sys_user?sysparm_limit=1"
        
        async with session.get(url) as response:
            if response.status == 200:
                return create_success_response(
                    data={
                        "connected": True,
                        "instance_url": settings.servicenow_instance_url,
                        "username": settings.servicenow_username,
                        "status_code": response.status
                    },
                    message="ServiceNow connection successful"
                )
            else:
                error_text = await response.text()
                return create_error_response(
                    error=f"Connection failed: {response.status} - {error_text}",
                    code="SERVICENOW_CONNECTION_ERROR"
                )
    finally:
        await servicenow.close_session()
