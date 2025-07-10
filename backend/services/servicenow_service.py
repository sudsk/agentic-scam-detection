import aiohttp
import json
import base64
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ServiceNowService:
    def __init__(self, instance_url: str, username: str, password: str, api_key: Optional[str] = None):
        self.instance_url = instance_url
        self.username = username
        self.password = password
        self.api_key = api_key  # Optional, for future API key support
        self.session = None
    
    async def _get_session(self):
        if self.session is None or self.session.closed:
            # Use Basic Authentication
            credentials = f"{self.username}:{self.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Basic {encoded_credentials}'
            }
            
            # If API key is provided, add it as well (some ServiceNow instances support both)
            if self.api_key:
                headers['X-UserToken'] = self.api_key
            
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def create_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create incident in ServiceNow"""
        try:
            session = await self._get_session()
            url = f"{self.instance_url}/api/now/table/incident"
            
            logger.info(f"Creating ServiceNow incident at: {url}")
            logger.debug(f"Incident data: {json.dumps(incident_data, indent=2)}")
            
            async with session.post(url, json=incident_data) as response:
                if response.status == 201:
                    result = await response.json()
                    logger.info(f"Successfully created incident: {result['result']['number']}")
                    return {
                        "success": True,
                        "incident_number": result["result"]["number"],
                        "incident_sys_id": result["result"]["sys_id"],
                        "incident_url": f"{self.instance_url}/nav_to.do?uri=incident.do?sys_id={result['result']['sys_id']}",
                        "state": result["result"].get("state"),
                        "priority": result["result"].get("priority"),
                        "created_at": result["result"].get("sys_created_on")
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"ServiceNow API error: {response.status} - {error_text}")
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
    
    async def get_incident(self, incident_number: str) -> Dict[str, Any]:
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
    
    async def update_incident(self, incident_sys_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def get_incidents(self, limit: int = 50, filters: Dict = None) -> Dict[str, Any]:
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
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test ServiceNow connection"""
        try:
            session = await self._get_session()
            url = f"{self.instance_url}/api/now/table/sys_user?sysparm_limit=1"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return {
                        "success": True,
                        "status_code": response.status,
                        "message": "Connection successful"
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "status_code": response.status,
                        "error": f"Connection failed: {response.status} - {error_text}"
                    }
        except Exception as e:
            logger.error(f"Error testing ServiceNow connection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close_session(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
