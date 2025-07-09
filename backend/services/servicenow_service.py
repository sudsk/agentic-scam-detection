import aiohttp
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ServiceNowService:
    def __init__(self, instance_url: str, api_key: str):
        self.instance_url = instance_url
        self.api_key = api_key
        self.session = None
    
    async def _get_session(self):
        if self.session is None or self.session.closed:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-UserToken': self.api_key
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def create_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
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
                        "incident_url": f"{self.instance_url}/nav_to.do?uri=incident.do?sys_id={result['result']['sys_id']}"
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
    
    async def close_session(self):
        if self.session:
            await self.session.close()
