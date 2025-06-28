# agents/shared/base_agent.py
"""
Base Agent Class - Common functionality for all fraud detection agents
Provides standard interface and shared utilities
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class AgentCapability(Enum):
    """Agent capability enumeration"""
    AUDIO_PROCESSING = "audio_processing"
    FRAUD_DETECTION = "fraud_detection"
    POLICY_GUIDANCE = "policy_guidance"
    CASE_MANAGEMENT = "case_management"
    REAL_TIME_PROCESSING = "real_time_processing"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_SCORING = "risk_scoring"
    ESCALATION_MANAGEMENT = "escalation_management"

class BaseAgent(ABC):
    """
    Base class for all fraud detection agents
    Provides common functionality and enforces interface consistency
    """
    
    def __init__(self, agent_type: str, agent_name: str):
        """
        Initialize base agent
        
        Args:
            agent_type: Type identifier for the agent
            agent_name: Human-readable name for the agent
        """
        self.agent_id = f"{agent_type}_{str(uuid.uuid4())[:8]}"
        self.agent_type = agent_type
        self.agent_name = agent_name
        self.status = AgentStatus.ACTIVE
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "last_processing_time": 0.0
        }
        
        # Configuration
        self.config = self._get_default_config()
        
        # Capabilities
        self.capabilities = self._get_capabilities()
        
        logger.info(f"✅ {self.agent_name} ({self.agent_id}) initialized")
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the agent"""
        pass
    
    @abstractmethod
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get list of agent capabilities"""
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Main processing method - must be implemented by each agent
        
        Args:
            input_data: Input data to process
            
        Returns:
            Dict containing processing results
        """
        pass
    
    def update_metrics(self, processing_time: float, success: bool = True):
        """Update agent performance metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["last_processing_time"] = processing_time
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update average processing time
        total_successful = self.metrics["successful_requests"]
        if total_successful > 0:
            current_avg = self.metrics["average_processing_time"]
            self.metrics["average_processing_time"] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
        
        self.last_activity = datetime.now()
    
    def set_status(self, status: AgentStatus, reason: Optional[str] = None):
        """Set agent status"""
        old_status = self.status
        self.status = status
        self.last_activity = datetime.now()
        
        logger.info(f"🔄 {self.agent_name} status changed: {old_status.value} → {status.value}")
        if reason:
            logger.info(f"   Reason: {reason}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update agent configuration"""
        self.config.update(new_config)
        self.last_activity = datetime.now()
        logger.info(f"⚙️ {self.agent_name} configuration updated")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "config": self.config,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds()
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get agent health status"""
        is_healthy = (
            self.status == AgentStatus.ACTIVE and
            (datetime.now() - self.last_activity).total_seconds() < 3600  # Active within last hour
        )
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "healthy": is_healthy,
            "status": self.status.value,
            "last_activity": self.last_activity.isoformat(),
            "error_rate": (
                self.metrics["failed_requests"] / max(self.metrics["total_requests"], 1)
            ),
            "timestamp": datetime.now().isoformat()
        }
    
    def log_activity(self, activity: str, details: Optional[Dict] = None):
        """Log agent activity"""
        self.last_activity = datetime.now()
        
        log_message = f"🔍 {self.agent_name}: {activity}"
        if details:
            log_message += f" - Details: {details}"
        
        logger.info(log_message)
    
    def handle_error(self, error: Exception, context: Optional[str] = None):
        """Handle and log errors consistently"""
        error_msg = f"❌ {self.agent_name} error"
        if context:
            error_msg += f" in {context}"
        error_msg += f": {str(error)}"
        
        logger.error(error_msg)
        self.metrics["failed_requests"] += 1
        self.last_activity = datetime.now()
        
        # Set status to error if too many failures
        error_rate = self.metrics["failed_requests"] / max(self.metrics["total_requests"], 1)
        if error_rate > 0.5 and self.metrics["total_requests"] > 10:
            self.set_status(AgentStatus.ERROR, f"High error rate: {error_rate:.2%}")
    
    def reset_metrics(self):
        """Reset agent metrics"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "last_processing_time": 0.0
        }
        logger.info(f"📊 {self.agent_name} metrics reset")
    
    def __str__(self) -> str:
        return f"{self.agent_name} ({self.agent_id}) - Status: {self.status.value}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.agent_id}>"

class ProcessingResult:
    """Standard result class for agent processing"""
    
    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        processing_time: Optional[float] = None,
        agent_id: Optional[str] = None
    ):
        self.success = success
        self.data = data or {}
        self.error = error
        self.processing_time = processing_time
        self.agent_id = agent_id
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "processing_time": self.processing_time,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp
        }

class AgentRegistry:
    """Registry for managing all agents in the system"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, List[str]] = {}
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.agent_id] = agent
        
        if agent.agent_type not in self.agent_types:
            self.agent_types[agent.agent_type] = []
        self.agent_types[agent.agent_type].append(agent.agent_id)
        
        logger.info(f"📝 Registered agent: {agent.agent_name} ({agent.agent_id})")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            del self.agents[agent_id]
            
            if agent.agent_type in self.agent_types:
                self.agent_types[agent.agent_type].remove(agent_id)
                if not self.agent_types[agent.agent_type]:
                    del self.agent_types[agent.agent_type]
            
            logger.info(f"🗑️ Unregistered agent: {agent.agent_name} ({agent_id})")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get all agents of a specific type"""
        agent_ids = self.agent_types.get(agent_type, [])
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents"""
        return list(self.agents.values())
    
    def get_healthy_agents(self) -> List[BaseAgent]:
        """Get all healthy agents"""
        return [agent for agent in self.agents.values() if agent.get_health_check()["healthy"]]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_agents = len(self.agents)
        healthy_agents = len(self.get_healthy_agents())
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "agent_types": len(self.agent_types),
            "system_healthy": healthy_agents == total_agents,
            "agents_by_type": {
                agent_type: len(agent_ids) 
                for agent_type, agent_ids in self.agent_types.items()
            },
            "timestamp": datetime.now().isoformat()
        }

# Global agent registry
agent_registry = AgentRegistry()
