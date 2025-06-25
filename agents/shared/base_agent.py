import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from google.cloud.adk.v2 import Agent, AgentConfig, Message, Tool
from google.cloud.adk.v2.types import AgentCapability, MemoryType
import uuid

logger = logging.getLogger(__name__)

class BaseScamDetectionAgent(Agent, ABC):
    """
    Base class for all Scam Detection agents using latest ADK v2.0
    """
    
    def __init__(self, agent_id: str, capabilities: List[str], memory_type: str = "contextual"):
        # Configure agent with latest ADK v2.0
        config = AgentConfig(
            agent_id=agent_id,
            agent_name=f"ScamDetection_{agent_id}",
            capabilities=[AgentCapability(name=cap) for cap in capabilities],
            memory_type=MemoryType(memory_type),
            version="2.0.0",
            runtime_config={
                "max_memory_size": "1GB",
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "enable_logging": True,
                "enable_monitoring": True
            }
        )
        
        super().__init__(config)
        
        # Agent state management
        self.session_data = {}
        self.performance_metrics = {
            "messages_processed": 0,
            "errors_count": 0,
            "avg_response_time": 0.0,
            "last_updated": datetime.now()
        }
        
        # Initialize agent-specific tools
        self.tools = self._initialize_tools()
        
        logger.info(f"✅ {agent_id} initialized with ADK v2.0")
    
    @abstractmethod
    def _initialize_tools(self) -> List[Tool]:
        """Initialize agent-specific tools"""
        pass
    
    @abstractmethod
    async def _process_message_impl(self, message: Message) -> Message:
        """Agent-specific message processing implementation"""
        pass
    
    async def process_message(self, message: Message) -> Message:
        """
        Main message processing with monitoring and error handling
        """
        start_time = datetime.now()
        
        try:
            # Update metrics
            self.performance_metrics["messages_processed"] += 1
            
            # Process message
            response = await self._process_message_impl(message)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_response_time(processing_time)
            
            # Log successful processing
            logger.debug(f"{self.config.agent_id} processed message in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            self.performance_metrics["errors_count"] += 1
            logger.error(f"Error in {self.config.agent_id}: {str(e)}")
            
            return self._create_error_response(message, str(e))
    
    def _update_response_time(self, processing_time: float):
        """Update average response time"""
        current_avg = self.performance_metrics["avg_response_time"]
        count = self.performance_metrics["messages_processed"]
        
        new_avg = ((current_avg * (count - 1)) + processing_time) / count
        self.performance_metrics["avg_response_time"] = new_avg
        self.performance_metrics["last_updated"] = datetime.now()
    
    def _create_error_response(self, original_message: Message, error: str) -> Message:
        """Create standardized error response"""
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="error",
            data={
                "error": error,
                "original_message_id": original_message.message_id,
                "agent_id": self.config.agent_id,
                "timestamp": datetime.now().isoformat()
            },
            source_agent=self.config.agent_id
        )
    
    def get_performance_metrics(self) -> Dict:
        """Get agent performance metrics"""
        return self.performance_metrics.copy()
