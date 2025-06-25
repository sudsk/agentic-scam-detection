"""
Shared Base Agent Class for HSBC Scam Detection System
Provides common functionality for all specialized agents
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class MessageType(Enum):
    """Message type enumeration"""
    AUDIO_STREAM = "audio_stream"
    TRANSCRIPT = "transcript"
    FRAUD_ANALYSIS = "fraud_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    POLICY_GUIDANCE = "policy_guidance"
    CASE_UPDATE = "case_update"
    ALERT = "alert"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class Message:
    """Standard message format for inter-agent communication"""
    message_id: str
    message_type: str
    source_agent: str
    target_agent: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str = None
    data: Dict[str, Any] = None
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    ttl_seconds: Optional[int] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.data is None:
            self.data = {}
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create message from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string"""
        return cls.from_dict(json.loads(json_str))

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    processing_time_ms: int
    confidence_level: float

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    agent_id: str
    messages_processed: int = 0
    messages_failed: int = 0
    average_processing_time_ms: float = 0.0
    last_heartbeat: Optional[str] = None
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_rate: float = 0.0
    
    def update_processing_time(self, processing_time_ms: float):
        """Update average processing time"""
        total_processed = self.messages_processed + self.messages_failed
        if total_processed > 0:
            self.average_processing_time_ms = (
                (self.average_processing_time_ms * (total_processed - 1) + processing_time_ms)
                / total_processed
            )
        else:
            self.average_processing_time_ms = processing_time_ms
    
    def calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        total = self.messages_processed + self.messages_failed
        if total > 0:
            self.error_rate = self.messages_failed / total
        return self.error_rate

class BaseAgent(ABC):
    """
    Abstract base class for all HSBC Scam Detection agents
    Provides common functionality and interface
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        capabilities: List[AgentCapability],
        config: Optional[Dict] = None
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.capabilities = capabilities
        self.config = config or {}
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics(agent_id=agent_id)
        self.start_time = datetime.now()
        
        # Message handling
        self.message_queue = asyncio.Queue()
        self.processing_tasks = set()
        self.heartbeat_interval = self.config.get('heartbeat_interval', 30)
        
        # Session management
        self.active_sessions = {}
        self.session_data = {}
        
        # Error handling
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        self.logger.info(f"🤖 Agent {agent_id} ({agent_name}) initializing...")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            # Perform agent-specific initialization
            await self._initialize_agent()
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            # Start message processing
            asyncio.create_task(self._process_message_queue())
            
            self.status = AgentStatus.READY
            self.logger.info(f"✅ Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"❌ Failed to initialize agent {self.agent_id}: {e}")
            return False
    
    @abstractmethod
    async def _initialize_agent(self):
        """Agent-specific initialization logic"""
        pass
    
    @abstractmethod
    async def _process_message(self, message: Message) -> Optional[Message]:
        """Process a single message - implemented by each agent"""
        pass
    
    async def send_message(self, message: Message) -> bool:
        """Send a message to another agent or external system"""
        try:
            # Add message to queue for processing
            await self.message_queue.put(message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process an incoming message with error handling and metrics"""
        start_time = datetime.now()
        
        try:
            # Validate message
            if not self._validate_message(message):
                raise ValueError("Invalid message format")
            
            # Check if agent can handle this message type
            if not self._can_handle_message(message):
                self.logger.warning(f"Cannot handle message type: {message.message_type}")
                return self._create_error_response(message, "Unsupported message type")
            
            # Update session if needed
            if message.session_id:
                await self._update_session(message.session_id, message)
            
            # Process the message
            self.status = AgentStatus.PROCESSING
            response = await self._process_message_with_retry(message)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.messages_processed += 1
            self.metrics.update_processing_time(processing_time)
            
            self.status = AgentStatus.READY
            return response
            
        except Exception as e:
            # Update error metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.messages_failed += 1
            self.metrics.update_processing_time(processing_time)
            self.metrics.calculate_error_rate()
            
            self.logger.error(f"Error processing message {message.message_id}: {e}")
            self.status = AgentStatus.READY
            
            return self._create_error_response(message, str(e))
    
    async def _process_message_with_retry(self, message: Message) -> Optional[Message]:
        """Process message with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self._process_message(message)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All retry attempts failed for message {message.message_id}")
        
        raise last_exception
    
    def _validate_message(self, message: Message) -> bool:
        """Validate message format"""
        required_fields = ['message_id', 'message_type', 'source_agent', 'timestamp']
        return all(hasattr(message, field) and getattr(message, field) for field in required_fields)
    
    def _can_handle_message(self, message: Message) -> bool:
        """Check if agent can handle this message type"""
        # By default, check against capabilities
        for capability in self.capabilities:
            if message.message_type in capability.input_types:
                return True
        return False
    
    def _create_error_response(self, original_message: Message, error: str) -> Message:
        """Create standardized error response"""
        return Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR.value,
            source_agent=self.agent_id,
            target_agent=original_message.source_agent,
            session_id=original_message.session_id,
            correlation_id=original_message.message_id,
            data={
                'error': error,
                'original_message_id': original_message.message_id,
                'agent_id': self.agent_id
            }
        )
    
    async def _update_session(self, session_id: str, message: Message):
        """Update session data"""
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                'created_at': datetime.now().isoformat(),
                'messages': [],
                'metadata': {}
            }
        
        # Store message reference (not full message to save memory)
        self.session_data[session_id]['messages'].append({
            'message_id': message.message_id,
            'message_type': message.message_type,
            'timestamp': message.timestamp
        })
        
        # Keep only last 100 messages per session
        if len(self.session_data[session_id]['messages']) > 100:
            self.session_data[session_id]['messages'] = self.session_data[session_id]['messages'][-100:]
    
    async def _process_message_queue(self):
        """Process messages from the queue"""
        while True:
            try:
                # Get message from queue
                message = await self.message_queue.get()
                
                # Create task for processing
                task = asyncio.create_task(self.process_message(message))
                self.processing_tasks.add(task)
                
                # Clean up completed tasks
                task.add_done_callback(self.processing_tasks.discard)
                
            except Exception as e:
                self.logger.error(f"Error in message queue processing: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat"""
        while self.status != AgentStatus.SHUTDOWN:
            try:
                # Update metrics
                self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                self.metrics.last_heartbeat = datetime.now().isoformat()
                
                # Send heartbeat message
                heartbeat = Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT.value,
                    source_agent=self.agent_id,
                    data={
                        'status': self.status.value,
                        'metrics': asdict(self.metrics),
                        'active_sessions': len(self.active_sessions)
                    }
                )
                
                # In a real implementation, this would be sent to a monitoring service
                self.logger.debug(f"💓 Heartbeat from {self.agent_id}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        self.logger.info(f"🔄 Shutting down agent {self.agent_id}...")
        
        self.status = AgentStatus.SHUTDOWN
        
        # Wait for processing tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Perform agent-specific cleanup
        await self._cleanup_agent()
        
        self.logger.info(f"✅ Agent {self.agent_id} shutdown complete")
    
    async def _cleanup_agent(self):
        """Agent-specific cleanup logic"""
        pass
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'status': self.status.value,
            'capabilities': [asdict(cap) for cap in self.capabilities],
            'metrics': asdict(self.metrics),
            'active_sessions': len(self.active_sessions),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }
    
    def get_health(self) -> Dict:
        """Get agent health status"""
        is_healthy = (
            self.status in [AgentStatus.READY, AgentStatus.PROCESSING] and
            self.metrics.error_rate < 0.1 and  # Less than 10% error rate
            (datetime.now() - self.start_time).total_seconds() > 10  # Running for at least 10 seconds
        )
        
        return {
            'agent_id': self.agent_id,
            'healthy': is_healthy,
            'status': self.status.value,
            'error_rate': self.metrics.error_rate,
            'last_heartbeat': self.metrics.last_heartbeat,
            'uptime_seconds': self.metrics.uptime_seconds
        }
