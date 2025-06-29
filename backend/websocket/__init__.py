# backend/websocket/__init__.py 

"""
WebSocket module for real-time communication
Provides connection management and specialized handlers for fraud detection
"""

from .connection_manager import ConnectionManager, WebSocketConnection
from .fraud_detection_handler import FraudDetectionWebSocketHandler

__all__ = [
    'ConnectionManager',
    'WebSocketConnection', 
    'FraudDetectionWebSocketHandler'
]
