# backend/websocket/__init__.py 

"""
WebSocket module for real-time communication
Provides connection management for WebSocket functionality
"""

from .connection_manager import ConnectionManager, WebSocketConnection

__all__ = [
    'ConnectionManager',
    'WebSocketConnection'
]
