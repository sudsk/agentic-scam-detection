# backend/websocket/connection_manager.py 

import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass

from ..utils import get_current_timestamp, log_agent_activity

logger = logging.getLogger(__name__)

@dataclass
class WebSocketConnection:
    """WebSocket connection data model"""
    websocket: WebSocket
    client_id: str
    connected_at: str
    last_ping: Optional[str] = None
    metadata: Optional[Dict] = None

class ConnectionManager:
    """
    WebSocket connection manager for handling multiple client connections
    Supports broadcasting, personal messaging, and connection lifecycle management
    """
    
    def __init__(self):
        self.active_connections: List[WebSocketConnection] = []
        self.client_connections: Dict[str, WebSocketConnection] = {}
        self.connection_count = 0
        self.total_connections = 0
        
        logger.info("🔌 ConnectionManager initialized")
    
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Optional[Dict] = None) -> bool:
        """
        Connect a new WebSocket client
        
        Args:
            websocket: WebSocket instance
            client_id: Unique client identifier
            metadata: Optional client metadata
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Accept the WebSocket connection
            await websocket.accept()
            
            # Check if client is already connected
            if client_id in self.client_connections:
                logger.warning(f"⚠️ Client {client_id} already connected, replacing connection")
                await self.disconnect(client_id)
            
            # Create connection object
            connection = WebSocketConnection(
                websocket=websocket,
                client_id=client_id,
                connected_at=get_current_timestamp(),
                metadata=metadata or {}
            )
            
            # Add to active connections
            self.active_connections.append(connection)
            self.client_connections[client_id] = connection
            self.connection_count += 1
            self.total_connections += 1
            
            logger.info(f"✅ WebSocket connected: {client_id} (Total: {self.connection_count})")
            log_agent_activity("websocket", f"Client connected: {client_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect WebSocket for {client_id}: {e}")
            return False
    
    def disconnect(self, client_id: str) -> bool:
        """
        Disconnect a WebSocket client
        
        Args:
            client_id: Client identifier to disconnect
            
        Returns:
            bool: True if disconnection successful, False if client not found
        """
        try:
            if client_id not in self.client_connections:
                logger.warning(f"⚠️ Attempted to disconnect unknown client: {client_id}")
                return False
            
            connection = self.client_connections[client_id]
            
            # Remove from active connections
            if connection in self.active_connections:
                self.active_connections.remove(connection)
            
            # Remove from client mapping
            del self.client_connections[client_id]
            self.connection_count -= 1
            
            logger.info(f"🔌 WebSocket disconnected: {client_id} (Remaining: {self.connection_count})")
            log_agent_activity("websocket", f"Client disconnected: {client_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting client {client_id}: {e}")
            return False
    
    async def disconnect_all(self) -> int:
        """
        Disconnect all active WebSocket connections
        
        Returns:
            int: Number of connections that were disconnected
        """
        disconnected_count = 0
        
        # Create a copy to avoid modification during iteration
        clients_to_disconnect = list(self.client_connections.keys())
        
        for client_id in clients_to_disconnect:
            try:
                connection = self.client_connections[client_id]
                await connection.websocket.close()
                self.disconnect(client_id)
                disconnected_count += 1
            except Exception as e:
                logger.error(f"❌ Error closing connection for {client_id}: {e}")
        
        logger.info(f"🧹 Disconnected all WebSocket connections: {disconnected_count}")
        return disconnected_count
    
    async def send_personal_message(self, message: str, client_id: str) -> bool:
        """
        Send a personal message to a specific client
        
        Args:
            message: Message to send
            client_id: Target client identifier
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            if client_id not in self.client_connections:
                logger.warning(f"⚠️ Cannot send message to unknown client: {client_id}")
                return False
            
            connection = self.client_connections[client_id]
            await connection.websocket.send_text(message)
            
            logger.debug(f"📤 Personal message sent to {client_id}")
            return True
            
        except WebSocketDisconnect:
            logger.warning(f"🔌 Client {client_id} disconnected during message send")
            self.disconnect(client_id)
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send personal message to {client_id}: {e}")
            return False
    
    async def send_personal_json(self, data: Dict, client_id: str) -> bool:
        """
        Send JSON data to a specific client
        
        Args:
            data: Dictionary to send as JSON
            client_id: Target client identifier
            
        Returns:
            bool: True if data sent successfully, False otherwise
        """
        try:
            message = json.dumps(data)
            return await self.send_personal_message(message, client_id)
        except Exception as e:
            logger.error(f"❌ Failed to send JSON to {client_id}: {e}")
            return False
    
    async def broadcast(self, message: str, exclude_clients: Optional[Set[str]] = None) -> int:
        """
        Broadcast a message to all connected clients
        
        Args:
            message: Message to broadcast
            exclude_clients: Set of client IDs to exclude from broadcast
            
        Returns:
            int: Number of clients that received the message
        """
        exclude_clients = exclude_clients or set()
        sent_count = 0
        failed_clients = []
        
        for connection in self.active_connections.copy():  # Copy to avoid modification during iteration
            if connection.client_id in exclude_clients:
                continue
                
            try:
                await connection.websocket.send_text(message)
                sent_count += 1
            except WebSocketDisconnect:
                logger.warning(f"🔌 Client {connection.client_id} disconnected during broadcast")
                failed_clients.append(connection.client_id)
            except Exception as e:
                logger.error(f"❌ Failed to broadcast to {connection.client_id}: {e}")
                failed_clients.append(connection.client_id)
        
        # Clean up failed connections
        for client_id in failed_clients:
            self.disconnect(client_id)
        
        logger.debug(f"📡 Broadcast sent to {sent_count} clients")
        return sent_count
    
    async def broadcast_json(self, data: Dict, exclude_clients: Optional[Set[str]] = None) -> int:
        """
        Broadcast JSON data to all connected clients
        
        Args:
            data: Dictionary to broadcast as JSON
            exclude_clients: Set of client IDs to exclude from broadcast
            
        Returns:
            int: Number of clients that received the data
        """
        try:
            message = json.dumps(data)
            return await self.broadcast(message, exclude_clients)
        except Exception as e:
            logger.error(f"❌ Failed to broadcast JSON: {e}")
            return 0
    
    def get_connection_count(self) -> int:
        """Get the current number of active connections"""
        return self.connection_count
    
    def get_total_connections(self) -> int:
        """Get the total number of connections since startup"""
        return self.total_connections
    
    def get_connected_clients(self) -> List[str]:
        """Get list of currently connected client IDs"""
        return list(self.client_connections.keys())
    
    def is_client_connected(self, client_id: str) -> bool:
        """Check if a specific client is currently connected"""
        return client_id in self.client_connections
    
    def get_client_info(self, client_id: str) -> Optional[Dict]:
        """
        Get information about a specific client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dict with client information or None if not found
        """
        if client_id not in self.client_connections:
            return None
        
        connection = self.client_connections[client_id]
        return {
            "client_id": client_id,
            "connected_at": connection.connected_at,
            "last_ping": connection.last_ping,
            "metadata": connection.metadata
        }
    
    def get_connection_stats(self) -> Dict:
        """Get detailed connection statistics"""
        return {
            "active_connections": self.connection_count,
            "total_connections_since_startup": self.total_connections,
            "connected_clients": self.get_connected_clients(),
            "manager_status": "operational",
            "last_updated": get_current_timestamp()
        }
    
    async def ping_all_clients(self) -> Dict[str, bool]:
        """
        Send ping to all connected clients to check connection health
        
        Returns:
            Dict mapping client_id to ping success status
        """
        ping_results = {}
        ping_message = json.dumps({
            "type": "ping",
            "timestamp": get_current_timestamp(),
            "server": "connection_manager"
        })
        
        for client_id in list(self.client_connections.keys()):
            try:
                success = await self.send_personal_message(ping_message, client_id)
                ping_results[client_id] = success
                
                if success:
                    # Update last ping time
                    self.client_connections[client_id].last_ping = get_current_timestamp()
                    
            except Exception as e:
                logger.error(f"❌ Ping failed for {client_id}: {e}")
                ping_results[client_id] = False
        
        return ping_results
    
    async def cleanup_stale_connections(self, max_age_minutes: int = 60) -> int:
        """
        Clean up stale connections that haven't been active
        
        Args:
            max_age_minutes: Maximum age in minutes before connection is considered stale
            
        Returns:
            int: Number of connections cleaned up
        """
        cleanup_count = 0
        current_time = datetime.now()
        stale_clients = []
        
        for client_id, connection in self.client_connections.items():
            try:
                connected_time = datetime.fromisoformat(connection.connected_at.replace('Z', '+00:00'))
                age_minutes = (current_time - connected_time).total_seconds() / 60
                
                if age_minutes > max_age_minutes:
                    stale_clients.append(client_id)
                    
            except Exception as e:
                logger.error(f"❌ Error checking connection age for {client_id}: {e}")
                stale_clients.append(client_id)  # Clean up problematic connections
        
        # Clean up stale connections
        for client_id in stale_clients:
            try:
                await self.client_connections[client_id].websocket.close()
                self.disconnect(client_id)
                cleanup_count += 1
            except Exception as e:
                logger.error(f"❌ Error cleaning up stale connection {client_id}: {e}")
        
        if cleanup_count > 0:
            logger.info(f"🧹 Cleaned up {cleanup_count} stale WebSocket connections")
        
        return cleanup_count

# Export the connection manager class
__all__ = ['ConnectionManager', 'WebSocketConnection']
