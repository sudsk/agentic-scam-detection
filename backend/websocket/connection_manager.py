"""
WebSocket Connection Manager for HSBC Scam Detection Agent
Handles real-time communication with frontend clients
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, dict] = {}
        self.message_history: Dict[str, List[dict]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            'connected_at': datetime.now(),
            'messages_sent': 0,
            'messages_received': 0,
            'last_activity': datetime.now()
        }
        self.message_history[client_id] = []
        
        logger.info(f"✅ Client {client_id} connected. Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_personal_message(
            json.dumps({
                'type': 'connection_established',
                'data': {
                    'client_id': client_id,
                    'server_time': datetime.now().isoformat(),
                    'message': 'Connected to HSBC Scam Detection Agent System'
                }
            }),
            client_id
        )
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Keep metadata for analytics but mark as disconnected
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]['disconnected_at'] = datetime.now()
                session_duration = (
                    self.connection_metadata[client_id]['disconnected_at'] - 
                    self.connection_metadata[client_id]['connected_at']
                ).total_seconds()
                self.connection_metadata[client_id]['session_duration'] = session_duration
            
            logger.info(f"❌ Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(message)
                
                # Update metadata
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id]['messages_sent'] += 1
                    self.connection_metadata[client_id]['last_activity'] = datetime.now()
                
                # Store message in history (keep last 100 messages)
                if client_id in self.message_history:
                    self.message_history[client_id].append({
                        'type': 'sent',
                        'message': message[:200] + '...' if len(message) > 200 else message,
                        'timestamp': datetime.now().isoformat()
                    })
                    # Keep only last 100 messages
                    if len(self.message_history[client_id]) > 100:
                        self.message_history[client_id] = self.message_history[client_id][-100:]
                
                logger.debug(f"📤 Message sent to {client_id}")
                
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
        else:
            logger.warning(f"Attempted to send message to non-existent client: {client_id}")
    
    async def broadcast(self, message: str, exclude_clients: Optional[List[str]] = None):
        """Broadcast a message to all connected clients"""
        exclude_clients = exclude_clients or []
        
        if not self.active_connections:
            logger.info("No active connections for broadcast")
            return
        
        successful_sends = 0
        failed_sends = 0
        
        for client_id, websocket in list(self.active_connections.items()):
            if client_id not in exclude_clients:
                try:
                    await websocket.send_text(message)
                    successful_sends += 1
                    
                    # Update metadata
                    if client_id in self.connection_metadata:
                        self.connection_metadata[client_id]['messages_sent'] += 1
                        self.connection_metadata[client_id]['last_activity'] = datetime.now()
                        
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    self.disconnect(client_id)
                    failed_sends += 1
        
        logger.info(f"📡 Broadcast complete: {successful_sends} successful, {failed_sends} failed")
    
    async def send_system_alert(self, alert_type: str, message: str, priority: str = "normal"):
        """Send a system-wide alert to all connected clients"""
        alert_message = json.dumps({
            'type': 'system_alert',
            'data': {
                'alert_type': alert_type,
                'message': message,
                'priority': priority,
                'timestamp': datetime.now().isoformat()
            }
        })
        
        await self.broadcast(alert_message)
        logger.info(f"🚨 System alert sent: {alert_type} - {message}")
    
    async def send_fraud_alert(self, client_id: str, risk_score: float, scam_type: str):
        """Send a fraud alert to a specific client"""
        alert_message = json.dumps({
            'type': 'fraud_alert',
            'data': {
                'risk_score': risk_score,
                'scam_type': scam_type,
                'timestamp': datetime.now().isoformat(),
                'alert_level': 'HIGH' if risk_score >= 80 else 'MEDIUM' if risk_score >= 40 else 'LOW'
            }
        })
        
        await self.send_personal_message(alert_message, client_id)
        logger.warning(f"🚨 Fraud alert sent to {client_id}: {scam_type} - {risk_score}%")
    
    async def ping_clients(self):
        """Send ping to all clients to check connection health"""
        ping_message = json.dumps({
            'type': 'ping',
            'timestamp': datetime.now().isoformat()
        })
        
        await self.broadcast(ping_message)
        logger.debug(f"🏓 Ping sent to {len(self.active_connections)} clients")
    
    async def disconnect_all(self):
        """Disconnect all clients (used during shutdown)"""
        disconnection_message = json.dumps({
            'type': 'server_shutdown',
            'data': {
                'message': 'Server is shutting down',
                'timestamp': datetime.now().isoformat()
            }
        })
        
        # Send shutdown message to all clients
        await self.broadcast(disconnection_message)
        
        # Close all connections
        for client_id in list(self.active_connections.keys()):
            try:
                websocket = self.active_connections[client_id]
                await websocket.close(code=1000, reason="Server shutdown")
            except Exception as e:
                logger.error(f"Error closing connection for {client_id}: {e}")
            finally:
                self.disconnect(client_id)
        
        logger.info("🔌 All WebSocket connections closed")
    
    def get_connection_stats(self) -> dict:
        """Get statistics about active connections"""
        active_count = len(self.active_connections)
        total_messages_sent = sum(
            metadata.get('messages_sent', 0) 
            for metadata in self.connection_metadata.values()
        )
        total_messages_received = sum(
            metadata.get('messages_received', 0) 
            for metadata in self.connection_metadata.values()
        )
        
        # Calculate average session duration for disconnected clients
        disconnected_sessions = [
            metadata for metadata in self.connection_metadata.values()
            if 'disconnected_at' in metadata
        ]
        
        avg_session_duration = 0
        if disconnected_sessions:
            total_duration = sum(
                session.get('session_duration', 0) 
                for session in disconnected_sessions
            )
            avg_session_duration = total_duration / len(disconnected_sessions)
        
        return {
            'active_connections': active_count,
            'total_connections_ever': len(self.connection_metadata),
            'total_messages_sent': total_messages_sent,
            'total_messages_received': total_messages_received,
            'average_session_duration_seconds': avg_session_duration,
            'connection_details': {
                client_id: {
                    'connected_at': metadata['connected_at'].isoformat(),
                    'messages_sent': metadata.get('messages_sent', 0),
                    'messages_received': metadata.get('messages_received', 0),
                    'last_activity': metadata.get('last_activity', datetime.now()).isoformat(),
                    'is_active': client_id in self.active_connections
                }
                for client_id, metadata in self.connection_metadata.items()
            }
        }
    
    def get_client_history(self, client_id: str, limit: int = 50) -> List[dict]:
        """Get message history for a specific client"""
        if client_id in self.message_history:
            return self.message_history[client_id][-limit:]
        return []
    
    async def handle_client_message(self, client_id: str, message: str):
        """Handle incoming message from client and update metadata"""
        if client_id in self.connection_metadata:
            self.connection_metadata[client_id]['messages_received'] += 1
            self.connection_metadata[client_id]['last_activity'] = datetime.now()
        
        # Store message in history
        if client_id in self.message_history:
            self.message_history[client_id].append({
                'type': 'received',
                'message': message[:200] + '...' if len(message) > 200 else message,
                'timestamp': datetime.now().isoformat()
            })
            # Keep only last 100 messages
            if len(self.message_history[client_id]) > 100:
                self.message_history[client_id] = self.message_history[client_id][-100:]
        
        logger.debug(f"📥 Message received from {client_id}")
    
    async def cleanup_inactive_connections(self, timeout_seconds: int = 300):
        """Clean up connections that have been inactive for too long"""
        current_time = datetime.now()
        inactive_clients = []
        
        for client_id, metadata in self.connection_metadata.items():
            if client_id in self.active_connections:
                last_activity = metadata.get('last_activity', metadata['connected_at'])
                if (current_time - last_activity).total_seconds() > timeout_seconds:
                    inactive_clients.append(client_id)
        
        for client_id in inactive_clients:
            try:
                await self.send_personal_message(
                    json.dumps({
                        'type': 'timeout_warning',
                        'data': {
                            'message': 'Connection will be closed due to inactivity',
                            'timestamp': current_time.isoformat()
                        }
                    }),
                    client_id
                )
                
                # Wait a bit then disconnect
                await asyncio.sleep(5)
                if client_id in self.active_connections:
                    await self.active_connections[client_id].close(code=1000, reason="Inactive timeout")
                    self.disconnect(client_id)
                    
            except Exception as e:
                logger.error(f"Error cleaning up inactive connection {client_id}: {e}")
                self.disconnect(client_id)
        
        if inactive_clients:
            logger.info(f"🧹 Cleaned up {len(inactive_clients)} inactive connections")
    
    def is_client_connected(self, client_id: str) -> bool:
        """Check if a specific client is currently connected"""
        return client_id in self.active_connections
