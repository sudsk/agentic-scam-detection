# backend/services/mock_database.py
"""
Centralized Mock Database Service
Consolidates all mock CRUD operations to eliminate redundancy
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json

from ..utils import (
    generate_uuid, generate_case_id, get_current_timestamp,
    calculate_percentage, safe_int_conversion
)
from ..api.models import CaseStatus, Priority, ScamType

logger = logging.getLogger(__name__)

@dataclass
class MockRecord:
    """Base mock database record"""
    id: str
    created_at: str = field(default_factory=get_current_timestamp)
    updated_at: str = field(default_factory=get_current_timestamp)
    data: Dict[str, Any] = field(default_factory=dict)

class MockDatabase:
    """
    Centralized mock database for development and testing
    Provides consistent CRUD operations across all API routes
    """
    
    def __init__(self):
        # Storage collections
        self.collections: Dict[str, Dict[str, MockRecord]] = defaultdict(dict)
        
        # Index storage for fast lookups
        self.indexes: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # Statistics
        self.stats = {
            "total_operations": 0,
            "operations_by_type": defaultdict(int),
            "collections_stats": defaultdict(lambda: {"count": 0, "last_access": None})
        }
        
        # Initialize with sample data
        self._initialize_sample_data()
        
        logger.info("🗄️ Mock Database initialized")
    
    def _initialize_sample_data(self):
        """Initialize database with sample data"""
        
        # Sample audio files
        audio_samples = [
            {
                "file_id": "audio_001",
                "filename": "investment_scam_live_call.wav",
                "size_bytes": 1024000,
                "duration_seconds": 45.0,
                "format": "wav",
                "session_id": "session_001",
                "status": "completed"
            },
            {
                "file_id": "audio_002", 
                "filename": "romance_scam_live_call.wav",
                "size_bytes": 856000,
                "duration_seconds": 38.5,
                "format": "wav",
                "session_id": "session_002", 
                "status": "completed"
            },
            {
                "file_id": "audio_003",
                "filename": "legitimate_call.wav", 
                "size_bytes": 612000,
                "duration_seconds": 25.0,
                "format": "wav",
                "session_id": "session_003",
                "status": "completed"
            }
        ]
        
        for audio in audio_samples:
            self.create("audio_files", audio["file_id"], audio)
        
        # Sample fraud cases
        case_samples = [
            {
                "case_id": "HSBC-FD-20240115-ABC123",
                "session_id": "session_001",
                "customer_id": "CUST001234", 
                "fraud_type": "investment_scam",
                "risk_score": 87.0,
                "description": "High-risk investment scam with guaranteed returns",
                "status": "open",
                "priority": "high",
                "assigned_to": "fraud.team@hsbc.com"
            },
            {
                "case_id": "HSBC-FD-20240115-DEF456",
                "session_id": "session_002",
                "customer_id": "CUST005678",
                "fraud_type": "romance_scam", 
                "risk_score": 72.0,
                "description": "Romance scam with overseas emergency request",
                "status": "investigating",
                "priority": "medium",
                "assigned_to": "fraud.specialist@hsbc.com"
            }
        ]
        
        for case in case_samples:
            self.create("fraud_cases", case["case_id"], case)
        
        # Sample fraud sessions
        session_samples = [
            {
                "session_id": "session_001",
                "customer_id": "CUST001234",
                "agent_id": "agent_001",
                "risk_score": 87.0,
                "scam_type": "investment_scam",
                "status": "completed"
            },
            {
                "session_id": "session_002", 
                "customer_id": "CUST005678",
                "agent_id": "agent_002",
                "risk_score": 72.0,
                "scam_type": "romance_scam", 
                "status": "completed"
            },
            {
                "session_id": "session_003",
                "customer_id": "CUST009876",
                "agent_id": "agent_001", 
                "risk_score": 12.0,
                "scam_type": "none",
                "status": "completed"
            }
        ]
        
        for session in session_samples:
            self.create("fraud_sessions", session["session_id"], session)
        
        logger.info(f"✅ Initialized mock database with sample data")
    
    def _update_stats(self, operation: str, collection: str):
        """Update operation statistics"""
        self.stats["total_operations"] += 1
        self.stats["operations_by_type"][operation] += 1
        self.stats["collections_stats"][collection]["last_access"] = get_current_timestamp()
    
    def _build_indexes(self, collection: str, record_id: str, data: Dict[str, Any]):
        """Build indexes for fast lookups"""
        # Index common fields
        indexable_fields = ["status", "priority", "customer_id", "session_id", "agent_id", "fraud_type"]
        
        for field in indexable_fields:
            if field in data:
                field_value = str(data[field]).lower()
                if record_id not in self.indexes[collection][field_value]:
                    self.indexes[collection][field_value].append(record_id)
    
    def _remove_from_indexes(self, collection: str, record_id: str, data: Dict[str, Any]):
        """Remove record from indexes"""
        for field_value, record_ids in self.indexes[collection].items():
            if record_id in record_ids:
                record_ids.remove(record_id)
    
    # ===== CRUD OPERATIONS =====
    
    def create(self, collection: str, record_id: str, data: Dict[str, Any]) -> MockRecord:
        """Create a new record"""
        self._update_stats("create", collection)
        
        record = MockRecord(
            id=record_id,
            data=data.copy()
        )
        
        self.collections[collection][record_id] = record
        self._build_indexes(collection, record_id, data)
        
        # Update collection stats
        self.stats["collections_stats"][collection]["count"] += 1
        
        logger.debug(f"📝 Created record in {collection}: {record_id}")
        return record
    
    def get(self, collection: str, record_id: str) -> Optional[MockRecord]:
        """Get a record by ID"""
        self._update_stats("get", collection)
        
        record = self.collections[collection].get(record_id)
        if record:
            logger.debug(f"📖 Retrieved record from {collection}: {record_id}")
        else:
            logger.debug(f"❌ Record not found in {collection}: {record_id}")
        
        return record
    
    def update(self, collection: str, record_id: str, updates: Dict[str, Any]) -> Optional[MockRecord]:
        """Update an existing record"""
        self._update_stats("update", collection)
        
        record = self.collections[collection].get(record_id)
        if not record:
            logger.warning(f"❌ Cannot update non-existent record in {collection}: {record_id}")
            return None
        
        # Remove from old indexes
        self._remove_from_indexes(collection, record_id, record.data)
        
        # Update data
        record.data.update(updates)
        record.updated_at = get_current_timestamp()
        
        # Rebuild indexes
        self._build_indexes(collection, record_id, record.data)
        
        logger.debug(f"📝 Updated record in {collection}: {record_id}")
        return record
    
    def delete(self, collection: str, record_id: str) -> bool:
        """Delete a record"""
        self._update_stats("delete", collection)
        
        record = self.collections[collection].get(record_id)
        if not record:
            logger.warning(f"❌ Cannot delete non-existent record in {collection}: {record_id}")
            return False
        
        # Remove from indexes
        self._remove_from_indexes(collection, record_id, record.data)
        
        # Delete record
        del self.collections[collection][record_id]
        
        # Update collection stats
        self.stats["collections_stats"][collection]["count"] -= 1
        
        logger.debug(f"🗑️ Deleted record from {collection}: {record_id}")
        return True
    
    def list_records(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: Optional[str] = None,
        sort_desc: bool = False
    ) -> Tuple[List[MockRecord], int]:
        """List records with filtering and pagination"""
        self._update_stats("list", collection)
        
        records = list(self.collections[collection].values())
        
        # Apply filters
        if filters:
            filtered_records = []
            for record in records:
                match = True
                for key, value in filters.items():
                    record_value = record.data.get(key)
                    if record_value != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            records = filtered_records
        
        total_count = len(records)
        
        # Apply sorting
        if sort_by:
            def sort_key(record):
                value = record.data.get(sort_by, record.created_at if sort_by == 'created_at' else '')
                return value if value is not None else ''
            
            records.sort(key=sort_key, reverse=sort_desc)
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_records = records[start_idx:end_idx]
        
        logger.debug(f"📋 Listed {len(paginated_records)}/{total_count} records from {collection}")
        return paginated_records, total_count
    
    def search(
        self,
        collection: str,
        query: str,
        search_fields: List[str] = None
    ) -> List[MockRecord]:
        """Search records by text query"""
        self._update_stats("search", collection)
        
        search_fields = search_fields or ["description", "filename", "text", "message"]
        query_lower = query.lower()
        results = []
        
        for record in self.collections[collection].values():
            for field in search_fields:
                field_value = record.data.get(field, "")
                if isinstance(field_value, str) and query_lower in field_value.lower():
                    results.append(record)
                    break
        
        logger.debug(f"🔍 Search in {collection} found {len(results)} results for: {query}")
        return results
    
    def count(self, collection: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filtering"""
        self._update_stats("count", collection)
        
        if not filters:
            count = len(self.collections[collection])
        else:
            count = 0
            for record in self.collections[collection].values():
                match = True
                for key, value in filters.items():
                    if record.data.get(key) != value:
                        match = False
                        break
                if match:
                    count += 1
        
        logger.debug(f"📊 Count in {collection}: {count}")
        return count
    
    def exists(self, collection: str, record_id: str) -> bool:
        """Check if record exists"""
        exists = record_id in self.collections[collection]
        logger.debug(f"❓ Record exists in {collection} {record_id}: {exists}")
        return exists
    
    # ===== SPECIALIZED OPERATIONS =====
    
    def get_by_field(self, collection: str, field: str, value: Any) -> List[MockRecord]:
        """Get records by field value using indexes"""
        self._update_stats("get_by_field", collection)
        
        field_value = str(value).lower()
        record_ids = self.indexes[collection].get(field_value, [])
        
        records = []
        for record_id in record_ids:
            record = self.collections[collection].get(record_id)
            if record:
                records.append(record)
        
        logger.debug(f"🔎 Found {len(records)} records in {collection} where {field}={value}")
        return records
    
    def get_recent(self, collection: str, limit: int = 10) -> List[MockRecord]:
        """Get most recent records"""
        records = list(self.collections[collection].values())
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]
    
    def bulk_create(self, collection: str, records_data: List[Dict[str, Any]]) -> List[MockRecord]:
        """Create multiple records"""
        created_records = []
        for data in records_data:
            record_id = data.get('id') or generate_uuid()
            record = self.create(collection, record_id, data)
            created_records.append(record)
        
        logger.info(f"📦 Bulk created {len(created_records)} records in {collection}")
        return created_records
    
    def bulk_update(self, collection: str, updates: Dict[str, Dict[str, Any]]) -> List[MockRecord]:
        """Update multiple records"""
        updated_records = []
        for record_id, update_data in updates.items():
            record = self.update(collection, record_id, update_data)
            if record:
                updated_records.append(record)
        
        logger.info(f"📦 Bulk updated {len(updated_records)} records in {collection}")
        return updated_records
    
    # ===== ANALYTICS OPERATIONS =====
    
    def aggregate(self, collection: str, group_by: str) -> Dict[str, int]:
        """Aggregate records by field"""
        self._update_stats("aggregate", collection)
        
        aggregation = defaultdict(int)
        for record in self.collections[collection].values():
            field_value = record.data.get(group_by, "unknown")
            aggregation[str(field_value)] += 1
        
        logger.debug(f"📊 Aggregated {collection} by {group_by}: {dict(aggregation)}")
        return dict(aggregation)
    
    def get_statistics(self, collection: str) -> Dict[str, Any]:
        """Get collection statistics"""
        records = list(self.collections[collection].values())
        total_count = len(records)
        
        if total_count == 0:
            return {"total_count": 0}
        
        # Calculate basic stats
        stats = {
            "total_count": total_count,
            "created_today": 0,
            "updated_today": 0,
            "oldest_record": None,
            "newest_record": None
        }
        
        today = datetime.now().date()
        oldest_time = None
        newest_time = None
        
        for record in records:
            # Parse timestamps
            try:
                created_time = datetime.fromisoformat(record.created_at.replace('Z', '+00:00'))
                updated_time = datetime.fromisoformat(record.updated_at.replace('Z', '+00:00'))
                
                # Count today's records
                if created_time.date() == today:
                    stats["created_today"] += 1
                if updated_time.date() == today:
                    stats["updated_today"] += 1
                
                # Track oldest and newest
                if oldest_time is None or created_time < oldest_time:
                    oldest_time = created_time
                    stats["oldest_record"] = record.created_at
                
                if newest_time is None or created_time > newest_time:
                    newest_time = created_time
                    stats["newest_record"] = record.created_at
                    
            except ValueError:
                continue
        
        # Add field-specific stats
        if collection == "fraud_cases":
            stats.update(self._get_case_statistics())
        elif collection == "fraud_sessions":
            stats.update(self._get_session_statistics())
        elif collection == "audio_files":
            stats.update(self._get_audio_statistics())
        
        return stats
    
    def _get_case_statistics(self) -> Dict[str, Any]:
        """Get fraud case specific statistics"""
        cases = list(self.collections["fraud_cases"].values())
        
        if not cases:
            return {}
        
        # Count by status
        status_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        fraud_type_counts = defaultdict(int)
        risk_scores = []
        
        for case in cases:
            data = case.data
            status_counts[data.get("status", "unknown")] += 1
            priority_counts[data.get("priority", "unknown")] += 1
            fraud_type_counts[data.get("fraud_type", "unknown")] += 1
            
            risk_score = data.get("risk_score")
            if risk_score is not None:
                risk_scores.append(float(risk_score))
        
        stats = {
            "by_status": dict(status_counts),
            "by_priority": dict(priority_counts),
            "by_fraud_type": dict(fraud_type_counts),
            "open_cases": status_counts.get("open", 0),
            "resolved_cases": status_counts.get("resolved", 0),
            "high_priority_cases": priority_counts.get("high", 0) + priority_counts.get("critical", 0)
        }
        
        if risk_scores:
            stats.update({
                "average_risk_score": sum(risk_scores) / len(risk_scores),
                "min_risk_score": min(risk_scores),
                "max_risk_score": max(risk_scores),
                "high_risk_cases": len([s for s in risk_scores if s >= 80])
            })
        
        return stats
    
    def _get_session_statistics(self) -> Dict[str, Any]:
        """Get fraud session specific statistics"""
        sessions = list(self.collections["fraud_sessions"].values())
        
        if not sessions:
            return {}
        
        # Count by scam type and status
        scam_type_counts = defaultdict(int)
        status_counts = defaultdict(int)
        risk_scores = []
        
        for session in sessions:
            data = session.data
            scam_type_counts[data.get("scam_type", "unknown")] += 1
            status_counts[data.get("status", "unknown")] += 1
            
            risk_score = data.get("risk_score")
            if risk_score is not None:
                risk_scores.append(float(risk_score))
        
        stats = {
            "by_scam_type": dict(scam_type_counts),
            "by_status": dict(status_counts),
            "fraud_detected": len([s for s in sessions if s.data.get("scam_type", "none") != "none"]),
            "completed_sessions": status_counts.get("completed", 0)
        }
        
        if risk_scores:
            stats.update({
                "average_risk_score": sum(risk_scores) / len(risk_scores),
                "fraud_detection_rate": calculate_percentage(
                    len([s for s in risk_scores if s >= 40]), 
                    len(risk_scores)
                )
            })
        
        return stats
    
    def _get_audio_statistics(self) -> Dict[str, Any]:
        """Get audio file specific statistics"""
        audio_files = list(self.collections["audio_files"].values())
        
        if not audio_files:
            return {}
        
        # Count by format and status
        format_counts = defaultdict(int)
        status_counts = defaultdict(int)
        total_size = 0
        total_duration = 0
        
        for audio in audio_files:
            data = audio.data
            format_counts[data.get("format", "unknown")] += 1
            status_counts[data.get("status", "unknown")] += 1
            
            size_bytes = data.get("size_bytes", 0)
            if size_bytes:
                total_size += int(size_bytes)
            
            duration = data.get("duration_seconds", 0)
            if duration:
                total_duration += float(duration)
        
        return {
            "by_format": dict(format_counts),
            "by_status": dict(status_counts),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_duration_minutes": round(total_duration / 60, 2),
            "completed_files": status_counts.get("completed", 0)
        }
    
    # ===== DATABASE MANAGEMENT =====
    
    def clear_collection(self, collection: str) -> int:
        """Clear all records from a collection"""
        count = len(self.collections[collection])
        self.collections[collection].clear()
        self.indexes[collection].clear()
        self.stats["collections_stats"][collection]["count"] = 0
        
        logger.warning(f"🧹 Cleared {count} records from {collection}")
        return count
    
    def drop_collection(self, collection: str) -> bool:
        """Drop entire collection"""
        if collection in self.collections:
            count = len(self.collections[collection])
            del self.collections[collection]
            del self.indexes[collection]
            del self.stats["collections_stats"][collection]
            
            logger.warning(f"🗑️ Dropped collection {collection} with {count} records")
            return True
        return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get overall database information"""
        total_records = sum(len(collection) for collection in self.collections.values())
        
        return {
            "total_collections": len(self.collections),
            "total_records": total_records,
            "collections": {
                name: {
                    "record_count": len(records),
                    "last_access": self.stats["collections_stats"][name]["last_access"]
                }
                for name, records in self.collections.items()
            },
            "operation_stats": dict(self.stats["operations_by_type"]),
            "total_operations": self.stats["total_operations"]
        }
    
    def backup_data(self) -> Dict[str, Any]:
        """Create a backup of all data"""
        backup = {
            "timestamp": get_current_timestamp(),
            "collections": {},
            "stats": self.stats
        }
        
        for collection_name, records in self.collections.items():
            backup["collections"][collection_name] = {
                record_id: {
                    "id": record.id,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "data": record.data
                }
                for record_id, record in records.items()
            }
        
        logger.info(f"💾 Created database backup")
        return backup
    
    def restore_data(self, backup: Dict[str, Any]) -> bool:
        """Restore data from backup"""
        try:
            # Clear existing data
            self.collections.clear()
            self.indexes.clear()
            
            # Restore collections
            for collection_name, records_data in backup["collections"].items():
                for record_id, record_data in records_data.items():
                    record = MockRecord(
                        id=record_data["id"],
                        created_at=record_data["created_at"],
                        updated_at=record_data["updated_at"],
                        data=record_data["data"]
                    )
                    self.collections[collection_name][record_id] = record
                    self._build_indexes(collection_name, record_id, record_data["data"])
            
            # Restore stats if available
            if "stats" in backup:
                self.stats.update(backup["stats"])
            
            logger.info(f"📥 Restored database from backup")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restore backup: {e}")
            return False

# ===== SPECIALIZED SERVICE CLASSES =====

class AudioFileService:
    """Specialized service for audio file operations"""
    
    def __init__(self, db: MockDatabase):
        self.db = db
        self.collection = "audio_files"
    
    def store_audio_metadata(self, file_id: str, metadata: Dict[str, Any]) -> MockRecord:
        """Store audio file metadata"""
        return self.db.create(self.collection, file_id, metadata)
    
    def get_audio_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get audio file metadata"""
        record = self.db.get(self.collection, file_id)
        return record.data if record else None
    
    def update_processing_status(self, file_id: str, status: str, progress: int = None) -> bool:
        """Update audio processing status"""
        updates = {"status": status}
        if progress is not None:
            updates["progress_percentage"] = progress
        
        record = self.db.update(self.collection, file_id, updates)
        return record is not None
    
    def get_files_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all audio files for a session"""
        records = self.db.get_by_field(self.collection, "session_id", session_id)
        return [record.data for record in records]

class CaseService:
    """Specialized service for case management operations"""
    
    def __init__(self, db: MockDatabase):
        self.db = db
        self.collection = "fraud_cases"
    
    def create_case(self, case_data: Dict[str, Any]) -> MockRecord:
        """Create a new fraud case"""
        case_id = case_data.get("case_id") or generate_case_id()
        return self.db.create(self.collection, case_id, case_data)
    
    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get case by ID"""
        record = self.db.get(self.collection, case_id)
        return record.data if record else None
    
    def update_case(self, case_id: str, updates: Dict[str, Any]) -> bool:
        """Update case"""
        record = self.db.update(self.collection, case_id, updates)
        return record is not None
    
    def get_cases_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get cases by status"""
        records = self.db.get_by_field(self.collection, "status", status)
        return [record.data for record in records]
    
    def get_cases_by_customer(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get cases by customer"""
        records = self.db.get_by_field(self.collection, "customer_id", customer_id)
        return [record.data for record in records]

class SessionService:
    """Specialized service for session management operations"""
    
    def __init__(self, db: MockDatabase):
        self.db = db
        self.collection = "fraud_sessions"
    
    def create_session(self, session_data: Dict[str, Any]) -> MockRecord:
        """Create a new session"""
        session_id = session_data.get("session_id") or f"session_{generate_uuid()[:8]}"
        return self.db.create(self.collection, session_id, session_data)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        record = self.db.get(self.collection, session_id)
        return record.data if record else None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session"""
        record = self.db.update(self.collection, session_id, updates)
        return record is not None
    
    def get_customer_sessions(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a customer"""
        records = self.db.get_by_field(self.collection, "customer_id", customer_id)
        return [record.data for record in records]

# ===== GLOBAL INSTANCE =====

# Create global mock database instance
mock_db = MockDatabase()

# Create specialized services
audio_service = AudioFileService(mock_db)
case_service = CaseService(mock_db)
session_service = SessionService(mock_db)

# ===== EXPORT ALL =====

__all__ = [
    'MockDatabase', 'MockRecord', 'AudioFileService', 'CaseService', 'SessionService',
    'mock_db', 'audio_service', 'case_service', 'session_service'
]
