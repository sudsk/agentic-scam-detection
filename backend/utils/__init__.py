# backend/utils/__init__.py
"""
Shared utilities for HSBC Fraud Detection System
Consolidates common helper functions to eliminate redundancy
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import re

logger = logging.getLogger(__name__)

# ===== DATE/TIME UTILITIES =====

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def get_current_date_string() -> str:
    """Get current date in YYYYMMDD format"""
    return datetime.now().strftime("%Y%m%d")

def format_duration(start_time: datetime, end_time: Optional[datetime] = None) -> float:
    """Calculate duration in seconds between timestamps"""
    end_time = end_time or datetime.now()
    return (end_time - start_time).total_seconds()

def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime object"""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        return datetime.now()

# ===== ID GENERATION UTILITIES =====

def generate_uuid() -> str:
    """Generate a standard UUID4 string"""
    return str(uuid.uuid4())

def generate_short_id(length: int = 8) -> str:
    """Generate a short uppercase ID"""
    return str(uuid.uuid4())[:length].upper()

def generate_session_id(prefix: str = "session") -> str:
    """Generate a session ID with timestamp"""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generate_short_id(6)}"

def generate_case_id(prefix: str = "FD") -> str:
    """Generate a case ID with date and unique identifier"""
    date_str = get_current_date_string()
    unique_id = generate_short_id(8)
    return f"{prefix}-{date_str}-{unique_id}"

# ===== RESPONSE FORMATTING UTILITIES =====

def create_success_response(
    data: Any = None, 
    message: str = "Success",
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": timestamp or get_current_timestamp()
    }

def create_error_response(
    error: str,
    code: Optional[str] = None,
    details: Optional[Dict] = None,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "status": "error",
        "error": error,
        "code": code,
        "details": details or {},
        "timestamp": timestamp or get_current_timestamp()
    }

def create_processing_response(
    session_id: str,
    status: str = "processing",
    progress: Optional[int] = None,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized processing response"""
    response = {
        "session_id": session_id,
        "status": status,
        "timestamp": get_current_timestamp()
    }
    
    if progress is not None:
        response["progress_percentage"] = progress
    if message:
        response["message"] = message
        
    return response

# ===== FILE UTILITIES =====

def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension against allowed list"""
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower()
    return file_extension in [ext.lower() for ext in allowed_extensions]

def get_file_size_mb(file_size_bytes: int) -> float:
    """Convert file size from bytes to MB"""
    return file_size_bytes / (1024 * 1024)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove unsafe characters
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    # Ensure it doesn't start with dots or dashes
    sanitized = re.sub(r'^[\.\-]+', '', sanitized)
    return sanitized[:255]  # Limit length

# ===== VALIDATION UTILITIES =====

def validate_risk_score(score: float) -> bool:
    """Validate risk score is between 0 and 100"""
    return 0 <= score <= 100

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format"""
    if not session_id or len(session_id) < 5:
        return False
    return re.match(r'^[a-zA-Z0-9_\-]+$', session_id) is not None

def validate_email(email: str) -> bool:
    """Basic email validation"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None

# ===== CONFIGURATION UTILITIES =====

def get_priority_from_risk_score(risk_score: float, thresholds: Dict[str, float]) -> str:
    """Determine priority based on risk score and thresholds"""
    if risk_score >= thresholds.get("critical", 80):
        return "critical"
    elif risk_score >= thresholds.get("high", 60):
        return "high"
    elif risk_score >= thresholds.get("medium", 40):
        return "medium"
    else:
        return "low"

def get_risk_level_from_score(risk_score: float) -> str:
    """Get risk level string from numeric score"""
    if risk_score >= 80:
        return "CRITICAL"
    elif risk_score >= 60:
        return "HIGH"
    elif risk_score >= 40:
        return "MEDIUM"
    elif risk_score >= 20:
        return "LOW"
    else:
        return "MINIMAL"

# ===== LOGGING UTILITIES =====

def log_agent_activity(
    agent_id: str, 
    activity: str, 
    details: Optional[Dict] = None,
    level: str = "info"
) -> None:
    """Standardized agent activity logging"""
    log_message = f"🤖 {agent_id}: {activity}"
    if details:
        log_message += f" - {details}"
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(log_message)

def log_performance_metric(
    component: str,
    metric_name: str,
    value: float,
    unit: str = ""
) -> None:
    """Log performance metrics consistently"""
    logger.info(f"📊 {component} - {metric_name}: {value}{unit}")

# ===== DATA PROCESSING UTILITIES =====

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with default"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value: Any, default: int = 0) -> int:
    """Safely convert value to int with default"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text (basic implementation)"""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [word for word in words if len(word) >= min_length]

# ===== STATISTICAL UTILITIES =====

def calculate_percentage(part: float, total: float, decimal_places: int = 2) -> float:
    """Calculate percentage with safe division"""
    if total == 0:
        return 0.0
    return round((part / total) * 100, decimal_places)

def calculate_average(values: List[float]) -> float:
    """Calculate average with safe division"""
    if not values:
        return 0.0
    return sum(values) / len(values)

def calculate_success_rate(successful: int, total: int) -> float:
    """Calculate success rate as percentage"""
    return calculate_percentage(successful, total)

# ===== CACHE KEY UTILITIES =====

def generate_cache_key(*parts: str) -> str:
    """Generate consistent cache key from parts"""
    sanitized_parts = [str(part).replace(':', '_').replace(' ', '_') for part in parts]
    return ':'.join(sanitized_parts)

def generate_session_cache_key(session_id: str, data_type: str) -> str:
    """Generate cache key for session data"""
    return generate_cache_key("session", session_id, data_type)

# ===== EXPORT ALL UTILITIES =====

__all__ = [
    # Time utilities
    'get_current_timestamp', 'get_current_date_string', 'format_duration', 'parse_iso_timestamp',
    
    # ID generation
    'generate_uuid', 'generate_short_id', 'generate_session_id', 'generate_case_id',
    
    # Response formatting
    'create_success_response', 'create_error_response', 'create_processing_response',
    
    # File utilities
    'validate_file_extension', 'get_file_size_mb', 'sanitize_filename',
    
    # Validation
    'validate_risk_score', 'validate_session_id', 'validate_email',
    
    # Configuration
    'get_priority_from_risk_score', 'get_risk_level_from_score',
    
    # Logging
    'log_agent_activity', 'log_performance_metric',
    
    # Data processing
    'safe_float_conversion', 'safe_int_conversion', 'truncate_text', 'extract_keywords',
    
    # Statistics
    'calculate_percentage', 'calculate_average', 'calculate_success_rate',
    
    # Cache
    'generate_cache_key', 'generate_session_cache_key'
]
