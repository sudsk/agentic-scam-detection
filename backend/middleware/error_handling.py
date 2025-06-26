# backend/middleware/error_handling.py
"""
Centralized error handling middleware and decorators
Eliminates redundant error handling patterns across API routes
"""

import logging
import traceback
from functools import wraps
from typing import Callable, Any, Dict, Optional
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..utils import create_error_response, get_current_timestamp
from ..api.models import ErrorResponse

logger = logging.getLogger(__name__)

# ===== CUSTOM EXCEPTIONS =====

class FraudDetectionError(Exception):
    """Base exception for fraud detection system"""
    def __init__(self, message: str, code: str = None, details: Dict = None):
        self.message = message
        self.code = code or "FRAUD_DETECTION_ERROR"
        self.details = details or {}
        super().__init__(self.message)

class AgentProcessingError(FraudDetectionError):
    """Exception for agent processing errors"""
    def __init__(self, agent_id: str, message: str, details: Dict = None):
        self.agent_id = agent_id
        super().__init__(
            message=f"Agent {agent_id}: {message}",
            code="AGENT_PROCESSING_ERROR",
            details={"agent_id": agent_id, **(details or {})}
        )

class ValidationError(FraudDetectionError):
    """Exception for validation errors"""
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(
            message=f"Validation error for {field}: {message}",
            code="VALIDATION_ERROR",
            details={"field": field, "value": str(value), "message": message}
        )

class ConfigurationError(FraudDetectionError):
    """Exception for configuration errors"""
    def __init__(self, component: str, message: str):
        super().__init__(
            message=f"Configuration error in {component}: {message}",
            code="CONFIGURATION_ERROR",
            details={"component": component}
        )

class ExternalServiceError(FraudDetectionError):
    """Exception for external service errors"""
    def __init__(self, service: str, message: str, status_code: int = None):
        self.service = service
        self.status_code = status_code
        super().__init__(
            message=f"External service {service}: {message}",
            code="EXTERNAL_SERVICE_ERROR",
            details={"service": service, "status_code": status_code}
        )

# ===== ERROR MAPPING =====

ERROR_STATUS_MAP = {
    "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
    "AGENT_PROCESSING_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "CONFIGURATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "EXTERNAL_SERVICE_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
    "FRAUD_DETECTION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "FILE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "PERMISSION_DENIED": status.HTTP_403_FORBIDDEN,
    "RATE_LIMIT_EXCEEDED": status.HTTP_429_TOO_MANY_REQUESTS
}

# ===== MIDDLEWARE =====

async def error_handling_middleware(request: Request, call_next: Callable) -> Response:
    """
    Global error handling middleware for FastAPI
    Catches and standardizes all unhandled exceptions
    """
    try:
        response = await call_next(request)
        return response
        
    except HTTPException as e:
        # Let FastAPI handle HTTPExceptions normally
        raise e
        
    except FraudDetectionError as e:
        # Handle our custom exceptions
        logger.error(f"FraudDetectionError: {e.message}", extra={
            "code": e.code,
            "details": e.details,
            "path": request.url.path,
            "method": request.method
        })
        
        status_code = ERROR_STATUS_MAP.get(e.code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        error_response = create_error_response(
            error=e.message,
            code=e.code,
            details=e.details
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
        
    except ValidationError as e:
        # Handle Pydantic validation errors
        logger.error(f"Validation error: {str(e)}", extra={
            "path": request.url.path,
            "method": request.method
        })
        
        error_response = create_error_response(
            error="Validation failed",
            code="VALIDATION_ERROR",
            details={"validation_errors": str(e)}
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response
        )
        
    except Exception as e:
        # Handle all other unexpected exceptions
        logger.error(f"Unhandled exception: {str(e)}", extra={
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc()
        })
        
        error_response = create_error_response(
            error="Internal server error",
            code="INTERNAL_SERVER_ERROR",
            details={"exception_type": type(e).__name__}
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )

# ===== DECORATORS =====

def handle_agent_errors(agent_name: str = None):
    """
    Decorator for handling agent processing errors
    Standardizes error responses and logging for agent operations
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = get_current_timestamp()
            agent_id = agent_name or func.__name__
            
            try:
                logger.info(f"🤖 {agent_id}: Starting processing")
                result = await func(*args, **kwargs)
                logger.info(f"✅ {agent_id}: Processing completed successfully")
                return result
                
            except FraudDetectionError:
                # Re-raise our custom exceptions
                raise
                
            except Exception as e:
                logger.error(f"❌ {agent_id}: Processing failed - {str(e)}")
                raise AgentProcessingError(
                    agent_id=agent_id,
                    message=str(e),
                    details={
                        "function": func.__name__,
                        "start_time": start_time,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = get_current_timestamp()
            agent_id = agent_name or func.__name__
            
            try:
                logger.info(f"🤖 {agent_id}: Starting processing")
                result = func(*args, **kwargs)
                logger.info(f"✅ {agent_id}: Processing completed successfully")
                return result
                
            except FraudDetectionError:
                # Re-raise our custom exceptions
                raise
                
            except Exception as e:
                logger.error(f"❌ {agent_id}: Processing failed - {str(e)}")
                raise AgentProcessingError(
                    agent_id=agent_id,
                    message=str(e),
                    details={
                        "function": func.__name__,
                        "start_time": start_time,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                )
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def handle_api_errors(operation_name: str = None):
    """
    Decorator for handling API route errors
    Provides consistent error responses for FastAPI routes
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            operation = operation_name or func.__name__
            
            try:
                logger.debug(f"🌐 API: {operation} - Starting")
                result = await func(*args, **kwargs)
                logger.debug(f"✅ API: {operation} - Completed")
                return result
                
            except HTTPException:
                # Re-raise FastAPI HTTPExceptions
                raise
                
            except FraudDetectionError as e:
                logger.error(f"❌ API: {operation} - FraudDetectionError: {e.message}")
                status_code = ERROR_STATUS_MAP.get(e.code, status.HTTP_500_INTERNAL_SERVER_ERROR)
                raise HTTPException(
                    status_code=status_code,
                    detail={
                        "error": e.message,
                        "code": e.code,
                        "details": e.details,
                        "operation": operation,
                        "timestamp": get_current_timestamp()
                    }
                )
                
            except ValidationError as e:
                logger.error(f"❌ API: {operation} - Validation error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "Validation failed",
                        "code": "VALIDATION_ERROR",
                        "details": {"validation_errors": str(e)},
                        "operation": operation,
                        "timestamp": get_current_timestamp()
                    }
                )
                
            except Exception as e:
                logger.error(f"❌ API: {operation} - Unexpected error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Internal server error",
                        "code": "INTERNAL_SERVER_ERROR",
                        "details": {"exception_type": type(e).__name__},
                        "operation": operation,
                        "timestamp": get_current_timestamp()
                    }
                )
        
        return wrapper
    return decorator

def validate_input(**validation_rules):
    """
    Decorator for input validation
    Provides consistent validation with custom error messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Validate inputs based on rules
            for param_name, rules in validation_rules.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    
                    # Check required
                    if rules.get('required', False) and value is None:
                        raise ValidationError(param_name, "Field is required")
                    
                    # Check type
                    if value is not None and 'type' in rules:
                        expected_type = rules['type']
                        if not isinstance(value, expected_type):
                            raise ValidationError(
                                param_name, 
                                f"Expected {expected_type.__name__}, got {type(value).__name__}",
                                value
                            )
                    
                    # Check min/max for numbers
                    if isinstance(value, (int, float)):
                        if 'min' in rules and value < rules['min']:
                            raise ValidationError(param_name, f"Value must be >= {rules['min']}", value)
                        if 'max' in rules and value > rules['max']:
                            raise ValidationError(param_name, f"Value must be <= {rules['max']}", value)
                    
                    # Check length for strings
                    if isinstance(value, str):
                        if 'min_length' in rules and len(value) < rules['min_length']:
                            raise ValidationError(param_name, f"Length must be >= {rules['min_length']}", value)
                        if 'max_length' in rules and len(value) > rules['max_length']:
                            raise ValidationError(param_name, f"Length must be <= {rules['max_length']}", value)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def require_session_id(func: Callable) -> Callable:
    """
    Decorator to ensure session_id is provided and valid
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        session_id = kwargs.get('session_id')
        
        if not session_id:
            raise ValidationError("session_id", "Session ID is required")
        
        if not isinstance(session_id, str) or len(session_id) < 5:
            raise ValidationError("session_id", "Invalid session ID format", session_id)
        
        return await func(*args, **kwargs)
    
    return wrapper

def log_performance(operation_name: str = None):
    """
    Decorator to log performance metrics
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import time
            operation = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"📊 Performance: {operation} completed in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.warning(f"📊 Performance: {operation} failed after {duration:.3f}s")
                raise
        
        return wrapper
    return decorator

# ===== UTILITY FUNCTIONS =====

def create_http_exception(
    status_code: int,
    message: str,
    code: str = None,
    details: Dict = None
) -> HTTPException:
    """Create standardized HTTPException"""
    return HTTPException(
        status_code=status_code,
        detail={
            "error": message,
            "code": code or "HTTP_ERROR",
            "details": details or {},
            "timestamp": get_current_timestamp()
        }
    )

def handle_external_service_error(
    service_name: str,
    error: Exception,
    fallback_message: str = None
) -> ExternalServiceError:
    """Handle errors from external services consistently"""
    message = fallback_message or f"Service unavailable: {str(error)}"
    
    status_code = None
    if hasattr(error, 'status_code'):
        status_code = error.status_code
    elif hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        status_code = error.response.status_code
    
    return ExternalServiceError(
        service=service_name,
        message=message,
        status_code=status_code
    )

def safe_execute(
    func: Callable,
    *args,
    fallback_value: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling
    Returns fallback_value if function fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"Safe execution failed for {func.__name__}: {str(e)}")
        return fallback_value

# ===== CONTEXTUAL ERROR HANDLERS =====

class ErrorContext:
    """Context manager for enhanced error handling"""
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = get_current_timestamp()
        logger.debug(f"🔧 Starting: {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logger.debug(f"✅ Completed: {self.operation}")
        else:
            logger.error(f"❌ Failed: {self.operation} - {str(exc_val)}", extra={
                **self.context,
                "exception_type": exc_type.__name__,
                "start_time": self.start_time
            })
        return False  # Don't suppress exceptions

# ===== EXPORT ALL =====

__all__ = [
    # Exceptions
    'FraudDetectionError', 'AgentProcessingError', 'ValidationError', 
    'ConfigurationError', 'ExternalServiceError',
    
    # Middleware
    'error_handling_middleware',
    
    # Decorators
    'handle_agent_errors', 'handle_api_errors', 'validate_input',
    'require_session_id', 'log_performance',
    
    # Utilities
    'create_http_exception', 'handle_external_service_error', 'safe_execute',
    
    # Context
    'ErrorContext'
]
