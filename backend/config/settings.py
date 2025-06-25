"""
Configuration settings for HSBC Scam Detection Agent Backend
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # CORS Configuration
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ]
    
    # Agent Configuration
    max_concurrent_sessions: int = 500
    agent_timeout_seconds: int = 30
    risk_threshold_high: int = 80
    risk_threshold_medium: int = 40
    
    # Audio Processing
    max_audio_file_size_mb: int = 100
    supported_audio_formats: List[str] = [".wav", ".mp3", ".m4a", ".flac"]
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    
    # Google Cloud Configuration
    gcp_project_id: Optional[str] = None
    gcp_region: str = "europe-west2"
    google_application_credentials: Optional[str] = None
    
    # Speech-to-Text Configuration
    speech_model: str = "telephony"
    speech_language_code: str = "en-GB"
    enable_speaker_diarization: bool = True
    enable_word_confidence: bool = True
    
    # Vertex AI Configuration
    vertex_ai_location: str = "europe-west2"
    vertex_ai_model_name: str = "gemini-pro"
    vertex_ai_endpoint_name: Optional[str] = None
    
    # Database Configuration
    firestore_collection_sessions: str = "fraud_sessions"
    firestore_collection_cases: str = "fraud_cases"
    firestore_collection_metrics: str = "system_metrics"
    
    # Pub/Sub Configuration
    pubsub_topic_audio: str = "hsbc-fraud-audio"
    pubsub_topic_transcripts: str = "hsbc-fraud-transcripts"
    pubsub_topic_alerts: str = "hsbc-fraud-alerts"
    
    # Redis Configuration (for caching and session management)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Security Configuration
    secret_key: str = "hsbc-scam-detection-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Monitoring and Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 10
    
    # WebSocket Configuration
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    websocket_max_connections: int = 1000
    
    # External API Configuration
    quantexa_api_url: Optional[str] = None
    quantexa_api_key: Optional[str] = None
    quantexa_timeout_seconds: int = 30
    
    hsbc_core_banking_url: Optional[str] = None
    hsbc_core_banking_certificate: Optional[str] = None
    hsbc_core_banking_timeout_seconds: int = 60
    
    # Feature Flags
    enable_learning_agent: bool = True
    enable_compliance_agent: bool = True
    enable_case_management: bool = True
    enable_real_time_transcription: bool = True
    enable_mock_mode: bool = True  # For demo purposes
    
    # Demo Configuration
    demo_mode: bool = True
    demo_sample_data_path: str = "data/sample_audio"
    demo_delay_simulation: bool = True
    demo_delay_seconds: float = 1.0
    
    @validator('allowed_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('supported_audio_formats', pre=True)
    def parse_audio_formats(cls, v):
        """Parse supported audio formats from environment variable"""
        if isinstance(v, str):
            return [fmt.strip() for fmt in v.split(',')]
        return v
    
    @validator('google_application_credentials')
    def validate_gcp_credentials(cls, v):
        """Validate Google Cloud credentials path"""
        if v and not os.path.exists(v):
            raise ValueError(f"Google Application Credentials file not found: {v}")
        return v
    
    @validator('max_audio_file_size_mb')
    def validate_max_file_size(cls, v):
        """Validate maximum file size is reasonable"""
        if v <= 0 or v > 1000:
            raise ValueError("Max audio file size must be between 1 and 1000 MB")
        return v
    
    @validator('risk_threshold_high', 'risk_threshold_medium')
    def validate_risk_thresholds(cls, v):
        """Validate risk thresholds are between 0 and 100"""
        if v < 0 or v > 100:
            raise ValueError("Risk thresholds must be between 0 and 100")
        return v
    
    def get_max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes"""
        return self.max_audio_file_size_mb * 1024 * 1024
    
    def is_audio_format_supported(self, filename: str) -> bool:
        """Check if audio file format is supported"""
        return any(filename.lower().endswith(fmt) for fmt in self.supported_audio_formats)
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Development settings override
class DevelopmentSettings(Settings):
    """Development-specific settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    demo_mode: bool = True
    enable_mock_mode: bool = True
    
    # Relaxed security for development
    allowed_origins: List[str] = ["*"]
    
    # Lower thresholds for testing
    max_concurrent_sessions: int = 10
    rate_limit_requests_per_minute: int = 1000

# Production settings override
class ProductionSettings(Settings):
    """Production-specific settings"""
    debug: bool = False
    log_level: str = "INFO"
    demo_mode: bool = False
    enable_mock_mode: bool = False
    
    # Strict security for production
    allowed_origins: List[str] = [
        "https://hsbc-fraud-detection.internal",
        "https://hsbc-agent-desktop.internal"
    ]
    
    # Production performance settings
    max_concurrent_sessions: int = 500
    rate_limit_requests_per_minute: int = 100
    
    # Required in production
    gcp_project_id: str
    google_application_credentials: str
    quantexa_api_url: str
    quantexa_api_key: str
    hsbc_core_banking_url: str

# Testing settings override
class TestingSettings(Settings):
    """Testing-specific settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    demo_mode: bool = True
    enable_mock_mode: bool = True
    
    # Test database
    firestore_collection_sessions: str = "test_fraud_sessions"
    firestore_collection_cases: str = "test_fraud_cases"
    firestore_collection_metrics: str = "test_system_metrics"
    
    # Disable external services for testing
    enable_learning_agent: bool = False
    enable_case_management: bool = False

def get_settings_for_environment(environment: str = None) -> Settings:
    """Get settings based on environment"""
    environment = environment or os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# Export commonly used settings
settings = get_settings()
