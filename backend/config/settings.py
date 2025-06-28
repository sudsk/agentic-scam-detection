# backend/config/settings.py
"""
Centralized Configuration for HSBC Scam Detection Agent Backend
Single source of truth for all configuration values
FIXED: Updated for Pydantic v2 compatibility
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any

# Fixed imports for Pydantic v2
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # ===== API CONFIGURATION =====
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    environment: str = "development"
    
    # ===== CORS CONFIGURATION =====
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ]
    
    # ===== FRAUD DETECTION THRESHOLDS (CENTRALIZED) =====
    # Risk score thresholds used across ALL agents
    risk_threshold_critical: int = 80
    risk_threshold_high: int = 60
    risk_threshold_medium: int = 40
    risk_threshold_low: int = 20
    
    # Confidence thresholds
    confidence_threshold_high: float = 0.9
    confidence_threshold_medium: float = 0.7
    confidence_threshold_low: float = 0.5
    
    # Pattern weights for fraud detection
    pattern_weights: Dict[str, float] = {
        "authority_impersonation": 35,
        "credential_requests": 40,
        "investment_fraud": 40,  # Increased from 30
        "romance_exploitation": 30,  # Increased from 25
        "urgency_pressure": 25,  # Increased from 20
        "third_party_instructions": 30,  # Increased from 25
        "financial_pressure": 25
    }
    
    # ===== AGENT CONFIGURATION =====
    max_concurrent_sessions: int = 500
    agent_timeout_seconds: int = 30
    agent_retry_attempts: int = 3
    agent_processing_timeout: int = 60
    
    # Agent-specific settings
    audio_agent_config: Dict[str, Any] = {
        "sample_rate": 16000,
        "channels": 1,
        "language_code": "en-GB",
        "enable_speaker_diarization": True,
        "confidence_threshold": 0.8,
        "max_audio_duration_seconds": 300,
        "chunk_size_seconds": 30
    }
    
    fraud_agent_config: Dict[str, Any] = {
        "max_processing_time_seconds": 5.0,
        "min_text_length": 3,
        "max_text_length": 10000
    }
    
    policy_agent_config: Dict[str, Any] = {
        "policy_version": "2.1.0",
        "include_customer_education": True,
        "max_guidance_items": 10
    }
    
    case_agent_config: Dict[str, Any] = {
        "case_id_prefix": "FD",
        "case_retention_days": 2555,  # 7 years
        "max_cases_per_agent": 50,
        "case_timeout_hours": 48,
        "auto_escalation_enabled": True
    }
    
    # Team assignments for case escalation
    team_assignments: Dict[str, str] = {
        "critical": "financial-crime-immediate@bank.com",
        "high": "fraud-investigation-team@bank.com", 
        "medium": "fraud-prevention-team@bank.com",
        "low": "customer-service-enhanced@bank.com"
    }
    
    # ===== AUDIO PROCESSING CONFIGURATION =====
    max_audio_file_size_mb: int = 100
    supported_audio_formats: List[str] = [".wav", ".mp3", ".m4a", ".flac"]
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    
    # ===== SCAM TYPE CONFIGURATION =====
    scam_type_keywords: Dict[str, List[str]] = {
        "investment_scam": [
            "investment", "trading", "forex", "crypto", "bitcoin", "profit",
            "returns", "margin call", "platform", "advisor", "broker", "guaranteed"
        ],
        "romance_scam": [
            "online", "dating", "relationship", "boyfriend", "girlfriend",
            "military", "overseas", "emergency", "medical", "visa", "stuck", "stranded"
        ],
        "impersonation_scam": [
            "bank", "security", "fraud", "police", "tax", "hmrc",
            "government", "investigation", "verification", "account compromised"
        ],
        "authority_scam": [
            "police", "court", "tax", "fine", "penalty", "arrest",
            "legal action", "warrant", "prosecution", "bailiff"
        ]
    }
    
    # ===== DATABASE CONFIGURATION =====
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "hsbc_fraud_detection"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres123"
    
    # Firestore collections
    firestore_collection_sessions: str = "fraud_sessions"
    firestore_collection_cases: str = "fraud_cases" 
    firestore_collection_metrics: str = "system_metrics"
    firestore_collection_audio: str = "audio_files"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ttl_sessions: int = 3600  # 1 hour
    redis_ttl_cache: int = 1800     # 30 minutes
    
    # ===== GOOGLE CLOUD CONFIGURATION =====
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
    
    # Pub/Sub Configuration
    pubsub_topic_audio: str = "hsbc-fraud-audio"
    pubsub_topic_transcripts: str = "hsbc-fraud-transcripts"
    pubsub_topic_alerts: str = "hsbc-fraud-alerts"
    
    # ===== SECURITY CONFIGURATION =====
    secret_key: str = "hsbc-scam-detection-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    
    # ===== MONITORING & LOGGING =====
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # ===== RATE LIMITING =====
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 10
    
    # ===== WEBSOCKET CONFIGURATION =====
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    websocket_max_connections: int = 1000
    
    # ===== EXTERNAL API CONFIGURATION =====
    quantexa_api_url: Optional[str] = None
    quantexa_api_key: Optional[str] = None
    quantexa_timeout_seconds: int = 30
    
    hsbc_core_banking_url: Optional[str] = None
    hsbc_core_banking_certificate: Optional[str] = None
    hsbc_core_banking_timeout_seconds: int = 60
    
    # ===== FEATURE FLAGS =====
    enable_learning_agent: bool = True
    enable_compliance_agent: bool = True
    enable_case_management: bool = True
    enable_real_time_transcription: bool = True
    enable_mock_mode: bool = True
    
    # ===== DEMO CONFIGURATION =====
    demo_mode: bool = True
    demo_sample_data_path: str = "data/sample_audio"
    demo_delay_simulation: bool = True
    demo_delay_seconds: float = 1.0
    
    # ===== PERFORMANCE CONFIGURATION =====
    max_concurrent_audio_processing: int = 10
    max_concurrent_fraud_analysis: int = 20
    cache_ttl_seconds: int = 3600
    
    # ===== VALIDATION =====
    @field_validator('allowed_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @field_validator('supported_audio_formats', mode='before')
    @classmethod
    def parse_audio_formats(cls, v):
        if isinstance(v, str):
            return [fmt.strip() for fmt in v.split(',')]
        return v
    
    @field_validator('google_application_credentials')
    @classmethod
    def validate_gcp_credentials(cls, v):
        if v and not os.path.exists(v):
            raise ValueError(f"Google Application Credentials file not found: {v}")
        return v
    
    @field_validator('max_audio_file_size_mb')
    @classmethod
    def validate_max_file_size(cls, v):
        if v <= 0 or v > 1000:
            raise ValueError("Max audio file size must be between 1 and 1000 MB")
        return v
    
    @field_validator('risk_threshold_critical', 'risk_threshold_high', 'risk_threshold_medium', 'risk_threshold_low')
    @classmethod
    def validate_risk_thresholds(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Risk thresholds must be between 0 and 100")
        return v
    
    # ===== DERIVED PROPERTIES =====
    
    @property
    def risk_thresholds(self) -> Dict[str, int]:
        """Get risk thresholds as dictionary"""
        return {
            "critical": self.risk_threshold_critical,
            "high": self.risk_threshold_high,
            "medium": self.risk_threshold_medium,
            "low": self.risk_threshold_low
        }
    
    @property
    def confidence_thresholds(self) -> Dict[str, float]:
        """Get confidence thresholds as dictionary"""
        return {
            "high": self.confidence_threshold_high,
            "medium": self.confidence_threshold_medium,
            "low": self.confidence_threshold_low
        }
    
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
    
    def get_postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
        base_config = {
            "risk_thresholds": self.risk_thresholds,
            "confidence_thresholds": self.confidence_thresholds,
            "pattern_weights": self.pattern_weights,
            "team_assignments": self.team_assignments,
            "processing_timeout": self.agent_processing_timeout,
            "retry_attempts": self.agent_retry_attempts
        }
        
        agent_configs = {
            "audio_processor": {**base_config, **self.audio_agent_config},
            "fraud_detection": {**base_config, **self.fraud_agent_config},
            "policy_guidance": {**base_config, **self.policy_agent_config},
            "case_management": {**base_config, **self.case_agent_config}
        }
        
        return agent_configs.get(agent_type, base_config)
    
    # Model configuration for Pydantic v2
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields
    }

# ===== ENVIRONMENT-SPECIFIC SETTINGS =====

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
    gcp_project_id: str = Field(..., description="GCP Project ID is required in production")
    google_application_credentials: str = Field(..., description="GCP credentials required in production")
    quantexa_api_url: str = Field(..., description="Quantexa API URL required in production")
    quantexa_api_key: str = Field(..., description="Quantexa API key required in production")
    hsbc_core_banking_url: str = Field(..., description="HSBC Core Banking URL required in production")

class TestingSettings(Settings):
    """Testing-specific settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    demo_mode: bool = True
    enable_mock_mode: bool = True
    
    # Test database
    postgres_db: str = "hsbc_fraud_test"
    firestore_collection_sessions: str = "test_fraud_sessions"
    firestore_collection_cases: str = "test_fraud_cases"
    firestore_collection_metrics: str = "test_system_metrics"
    
    # Disable external services for testing
    enable_learning_agent: bool = False
    enable_case_management: bool = False

# ===== SETTINGS FACTORY =====

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

def get_settings_for_environment(environment: str = None) -> Settings:
    """Get settings based on environment"""
    environment = environment or os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

def create_agent_config(agent_type: str, settings: Settings = None) -> Dict[str, Any]:
    """Create agent configuration from centralized settings"""
    settings = settings or get_settings()
    return settings.get_agent_config(agent_type)

# Export settings instance
settings = get_settings()
