# backend/config/settings.py - FIXED VERSION
"""
Updated configuration settings with pattern_weights for fraud detection
FIXED: Added missing pattern_weights attribute that the fraud detection agent needs
"""

import os
from pathlib import Path
from typing import Dict, Any, List

class Settings:
    """Application settings with real transcription configuration"""
    
    def __init__(self):
        # Base configuration
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        
        # CORS settings - FIXED: Added wildcard for development like original
        self.allowed_origins = [
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "*"  # FIXED: Allow all origins for development (like original settings)
        ]
        
        # FIXED: Add missing pattern_weights that the fraud detection agent needs
        self.pattern_weights: Dict[str, float] = {
            "authority_impersonation": 35,
            "credential_requests": 40,
            "investment_fraud": 40,
            "romance_exploitation": 30,
            "urgency_pressure": 25,
            "third_party_instructions": 30,
            "financial_pressure": 25
        }
        
        # FIXED: Add missing confidence thresholds that agents need
        self.confidence_threshold_high = 0.9
        self.confidence_threshold_medium = 0.7
        self.confidence_threshold_low = 0.5
        
        # FIXED: Add missing agent configuration attributes
        self.max_concurrent_sessions = 500
        self.agent_timeout_seconds = 30
        self.agent_retry_attempts = 3
        self.agent_processing_timeout = 60
        
        # FIXED: Add missing risk thresholds that agents need
        self.risk_threshold_critical = 80
        self.risk_threshold_high = 60
        self.risk_threshold_medium = 40
        self.risk_threshold_low = 20
        
        # FIXED: Add missing team assignments
        self.team_assignments = {
            "critical": "financial-crime-immediate@bank.com",
            "high": "fraud-investigation-team@bank.com", 
            "medium": "fraud-prevention-team@bank.com",
            "low": "customer-service-enhanced@bank.com"
        }
        
        # FIXED: Add missing scam type keywords
        self.scam_type_keywords = {
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
        
        # Audio file settings
        self.supported_audio_formats = [".wav", ".mp3", ".m4a", ".flac"]
        self.max_audio_file_size_mb = 100
        
        # FIXED: Add missing demo configuration attributes from original settings
        self.demo_mode = True
        self.demo_sample_data_path = "data/sample_audio"
        self.demo_delay_simulation = True
        self.demo_delay_seconds = 1.0
        
        # Paths
        self.data_path = Path("data")
        self.sample_audio_path = self.data_path / "sample_audio"
        self.uploads_path = Path("uploads")
        
        # WebSocket settings
        self.websocket_timeout = 300  # 5 minutes
        self.max_connections = 100
        
        # FIXED: Add additional missing attributes from original settings
        self.environment = "development"
        self.enable_mock_mode = True
        self.enable_learning_agent = True
        self.enable_compliance_agent = True
        self.enable_case_management = True
        self.enable_real_time_transcription = True
        
        # Agent configurations - UPDATED for real transcription
        self._setup_agent_configs()
    
    def _setup_agent_configs(self):
        """Setup agent configurations with real transcription settings"""
        
        # Audio Processing Agent Configuration - GOOGLE STT ONLY
        self.audio_agent_config = {
            # === TRANSCRIPTION ENGINE SETTINGS ===
            "transcription_engine": "google_stt",  # Only Google Speech-to-Text supported
            
            # === AUDIO PROCESSING SETTINGS ===
            "chunk_duration_seconds": float(os.getenv("CHUNK_DURATION", "2.0")),
            "overlap_seconds": float(os.getenv("CHUNK_OVERLAP", "0.5")),
            "processing_delay_ms": int(os.getenv("PROCESSING_DELAY", "200")),  # Fast cloud processing
            
            # === AUDIO FORMAT SETTINGS ===
            "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),
            "channels": int(os.getenv("AUDIO_CHANNELS", "1")),
            "audio_format": "wav",
            
            # === PATHS ===
            "audio_base_path": str(self.sample_audio_path),
            "temp_audio_path": str(self.uploads_path / "temp_audio"),
            
            # === SPEECH DETECTION ===
            "min_speech_duration": float(os.getenv("MIN_SPEECH_DURATION", "0.5")),
            "silence_threshold": float(os.getenv("SILENCE_THRESHOLD", "0.01")),
            
            # === SPEAKER IDENTIFICATION ===
            "enable_speaker_diarization": os.getenv("ENABLE_SPEAKER_DIARIZATION", "true").lower() == "true",  # Enable by default for Google STT
            "default_speaker_alternation": True,
            
            # === GOOGLE STT SPECIFIC SETTINGS ===
            "language_code": os.getenv("GOOGLE_LANGUAGE_CODE", "en-GB"),
            "encoding": "LINEAR16",
            "enable_automatic_punctuation": True,
            "enable_word_time_offsets": True,
            "enable_word_confidence": True,
            "use_enhanced": True,  # Use enhanced model for better accuracy
            "model": os.getenv("GOOGLE_STT_MODEL", "telephony"),  # Optimized for phone calls
            
            # === PERFORMANCE SETTINGS ===
            "max_concurrent_transcriptions": int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "5")),  # Higher for cloud
            "transcription_timeout": int(os.getenv("TRANSCRIPTION_TIMEOUT", "30")),  # Longer for network calls
            
            # === LOGGING ===
            "log_transcription_details": os.getenv("LOG_TRANSCRIPTION_DETAILS", "true").lower() == "true",
            "save_transcription_logs": os.getenv("SAVE_TRANSCRIPTION_LOGS", "false").lower() == "true"
        }
        
        # Fraud Detection Agent Configuration
        self.fraud_agent_config = {
            "confidence_threshold": 0.7,
            "risk_escalation_threshold": 80,
            "pattern_weights": self.pattern_weights,  # FIXED: Use the class attribute
            "scam_type_mapping": self.scam_type_keywords,  # FIXED: Use the class attribute
            "risk_thresholds": {
                "critical": self.risk_threshold_critical,
                "high": self.risk_threshold_high,
                "medium": self.risk_threshold_medium,
                "low": self.risk_threshold_low
            }
        }
        
        # Policy Guidance Agent Configuration
        self.policy_agent_config = {
            "guidance_detail_level": "detailed",
            "include_regulatory_info": True,
            "max_recommended_actions": 6,
            "include_customer_education": True,
            "policy_version": "2.1.0",
            "risk_thresholds": {
                "critical": self.risk_threshold_critical,
                "high": self.risk_threshold_high,
                "medium": self.risk_threshold_medium,
                "low": self.risk_threshold_low
            },
            "team_assignments": self.team_assignments
        }
        
        # Case Management Agent Configuration
        self.case_agent_config = {
            "auto_create_threshold": 60,
            "case_id_prefix": "FD",
            "case_retention_days": 2555,  # 7 years
            "max_cases_per_agent": 50,
            "case_timeout_hours": 48,
            "auto_escalation_enabled": True,
            "priority_mapping": {
                "low": (0, 39),
                "medium": (40, 69), 
                "high": (70, 89),
                "critical": (90, 100)
            },
            "default_assignee": "fraud_team",
            "include_transcription": True,
            "risk_thresholds": {
                "critical": self.risk_threshold_critical,
                "high": self.risk_threshold_high,
                "medium": self.risk_threshold_medium,
                "low": self.risk_threshold_low
            },
            "team_assignments": self.team_assignments
        }
        
        # System Orchestrator Configuration  
        self.orchestrator_config = {
            "agent_coordination": True,
            "parallel_processing": True,
            "error_handling": "graceful_degradation",
            "performance_monitoring": True
        }
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
        # Base configuration that all agents need
        base_config = {
            "risk_thresholds": self.risk_thresholds,
            "confidence_thresholds": self.confidence_thresholds,
            "pattern_weights": self.pattern_weights,
            "team_assignments": self.team_assignments,
            "processing_timeout": self.agent_processing_timeout,
            "retry_attempts": self.agent_retry_attempts,
            "timeout_seconds": self.agent_timeout_seconds,
            "max_concurrent_sessions": self.max_concurrent_sessions
        }
        
        configs = {
            "audio_processor": {**base_config, **self.audio_agent_config},
            "fraud_detection": {**base_config, **self.fraud_agent_config},
            "policy_guidance": {**base_config, **self.policy_agent_config},
            "case_management": {**base_config, **self.case_agent_config},
            "orchestrator": {**base_config, **self.orchestrator_config}
        }
        return configs.get(agent_type, base_config)
    
    def get_transcription_config(self) -> Dict[str, Any]:
        """Get transcription-specific configuration for Google STT"""
        return {
            "engine": "google_stt",
            "chunk_duration": self.audio_agent_config["chunk_duration_seconds"],
            "sample_rate": self.audio_agent_config["sample_rate"],
            "channels": self.audio_agent_config["channels"],
            "processing_delay": self.audio_agent_config["processing_delay_ms"],
            "language_code": self.audio_agent_config["language_code"],
            "model": self.audio_agent_config["model"],
            "enable_speaker_diarization": self.audio_agent_config["enable_speaker_diarization"]
        }
    
    # FIXED: Add properties that agents expect to access directly
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
    
    def parse_cors_origins(self, origins_input) -> List[str]:
        """Parse CORS origins from string or list (like original settings)"""
        if isinstance(origins_input, str):
            return [origin.strip() for origin in origins_input.split(',')]
        return origins_input
    
    def validate_transcription_setup(self) -> Dict[str, Any]:
        """Validate Google Speech-to-Text setup"""
        validation_result = {
            "engine": "google_stt",
            "status": "unknown",
            "requirements_met": False,
            "missing_dependencies": [],
            "recommendations": []
        }
        
        try:
            from google.cloud import speech
            
            # Check for credentials
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and os.path.exists(creds_path):
                validation_result.update({
                    "status": "ready",
                    "requirements_met": True,
                    "recommendations": [
                        "Google Speech-to-Text credentials found",
                        f"Using model: {self.audio_agent_config['model']}",
                        f"Language: {self.audio_agent_config['language_code']}"
                    ]
                })
            else:
                validation_result.update({
                    "status": "missing_credentials",
                    "requirements_met": False,
                    "missing_dependencies": ["Google Cloud credentials"],
                    "recommendations": [
                        "Set GOOGLE_APPLICATION_CREDENTIALS environment variable",
                        "Download service account JSON from Google Cloud Console",
                        "Ensure the service account has Speech-to-Text API permissions"
                    ]
                })
        except ImportError:
            validation_result.update({
                "status": "missing_dependencies",
                "requirements_met": False,
                "missing_dependencies": ["google-cloud-speech"],
                "recommendations": ["Install with: pip install google-cloud-speech"]
            })
        
        return validation_result

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def get_transcription_validation() -> Dict[str, Any]:
    """Get transcription setup validation"""
    settings = get_settings()
    return settings.validate_transcription_setup()

# Environment variable examples for Google Speech-to-Text setup
ENV_EXAMPLES = {
    "google_stt_production": {
        "GOOGLE_LANGUAGE_CODE": "en-GB",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json", 
        "GOOGLE_STT_MODEL": "telephony",  # Optimized for phone calls
        "CHUNK_DURATION": "2.0",
        "PROCESSING_DELAY": "200",  # Fast cloud processing
        "ENABLE_SPEAKER_DIARIZATION": "true"
    },
    "google_stt_development": {
        "GOOGLE_LANGUAGE_CODE": "en-US",
        "GOOGLE_APPLICATION_CREDENTIALS": "./credentials/dev-service-account.json",
        "GOOGLE_STT_MODEL": "latest_long",  # Better for longer audio
        "CHUNK_DURATION": "3.0",
        "PROCESSING_DELAY": "300",
        "ENABLE_SPEAKER_DIARIZATION": "true"
    }
}
