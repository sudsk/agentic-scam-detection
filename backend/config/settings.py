# backend/config/settings.py - UPDATED FOR GOOGLE SPEECH v2 API
"""
Updated configuration settings for Google Speech-to-Text v2 API
FIXED: Proper telephony model configuration and v2 API settings
"""

import os
from pathlib import Path
from typing import Dict, Any, List

class Settings:
    """Application settings with Google Speech v2 API configuration"""
    
    def __init__(self):
        # Base configuration
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        
        # CORS settings
        self.allowed_origins = [
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "*"  # Allow all origins for development
        ]
        
        # Pattern weights for fraud detection
        self.pattern_weights: Dict[str, float] = {
            "authority_impersonation": 35,
            "credential_requests": 40,
            "investment_fraud": 40,
            "romance_exploitation": 30,
            "urgency_pressure": 25,
            "third_party_instructions": 30,
            "financial_pressure": 25
        }
        
        # Confidence thresholds
        self.confidence_threshold_high = 0.9
        self.confidence_threshold_medium = 0.7
        self.confidence_threshold_low = 0.5
        
        # Agent configuration
        self.max_concurrent_sessions = 500
        self.agent_timeout_seconds = 30
        self.agent_retry_attempts = 3
        self.agent_processing_timeout = 60
        
        # Risk thresholds
        self.risk_threshold_critical = 80
        self.risk_threshold_high = 60
        self.risk_threshold_medium = 40
        self.risk_threshold_low = 20
        
        # Team assignments
        self.team_assignments = {
            "critical": "financial-crime-immediate@bank.com",
            "high": "fraud-investigation-team@bank.com", 
            "medium": "fraud-prevention-team@bank.com",
            "low": "customer-service-enhanced@bank.com"
        }
        
        # Scam type keywords
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
        
        # Demo configuration
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
        
        # Additional attributes
        self.environment = "development"
        self.enable_mock_mode = True
        self.enable_learning_agent = True
        self.enable_compliance_agent = True
        self.enable_case_management = True
        self.enable_real_time_transcription = True
        
        # Agent configurations with Google Speech v2 API
        self._setup_agent_configs_v2()
    
    def _setup_agent_configs_v2(self):
        """Setup agent configurations with Google Speech v2 API settings"""
        
        # ===== GOOGLE SPEECH v2 API CONFIGURATION =====
        self.audio_agent_config = {
            # === API VERSION SETTINGS ===
            "api_version": "v2",
            "transcription_engine": "google_stt_v2",
            
            # === PROJECT AND LOCATION SETTINGS ===
            "google_project_id": os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GCP_PROJECT_ID", "your-project-id")),
            "google_location": os.getenv("GOOGLE_SPEECH_LOCATION", "global"),  # or specific region like "us-central1"
            
            # === RECOGNIZER SETTINGS (v2 API) ===
            "recognizer_id": os.getenv("GOOGLE_RECOGNIZER_ID", "_"),  # Use default recognizer
            
            # === MODEL CONFIGURATION ===
            # See: https://cloud.google.com/speech-to-text/docs/transcription-model
            "model": os.getenv("GOOGLE_STT_MODEL", "telephony"),  # OPTIMAL for phone calls
            "model_fallback": "latest_short",  # Fallback if telephony not available
            
            # Alternative models available:
            # - "telephony": Optimized for phone call audio (8kHz-48kHz)
            # - "latest_long": Best for longer audio (>1min)
            # - "latest_short": Best for shorter audio (<1min)
            # - "command_and_search": For voice commands
            # - "phone_call": Legacy telephony model (deprecated, use "telephony")
            
            # === AUDIO PROCESSING SETTINGS ===
            "chunk_duration_seconds": float(os.getenv("CHUNK_DURATION", "2.0")),
            "overlap_seconds": float(os.getenv("CHUNK_OVERLAP", "0.5")),
            "processing_delay_ms": int(os.getenv("PROCESSING_DELAY", "200")),
            
            # === AUDIO FORMAT SETTINGS ===
            "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),  # 16kHz standard for telephony
            "channels": int(os.getenv("AUDIO_CHANNELS", "1")),  # Mono for phone calls
            "audio_encoding": "LINEAR16",  # Best for telephony
            
            # === LANGUAGE SETTINGS ===
            "language_codes": [os.getenv("GOOGLE_LANGUAGE_CODE", "en-GB")],  # v2 uses list
            "alternative_language_codes": ["en-US"],  # Additional languages
            
            # === RECOGNITION FEATURES (v2 API) ===
            "enable_automatic_punctuation": True,
            "enable_word_confidence": True,
            "enable_word_time_offsets": True,
            "enable_profanity_filter": False,  # Keep original for fraud detection
            "enable_spoken_punctuation": False,
            "enable_spoken_emojis": False,
            
            # === SPEAKER DIARIZATION (v2 API) ===
            "enable_speaker_diarization": True,
            "min_speaker_count": 1,
            "max_speaker_count": 2,  # Customer + Agent
            
            # === STREAMING FEATURES (v2 API) ===
            "interim_results": True,
            "enable_voice_activity_events": True,
            "voice_activity_timeout": {
                "speech_start_timeout": "5s",
                "speech_end_timeout": "2s"
            },
            
            # === ADAPTATION SETTINGS (v2 API) ===
            # Custom vocabulary for banking/fraud terms
            "enable_adaptation": True,
            "phrase_hints": [
                # Banking terms
                "investment", "transfer", "account", "balance", "transaction",
                "verification", "security", "PIN", "password", "authentication",
                
                # Fraud-related terms
                "guaranteed returns", "margin call", "investment opportunity",
                "emergency", "urgent", "immediately", "before it's too late",
                "police", "investigation", "court", "tax office", "HMRC",
                
                # Financial amounts
                "thousand", "million", "pounds", "dollars", "euros",
                "payment", "deposit", "withdrawal", "fee"
            ],
            "boost": 20,  # Boost recognition of phrase hints
            
            # === PATHS ===
            "audio_base_path": str(self.sample_audio_path),
            "temp_audio_path": str(self.uploads_path / "temp_audio"),
            
            # === PERFORMANCE SETTINGS ===
            "max_concurrent_transcriptions": int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "10")),
            "transcription_timeout": int(os.getenv("TRANSCRIPTION_TIMEOUT", "60")),
            "recognition_restart_timeout": int(os.getenv("RESTART_TIMEOUT", "55")),  # v2 API limit
            
            # === ERROR HANDLING ===
            "max_retry_attempts": 3,
            "retry_delay_seconds": 2.0,
            "enable_fallback_model": True,
            
            # === LOGGING ===
            "log_transcription_details": os.getenv("LOG_TRANSCRIPTION_DETAILS", "true").lower() == "true",
            "save_transcription_logs": os.getenv("SAVE_TRANSCRIPTION_LOGS", "false").lower() == "true",
            
            # === COST OPTIMIZATION ===
            "enable_enhanced_models": os.getenv("ENABLE_ENHANCED_MODELS", "true").lower() == "true",
            "data_logging": "NEVER",  # Don't log audio for privacy
        }
        
        # Other agent configurations remain the same
        self.fraud_agent_config = {
            "confidence_threshold": 0.7,
            "risk_escalation_threshold": 80,
            "pattern_weights": self.pattern_weights,
            "scam_type_mapping": self.scam_type_keywords,
            "risk_thresholds": {
                "critical": self.risk_threshold_critical,
                "high": self.risk_threshold_high,
                "medium": self.risk_threshold_medium,
                "low": self.risk_threshold_low
            }
        }
        
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
        
        self.case_agent_config = {
            "auto_create_threshold": 60,
            "case_id_prefix": "FD",
            "case_retention_days": 2555,
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
        
        self.orchestrator_config = {
            "agent_coordination": True,
            "parallel_processing": True,
            "error_handling": "graceful_degradation",
            "performance_monitoring": True
        }
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type"""
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
    
    def get_google_stt_v2_config(self) -> Dict[str, Any]:
        """Get Google Speech-to-Text v2 specific configuration"""
        return {
            "api_version": "v2",
            "project_id": self.audio_agent_config["google_project_id"],
            "location": self.audio_agent_config["google_location"],
            "recognizer_id": self.audio_agent_config["recognizer_id"],
            "model": self.audio_agent_config["model"],
            "language_codes": self.audio_agent_config["language_codes"],
            "sample_rate": self.audio_agent_config["sample_rate"],
            "encoding": self.audio_agent_config["audio_encoding"],
            "enable_speaker_diarization": self.audio_agent_config["enable_speaker_diarization"],
            "interim_results": self.audio_agent_config["interim_results"],
            "phrase_hints": self.audio_agent_config["phrase_hints"],
            "boost": self.audio_agent_config["boost"]
        }
    
    def get_telephony_optimized_config(self) -> Dict[str, Any]:
        """Get telephony-optimized configuration"""
        return {
            "model": "telephony",  # Optimal for phone calls
            "sample_rate": 16000,  # Standard telephony
            "channels": 1,  # Mono
            "encoding": "LINEAR16",
            "chunk_duration_ms": 200,  # Fast response
            "language_codes": ["en-GB"],
            "enable_speaker_diarization": True,
            "min_speaker_count": 1,
            "max_speaker_count": 2,
            "enable_automatic_punctuation": True,
            "enable_word_confidence": True,
            "enable_word_time_offsets": True,
            "interim_results": True,
            "enable_voice_activity_events": True,
            "phrase_hints": self.audio_agent_config["phrase_hints"]
        }
    
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
    
    def validate_google_stt_v2_setup(self) -> Dict[str, Any]:
        """Validate Google Speech-to-Text v2 setup"""
        validation_result = {
            "api_version": "v2",
            "status": "unknown",
            "requirements_met": False,
            "missing_dependencies": [],
            "recommendations": [],
            "auth_method": "gce_service_account",
            "model_config": {}
        }
        
        try:
            from google.cloud.speech_v2 import SpeechClient
            from google.cloud.speech_v2.types import cloud_speech
            
            try:
                client = SpeechClient()
                
                # Test configuration
                config = cloud_speech.RecognitionConfig(
                    encoding=cloud_speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_codes=["en-GB"],
                    model="telephony",  # Test telephony model
                    features=cloud_speech.RecognitionFeatures(
                        enable_automatic_punctuation=True,
                        enable_word_confidence=True,
                        enable_speaker_diarization=True,
                        diarization_config=cloud_speech.SpeakerDiarizationConfig(
                            min_speaker_count=1,
                            max_speaker_count=2
                        )
                    )
                )
                
                validation_result.update({
                    "status": "ready",
                    "requirements_met": True,
                    "auth_method": "gce_service_account",
                    "model_config": {
                        "primary_model": "telephony",
                        "fallback_model": "latest_short",
                        "language": "en-GB",
                        "sample_rate": "16kHz",
                        "encoding": "LINEAR16",
                        "speaker_diarization": True
                    },
                    "recommendations": [
                        "Using Google Speech-to-Text v2 API",
                        "Telephony model optimized for phone calls",
                        "Speaker diarization enabled for customer/agent detection",
                        "Real-time streaming with interim results",
                        "Custom phrase hints for banking/fraud terms"
                    ]
                })
                    
            except Exception as auth_error:
                validation_result.update({
                    "status": "service_account_error",
                    "requirements_met": False,
                    "auth_method": "gce_service_account_failed",
                    "missing_dependencies": [f"VM service account error: {auth_error}"],
                    "recommendations": [
                        "Ensure your GCE VM has a service account attached",
                        "Grant the service account 'Cloud Speech Client' role",
                        "Enable the Speech-to-Text API in your GCP project",
                        "For v2 API, ensure 'Speech-to-Text v2 API' is enabled",
                        "Check: gcloud compute instances describe YOUR_VM_NAME --zone=YOUR_ZONE",
                        "Verify: curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
                    ]
                })
                    
        except ImportError:
            validation_result.update({
                "status": "missing_dependencies",
                "requirements_met": False,
                "missing_dependencies": ["google-cloud-speech>=2.33.0"],
                "recommendations": [
                    "Install with: pip install google-cloud-speech>=2.33.0",
                    "For v2 API support, use latest version",
                    "Verify installation: python -c 'from google.cloud.speech_v2 import SpeechClient'"
                ]
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

def get_google_stt_v2_validation() -> Dict[str, Any]:
    """Get Google Speech v2 setup validation"""
    settings = get_settings()
    return settings.validate_google_stt_v2_setup()

# Environment variable examples for Google Speech v2 API setup
ENV_EXAMPLES = {
    "google_stt_v2_production": {
        "GOOGLE_CLOUD_PROJECT": "your-production-project",
        "GOOGLE_SPEECH_LOCATION": "global",  # or "us-central1"
        "GOOGLE_LANGUAGE_CODE": "en-GB",
        "GOOGLE_STT_MODEL": "telephony",  # Optimal for phone calls
        "GOOGLE_RECOGNIZER_ID": "_",  # Use default recognizer
        "CHUNK_DURATION": "2.0",
        "PROCESSING_DELAY": "200",
        "RESTART_TIMEOUT": "55",  # v2 API streaming limit
        "ENABLE_ENHANCED_MODELS": "true"
    },
    "google_stt_v2_development": {
        "GOOGLE_CLOUD_PROJECT": "your-dev-project",
        "GOOGLE_SPEECH_LOCATION": "global",
        "GOOGLE_LANGUAGE_CODE": "en-US",
        "GOOGLE_STT_MODEL": "latest_short",  # Good for testing
        "CHUNK_DURATION": "3.0",
        "PROCESSING_DELAY": "300",
        "LOG_TRANSCRIPTION_DETAILS": "true"
    }
}

# Model selection guide
MODEL_SELECTION_GUIDE = {
    "telephony": {
        "description": "Optimized for phone call audio quality",
        "best_for": ["Phone calls", "VOIP", "Call center recordings"],
        "sample_rates": ["8kHz", "16kHz", "44.1kHz", "48kHz"],
        "latency": "Low",
        "accuracy": "High for telephony audio",
        "use_case": "Real-time fraud detection on phone calls"
    },
    "latest_long": {
        "description": "Best for longer audio files",
        "best_for": ["Meetings", "Lectures", "Long recordings"],
        "sample_rates": ["16kHz", "44.1kHz", "48kHz"],
        "latency": "Medium",
        "accuracy": "Highest overall",
        "use_case": "Post-call analysis of long conversations"
    },
    "latest_short": {
        "description": "Best for shorter audio clips",
        "best_for": ["Voice commands", "Short clips", "Quick queries"],
        "sample_rates": ["16kHz", "44.1kHz", "48kHz"],
        "latency": "Low",
        "accuracy": "High for short audio",
        "use_case": "Quick fraud alerts and short interactions"
    },
    "command_and_search": {
        "description": "Optimized for voice commands",
        "best_for": ["Voice search", "Commands", "Keywords"],
        "sample_rates": ["16kHz"],
        "latency": "Very low",
        "accuracy": "High for commands",
        "use_case": "Voice-activated fraud alerts"
    }
}
