# backend/config/settings.py - UPDATED FOR GOOGLE SPEECH v1p1beta1 API
"""
Updated configuration settings for Google Speech-to-Text v1p1beta1 API
OPTIMIZED: Telephony model configuration and v1p1beta1 API settings for live calls
"""

import os
from pathlib import Path
from typing import Dict, Any, List

class Settings:
    """Application settings with Google Speech v1p1beta1 API configuration"""
    
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
        
        # Agent configurations with Google Speech v1p1beta1 API
        self._setup_agent_configs_v1p1beta1()
    
    def _setup_agent_configs_v1p1beta1(self):
        """Setup agent configurations with Google Speech v1p1beta1 API settings"""
        
        # ===== GOOGLE SPEECH v1p1beta1 API CONFIGURATION =====
        self.audio_agent_config = {
            # === API VERSION SETTINGS ===
            "api_version": "v1p1beta1",
            "transcription_engine": "google_stt_v1p1beta1_telephony",
            
            # === PROJECT SETTINGS ===
            "google_project_id": os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GCP_PROJECT_ID", "your-project-id")),
            
            # === MODEL CONFIGURATION ===
            # v1p1beta1 optimal models for telephony
            "model": os.getenv("GOOGLE_STT_MODEL", "phone_call"),  # OPTIMAL for phone calls
            "model_fallback": "default",  # Fallback if phone_call not available
            
            # Available models in v1p1beta1:
            # - "phone_call": Best for audio from phone calls (8kHz-16kHz) ✅ RECOMMENDED
            # - "video": Best for video with multiple speakers (premium model)
            # - "command_and_search": Best for short queries
            # - "default": General purpose model
            
            # === AUDIO PROCESSING SETTINGS ===
            "chunk_duration_seconds": float(os.getenv("CHUNK_DURATION", "0.2")),  # 200ms
            "overlap_seconds": float(os.getenv("CHUNK_OVERLAP", "0.0")),
            "processing_delay_ms": int(os.getenv("PROCESSING_DELAY", "20")),  # Minimal delay
            
            # === AUDIO FORMAT SETTINGS ===
            "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),  # 16kHz for high quality
            "channels": int(os.getenv("AUDIO_CHANNELS", "1")),  # Mono for phone calls
            "audio_encoding": "LINEAR16",  # Best quality for telephony
            
            # === LANGUAGE SETTINGS ===
            "language_code": os.getenv("GOOGLE_LANGUAGE_CODE", "en-GB"),  # Primary language
            "alternative_language_codes": ["en-US"],  # Additional languages
            
            # === RECOGNITION FEATURES (v1p1beta1) ===
            "enable_automatic_punctuation": True,
            "enable_word_confidence": True,
            "enable_word_time_offsets": True,
            "enable_profanity_filter": False,  # Keep original for fraud detection
            
            # === SPEAKER DIARIZATION (v1p1beta1) ===
            "enable_speaker_diarization": True,
            "min_speaker_count": 1,
            "max_speaker_count": 2,  # Customer + Agent
            
            # === STREAMING FEATURES (v1p1beta1) ===
            "interim_results": True,
            "single_utterance": False,  # Continuous recognition
            
            # === SPEECH ADAPTATION (v1p1beta1) ===
            # Custom vocabulary for banking/fraud terms
            "enable_speech_adaptation": True,
            "speech_contexts": [
                {
                    "phrases": [
                        # Banking terms
                        "investment", "transfer", "account", "balance", "transaction",
                        "verification", "security", "PIN", "password", "authentication",
                        "sort code", "account number", "overdraft", "mortgage",
                        
                        # Fraud-related terms
                        "guaranteed returns", "margin call", "investment opportunity",
                        "emergency", "urgent", "immediately", "before it's too late",
                        "police", "investigation", "court", "tax office", "HMRC",
                        "arrest warrant", "legal action", "prosecution", "bailiff",
                        
                        # Romance scam terms
                        "military", "overseas", "deployed", "medical emergency",
                        "stuck", "stranded", "visa", "customs", "hospital",
                        
                        # Authority impersonation
                        "bank security", "fraud department", "verification call",
                        "account compromised", "suspicious activity", "block account",
                        
                        # Financial amounts
                        "thousand", "million", "pounds", "dollars", "euros",
                        "payment", "deposit", "withdrawal", "fee", "charge"
                    ],
                    "boost": 15  # Boost recognition of these terms
                }
            ],
            
            # === PATHS ===
            "audio_base_path": str(self.sample_audio_path),
            "temp_audio_path": str(self.uploads_path / "temp_audio"),
            
            # === PERFORMANCE SETTINGS ===
            "max_concurrent_transcriptions": int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "50")),
            "transcription_timeout": int(os.getenv("TRANSCRIPTION_TIMEOUT", "300")),  # 5 minutes
            "recognition_restart_timeout": int(os.getenv("RESTART_TIMEOUT", "240")),  # 4 minutes for stability
            
            # === ERROR HANDLING ===
            "max_retry_attempts": 3,
            "retry_delay_seconds": 2.0,
            "enable_fallback_model": True,
            
            # === LOGGING ===
            "log_transcription_details": os.getenv("LOG_TRANSCRIPTION_DETAILS", "true").lower() == "true",
            "save_transcription_logs": os.getenv("SAVE_TRANSCRIPTION_LOGS", "false").lower() == "true",
            
            # === TELEPHONY SPECIFIC SETTINGS ===
            "telephony_integration": {
                "supported_codecs": ["PCMU", "PCMA", "LINEAR16"],
                "supported_sample_rates": [8000, 16000],
                "buffer_size_ms": 200,
                "jitter_buffer_ms": 50,
                "vad_enabled": True,  # Voice Activity Detection
                "noise_reduction": True,
                "echo_cancellation": True
            },
            
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
    
    def get_google_stt_v1p1beta1_config(self) -> Dict[str, Any]:
        """Get Google Speech-to-Text v1p1beta1 specific configuration"""
        return {
            "api_version": "v1p1beta1",
            "project_id": self.audio_agent_config["google_project_id"],
            "model": self.audio_agent_config["model"],
            "language_code": self.audio_agent_config["language_code"],
            "sample_rate": self.audio_agent_config["sample_rate"],
            "encoding": self.audio_agent_config["audio_encoding"],
            "enable_speaker_diarization": self.audio_agent_config["enable_speaker_diarization"],
            "interim_results": self.audio_agent_config["interim_results"],
            "speech_contexts": self.audio_agent_config["speech_contexts"],
            "enable_automatic_punctuation": self.audio_agent_config["enable_automatic_punctuation"],
            "enable_word_confidence": self.audio_agent_config["enable_word_confidence"],
            "enable_word_time_offsets": self.audio_agent_config["enable_word_time_offsets"]
        }
    
    def get_telephony_optimized_config(self) -> Dict[str, Any]:
        """Get telephony-optimized configuration for v1p1beta1"""
        return {
            "model": "phone_call",  # Optimal for phone calls
            "sample_rate": 16000,  # High quality telephony
            "channels": 1,  # Mono
            "encoding": "LINEAR16",
            "chunk_duration_ms": 200,  # Fast response
            "language_code": "en-GB",
            "enable_speaker_diarization": True,
            "min_speaker_count": 1,
            "max_speaker_count": 2,
            "enable_automatic_punctuation": True,
            "enable_word_confidence": True,
            "enable_word_time_offsets": True,
            "interim_results": True,
            "single_utterance": False,
            "speech_contexts": self.audio_agent_config["speech_contexts"],
            "telephony_integration": self.audio_agent_config["telephony_integration"]
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
    
    def validate_google_stt_v1p1beta1_setup(self) -> Dict[str, Any]:
        """Validate Google Speech-to-Text v1p1beta1 setup"""
        validation_result = {
            "api_version": "v1p1beta1",
            "status": "unknown",
            "requirements_met": False,
            "missing_dependencies": [],
            "recommendations": [],
            "auth_method": "service_account",
            "model_config": {}
        }
        
        try:
            from google.cloud import speech_v1p1beta1 as speech
            
            try:
                client = speech.SpeechClient()
                
                # Test telephony configuration
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-GB",
                    model="phone_call",  # Test telephony model
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    diarization_config=speech.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=1,
                        max_speaker_count=2
                    )
                )
                
                validation_result.update({
                    "status": "ready",
                    "requirements_met": True,
                    "auth_method": "service_account_authenticated",
                    "model_config": {
                        "primary_model": "phone_call",
                        "fallback_model": "default",
                        "language": "en-GB",
                        "sample_rate": "16kHz",
                        "encoding": "LINEAR16",
                        "speaker_diarization": True,
                        "streaming_enabled": True,
                        "max_streaming_duration": "5 minutes"
                    },
                    "recommendations": [
                        "Using Google Speech-to-Text v1p1beta1 API",
                        "phone_call model optimized for telephony",
                        "Speaker diarization enabled for customer/agent detection",
                        "Real-time streaming with 5-minute limit",
                        "Speech adaptation configured for banking/fraud terms",
                        "Automatic punctuation and word confidence enabled"
                    ]
                })
                    
            except Exception as auth_error:
                validation_result.update({
                    "status": "authentication_error",
                    "requirements_met": False,
                    "auth_method": "service_account_failed",
                    "missing_dependencies": [f"Authentication error: {auth_error}"],
                    "recommendations": [
                        "Ensure Google Cloud credentials are properly configured",
                        "Set GOOGLE_APPLICATION_CREDENTIALS environment variable",
                        "Grant Speech Client role to service account",
                        "Enable Speech-to-Text API in your GCP project",
                        "For v1p1beta1, ensure Speech-to-Text API is enabled",
                        "Test: gcloud auth application-default print-access-token"
                    ]
                })
                    
        except ImportError:
            validation_result.update({
                "status": "missing_dependencies",
                "requirements_met": False,
                "missing_dependencies": ["google-cloud-speech>=2.33.0"],
                "recommendations": [
                    "Install with: pip install google-cloud-speech>=2.33.0",
                    "For v1p1beta1 support, use latest version",
                    "Verify installation: python -c 'from google.cloud import speech_v1p1beta1'"
                ]
            })
        
        return validation_result
    
    def get_telephony_integration_guide(self) -> Dict[str, Any]:
        """Get telephony integration setup guide"""
        return {
            "api_version": "v1p1beta1",
            "optimal_configuration": {
                "model": "phone_call",
                "sample_rate": "16000Hz (recommended) or 8000Hz",
                "encoding": "LINEAR16",
                "channels": "1 (mono)",
                "chunk_size": "200ms",
                "speaker_diarization": "enabled",
                "streaming_limit": "5 minutes (auto-restart)"
            },
            "integration_steps": [
                "1. Configure telephony system to stream audio in real-time",
                "2. Set up audio buffering with 200ms chunks",
                "3. Initialize Google Speech v1p1beta1 client",
                "4. Create streaming recognition session",
                "5. Feed audio chunks to buffer",
                "6. Process transcription results with speaker tags",
                "7. Trigger fraud analysis on customer speech",
                "8. Handle automatic recognition restarts"
            ],
            "telephony_providers": {
                "twilio": {
                    "websocket_streams": "supported",
                    "sample_rate": "8000Hz or 16000Hz",
                    "format": "mulaw, linear16",
                    "integration_complexity": "medium"
                },
                "asterisk": {
                    "real_time_streaming": "supported",
                    "sample_rate": "8000Hz, 16000Hz",
                    "format": "linear16, ulaw, alaw",
                    "integration_complexity": "high"
                },
                "freeswitch": {
                    "websocket_streaming": "supported",
                    "sample_rate": "8000Hz, 16000Hz, 48000Hz",
                    "format": "linear16",
                    "integration_complexity": "medium"
                }
            },
            "performance_considerations": {
                "latency": "< 200ms for real-time fraud detection",
                "throughput": "500+ concurrent calls supported",
                "reliability": "Auto-restart every 4 minutes",
                "accuracy": "phone_call model optimized for telephony",
                "speaker_separation": "Customer/agent diarization"
            }
        }

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def get_google_stt_v1p1beta1_validation() -> Dict[str, Any]:
    """Get Google Speech v1p1beta1 setup validation"""
    settings = get_settings()
    return settings.validate_google_stt_v1p1beta1_setup()

def get_telephony_integration_guide() -> Dict[str, Any]:
    """Get telephony integration guide"""
    settings = get_settings()
    return settings.get_telephony_integration_guide()

# Environment variable examples for Google Speech v1p1beta1 API setup
ENV_EXAMPLES = {
    "google_stt_v1p1beta1_production": {
        "GOOGLE_CLOUD_PROJECT": "your-production-project",
        "GOOGLE_LANGUAGE_CODE": "en-GB",
        "GOOGLE_STT_MODEL": "phone_call",  # Optimal for telephony
        "CHUNK_DURATION": "0.2",  # 200ms chunks
        "PROCESSING_DELAY": "20",  # Minimal delay
        "RESTART_TIMEOUT": "240",  # 4 minutes for stability
        "ENABLE_ENHANCED_MODELS": "true",
        "MAX_CONCURRENT_TRANSCRIPTIONS": "50"
    },
    "google_stt_v1p1beta1_development": {
        "GOOGLE_CLOUD_PROJECT": "your-dev-project",
        "GOOGLE_LANGUAGE_CODE": "en-US",
        "GOOGLE_STT_MODEL": "phone_call",
        "CHUNK_DURATION": "0.2",
        "PROCESSING_DELAY": "50",
        "LOG_TRANSCRIPTION_DETAILS": "true",
        "MAX_CONCURRENT_TRANSCRIPTIONS": "10"
    }
}

# Model selection guide for v1p1beta1
MODEL_SELECTION_GUIDE_V1P1BETA1 = {
    "phone_call": {
        "description": "Optimized for audio from phone calls",
        "best_for": ["Telephony systems", "Call centers", "VoIP calls", "8kHz-16kHz audio"],
        "sample_rates": ["8kHz", "16kHz"],
        "latency": "Low",
        "accuracy": "Highest for phone audio",
        "use_case": "✅ RECOMMENDED for live fraud detection on phone calls",
        "speaker_diarization": "Excellent",
        "streaming": "Full support with 5-minute limit"
    },
    "video": {
        "description": "Best for video with multiple speakers",
        "best_for": ["Video conferences", "Meetings", "Multiple speakers"],
        "sample_rates": ["16kHz", "44.1kHz", "48kHz"],
        "latency": "Medium",
        "accuracy": "High for video/multi-speaker",
        "use_case": "Not optimal for telephony fraud detection",
        "speaker_diarization": "Very good",
        "cost": "Premium model (higher cost)"
    },
    "command_and_search": {
        "description": "Best for short queries and commands",
        "best_for": ["Voice commands", "Short queries", "Voice search"],
        "sample_rates": ["16kHz"],
        "latency": "Very low",
        "accuracy": "High for short audio",
        "use_case": "Not suitable for continuous call transcription",
        "speaker_diarization": "Limited",
        "streaming": "Limited support"
    },
    "default": {
        "description": "General purpose model",
        "best_for": ["General audio", "Long-form content", "Mixed audio types"],
        "sample_rates": ["16kHz", "44.1kHz", "48kHz"],
        "latency": "Medium",
        "accuracy": "Good overall",
        "use_case": "Fallback option if phone_call model unavailable",
        "speaker_diarization": "Good",
        "streaming": "Full support"
    }
}
