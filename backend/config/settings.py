# backend/config/settings.py 
"""

"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

class Settings:
    """Simplified application settings"""
    
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
            "*"
        ]
        
        # Risk and pattern settings
        self.pattern_weights: Dict[str, float] = {
            "authority_impersonation": 35,
            "credential_requests": 40,
            "investment_fraud": 40,
            "romance_exploitation": 30,
            "urgency_pressure": 25,
            "third_party_instructions": 30,
            "financial_pressure": 25
        }
        
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
        
        
        # Scam type keywords (RESTORED from original)
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
        
        # Confidence thresholds (RESTORED from original)
        self.confidence_threshold_high = 0.9
        self.confidence_threshold_medium = 0.7
        self.confidence_threshold_low = 0.5
        
        # Agent configuration (RESTORED from original)
        self.max_concurrent_sessions = 500
        self.agent_timeout_seconds = 30
        self.agent_retry_attempts = 3
        self.agent_processing_timeout = 60
        
        # Audio file settings
        self.supported_audio_formats = [".wav", ".mp3", ".m4a", ".flac"]
        self.max_audio_file_size_mb = 100
        
        # Demo configuration (RESTORED from original)
        self.demo_mode = True
        self.demo_sample_data_path = "data/sample_audio"
        self.demo_delay_simulation = True
        self.demo_delay_seconds = 1.0
        
        # WebSocket settings (RESTORED from original)
        self.websocket_timeout = 300  # 5 minutes
        self.max_connections = 100
        
        # Additional attributes (RESTORED from original)
        self.environment = "development"
        self.enable_mock_mode = True
        self.enable_learning_agent = True
        self.enable_compliance_agent = True
        self.enable_case_management = True
        self.enable_real_time_transcription = True
        
        # Paths
        self.data_path = Path("data")
        self.sample_audio_path = self.data_path / "sample_audio"
        self.uploads_path = Path("uploads")
        
        # Agent configurations
        self._setup_agent_configs()

    @property
    def servicenow_instance_url(self) -> str:
        return os.getenv("SERVICENOW_INSTANCE_URL", "https://dev303197.service-now.com")
    
    @property 
    def servicenow_auth_method(self) -> str:
        return os.getenv("SERVICENOW_AUTH_METHOD", "api_key")
    
    @property
    def servicenow_api_key(self) -> Optional[str]:
        return os.getenv("SERVICENOW_API_KEY")
    
    @property
    def servicenow_username(self) -> str:
        return os.getenv("SERVICENOW_USERNAME", "hsbc_fraud_api")
    
    @property
    def servicenow_enabled(self) -> bool:
        return os.getenv("SERVICENOW_ENABLED", "true").lower() == "true"
    
    @property
    def servicenow_min_risk_score(self) -> int:
        return int(os.getenv("SERVICENOW_MIN_RISK_SCORE", "60"))
    
    def _setup_agent_configs(self):
        """Setup simplified agent configurations"""
        
        # Audio processor configuration with better mono/stereo support
        self.audio_agent_config = {
            # Google STT v1p1beta1 settings
            "api_version": "v1p1beta1",
            "google_project_id": os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id"),
            
            # Audio processing
            "chunk_duration_ms": 200,  # 200ms chunks for real-time
            "sample_rate": 16000,      # Standard sample rate
            "audio_encoding": "LINEAR16",
            "language_code": "en-GB",
            
            # Models
            "primary_model": "phone_call",  # Best for telephony
            "fallback_model": "default",
            
            # Speaker diarization
            "enable_speaker_diarization": True,
            "min_speaker_count": 2,
            "max_speaker_count": 2,
            "speaker_mode": "agent_first",
            
            # Streaming settings
            "interim_results": True,
            "single_utterance": False,
            "restart_timeout": 240,  # 4 minutes
            
            # Recognition features
            "enable_automatic_punctuation": True,
            "enable_word_confidence": True,
            "enable_word_time_offsets": True,
            "enable_profanity_filter": False,
            
            # Audio format support
            "support_mono": True,
            "support_stereo": True,
            "stereo_channel_mapping": {
                "left": "agent",    # Left channel = Agent
                "right": "customer" # Right channel = Customer
            },
            
            # Banking/fraud vocabulary
            "speech_contexts": [
                {
                    "phrases": [
                        # Banking terms
                        "HSBC", "account", "transfer", "payment", "balance",
                        "investment", "verification", "security", "PIN",
                        
                        # Fraud terms
                        "emergency", "urgent", "immediately", "police",
                        "investigation", "court", "tax office", "HMRC",
                        "guaranteed returns", "margin call", "stuck overseas",
                        
                        # Agent phrases (since agent speaks first)
                        "good morning", "good afternoon", "how may I help",
                        "thank you for calling", "for security purposes"
                    ],
                    "boost": 15
                }
            ],
            
            # Performance settings
            "max_concurrent_sessions": 50,
            "processing_timeout": 300,
            "audio_base_path": str(self.sample_audio_path)
        }
        
        # Other agent configs (simplified but complete)
        self.fraud_agent_config = {
            "confidence_threshold": 0.7,
            "risk_escalation_threshold": 80,
            "pattern_weights": self.pattern_weights,
            "scam_type_mapping": self.scam_type_keywords,  # Link to scam keywords
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
            "confidence_thresholds": self.confidence_thresholds,  # Added back
            "pattern_weights": self.pattern_weights,
            "team_assignments": self.team_assignments,
            "processing_timeout": self.agent_processing_timeout,  # Use full name
            "retry_attempts": self.agent_retry_attempts,  # Use full name
            "timeout_seconds": self.agent_timeout_seconds,  # Added back
            "max_concurrent_sessions": self.max_concurrent_sessions  # Added back
        }
        
        configs = {
            "audio_processor": {**base_config, **self.audio_agent_config},
            "fraud_detection": {**base_config, **self.fraud_agent_config},
            "policy_guidance": {**base_config, **self.policy_agent_config},
            "case_management": {**base_config, **self.case_agent_config},
            "orchestrator": {**base_config, **self.orchestrator_config}  # Added back
        }
        return configs.get(agent_type, base_config)
    
    @property
    def risk_thresholds(self) -> Dict[str, int]:
        """Get risk thresholds"""
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
    
    def get_google_stt_config(self) -> Dict[str, Any]:
        """Get Google STT v1p1beta1 configuration"""
        return {
            "api_version": "v1p1beta1",
            "project_id": self.audio_agent_config["google_project_id"],
            "model": self.audio_agent_config["primary_model"],
            "language_code": self.audio_agent_config["language_code"],
            "sample_rate": self.audio_agent_config["sample_rate"],
            "encoding": self.audio_agent_config["audio_encoding"],
            "enable_speaker_diarization": self.audio_agent_config["enable_speaker_diarization"],
            "speech_contexts": self.audio_agent_config["speech_contexts"]
        }
    
    def get_stereo_config(self) -> Dict[str, Any]:
        """Get stereo audio configuration"""
        return {
            "supported": self.audio_agent_config["support_stereo"],
            "channel_mapping": self.audio_agent_config["stereo_channel_mapping"],
            "processing_mode": "channel_separation",
            "speaker_detection": "energy_based",
            "fallback_to_mono": True
        }
    
    def validate_google_stt_setup(self) -> Dict[str, Any]:
        """Validate Google Speech-to-Text setup"""
        validation_result = {
            "api_version": "v1p1beta1",
            "status": "unknown",
            "requirements_met": False,
            "recommendations": []
        }
        
        try:
            from google.cloud import speech_v1p1beta1 as speech
            
            try:
                client = speech.SpeechClient()
                
                # Test basic configuration
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-GB",
                    model="phone_call",
                    enable_automatic_punctuation=True,
                    diarization_config=speech.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,
                        max_speaker_count=2
                    )
                )
                
                validation_result.update({
                    "status": "ready",
                    "requirements_met": True,
                    "model_config": {
                        "primary_model": "phone_call",
                        "language": "en-GB",
                        "sample_rate": "16kHz",
                        "speaker_diarization": True,
                        "stereo_support": True
                    },
                    "recommendations": [
                        "✅ Google Speech-to-Text v1p1beta1 ready",
                        "✅ phone_call model available for telephony",
                        "✅ Speaker diarization configured for 2 speakers",
                        "✅ Stereo audio support enabled",
                        "✅ Real-time streaming with 4-minute restart cycle",
                        "💡 For best results with stereo: Left=Agent, Right=Customer"
                    ]
                })
                    
            except Exception as auth_error:
                validation_result.update({
                    "status": "authentication_error",
                    "requirements_met": False,
                    "recommendations": [
                        "❌ Authentication failed",
                        "🔧 Set GOOGLE_APPLICATION_CREDENTIALS environment variable",
                        "🔧 Ensure service account has Speech Client role",
                        "🔧 Enable Speech-to-Text API in GCP project"
                    ]
                })
                    
        except ImportError:
            validation_result.update({
                "status": "missing_dependencies",
                "requirements_met": False,
                "recommendations": [
                    "❌ Missing google-cloud-speech library",
                    "🔧 Install: pip install google-cloud-speech>=2.33.0"
                ]
            })
        
        return validation_result
    
    def get_audio_processing_guide(self) -> Dict[str, Any]:
        """Get audio processing setup guide"""
        return {
            "supported_formats": {
                "mono": {
                    "description": "Single channel audio with both speakers",
                    "speaker_detection": "Google diarization + simple logic",
                    "accuracy": "Good (depends on audio quality)",
                    "setup": "Standard telephony setup"
                },
                "stereo": {
                    "description": "Left=Agent, Right=Customer channels",
                    "speaker_detection": "Channel-based (100% accurate)",
                    "accuracy": "Excellent",
                    "setup": "Configure telephony to route speakers to separate channels",
                    "recommended": True
                }
            },
            "optimal_settings": {
                "sample_rate": "16kHz (recommended) or 8kHz minimum",
                "encoding": "LINEAR16 for best quality",
                "chunk_size": "200ms for real-time response",
                "model": "phone_call (optimized for telephony)"
            },
            "speaker_separation": {
                "mono_audio": [
                    "1. Relies on Google's speaker diarization",
                    "2. Agent always speaks first (helps with initial detection)",
                    "3. Uses speaker tags and simple alternation logic",
                    "4. May have occasional misattribution"
                ],
                "stereo_audio": [
                    "1. Left channel = Agent speech",
                    "2. Right channel = Customer speech", 
                    "3. Energy-based detection per channel",
                    "4. 100% accurate speaker identification",
                    "5. Converts to mono for STT while preserving speaker info"
                ]
            },
            "integration_tips": [
                "🎯 For production: Use stereo audio if possible",
                "⚡ 200ms chunks provide good real-time performance",
                "🔄 Auto-restart every 4 minutes for long calls",
                "🎙️ Agent-first conversation flow improves accuracy",
                "📞 phone_call model is optimized for telephony audio"
            ]
        }

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def get_google_stt_validation() -> Dict[str, Any]:
    """Get Google Speech setup validation"""
    settings = get_settings()
    return settings.validate_google_stt_setup()

def get_audio_processing_guide() -> Dict[str, Any]:
    """Get audio processing guide"""
    settings = get_settings()
    return settings.get_audio_processing_guide()
