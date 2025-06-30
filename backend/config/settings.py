# backend/config/settings.py - UPDATED for Real Transcription
"""
Updated configuration settings for real server-side transcription
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
        
        # CORS settings
        self.allowed_origins = [
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001"
        ]
        
        # Paths
        self.data_path = Path("data")
        self.sample_audio_path = self.data_path / "sample_audio"
        self.uploads_path = Path("uploads")
        
        # WebSocket settings
        self.websocket_timeout = 300  # 5 minutes
        self.max_connections = 100
        
        # Agent configurations - UPDATED for real transcription
        self._setup_agent_configs()
    
    def _setup_agent_configs(self):
        """Setup agent configurations with real transcription settings"""
        
        # Audio Processing Agent Configuration - UPDATED
        self.audio_agent_config = {
            # === TRANSCRIPTION ENGINE SETTINGS ===
            "transcription_engine": os.getenv("TRANSCRIPTION_ENGINE", "mock_stt"),
            # Options: "mock_stt", "whisper", "google_stt"
            
            # === AUDIO PROCESSING SETTINGS ===
            "chunk_duration_seconds": float(os.getenv("CHUNK_DURATION", "2.0")),
            "overlap_seconds": float(os.getenv("CHUNK_OVERLAP", "0.5")),
            "processing_delay_ms": int(os.getenv("PROCESSING_DELAY", "300")),
            
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
            "enable_speaker_diarization": os.getenv("ENABLE_SPEAKER_DIARIZATION", "false").lower() == "true",
            "default_speaker_alternation": True,
            
            # === WHISPER SPECIFIC SETTINGS ===
            "whisper_model": os.getenv("WHISPER_MODEL", "base"),  # tiny, base, small, medium, large
            "whisper_device": os.getenv("WHISPER_DEVICE", "cpu"),  # cpu, cuda
            "whisper_language": os.getenv("WHISPER_LANGUAGE", "en"),
            
            # === GOOGLE STT SPECIFIC SETTINGS ===
            "google_language_code": os.getenv("GOOGLE_LANGUAGE_CODE", "en-GB"),
            "google_encoding": "LINEAR16",
            "google_enable_punctuation": True,
            "google_enable_word_confidence": True,
            
            # === PERFORMANCE SETTINGS ===
            "max_concurrent_transcriptions": int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "3")),
            "transcription_timeout": int(os.getenv("TRANSCRIPTION_TIMEOUT", "10")),
            
            # === FALLBACK SETTINGS ===
            "enable_fallback": True,
            "fallback_engine": "mock_stt",  # Fallback if primary engine fails
            
            # === LOGGING ===
            "log_transcription_details": os.getenv("LOG_TRANSCRIPTION_DETAILS", "true").lower() == "true",
            "save_transcription_logs": os.getenv("SAVE_TRANSCRIPTION_LOGS", "false").lower() == "true"
        }
        
        # Fraud Detection Agent Configuration
        self.fraud_agent_config = {
            "confidence_threshold": 0.7,
            "risk_escalation_threshold": 80,
            "pattern_weights": {
                "urgency_patterns": 25,
                "guaranteed_returns": 30,
                "third_party_instructions": 20,
                "emotional_manipulation": 15,
                "tech_complexity": 10
            },
            "scam_type_mapping": {
                "investment": ["guaranteed", "returns", "profit", "investment", "trading"],
                "romance": ["love", "relationship", "emergency", "hospital", "stuck"],
                "impersonation": ["bank", "security", "verify", "account", "suspicious"]
            }
        }
        
        # Policy Guidance Agent Configuration
        self.policy_agent_config = {
            "guidance_detail_level": "detailed",
            "include_regulatory_info": True,
            "max_recommended_actions": 6,
            "include_customer_education": True
        }
        
        # Case Management Agent Configuration
        self.case_agent_config = {
            "auto_create_threshold": 60,
            "priority_mapping": {
                "low": (0, 39),
                "medium": (40, 69), 
                "high": (70, 89),
                "critical": (90, 100)
            },
            "default_assignee": "fraud_team",
            "include_transcription": True
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
        configs = {
            "audio_processor": self.audio_agent_config,
            "fraud_detector": self.fraud_agent_config,
            "policy_guide": self.policy_agent_config,
            "case_manager": self.case_agent_config,
            "orchestrator": self.orchestrator_config
        }
        return configs.get(agent_type, {})
    
    def get_transcription_config(self) -> Dict[str, Any]:
        """Get transcription-specific configuration"""
        return {
            "engine": self.audio_agent_config["transcription_engine"],
            "chunk_duration": self.audio_agent_config["chunk_duration_seconds"],
            "sample_rate": self.audio_agent_config["sample_rate"],
            "channels": self.audio_agent_config["channels"],
            "processing_delay": self.audio_agent_config["processing_delay_ms"],
            "whisper_model": self.audio_agent_config["whisper_model"],
            "google_language": self.audio_agent_config["google_language_code"]
        }
    
    def validate_transcription_setup(self) -> Dict[str, Any]:
        """Validate transcription engine setup"""
        engine = self.audio_agent_config["transcription_engine"]
        validation_result = {
            "engine": engine,
            "status": "unknown",
            "requirements_met": False,
            "missing_dependencies": [],
            "recommendations": []
        }
        
        if engine == "mock_stt":
            validation_result.update({
                "status": "ready",
                "requirements_met": True,
                "recommendations": ["Mock transcription ready - no dependencies needed"]
            })
        
        elif engine == "whisper":
            try:
                import whisper
                validation_result.update({
                    "status": "ready", 
                    "requirements_met": True,
                    "recommendations": [f"Whisper model '{self.audio_agent_config['whisper_model']}' will be loaded"]
                })
            except ImportError:
                validation_result.update({
                    "status": "missing_dependencies",
                    "requirements_met": False,
                    "missing_dependencies": ["openai-whisper"],
                    "recommendations": ["Install with: pip install openai-whisper"]
                })
        
        elif engine == "google_stt":
            try:
                from google.cloud import speech
                # Check for credentials
                creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if creds_path and os.path.exists(creds_path):
                    validation_result.update({
                        "status": "ready",
                        "requirements_met": True,
                        "recommendations": ["Google Speech-to-Text credentials found"]
                    })
                else:
                    validation_result.update({
                        "status": "missing_credentials",
                        "requirements_met": False,
                        "missing_dependencies": ["Google Cloud credentials"],
                        "recommendations": [
                            "Set GOOGLE_APPLICATION_CREDENTIALS environment variable",
                            "Download service account JSON from Google Cloud Console"
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

# Environment variable examples for different setups
ENV_EXAMPLES = {
    "mock_transcription": {
        "TRANSCRIPTION_ENGINE": "mock_stt",
        "CHUNK_DURATION": "2.0",
        "PROCESSING_DELAY": "300"
    },
    "whisper_transcription": {
        "TRANSCRIPTION_ENGINE": "whisper", 
        "WHISPER_MODEL": "base",  # or "small", "medium", "large"
        "WHISPER_DEVICE": "cpu",  # or "cuda" for GPU
        "CHUNK_DURATION": "3.0",  # Longer chunks for better accuracy
        "PROCESSING_DELAY": "800"  # More processing time needed
    },
    "google_stt": {
        "TRANSCRIPTION_ENGINE": "google_stt",
        "GOOGLE_LANGUAGE_CODE": "en-GB",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json",
        "CHUNK_DURATION": "2.0",
        "PROCESSING_DELAY": "200"  # Fast cloud processing
    }
}
