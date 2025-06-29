# backend/agents/audio_processor/agent.py - FIXED CONFIGURATION
"""
Audio Processing Agent - FIXED: Added missing configuration keys
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings

logger = logging.getLogger(__name__)

class AudioProcessorAgent(BaseAgent):
    """Audio processing agent for transcription and speaker identification"""
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Audio Processing Agent"
        )
        
        # Initialize audio processing capabilities
        self._initialize_audio_capabilities()
        
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"🎵 {self.agent_name} ready for transcription and speaker diarization")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with all required keys"""
        settings = get_settings()
        
        # Get base agent config from settings
        base_config = settings.get_agent_config("audio_processor")
        
        # Add audio-specific configuration with all required keys
        audio_config = {
            **base_config,
            "speech_language_code": "en-GB",  # FIXED: Added missing key
            "sample_rate": 16000,
            "channels": 1,
            "enable_speaker_diarization": True,
            "confidence_threshold": 0.8,
            "max_audio_duration_seconds": 300,
            "chunk_size_seconds": 30,
            "supported_formats": [".wav", ".mp3", ".m4a", ".flac"],
            "max_file_size_mb": 100
        }
        
        return audio_config
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get audio processing agent capabilities"""
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_audio_capabilities(self):
        """Initialize audio processing capabilities"""
        # Load supported formats from config
        self.supported_formats = self.config.get("supported_formats", [".wav", ".mp3", ".m4a", ".flac"])
        
        # Load sample scenarios for demo
        self.sample_scenarios = self._load_sample_scenarios()
        
        logger.info(f"📋 Loaded {len(self.sample_scenarios)} sample scenarios")
        logger.info(f"🎵 Audio config: Language={self.config.get('speech_language_code')}, Sample Rate={self.config.get('sample_rate')}")
    
    def _load_sample_scenarios(self) -> Dict[str, List[Dict]]:
        """Load sample transcription scenarios for demo purposes"""
        return {
            "investment": [
                {"speaker": "customer", "start": 0.0, "end": 5.0, 
                 "text": "My investment advisor just called saying there's a margin call on my trading account."},
                {"speaker": "agent", "start": 5.5, "end": 7.0, 
                 "text": "I see. Can you tell me more about this advisor?"},
                {"speaker": "customer", "start": 7.5, "end": 12.0, 
                 "text": "He's been guaranteeing thirty-five percent monthly returns and says I need to transfer fifteen thousand pounds immediately."},
                {"speaker": "agent", "start": 12.5, "end": 14.0,
                 "text": "I understand. Can you tell me how you first met this advisor?"},
                {"speaker": "customer", "start": 14.5, "end": 18.0,
                 "text": "He called me initially about cryptocurrency, then moved me to forex trading."},
                {"speaker": "agent", "start": 18.5, "end": 21.0,
                 "text": "Have you been able to withdraw any profits from this investment?"},
                {"speaker": "customer", "start": 21.5, "end": 25.0,
                 "text": "He says it's better to reinvest the profits for compound gains."}
            ],
            "romance": [
                {"speaker": "customer", "start": 0.0, "end": 4.0,
                 "text": "I need to send four thousand pounds to Turkey urgently."},
                {"speaker": "agent", "start": 4.5, "end": 6.0,
                 "text": "May I ask who this payment is for?"},
                {"speaker": "customer", "start": 6.5, "end": 10.0,
                 "text": "My partner Alex is stuck in Istanbul. We've been together for seven months online."},
                {"speaker": "agent", "start": 10.5, "end": 12.0,
                 "text": "Have you and Alex met in person?"},
                {"speaker": "customer", "start": 12.5, "end": 16.0,
                 "text": "Not yet, but we were planning to meet next month before this emergency happened."},
                {"speaker": "agent", "start": 16.5, "end": 18.0,
                 "text": "What kind of emergency is Alex facing?"},
                {"speaker": "customer", "start": 18.5, "end": 22.0,
                 "text": "He's in hospital and needs money for treatment before they'll help him properly."}
            ],
            "impersonation": [
                {"speaker": "customer", "start": 0.0, "end": 5.0,
                 "text": "Someone from bank security called saying my account has been compromised."},
                {"speaker": "agent", "start": 5.5, "end": 7.0,
                 "text": "Did they ask for any personal information?"},
                {"speaker": "customer", "start": 7.5, "end": 12.0,
                 "text": "Yes, they asked me to confirm my card details and PIN to verify my identity immediately."},
                {"speaker": "agent", "start": 12.5, "end": 15.0,
                 "text": "I need you to know that HSBC will never ask for your PIN over the phone."},
                {"speaker": "customer", "start": 15.5, "end": 18.0,
                 "text": "But they said fraudsters were trying to steal my money right now!"},
                {"speaker": "agent", "start": 18.5, "end": 21.0,
                 "text": "That person was the fraudster. Did you give them any information?"}
            ],
            "legitimate": [
                {"speaker": "customer", "start": 0.0, "end": 3.0,
                 "text": "Hi, I'd like to check my account balance please."},
                {"speaker": "agent", "start": 3.5, "end": 5.0,
                 "text": "Certainly, I can help you with that."},
                {"speaker": "customer", "start": 5.5, "end": 8.0,
                 "text": "I also want to set up a standing order for my rent."},
                {"speaker": "agent", "start": 8.5, "end": 10.0,
                 "text": "No problem. What's the amount and frequency?"},
                {"speaker": "customer", "start": 10.5, "end": 13.0,
                 "text": "Four hundred and fifty pounds monthly on the first of each month."}
            ]
        }
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Main processing method - transcribe audio file"""
        if isinstance(input_data, dict):
            file_path = input_data.get('file_path', input_data.get('audio_file_path', ''))
        else:
            file_path = str(input_data)
        
        return self.transcribe_audio(file_path)
    
    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Transcribe audio file with speaker diarization"""
        start_time = time.time()
        
        try:
            self.log_activity(f"Starting transcription", {"file_path": audio_file_path})
            
            if not audio_file_path:
                raise ValueError("No audio file path provided")
            
            filename = Path(audio_file_path).name.lower()
            scenario_type = self._determine_scenario_type(filename)
            segments = self.sample_scenarios.get(scenario_type, self.sample_scenarios["legitimate"])
            
            total_duration = max(seg["end"] for seg in segments) if segments else 0
            speakers_detected = list(set(s["speaker"] for s in segments))
            
            # Use the speech_language_code from config
            language_code = self.config.get("speech_language_code", "en-GB")
            
            result = {
                "file_path": audio_file_path,
                "total_duration": total_duration,
                "speaker_segments": segments,
                "speakers_detected": speakers_detected,
                "primary_language": language_code,  # FIXED: Use config value
                "confidence_score": 0.94,
                "processing_agent": self.agent_id,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=True)
            
            self.log_activity(f"Transcription complete", {
                "segments": len(segments),
                "duration": total_duration,
                "speakers": len(speakers_detected),
                "language": language_code,
                "processing_time": processing_time
            })
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, success=False)
            self.handle_error(e, "transcribe_audio")
            
            return {
                "error": str(e),
                "file_path": audio_file_path,
                "processing_agent": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _determine_scenario_type(self, filename: str) -> str:
        """Determine scenario type based on filename"""
        filename_lower = filename.lower()
        
        if "investment" in filename_lower:
            return "investment"
        elif "romance" in filename_lower:
            return "romance"
        elif "impersonation" in filename_lower:
            return "impersonation"
        else:
            return "legitimate"

# Create global instance
audio_processor_agent = AudioProcessorAgent()
