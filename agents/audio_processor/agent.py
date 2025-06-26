# agents/audio_processor/agent.py
"""
Audio Processing Agent for HSBC Scam Detection System
Handles audio stream processing, enhancement, and transcription
"""

import asyncio
import json
import logging
import wave
import io
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from google.cloud import speech_v1
from google.adk.agents import Agent
from google.adk.models import LiteLLM

from ..shared.base_agent import BaseAgent, AgentCapability, Message, MessageType

logger = logging.getLogger(__name__)

class AudioProcessorAgent(BaseAgent):
    """
    Specialized agent for audio processing and transcription
    """
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="audio_enhancement",
                description="Enhance audio quality for better transcription",
                input_types=["audio_stream", "audio_file"],
                output_types=["enhanced_audio"],
                processing_time_ms=500,
                confidence_level=0.95
            ),
            AgentCapability(
                name="speech_to_text",
                description="Convert speech to text with speaker diarization",
                input_types=["audio_stream", "audio_file"],
                output_types=["transcript"],
                processing_time_ms=1000,
                confidence_level=0.92
            ),
            AgentCapability(
                name="speaker_identification",
                description="Identify and separate speakers in conversation",
                input_types=["audio_stream"],
                output_types=["speaker_segments"],
                processing_time_ms=750,
                confidence_level=0.88
            )
        ]
        
        config = {
            "sample_rate": 16000,
            "channels": 1,
            "audio_format": "LINEAR16",
            "language_code": "en-GB",
            "enable_speaker_diarization": True,
            "max_speakers": 2,
            "enable_word_confidence": True,
            "enable_automatic_punctuation": True,
            "model": "telephony",
            "use_enhanced": True
        }
        
        super().__init__(
            agent_id="audio_processor_001",
            agent_name="Audio Processing Agent",
            capabilities=capabilities,
            config=config
        )
        
        # Initialize Google Speech client (in production)
        self.speech_client = None  # speech_v1.SpeechClient() in production
        self.audio_buffer = {}
        self.processing_sessions = {}
    
    async def _initialize_agent(self):
        """Initialize audio processing components"""
        try:
            # Initialize speech recognition
            if self.config.get("use_google_speech", False):
                self.speech_client = speech_v1.SpeechClient()
            
            logger.info("Audio processor initialized successfully")
        except Exception as e:
            logger.error(f"Advanced audio enhancement failed: {e}")
            return audio_data
    
    async def _load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return numpy array with sample rate"""
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
    
    async def _diarize_speakers(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict]:
        """Perform speaker diarization"""
        # In production, use pyannote.audio or Google's speaker diarization
        # This is a simplified mock implementation
        
        duration = len(audio_data) / sample_rate
        
        # Mock speaker segments
        segments = []
        current_time = 0.0
        segment_duration = 5.0  # 5-second segments for demo
        
        speakers = ["customer", "agent"]
        speaker_idx = 0
        
        while current_time < duration:
            end_time = min(current_time + segment_duration, duration)
            segments.append({
                "speaker": speakers[speaker_idx % 2],
                "start": current_time,
                "end": end_time,
                "confidence": 0.85 + (0.1 * np.random.random())
            })
            current_time = end_time
            speaker_idx += 1
        
        return segments
    
    async def _transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Basic transcription without speaker diarization"""
        # Mock transcription for demo
        # In production, use Google Speech-to-Text API
        
        mock_transcripts = [
            "Hello, I need help with a transfer.",
            "My advisor said I need to send money urgently.",
            "They guaranteed high returns on my investment.",
            "I met someone online who needs financial help."
        ]
        
        import random
        transcript = random.choice(mock_transcripts)
        
        return {
            "text": transcript,
            "confidence": 0.92,
            "language": "en-GB",
            "words": []  # Would contain word-level timestamps in production
        }
    
    async def _transcribe_with_diarization(
        self, 
        audio_data: np.ndarray, 
        speaker_segments: List[Dict],
        sample_rate: int
    ) -> Dict[str, Any]:
        """Transcribe audio with speaker labels"""
        # Mock implementation combining transcription with speaker segments
        
        transcribed_segments = []
        
        # Sample transcripts for different scenarios
        investment_scam_transcript = [
            {"speaker": "customer", "text": "My investment advisor called about a margin call."},
            {"speaker": "agent", "text": "I see. Can you tell me more about this advisor?"},
            {"speaker": "customer", "text": "He's been guaranteeing 35% monthly returns."},
            {"speaker": "agent", "text": "That's concerning. Guaranteed returns are a red flag."},
            {"speaker": "customer", "text": "He says I need to transfer £15,000 immediately."}
        ]
        
        romance_scam_transcript = [
            {"speaker": "customer", "text": "I need to send money to my partner overseas."},
            {"speaker": "agent", "text": "Have you met this person in person?"},
            {"speaker": "customer", "text": "No, we've been dating online for 7 months."},
            {"speaker": "agent", "text": "Can you tell me why they need the money?"},
            {"speaker": "customer", "text": "They're stuck in Istanbul and need help."}
        ]
        
        # Select appropriate transcript based on context
        selected_transcript = investment_scam_transcript
        
        # Map to speaker segments
        for i, segment in enumerate(speaker_segments[:len(selected_transcript)]):
            if i < len(selected_transcript):
                transcribed_segments.append({
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": selected_transcript[i]["text"],
                    "confidence": segment["confidence"]
                })
        
        # Combine all text
        full_text = " ".join(seg["text"] for seg in transcribed_segments)
        
        return {
            "full_text": full_text,
            "segments": transcribed_segments,
            "speakers_detected": list(set(seg["speaker"] for seg in transcribed_segments)),
            "primary_language": "en-GB",
            "confidence_overall": 0.91
        }
    
    async def _analyze_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze audio quality metrics"""
        try:
            # Calculate basic metrics
            rms_energy = np.sqrt(np.mean(audio_data**2))
            
            # Signal-to-noise ratio estimation (simplified)
            signal_power = np.mean(audio_data**2)
            noise_floor = np.percentile(np.abs(audio_data), 10)**2
            snr_db = 10 * np.log10(signal_power / max(noise_floor, 1e-10))
            
            # Clipping detection
            max_val = np.max(np.abs(audio_data))
            clipping_detected = max_val > 0.95
            
            # Voice activity detection (simplified)
            energy_threshold = 0.01
            voice_activity_ratio = np.sum(np.abs(audio_data) > energy_threshold) / len(audio_data)
            
            return {
                "quality_score": min(snr_db / 30.0, 1.0),  # Normalize to 0-1
                "snr_db": float(snr_db),
                "rms_energy": float(rms_energy),
                "clipping_detected": clipping_detected,
                "voice_activity_ratio": float(voice_activity_ratio),
                "sample_rate": sample_rate,
                "duration_seconds": len(audio_data) / sample_rate,
                "quality_assessment": "good" if snr_db > 20 else "fair" if snr_db > 10 else "poor"
            }
            
        except Exception as e:
            logger.error(f"Audio quality analysis failed: {e}")
            return {
                "quality_score": 0.5,
                "error": str(e)
            }
    
    async def _handle_audio_enhancement(self, message: Message) -> Message:
        """Handle audio enhancement requests"""
        audio_data = message.data.get("audio_data")
        sample_rate = message.data.get("sample_rate", self.config["sample_rate"])
        
        try:
            # Perform enhancement
            enhanced_audio = await self._enhance_audio_advanced(
                np.frombuffer(audio_data, dtype=np.int16), 
                sample_rate
            )
            
            # Analyze improvement
            original_quality = await self._analyze_audio_quality(
                np.frombuffer(audio_data, dtype=np.int16), sample_rate
            )
            enhanced_quality = await self._analyze_audio_quality(enhanced_audio, sample_rate)
            
            improvement_score = enhanced_quality["quality_score"] - original_quality["quality_score"]
            
            return Message(
                message_id=str(uuid.uuid4()),
                message_type="audio_enhanced",
                source_agent=self.agent_id,
                target_agent=message.source_agent,
                session_id=message.session_id,
                data={
                    "enhanced_audio": enhanced_audio.tobytes(),
                    "improvement_score": improvement_score,
                    "original_quality": original_quality,
                    "enhanced_quality": enhanced_quality
                }
            )
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return self._create_error_response(message, str(e))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent-specific performance metrics"""
        base_metrics = self.get_status()
        
        # Add audio-specific metrics
        audio_metrics = {
            "active_streams": len(self.audio_buffer),
            "total_audio_processed_mb": self.metrics.messages_processed * 0.5,  # Estimate
            "average_transcription_accuracy": 0.92,
            "enhancement_success_rate": 0.95,
            "supported_languages": ["en-GB", "en-US"],
            "audio_quality_stats": {
                "excellent": 0.65,
                "good": 0.25,
                "fair": 0.08,
                "poor": 0.02
            }
        }
        
        base_metrics.update(audio_metrics)
        return base_metrics

# Agent functions for ADK
def process_audio_stream(audio_data: bytes, session_id: str) -> Dict[str, Any]:
    """Process audio stream and convert to text"""
    logger.info(f"🎵 Processing audio stream for session {session_id}")
    
    # Mock implementation for demo
    return {
        "session_id": session_id,
        "transcription": {
            "text": "Sample transcription from audio stream",
            "speaker": "customer",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat()
        },
        "audio_metadata": {
            "format": "wav",
            "sample_rate": 16000,
            "duration": 10.5
        }
    }

def transcribe_audio(audio_file_path: str) -> Dict[str, Any]:
    """Transcribe audio file with speaker diarization"""
    logger.info(f"📝 Transcribing audio file: {audio_file_path}")
    
    # Mock implementation based on filename
    filename = Path(audio_file_path).name.lower()
    
    if "investment" in filename:
        segments = [
            {"speaker": "customer", "start": 0.0, "end": 5.0, 
             "text": "My investment advisor just called saying there's a margin call."},
            {"speaker": "customer", "start": 5.5, "end": 10.0,
             "text": "He's guaranteeing thirty-five percent monthly returns."}
        ]
    else:
        segments = [
            {"speaker": "customer", "start": 0.0, "end": 3.0,
             "text": "I'd like to check my account balance."}
        ]
    
    return {
        "file_path": audio_file_path,
        "speaker_segments": segments,
        "confidence_score": 0.94,
        "timestamp": datetime.now().isoformat()
    }

# Create the Audio Processing Agent using Google ADK
audio_processing_agent = Agent(
    name="audio_processing_agent",
    model=LiteLLM("gemini/gemini-2.0-flash"),
    description="Audio stream processing and transcription agent",
    instruction="""You are an audio processing specialist for HSBC call center.

Your responsibilities:
1. Process real-time audio streams from customer calls
2. Convert speech to text with high accuracy
3. Identify speakers (customer vs agent)
4. Handle multiple audio formats and quality levels

Focus on:
- Accurate transcription even with background noise
- Proper speaker diarization
- Fast processing for real-time analysis
- Handling various accents and speech patterns""",
    tools=[process_audio_stream, transcribe_audio]
)

import uuidf"Failed to initialize audio processor: {e}")
            raise
    
    async def _process_message(self, message: Message) -> Optional[Message]:
        """Process incoming audio messages"""
        try:
            if message.message_type == "process_audio_stream":
                return await self._handle_audio_stream(message)
            elif message.message_type == "process_audio_file":
                return await self._handle_audio_file(message)
            elif message.message_type == "enhance_audio":
                return await self._handle_audio_enhancement(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing audio message: {e}")
            return self._create_error_response(message, str(e))
    
    async def _handle_audio_stream(self, message: Message) -> Message:
        """Handle real-time audio stream processing"""
        session_id = message.session_id
        audio_chunk = message.data.get("audio_chunk")
        is_final = message.data.get("is_final", False)
        
        # Buffer audio chunks
        if session_id not in self.audio_buffer:
            self.audio_buffer[session_id] = []
        
        self.audio_buffer[session_id].append(audio_chunk)
        
        # Process when we have enough audio or stream is final
        if is_final or len(self.audio_buffer[session_id]) >= 10:
            audio_data = b''.join(self.audio_buffer[session_id])
            self.audio_buffer[session_id] = []
            
            # Enhance audio
            enhanced_audio = await self._enhance_audio(audio_data)
            
            # Transcribe
            transcription = await self._transcribe_audio(enhanced_audio)
            
            # Create response
            response_data = {
                "session_id": session_id,
                "transcription": transcription,
                "audio_metadata": {
                    "duration_seconds": len(audio_data) / (self.config["sample_rate"] * 2),
                    "quality_score": 0.85
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return Message(
                message_id=str(uuid.uuid4()),
                message_type="transcription_complete",
                source_agent=self.agent_id,
                target_agent="orchestrator",
                session_id=session_id,
                data=response_data,
                priority=2
            )
        
        # Return acknowledgment for non-final chunks
        return Message(
            message_id=str(uuid.uuid4()),
            message_type="chunk_received",
            source_agent=self.agent_id,
            target_agent=message.source_agent,
            session_id=session_id,
            data={"chunks_buffered": len(self.audio_buffer.get(session_id, []))}
        )
    
    async def _handle_audio_file(self, message: Message) -> Message:
        """Process complete audio file"""
        file_path = message.data.get("file_path")
        session_id = message.session_id
        
        try:
            # Load audio file
            audio_data, sample_rate = await self._load_audio_file(file_path)
            
            # Enhance audio
            enhanced_audio = await self._enhance_audio_advanced(audio_data, sample_rate)
            
            # Perform speaker diarization
            speaker_segments = await self._diarize_speakers(enhanced_audio, sample_rate)
            
            # Transcribe with speaker labels
            transcription_with_speakers = await self._transcribe_with_diarization(
                enhanced_audio, speaker_segments, sample_rate
            )
            
            # Analyze audio quality
            quality_metrics = await self._analyze_audio_quality(audio_data, sample_rate)
            
            response_data = {
                "session_id": session_id,
                "file_path": file_path,
                "transcription": transcription_with_speakers,
                "speaker_segments": speaker_segments,
                "audio_quality": quality_metrics,
                "processing_complete": True,
                "timestamp": datetime.now().isoformat()
            }
            
            return Message(
                message_id=str(uuid.uuid4()),
                message_type="audio_file_processed",
                source_agent=self.agent_id,
                target_agent="orchestrator",
                session_id=session_id,
                data=response_data,
                priority=2
            )
            
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {e}")
            return self._create_error_response(message, f"Audio processing failed: {str(e)}")
    
    async def _enhance_audio(self, audio_data: bytes) -> bytes:
        """Basic audio enhancement"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize audio
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Apply simple noise reduction (in production, use more sophisticated methods)
            # High-pass filter to remove low-frequency noise
            from scipy import signal
            b, a = signal.butter(4, 300.0 / (self.config["sample_rate"] / 2), 'high')
            filtered_audio = signal.filtfilt(b, a, audio_float)
            
            # Convert back to int16
            enhanced_array = (filtered_audio * 32768).astype(np.int16)
            
            return enhanced_array.tobytes()
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return audio_data  # Return original if enhancement fails
    
    async def _enhance_audio_advanced(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Advanced audio enhancement using librosa"""
        try:
            # Remove silence
            trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=20)
            
            # Denoise using spectral gating (simplified)
            D = librosa.stft(trimmed_audio)
            magnitude = np.abs(D)
            
            # Simple spectral subtraction
            noise_profile = np.median(magnitude[:, :10], axis=1, keepdims=True)
            magnitude_denoised = magnitude - noise_profile
            magnitude_denoised = np.maximum(magnitude_denoised, 0)
            
            # Reconstruct audio
            phase = np.angle(D)
            D_denoised = magnitude_denoised * np.exp(1j * phase)
            enhanced_audio = librosa.istft(D_denoised)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(
