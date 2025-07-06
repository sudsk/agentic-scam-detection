# backend/agents/audio_processor/agent.py - FIXED TIMING VERSION
"""
Audio Processing Agent - FIXED: Proper timing and no delays in transcription
Fixed: Audio simulation timing to prevent Google STT timeout errors
"""

import asyncio
import logging
import time
import wave
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
import json
import threading
import queue
from collections import deque
import io

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings
from ...utils import get_current_timestamp

logger = logging.getLogger(__name__)

class LiveAudioBuffer:
    """Thread-safe audio buffer for live streaming - FIXED timing"""
    
    def __init__(self, chunk_size: int = 3200):  # 200ms at 16kHz, 16-bit
        self.chunk_size = chunk_size
        self.buffer = queue.Queue(maxsize=200)
        self.is_active = False
        self.total_chunks_added = 0
        self.total_chunks_yielded = 0
        self.last_activity = time.time()
        
    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk from telephony system or file simulation"""
        if self.is_active and len(audio_data) > 0:
            try:
                self.last_activity = time.time()
                
                if self.buffer.full():
                    try:
                        self.buffer.get_nowait()  # Remove oldest chunk
                    except queue.Empty:
                        pass
                
                self.buffer.put(audio_data, block=False)
                self.total_chunks_added += 1
                    
            except queue.Full:
                logger.warning("⚠️ Audio buffer full, dropping chunk")
    
    def get_audio_chunks(self):
        """Generator for audio chunks - FIXED timing"""
        logger.info("🎵 Audio chunk generator started")
        
        consecutive_empty_reads = 0
        max_empty_reads = 50  # FIXED: Reduced from 100 to 50 (5 seconds)
        
        while self.is_active:
            try:
                chunk = self.buffer.get(timeout=0.1)
                
                if chunk and len(chunk) > 0:
                    self.total_chunks_yielded += 1
                    consecutive_empty_reads = 0
                    yield chunk
                else:
                    consecutive_empty_reads += 1
                    
            except queue.Empty:
                consecutive_empty_reads += 1
                
                # FIXED: Check if we should stop due to inactivity - reduced timeout
                if time.time() - self.last_activity > 5.0:  # FIXED: 5 seconds instead of 10
                    logger.info(f"🎵 Audio generator stopping: no new data for 5s")
                    break
                    
                continue
            except Exception as e:
                logger.error(f"❌ Error in audio chunk generator: {e}")
                break
        
        logger.info(f"🎵 Audio generator completed: {self.total_chunks_yielded} chunks yielded")
    
    def start(self):
        self.is_active = True
        self.last_activity = time.time()
        logger.info("▶️ Audio buffer started")
        
    def stop(self):
        self.is_active = False
        logger.info(f"⏹️ Audio buffer stopped - {self.total_chunks_added} added, {self.total_chunks_yielded} yielded")

class AudioProcessorAgent(BaseAgent):
    """
    SIMPLIFIED Audio Processor - FIXED timing for Google STT
    """
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Simplified Audio Processor (Google STT Only)"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, LiveAudioBuffer] = {}
        
        # FIXED: Audio settings for better timing
        self.chunk_duration_ms = 100  # FIXED: 100ms chunks instead of 200ms
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = int(self.sample_rate * self.channels * 2 * self.chunk_duration_ms / 1000)
        
        # Initialize Google STT
        self._initialize_google_stt()
              
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"📞 {self.agent_name} initialized (FIXED TIMING)")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        settings = get_settings()
        
        return {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio",
            "enable_interim_results": True,
            "single_utterance": False,
            "max_streaming_duration": 300,  # FIXED: 5 minutes instead of 10
            "restart_recognition_timeout": 120  # FIXED: 2 minutes instead of 5
        }
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get capabilities"""
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_google_stt(self):
        """Initialize Google Speech-to-Text v1p1beta1 API"""
        
        try:
            logger.info("🔄 Initializing Google Speech-to-Text v1p1beta1...")
            
            from google.cloud import speech_v1p1beta1 as speech
            
            self.google_client = speech.SpeechClient()
            self.speech_types = speech
            self.transcription_source = "google_stt_v1p1beta1_simplified"
            
            logger.info("✅ Google Speech-to-Text v1p1beta1 client initialized")
            
            # Test configuration
            try:
                test_config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB",
                    model="phone_call",
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    # FORCED 2-speaker diarization
                    diarization_config=speech.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,  # FORCE 2 speakers
                        max_speaker_count=2   # FORCE 2 speakers
                    )
                )
                logger.info("✅ Phone call model with FORCED 2-speaker diarization configured")
                self.telephony_config_available = True
                
            except Exception as config_error:
                logger.warning(f"⚠️ Telephony config issue: {config_error}")
                self.telephony_config_available = False
            
        except ImportError as e:
            logger.error(f"❌ Google STT v1p1beta1 import failed: {e}")
            self.google_client = None
            self.speech_types = None
            self.transcription_source = "unavailable"
            self.telephony_config_available = False
        except Exception as e:
            logger.error(f"❌ Google STT v1p1beta1 initialization failed: {e}")
            self.google_client = None
            self.speech_types = None
            self.transcription_source = "failed"
            self.telephony_config_available = False

    # ... [keep all the existing methods like process, _prepare_live_call, etc.] ...

    async def _live_streaming_with_restart(self, session_id: str) -> None:
        """Live streaming with automatic restart - FIXED timing"""
        session = self.active_sessions[session_id]
        restart_interval = self.config.get("restart_recognition_timeout", 120)  # FIXED: 2 minutes
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            try:
                logger.info(f"🔄 Starting recognition cycle for {session_id}")
                
                # FIXED: Run recognition for shorter interval to prevent timeout
                await asyncio.wait_for(
                    self._live_streaming_recognition(session_id),
                    timeout=restart_interval
                )
                
            except asyncio.TimeoutError:
                # Normal restart due to timeout
                session["recognition_restart_count"] += 1
                logger.info(f"🔄 Restarting recognition for {session_id} (restart #{session['recognition_restart_count']})")
                await asyncio.sleep(0.5)  # FIXED: Shorter delay
                
            except Exception as e:
                logger.error(f"❌ Recognition error for {session_id}: {e}")
                session["recognition_restart_count"] += 1
                await asyncio.sleep(1.0)  # FIXED: Shorter delay
                
                if session["recognition_restart_count"] > 3:  # FIXED: Fewer retries
                    logger.error(f"❌ Too many restart attempts for {session_id}, stopping")
                    break

    async def _live_streaming_recognition(self, session_id: str) -> None:
        """
        SIMPLIFIED v1p1beta1 streaming recognition - FIXED timing
        """
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        
        try:
            logger.info(f"🎙️ Starting SIMPLIFIED v1p1beta1 recognition for {session_id}")
            
            # Create SIMPLIFIED recognition config
            if self.telephony_config_available:
                recognition_config = self.speech_types.RecognitionConfig(
                    encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB",
                    model="phone_call",
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    # FORCED 2-speaker diarization
                    diarization_config=self.speech_types.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,
                        max_speaker_count=2,
                        speaker_tag=1
                    ),
                    use_enhanced=True
                )
                logger.info("📞 Using phone_call model with FORCED 2-speaker diarization")
            else:
                recognition_config = self.speech_types.RecognitionConfig(
                    encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB",
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    diarization_config=self.speech_types.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,
                        max_speaker_count=2
                    ),
                    use_enhanced=True
                )
                logger.info("📞 Using default model with FORCED 2-speaker diarization")
            
            # Create streaming config
            streaming_config = self.speech_types.StreamingRecognitionConfig(
                config=recognition_config,
                interim_results=True,
                single_utterance=False
            )
            
            # FIXED: Audio generator with immediate streaming
            def audio_generator():
                chunk_count = 0
                logger.info("📞 FIXED audio generator: Immediate streaming with proper timing")
                
                for audio_chunk in audio_buffer.get_audio_chunks():
                    if session_id not in self.active_sessions:
                        break
                    
                    if audio_chunk and len(audio_chunk) > 0:
                        chunk_count += 1
                        
                        # Create request with audio content
                        request = self.speech_types.StreamingRecognizeRequest()
                        request.audio_content = audio_chunk
                        yield request
                        
                        # FIXED: No delay - immediate streaming
                
                logger.info(f"🎵 FIXED audio generator completed: {chunk_count} chunks")
            
            # Start streaming recognition
            logger.info(f"🚀 Starting FIXED Google STT streaming for {session_id}")
            
            responses = self.google_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )
            
            # Process responses
            response_count = 0
            final_count = 0
            interim_count = 0
            
            for response in responses:
                if session_id not in self.active_sessions:
                    break
                
                response_count += 1
                
                # Count responses
                for result in response.results:
                    if result.is_final:
                        final_count += 1
                    else:
                        interim_count += 1
                
                # Process with SIMPLIFIED logic
                await self._process_streaming_response_simplified(session_id, response, websocket_callback)
            
            logger.info(f"✅ FIXED recognition completed for {session_id}")
            logger.info(f"📊 Stats: {response_count} responses, {final_count} final, {interim_count} interim")
            
        except Exception as e:
            logger.error(f"❌ FIXED streaming error for {session_id}: {e}")
            
            # Send error notification
            try:
                await websocket_callback({
                    "type": "streaming_error",
                    "data": {
                        "session_id": session_id,
                        "error": str(e),
                        "timestamp": get_current_timestamp()
                    }
                })
            except:
                pass
            
            raise

    # ... [keep the existing _process_streaming_response_simplified method] ...

    async def _simulate_audio_file(
        self,
        session_id: str,
        audio_path: Path,
        duration: float
    ) -> None:
        """Simulate audio file - FIXED with immediate streaming and proper timing"""
        
        try:
            logger.info(f"🎬 FIXED audio simulation: {audio_path.name}")
            
            # Read audio file
            with wave.open(str(audio_path), 'rb') as wav_file:
                actual_sample_rate = wav_file.getframerate()
                actual_channels = wav_file.getnchannels()
                actual_sample_width = wav_file.getsampwidth()
                
                logger.info(f"📁 Audio specs: {actual_sample_rate}Hz, {actual_channels}ch, {actual_sample_width*8}bit")
                
                # FIXED: Calculate chunks based on ACTUAL sample rate and smaller chunks
                frames_per_chunk = int(actual_sample_rate * self.chunk_duration_ms / 1000)  # 100ms chunks
                total_frames = wav_file.getnframes()
                total_chunks = int(total_frames / frames_per_chunk)
                
                logger.info(f"🎵 FIXED: Streaming {total_chunks} chunks (frames_per_chunk: {frames_per_chunk})")
                
                # Get audio buffer
                audio_buffer = self.audio_buffers.get(session_id)
                if not audio_buffer:
                    logger.error(f"❌ No audio buffer for session {session_id}")
                    return
                
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                
                # FIXED: Start immediate streaming with minimal delay
                logger.info("🚀 FIXED: Starting immediate audio streaming...")
                
                while chunk_count < total_chunks and session_id in self.active_sessions:
                    # Read audio chunk
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    
                    if not audio_chunk:
                        logger.warning(f"⚠️ No audio data at chunk {chunk_count}/{total_chunks}")
                        break
                    
                    # FIXED: Add to buffer IMMEDIATELY with no delay
                    audio_buffer.add_audio_chunk(audio_chunk)
                    
                    chunk_count += 1
                    
                    # Progress logging every 50 chunks (5 seconds at 100ms)
                    if chunk_count % 50 == 0:
                        elapsed_seconds = (chunk_count * self.chunk_duration_ms) / 1000
                        logger.info(f"🎵 FIXED Progress: {elapsed_seconds:.1f}s / {duration:.1f}s ({chunk_count}/{total_chunks})")
                    
                    # FIXED: Minimal timing control to maintain real-time pace
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = asyncio.get_event_loop().time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        # FIXED: Very small sleep to maintain timing without causing delays
                        await asyncio.sleep(min(wait_time, 0.05))  # Max 50ms sleep
                    
                    # FIXED: Don't wait if we're behind - keep streaming
                
                logger.info(f"✅ FIXED Audio simulation complete: {chunk_count} chunks streamed")
                
                # Signal completion
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    websocket_callback = session.get("websocket_callback")
                    if websocket_callback:
                        await websocket_callback({
                            "type": "audio_simulation_complete",
                            "data": {
                                "session_id": session_id,
                                "chunks_streamed": chunk_count,
                                "duration": duration,
                                "timestamp": get_current_timestamp()
                            }
                        })
                
                # FIXED: Keep buffer active for shorter time
                await asyncio.sleep(1.0)  # Only 1 second instead of 3
                audio_buffer.stop()
                
        except Exception as e:
            logger.error(f"❌ Error in FIXED audio simulation: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")

    # ... [keep all other existing methods unchanged] ...

# Create global instance
audio_processor_agent = AudioProcessorAgent()
