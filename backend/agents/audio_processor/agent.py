# backend/agents/audio_processor/agent.py - COMPLETE WORKING v2 API IMPLEMENTATION
"""
Audio Processing Agent for LIVE PHONE CALLS
COMPLETE: Correct Google Speech-to-Text v2 API implementation based on actual API structure
Real-time streaming recognition for telephony systems with progressive fraud detection
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
    """Thread-safe audio buffer for live streaming with proper chunk management"""
    
    def __init__(self, chunk_size: int = 3200):  # 200ms at 16kHz, 16-bit
        self.chunk_size = chunk_size
        self.buffer = queue.Queue(maxsize=100)  # Limit buffer size to prevent memory issues
        self.is_active = False
        self.total_chunks_added = 0
        self.total_chunks_yielded = 0
        
    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk from telephony system or file simulation"""
        if self.is_active and len(audio_data) > 0:
            try:
                # Don't block if buffer is full - drop old chunks to prevent lag
                if self.buffer.full():
                    try:
                        self.buffer.get_nowait()  # Remove oldest chunk
                    except queue.Empty:
                        pass
                
                self.buffer.put(audio_data, block=False)
                self.total_chunks_added += 1
                
                if self.total_chunks_added % 50 == 0:
                    logger.debug(f"📊 Audio buffer: {self.total_chunks_added} chunks added, queue size: {self.buffer.qsize()}")
                    
            except queue.Full:
                # Buffer is full, skip this chunk to maintain real-time performance
                logger.warning("⚠️ Audio buffer full, dropping chunk to maintain real-time performance")
    
    def get_audio_chunks(self):
        """Generator for audio chunks with improved flow control"""
        logger.info("🎵 Audio chunk generator started for live streaming")
        
        consecutive_empty_reads = 0
        max_empty_reads = 50  # Allow 5 seconds of no data (50 * 0.1s)
        
        while self.is_active and consecutive_empty_reads < max_empty_reads:
            try:
                # Get chunk with shorter timeout for more responsive streaming
                chunk = self.buffer.get(timeout=0.1)
                
                if chunk and len(chunk) > 0:
                    self.total_chunks_yielded += 1
                    consecutive_empty_reads = 0  # Reset counter on successful read
                    
                    if self.total_chunks_yielded <= 5 or self.total_chunks_yielded % 50 == 0:
                        logger.debug(f"🎵 Yielding chunk {self.total_chunks_yielded} (size: {len(chunk)} bytes)")
                    
                    yield chunk
                else:
                    consecutive_empty_reads += 1
                    
            except queue.Empty:
                # No audio available
                consecutive_empty_reads += 1
                continue
            except Exception as e:
                logger.error(f"❌ Error in audio chunk generator: {e}")
                break
        
        if consecutive_empty_reads >= max_empty_reads:
            logger.info(f"🎵 Audio generator stopping: no data for {max_empty_reads * 0.1}s")
        
        logger.info(f"🎵 Audio generator completed: {self.total_chunks_yielded} chunks yielded")
    
    def start(self):
        self.is_active = True
        logger.info("▶️ Audio buffer started")
        
    def stop(self):
        self.is_active = False
        logger.info(f"⏹️ Audio buffer stopped - {self.total_chunks_added} added, {self.total_chunks_yielded} yielded")
    
    def get_stats(self):
        """Get buffer statistics"""
        return {
            "is_active": self.is_active,
            "queue_size": self.buffer.qsize(),
            "total_chunks_added": self.total_chunks_added,
            "total_chunks_yielded": self.total_chunks_yielded,
            "chunk_size": self.chunk_size
        }

class AudioProcessorAgent(BaseAgent):
    """
    Live Audio Processor for Real-time Phone Calls
    COMPLETE: Correct Google Speech-to-Text v2 API implementation
    """
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Live Telephony Audio Processor (v2 API)"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, LiveAudioBuffer] = {}
        
        # Live streaming settings optimized for telephony
        self.chunk_duration_ms = 200  # 200ms chunks for good responsiveness
        self.sample_rate = 16000  # Standard telephony rate
        self.channels = 1  # Mono audio
        self.chunk_size = int(self.sample_rate * self.channels * 2 * self.chunk_duration_ms / 1000)  # 16-bit samples
        
        # Speaker tracking for multi-party calls
        self.speaker_states: Dict[str, str] = {}
        
        # Initialize Google STT v2 with correct implementation
        self._initialize_google_stt_v2_correct()
              
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"📞 {self.agent_name} initialized for live calls")
        logger.info(f"🎙️ Audio config: {self.chunk_duration_ms}ms chunks, {self.sample_rate}Hz, {self.chunk_size} bytes")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        settings = get_settings()
        
        return {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio",
            "enable_interim_results": True,
            "single_utterance": False,
            "max_streaming_duration": 300,
            "restart_recognition_timeout": 55  # Restart recognition every 55 seconds for v2 API stability
        }
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get capabilities"""
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_google_stt_v2_correct(self):
        """Initialize Google Speech-to-Text v2 API with correct structure understanding"""
        
        try:
            logger.info("🔄 Initializing Google Speech-to-Text v2 API (correct implementation)...")
            
            from google.cloud.speech_v2 import SpeechClient
            from google.cloud.speech_v2.types import cloud_speech
            
            self.google_client = SpeechClient()
            self.speech_v2 = cloud_speech
            self.transcription_source = "google_stt_v2_correct"
            
            logger.info("✅ Google Speech-to-Text v2 client initialized successfully")
            
            # The v2 API uses a different approach - we need to create a Recognizer first
            # Then use that recognizer for streaming recognition
            logger.info("📋 v2 API structure detected:")
            logger.info("  - Uses Recognizer objects with pre-configured settings")
            logger.info("  - RecognitionConfig doesn't have encoding field")
            logger.info("  - Uses ExplicitDecodingConfig or AutoDetectDecodingConfig")
            logger.info("  - Streaming uses recognizer path instead of inline config")
            
            # Test creating a recognizer configuration
            try:
                # v2 API uses ExplicitDecodingConfig for audio format specification
                explicit_config = cloud_speech.ExplicitDecodingConfig(
                    sample_rate_hertz=self.sample_rate,
                    audio_channel_count=self.channels,
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
                )
                logger.info("✅ v2 API: Successfully created ExplicitDecodingConfig")
                self.decoding_config = explicit_config
                self.use_explicit_decoding = True
                
            except Exception as explicit_error:
                logger.warning(f"⚠️ Could not create ExplicitDecodingConfig: {explicit_error}")
                
                # Try AutoDetectDecodingConfig as fallback
                try:
                    auto_config = cloud_speech.AutoDetectDecodingConfig()
                    logger.info("✅ v2 API: Using AutoDetectDecodingConfig as fallback")
                    self.decoding_config = auto_config
                    self.use_explicit_decoding = False
                    
                except Exception as auto_error:
                    logger.error(f"❌ Could not create any DecodingConfig: {auto_error}")
                    self.decoding_config = None
                    self.use_explicit_decoding = False
            
        except ImportError as e:
            logger.error(f"❌ Google STT v2 import failed: {e}")
            logger.error("Please install: pip install google-cloud-speech>=2.33.0")
            self.google_client = None
            self.speech_v2 = None
            self.transcription_source = "unavailable"
        except Exception as e:
            logger.error(f"❌ Google STT v2 initialization failed: {e}")
            self.google_client = None
            self.speech_v2 = None
            self.transcription_source = "failed"
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Main processing method"""
        if isinstance(input_data, dict):
            action = input_data.get('action', 'status')
            if action == 'start_live_call':
                return self._prepare_live_call(input_data)
            elif action == 'add_audio_chunk':
                return self._add_audio_chunk(input_data)
            elif action == 'stop_live_call':
                session_id = input_data.get('session_id')
                if session_id:
                    asyncio.create_task(self.stop_live_streaming(session_id))
                    return {"status": "stopping", "session_id": session_id}
        
        return {
            "agent_type": self.agent_type,
            "status": "ready_for_live_calls",
            "transcription_source": self.transcription_source,
            "active_calls": len(self.active_sessions),
            "agent_id": self.agent_id,
            "google_stt_v2_available": self.google_client is not None,
            "api_version": "v2_correct",
            "decoding_config_available": hasattr(self, 'decoding_config') and self.decoding_config is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    def _prepare_live_call(self, data: Dict) -> Dict[str, Any]:
        """Prepare for a live call with proper initialization"""
        session_id = data.get('session_id')
        if not session_id:
            return {"error": "session_id required"}
        
        if session_id in self.active_sessions:
            logger.warning(f"⚠️ Session {session_id} already exists, cleaning up first")
            asyncio.create_task(self.stop_live_streaming(session_id))
        
        # Initialize session
        self.active_sessions[session_id] = {
            "start_time": datetime.now(),
            "status": "ready",
            "total_audio_received": 0,
            "transcription_segments": [],
            "current_speaker": "customer",
            "recognition_restart_count": 0
        }
        
        # Initialize audio buffer with proper settings
        self.audio_buffers[session_id] = LiveAudioBuffer(self.chunk_size)
        self.audio_buffers[session_id].start()
        self.speaker_states[session_id] = "customer"
        
        logger.info(f"✅ Live call session {session_id} prepared and ready")
        
        return {
            "session_id": session_id,
            "status": "ready_for_live_audio",
            "chunk_size_bytes": self.chunk_size,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "buffer_active": True
        }
    
    def _add_audio_chunk(self, data: Dict) -> Dict[str, Any]:
        """Add audio chunk from telephony system or file simulation"""
        session_id = data.get('session_id')
        audio_data = data.get('audio_data')
        speaker = data.get('speaker', 'customer')
        
        if not session_id or session_id not in self.active_sessions:
            return {"error": "Invalid or unknown session"}
        
        if not audio_data:
            return {"error": "No audio data provided"}
        
        # Add to buffer
        self.audio_buffers[session_id].add_audio_chunk(audio_data)
        
        # Update session stats
        session = self.active_sessions[session_id]
        session["total_audio_received"] += len(audio_data)
        session["current_speaker"] = speaker
        self.speaker_states[session_id] = speaker
        
        return {
            "status": "audio_received",
            "bytes_received": len(audio_data),
            "total_bytes": session["total_audio_received"],
            "buffer_size": self.audio_buffers[session_id].buffer.qsize()
        }
    
    async def start_live_streaming(
        self, 
        session_id: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Start live streaming recognition for telephony using correct v2 API"""
        
        try:
            logger.info(f"📞 Starting live streaming recognition for session {session_id}")
            
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not prepared")
            
            if not self.google_client:
                raise Exception("Google Speech-to-Text v2 not available")
            
            if not hasattr(self, 'decoding_config') or self.decoding_config is None:
                raise Exception("DecodingConfig not available")
            
            # Get audio buffer
            audio_buffer = self.audio_buffers[session_id]
            if not audio_buffer.is_active:
                audio_buffer.start()
            
            # Update session status
            session = self.active_sessions[session_id]
            session["status"] = "streaming"
            session["websocket_callback"] = websocket_callback
            
            # Start streaming task with restart capability
            streaming_task = asyncio.create_task(
                self._live_streaming_with_restart_v2_correct(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            # Send confirmation
            await websocket_callback({
                "type": "live_streaming_started",
                "data": {
                    "session_id": session_id,
                    "processing_mode": "live_telephony_streaming_v2_correct",
                    "transcription_engine": self.transcription_source,
                    "sample_rate": self.sample_rate,
                    "chunk_duration_ms": self.chunk_duration_ms,
                    "api_version": "v2_correct",
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {
                "session_id": session_id,
                "status": "streaming",
                "processing_mode": "live_telephony_streaming_v2_correct"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting live streaming: {e}")
            return {"error": str(e)}
    
    async def _live_streaming_with_restart_v2_correct(self, session_id: str) -> None:
        """Live streaming with automatic restart for long calls using correct v2 API"""
        session = self.active_sessions[session_id]
        restart_interval = self.config.get("restart_recognition_timeout", 55)
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            try:
                logger.info(f"🔄 Starting recognition cycle for {session_id}")
                
                # Run recognition for the specified interval
                await asyncio.wait_for(
                    self._live_streaming_recognition_v2_correct(session_id),
                    timeout=restart_interval
                )
                
            except asyncio.TimeoutError:
                # Normal restart due to timeout
                session["recognition_restart_count"] += 1
                logger.info(f"🔄 Restarting recognition for {session_id} (restart #{session['recognition_restart_count']})")
                
                # Brief pause to allow cleanup
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Recognition error for {session_id}: {e}")
                # Try to restart after error
                session["recognition_restart_count"] += 1
                await asyncio.sleep(2.0)
                
                if session["recognition_restart_count"] > 5:
                    logger.error(f"❌ Too many restart attempts for {session_id}, stopping")
                    break
    
    async def _live_streaming_recognition_v2_correct(self, session_id: str) -> None:
        """
        CORRECT: v2 API streaming recognition using proper structure
        Based on actual v2 API capabilities from debug output
        """
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        
        try:
            logger.info(f"🎙️ Starting CORRECT v2 API recognition cycle for {session_id}")
            
            # Create recognizer configuration using v2 API structure
            # Note: Speaker diarization support is limited in v2 API
            
            # Step 1: Create basic recognition features (without diarization first)
            recognition_features = self.speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                enable_word_time_offsets=True
            )
            
            # Step 2: Create recognition config with telephony model
            recognition_config = self.speech_v2.RecognitionConfig(
                auto_decoding_config=self.decoding_config if not self.use_explicit_decoding else None,
                explicit_decoding_config=self.decoding_config if self.use_explicit_decoding else None,
                language_codes=["en-GB"],  # v2 uses language_codes (list)
                model="latest_short",  # Use latest_short instead of telephony for better compatibility
                features=recognition_features
            )
            
            logger.info("🎯 Using latest_short model (v2 API compatible)")
            
            # Step 3: Create streaming config
            streaming_features = self.speech_v2.StreamingRecognitionFeatures(
                interim_results=True,
                enable_voice_activity_events=True
            )
            
            streaming_config = self.speech_v2.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=streaming_features
            )
            
            # Step 4: Create the request generator (FIXED for better timing)
            def audio_generator_v2_correct():
                # First request with recognizer and config
                project_id = self._get_project_id()
                recognizer_path = f"projects/{project_id}/locations/global/recognizers/_"
                
                logger.info(f"🎯 Using recognizer path: {recognizer_path}")
                logger.info(f"🎯 Model: latest_short (v2 API)")
                logger.info(f"🎯 Decoding: {'Explicit' if self.use_explicit_decoding else 'Auto'}")
                
                yield self.speech_v2.StreamingRecognizeRequest(
                    recognizer=recognizer_path,
                    streaming_config=streaming_config
                )
                
                # Then send audio data with better timing
                chunk_count = 0
                last_yield_time = time.time()
                
                for audio_chunk in audio_buffer.get_audio_chunks():
                    if session_id not in self.active_sessions:
                        logger.info(f"🛑 Audio generator stopping for {session_id}")
                        break
                    
                    if audio_chunk and len(audio_chunk) > 0:
                        chunk_count += 1
                        current_time = time.time()
                        
                        # Log first few chunks and periodically
                        if chunk_count <= 3 or chunk_count % 50 == 0:
                            logger.debug(f"🎵 Sending audio chunk {chunk_count} ({len(audio_chunk)} bytes)")
                        
                        # Yield the audio chunk
                        yield self.speech_v2.StreamingRecognizeRequest(
                            audio=audio_chunk
                        )
                        
                        last_yield_time = current_time
                        
                        # Don't yield too frequently to avoid timeout
                        if chunk_count > 1:
                            time.sleep(0.02)  # Small delay between chunks
                
                logger.info(f"🎵 Audio generator completed: {chunk_count} chunks sent")
            
            # Step 5: Start streaming recognition
            logger.info(f"🚀 Starting Google STT v2 CORRECT streaming for {session_id}")
            
            responses = self.google_client.streaming_recognize(
                requests=audio_generator_v2_correct()
            )
            
            # Step 6: Process responses
            response_count = 0
            interim_count = 0
            final_count = 0
            
            for response in responses:
                if session_id not in self.active_sessions:
                    logger.info(f"🛑 Session {session_id} ended, stopping recognition")
                    break
                
                response_count += 1
                
                # Count interim vs final results
                for result in response.results:
                    if result.is_final:
                        final_count += 1
                    else:
                        interim_count += 1
                
                if response_count % 20 == 0:
                    logger.debug(f"📨 Processed {response_count} responses for {session_id} (Final: {final_count}, Interim: {interim_count})")
                
                await self._process_streaming_response_v2_correct(session_id, response, websocket_callback)
            
            logger.info(f"✅ CORRECT v2 API recognition cycle completed for {session_id}")
            logger.info(f"📊 Stats: {response_count} responses, {final_count} final, {interim_count} interim")
            
        except Exception as e:
            logger.error(f"❌ CORRECT v2 API streaming recognition error for {session_id}: {e}")
            logger.error(f"📋 Error details: {type(e).__name__}: {str(e)}")
            
            # Send error notification
            try:
                await websocket_callback({
                    "type": "streaming_error",
                    "data": {
                        "session_id": session_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "api_version": "v2_correct",
                        "model": "telephony",
                        "timestamp": get_current_timestamp()
                    }
                })
            except:
                pass  # Don't fail on callback errors
            
            raise  # Re-raise to trigger restart logic
    
    def _get_project_id(self) -> str:
        """Get Google Cloud project ID from environment or default"""
        import os
        return os.getenv('GOOGLE_CLOUD_PROJECT', os.getenv('GCP_PROJECT_ID', 'your-project-id'))
    
    async def _process_streaming_response_v2_correct(
        self, 
        session_id: str, 
        response, 
        websocket_callback: Callable
    ) -> None:
        """Process streaming response from correct v2 API"""
        
        try:
            # Handle streaming errors
            if hasattr(response, 'error') and response.error.code != 0:
                logger.error(f"❌ STT v2 streaming error for {session_id}: {response.error}")
                return
            
            # Process results from v2 API response
            for result in response.results:
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    
                    if transcript:
                        # Enhanced speaker detection for v2 API
                        detected_speaker = self._detect_speaker_v2_correct(session_id, alternative, result)
                        
                        # Get timing information from v2 API
                        start_time, end_time = self._extract_timing_v2_correct(alternative)
                        
                        # Create segment data
                        segment_data = {
                            "session_id": session_id,
                            "speaker": detected_speaker,
                            "text": transcript,
                            "confidence": getattr(alternative, 'confidence', 0.9),
                            "is_final": result.is_final,
                            "start": start_time,
                            "end": end_time,
                            "duration": end_time - start_time,
                            "processing_mode": "live_streaming_v2_correct",
                            "transcription_source": self.transcription_source,
                            "timestamp": get_current_timestamp()
                        }
                        
                        # Send to WebSocket
                        await websocket_callback({
                            "type": "transcription_segment",
                            "data": segment_data
                        })
                        
                        # Store in session for fraud analysis
                        session = self.active_sessions[session_id]
                        session["transcription_segments"].append(segment_data)
                        
                        # Trigger fraud analysis for customer speech
                        if result.is_final and detected_speaker == "customer":
                            await self._trigger_progressive_fraud_analysis(
                                session_id, transcript, websocket_callback
                            )
                        
                        # Log segment
                        status = "FINAL" if result.is_final else "interim"
                        logger.info(f"📝 {status}: {detected_speaker} - '{transcript[:50]}...'")
        
        except Exception as e:
            logger.error(f"❌ Error processing streaming response: {e}")
    
    def _detect_speaker_v2_correct(self, session_id: str, alternative, result) -> str:
        """Enhanced speaker detection for v2 API (with improved fallback logic)"""
        
        # Try v2 API speaker diarization if available
        if hasattr(result, 'speaker_info') and result.speaker_info:
            speaker_tag = result.speaker_info.speaker_tag
            logger.debug(f"🎯 v2 API result speaker_tag: {speaker_tag}")
            if speaker_tag == 1:
                return "customer"
            elif speaker_tag == 2:
                return "agent"
            else:
                return f"speaker_{speaker_tag}"
        
        # Try alternative level speaker info
        if hasattr(alternative, 'speaker_info') and alternative.speaker_info:
            speaker_tag = alternative.speaker_info.speaker_tag
            logger.debug(f"🎯 v2 API alternative speaker_tag: {speaker_tag}")
            if speaker_tag == 1:
                return "customer"
            elif speaker_tag == 2:
                return "agent"
            else:
                return f"speaker_{speaker_tag}"
        
        # Try words level speaker info
        if hasattr(alternative, 'words') and alternative.words:
            for word in alternative.words:
                if hasattr(word, 'speaker_info') and word.speaker_info:
                    speaker_tag = word.speaker_info.speaker_tag
                    logger.debug(f"🎯 v2 API word speaker_tag: {speaker_tag}")
                    if speaker_tag == 1:
                        return "customer"
                    elif speaker_tag == 2:
                        return "agent"
                    else:
                        return f"speaker_{speaker_tag}"
        
        # IMPROVED FALLBACK: Less frequent speaker switching for better demo experience
        session = self.active_sessions.get(session_id, {})
        segments = session.get("transcription_segments", [])
        
        # Start with customer for fraud detection priority
        if len(segments) == 0:
            return "customer"
        
        # Get the last speaker
        last_speaker = segments[-1].get("speaker", "customer") if segments else "customer"
        
        # Only switch speakers after accumulating significant speech
        # Count recent words to decide if it's time to switch
        recent_segments = segments[-3:] if len(segments) >= 3 else segments
        recent_words = sum(len(seg.get("text", "").split()) for seg in recent_segments)
        
        # Switch speaker every 8-15 words approximately (more realistic conversation flow)
        if recent_words > 12 and last_speaker == "customer":
            return "agent"
        elif recent_words > 8 and last_speaker == "agent":
            return "customer"
        else:
            return last_speaker  # Keep same speaker for continuity
    
    def _extract_timing_v2_correct(self, alternative) -> tuple:
        """Extract timing information from alternative in correct v2 API"""
        start_time = 0.0
        end_time = 0.0
        
        if hasattr(alternative, 'words') and alternative.words:
            first_word = alternative.words[0]
            last_word = alternative.words[-1]
            
            # v2 API uses start_offset and end_offset
            if hasattr(first_word, 'start_offset') and first_word.start_offset:
                start_time = first_word.start_offset.total_seconds()
            if hasattr(last_word, 'end_offset') and last_word.end_offset:
                end_time = last_word.end_offset.total_seconds()
                
            logger.debug(f"⏱️ v2 API timing: {start_time:.2f}s - {end_time:.2f}s")
        
        # Fallback: try alternative-level timing if available
        if start_time == 0.0 and end_time == 0.0:
            if hasattr(alternative, 'start_time') and alternative.start_time:
                start_time = alternative.start_time.total_seconds()
            if hasattr(alternative, 'end_time') and alternative.end_time:
                end_time = alternative.end_time.total_seconds()
        
        return start_time, end_time
    
    async def _trigger_progressive_fraud_analysis(
        self, 
        session_id: str, 
        customer_text: str, 
        websocket_callback: Callable
    ) -> None:
        """Trigger progressive fraud analysis as customer speaks"""
        
        try:
            # Add to session's customer speech history
            session = self.active_sessions[session_id]
            if "customer_speech_history" not in session:
                session["customer_speech_history"] = []
            
            session["customer_speech_history"].append(customer_text)
            
            # Combine recent customer speech for analysis
            recent_speech = " ".join(session["customer_speech_history"][-5:])
            
            if len(recent_speech.strip()) >= 20:  # Minimum text for analysis
                # Send to fraud detection system
                await websocket_callback({
                    "type": "progressive_fraud_analysis_trigger",
                    "data": {
                        "session_id": session_id,
                        "customer_text": recent_speech,
                        "text_length": len(recent_speech),
                        "segments_analyzed": len(session["customer_speech_history"]),
                        "timestamp": get_current_timestamp()
                    }
                })
                
                logger.info(f"🔍 Triggered fraud analysis: {len(recent_speech)} chars")
        
        except Exception as e:
            logger.error(f"❌ Error triggering fraud analysis: {e}")
    
    async def stop_live_streaming(self, session_id: str) -> Dict[str, Any]:
        """Stop live streaming for a session with proper cleanup"""
        try:
            logger.info(f"🛑 Stopping live streaming for {session_id}")
            
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            # Update session status
            session = self.active_sessions[session_id]
            session["status"] = "stopping"
            
            # Stop audio buffer
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].stop()
            
            # Cancel streaming task
            if session_id in self.streaming_tasks:
                task = self.streaming_tasks[session_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"✅ Streaming task cancelled for {session_id}")
                del self.streaming_tasks[session_id]
            
            # Cleanup session data
            del self.active_sessions[session_id]
            
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            
            if session_id in self.speaker_states:
                del self.speaker_states[session_id]
            
            logger.info(f"✅ Live streaming stopped and cleaned up for {session_id}")
            
            return {
                "session_id": session_id,
                "status": "stopped",
                "cleanup_complete": True,
                "timestamp": get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping live streaming for {session_id}: {e}")
            return {"error": str(e)}
    
    # ===== DEMO FILE PROCESSING METHODS =====
    
    async def start_realtime_processing(
        self,
        session_id: str,
        audio_filename: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Start real-time processing of demo audio file"""
        
        try:
            logger.info(f"🎬 Starting real-time demo processing: {audio_filename}")
            
            # Prepare session for live call
            prepare_result = self._prepare_live_call({"session_id": session_id})
            if "error" in prepare_result:
                return prepare_result
            
            # Load and validate audio file
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            if not audio_path.exists():
                return {"error": f"Audio file not found: {audio_filename}"}
            
            # Get audio file info
            try:
                with wave.open(str(audio_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    
                logger.info(f"📁 Demo file: {duration:.1f}s, {sample_rate}Hz")
            except Exception as e:
                logger.warning(f"⚠️ Could not read WAV info: {e}")
                duration = 30.0  # Default estimate
            
            # Start live streaming
            streaming_result = await self.start_live_streaming(session_id, websocket_callback)
            if "error" in streaming_result:
                return streaming_result
            
            # Start demo file simulation
            asyncio.create_task(
                self._simulate_live_audio_from_file(session_id, audio_path, duration)
            )
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "processing_mode": "demo_realtime_simulation",
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting realtime processing: {e}")
            return {"error": str(e)}
    
    async def _simulate_live_audio_from_file(
        self,
        session_id: str,
        audio_path: Path,
        duration: float
    ) -> None:
        """Simulate live audio streaming from demo file with proper timing"""
        
        try:
            logger.info(f"🎬 Simulating live audio stream from {audio_path.name}")
            
            # Read audio file
            with wave.open(str(audio_path), 'rb') as wav_file:
                # Verify audio format
                actual_sample_rate = wav_file.getframerate()
                actual_channels = wav_file.getnchannels()
                actual_sample_width = wav_file.getsampwidth()
                
                logger.info(f"📁 Audio file specs: {actual_sample_rate}Hz, {actual_channels}ch, {actual_sample_width*8}bit")
                
                if actual_channels != 1:
                    logger.warning(f"⚠️ Audio file is not mono, may affect processing")
                if actual_sample_width != 2:
                    logger.warning(f"⚠️ Audio file is not 16-bit, may affect processing")
                
                # Use the actual sample rate for calculations, not the target rate
                frames_per_chunk = int(actual_sample_rate * self.chunk_duration_ms / 1000)
                total_chunks = int(wav_file.getnframes() / frames_per_chunk)
                
                logger.info(f"🎵 Streaming {total_chunks} chunks at {self.chunk_duration_ms}ms intervals")
                logger.info(f"🎵 Chunk size: {frames_per_chunk} frames per chunk")
                
                # Stream chunks in real-time
                audio_buffer = self.audio_buffers.get(session_id)
                if not audio_buffer:
                    logger.error(f"❌ No audio buffer for session {session_id}")
                    return
                
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                
                # Start with a small delay to ensure recognition is ready
                await asyncio.sleep(0.5)
                
                while chunk_count < total_chunks and session_id in self.active_sessions:
                    # Read audio chunk
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    
                    if not audio_chunk:
                        break
                    
                    # Add to buffer for streaming recognition
                    audio_buffer.add_audio_chunk(audio_chunk)
                    
                    chunk_count += 1
                    
                    # Log progress every 5 seconds instead of every 25 chunks
                    elapsed_seconds = (chunk_count * self.chunk_duration_ms) / 1000
                    if elapsed_seconds > 0 and int(elapsed_seconds) % 5 == 0 and elapsed_seconds == int(elapsed_seconds):
                        logger.info(f"🎵 Demo streaming: {elapsed_seconds:.1f}s / {duration:.1f}s")
                    
                    # Maintain proper real-time timing
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = asyncio.get_event_loop().time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    elif wait_time < -0.5:  # If we're more than 500ms behind, catch up
                        logger.debug(f"⚠️ Audio streaming is {-wait_time:.1f}s behind real-time")
                
                logger.info(f"✅ Demo audio simulation complete: {chunk_count} chunks streamed")
                
                # Signal end of audio with a small delay
                await asyncio.sleep(1.0)
                
                # Send completion notification
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
                
                # Stop the audio buffer after a brief delay
                await asyncio.sleep(2.0)
                audio_buffer.stop()
                
        except Exception as e:
            logger.error(f"❌ Error simulating live audio: {e}")
    
    async def stop_realtime_processing(self, session_id: str) -> Dict[str, Any]:
        """Stop real-time processing"""
        return await self.stop_live_streaming(session_id)
    
    # ===== STATUS AND MANAGEMENT METHODS =====
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all active sessions"""
        
        sessions_status = {}
        for session_id, session in self.active_sessions.items():
            buffer_stats = self.audio_buffers[session_id].get_stats() if session_id in self.audio_buffers else {}
            
            sessions_status[session_id] = {
                "status": session.get("status", "unknown"),
                "start_time": session.get("start_time", "").isoformat() if session.get("start_time") else "",
                "total_audio_received": session.get("total_audio_received", 0),
                "transcription_segments": len(session.get("transcription_segments", [])),
                "current_speaker": session.get("current_speaker", "unknown"),
                "recognition_restart_count": session.get("recognition_restart_count", 0),
                "buffer_stats": buffer_stats,
                "has_streaming_task": session_id in self.streaming_tasks
            }
        
        return {
            "total_active_sessions": len(self.active_sessions),
            "transcription_engine": self.transcription_source,
            "google_stt_v2_available": self.google_client is not None,
            "sessions": sessions_status,
            "agent_id": self.agent_id,
            "timestamp": get_current_timestamp()
        }
    
    def get_session_transcription(self, session_id: str) -> Optional[List[Dict]]:
        """Get transcription segments for a session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].get("transcription_segments", [])
        return None
    
    async def cleanup_stale_sessions(self, max_age_hours: int = 2) -> int:
        """Clean up stale sessions"""
        current_time = datetime.now()
        stale_sessions = []
        
        for session_id, session in self.active_sessions.items():
            start_time = session.get("start_time")
            if start_time:
                age_hours = (current_time - start_time).total_seconds() / 3600
                if age_hours > max_age_hours:
                    stale_sessions.append(session_id)
        
        cleaned_count = 0
        for session_id in stale_sessions:
            try:
                await self.stop_live_streaming(session_id)
                cleaned_count += 1
            except Exception as e:
                logger.error(f"❌ Error cleaning stale session {session_id}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} stale sessions")
        
        return cleaned_count

# Create global instance
audio_processor_agent = AudioProcessorAgent()
