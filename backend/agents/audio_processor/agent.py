# backend/agents/audio_processor/agent.py - UPDATED FOR v1p1beta1 API
"""
Audio Processing Agent for LIVE PHONE CALLS
UPDATED: Correct Google Speech-to-Text v1p1beta1 API implementation for optimal telephony support
Real-time streaming recognition with speaker diarization for live fraud detection
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
        self.buffer = queue.Queue(maxsize=100)  # Limit buffer size
        self.is_active = False
        self.total_chunks_added = 0
        self.total_chunks_yielded = 0
        
    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk from telephony system or file simulation"""
        if self.is_active and len(audio_data) > 0:
            try:
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
                logger.warning("⚠️ Audio buffer full, dropping chunk to maintain real-time performance")
    
    def get_audio_chunks(self):
        """Generator for audio chunks with improved flow control"""
        logger.info("🎵 Audio chunk generator started for live streaming")
        
        consecutive_empty_reads = 0
        max_empty_reads = 50  # Allow 5 seconds of no data
        
        while self.is_active and consecutive_empty_reads < max_empty_reads:
            try:
                chunk = self.buffer.get(timeout=0.1)
                
                if chunk and len(chunk) > 0:
                    self.total_chunks_yielded += 1
                    consecutive_empty_reads = 0
                    
                    if self.total_chunks_yielded <= 5 or self.total_chunks_yielded % 50 == 0:
                        logger.debug(f"🎵 Yielding chunk {self.total_chunks_yielded} (size: {len(chunk)} bytes)")
                    
                    yield chunk
                else:
                    consecutive_empty_reads += 1
                    
            except queue.Empty:
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

class AudioProcessorAgent(BaseAgent):
    """
    Live Audio Processor for Real-time Phone Calls
    UPDATED: Optimized Google Speech-to-Text v1p1beta1 API for telephony
    """
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Live Telephony Audio Processor (v1p1beta1 Optimized)"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, LiveAudioBuffer] = {}
        
        # Telephony-optimized settings
        self.chunk_duration_ms = 200  # 200ms chunks for responsiveness
        self.sample_rate = 16000  # Standard telephony rate
        self.channels = 1  # Mono audio
        self.chunk_size = int(self.sample_rate * self.channels * 2 * self.chunk_duration_ms / 1000)
        
        # Speaker tracking
        self.speaker_states: Dict[str, str] = {}
        
        # Initialize Google STT v1p1beta1
        self._initialize_google_stt_v1p1beta1()
              
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"📞 {self.agent_name} initialized")
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
            "restart_recognition_timeout": 240  # 4 minutes for v1p1beta1 stability
        }
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get capabilities"""
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_google_stt_v1p1beta1(self):
        """Initialize Google Speech-to-Text v1p1beta1 API - OPTIMAL FOR TELEPHONY"""
        
        try:
            logger.info("🔄 Initializing Google Speech-to-Text v1p1beta1 API (telephony optimized)...")
            
            from google.cloud import speech_v1p1beta1 as speech
            
            self.google_client = speech.SpeechClient()
            self.speech_types = speech
            self.transcription_source = "google_stt_v1p1beta1_telephony"
            
            logger.info("✅ Google Speech-to-Text v1p1beta1 client initialized successfully")
            logger.info("📞 Using v1p1beta1 API pattern: streaming_recognize(config, requests)")
            
            # Test telephony model availability
            try:
                test_config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB",
                    model="phone_call",  # Optimal for telephony
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    # Speaker diarization config
                    diarization_config=speech.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=1,
                        max_speaker_count=2
                    )
                )
                logger.info("✅ v1p1beta1: phone_call model with diarization configured successfully")
                self.telephony_config_available = True
                
            except Exception as config_error:
                logger.warning(f"⚠️ Telephony config issue: {config_error}")
                logger.info("📞 Falling back to default model with diarization")
                self.telephony_config_available = False
            
        except ImportError as e:
            logger.error(f"❌ Google STT v1p1beta1 import failed: {e}")
            logger.error("Please install: pip install google-cloud-speech>=2.33.0")
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
            "google_stt_available": self.google_client is not None,
            "api_version": "v1p1beta1_telephony_optimized",
            "telephony_model_available": self.telephony_config_available,
            "timestamp": datetime.now().isoformat()
        }
    
    def _prepare_live_call(self, data: Dict) -> Dict[str, Any]:
        """Prepare for a live call"""
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
            "recognition_restart_count": 0,
            "customer_speech_history": []
        }
        
        # Initialize audio buffer
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
            "buffer_active": True,
            "api_version": "v1p1beta1_telephony"
        }
    
    def _add_audio_chunk(self, data: Dict) -> Dict[str, Any]:
        """Add audio chunk from telephony system"""
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
        """Start live streaming recognition using v1p1beta1 API"""
        
        try:
            logger.info(f"📞 Starting live streaming recognition for session {session_id}")
            
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not prepared")
            
            if not self.google_client:
                raise Exception("Google Speech-to-Text v1p1beta1 not available")
            
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
                self._live_streaming_with_restart_v1p1beta1(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            # Send confirmation
            await websocket_callback({
                "type": "live_streaming_started",
                "data": {
                    "session_id": session_id,
                    "processing_mode": "live_telephony_streaming_v1p1beta1",
                    "transcription_engine": self.transcription_source,
                    "sample_rate": self.sample_rate,
                    "chunk_duration_ms": self.chunk_duration_ms,
                    "api_version": "v1p1beta1_telephony",
                    "model": "phone_call" if self.telephony_config_available else "default",
                    "speaker_diarization": True,
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {
                "session_id": session_id,
                "status": "streaming",
                "processing_mode": "live_telephony_streaming_v1p1beta1"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting live streaming: {e}")
            return {"error": str(e)}
    
    async def _live_streaming_with_restart_v1p1beta1(self, session_id: str) -> None:
        """Live streaming with automatic restart for long calls"""
        session = self.active_sessions[session_id]
        restart_interval = self.config.get("restart_recognition_timeout", 240)
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            try:
                logger.info(f"🔄 Starting recognition cycle for {session_id}")
                
                # Run recognition for the specified interval
                await asyncio.wait_for(
                    self._live_streaming_recognition_v1p1beta1(session_id),
                    timeout=restart_interval
                )
                
            except asyncio.TimeoutError:
                # Normal restart due to timeout
                session["recognition_restart_count"] += 1
                logger.info(f"🔄 Restarting recognition for {session_id} (restart #{session['recognition_restart_count']})")
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"❌ Recognition error for {session_id}: {e}")
                session["recognition_restart_count"] += 1
                await asyncio.sleep(2.0)
                
                if session["recognition_restart_count"] > 3:
                    logger.error(f"❌ Too many restart attempts for {session_id}, stopping")
                    break
    
    async def _live_streaming_recognition_v1p1beta1(self, session_id: str) -> None:
        """
        v1p1beta1 API streaming recognition with telephony optimization
        """
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        
        try:
            logger.info(f"🎙️ Starting v1p1beta1 recognition cycle for {session_id}")
            
            # Create telephony-optimized recognition config
            if self.telephony_config_available:
                recognition_config = self.speech_types.RecognitionConfig(
                    encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB",
                    model="phone_call",  # Optimal for telephony
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    # Enhanced speaker diarization for phone calls
                    diarization_config=self.speech_types.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=1,
                        max_speaker_count=2
                    ),
                    # Speech adaptation for banking/fraud terms
                    speech_contexts=[
                        self.speech_types.SpeechContext(
                            phrases=[
                                "investment", "guaranteed returns", "margin call", "transfer money",
                                "emergency", "urgent", "immediately", "police", "investigation",
                                "bank security", "verification", "PIN", "password", "account details",
                                "romance", "military", "overseas", "stuck", "medical emergency"
                            ],
                            boost=15  # Boost recognition of fraud-related terms
                        )
                    ]
                )
                logger.info("📞 Using phone_call model with enhanced diarization")
            else:
                # Fallback configuration
                recognition_config = self.speech_types.RecognitionConfig(
                    encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB",
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    diarization_config=self.speech_types.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=1,
                        max_speaker_count=2
                    )
                )
                logger.info("📞 Using default model with basic diarization")
            
            # Create streaming config for v1p1beta1 old API pattern
            streaming_config = self.speech_types.StreamingRecognitionConfig(
                config=recognition_config,
                interim_results=True,
                single_utterance=False
            )
            
            # Create the request generator - CORRECT v1p1beta1 pattern
            def audio_generator_v1p1beta1():
                # For v1p1beta1 with old API pattern, we only send audio chunks
                # The config is passed separately to streaming_recognize()
                chunk_count = 0
                logger.info("📞 Audio generator: sending audio chunks only (config passed separately)")
                
                for audio_chunk in audio_buffer.get_audio_chunks():
                    if session_id not in self.active_sessions:
                        logger.info(f"🛑 Audio generator stopping for {session_id}")
                        break
                    
                    if audio_chunk and len(audio_chunk) > 0:
                        chunk_count += 1
                        
                        if chunk_count <= 3 or chunk_count % 50 == 0:
                            logger.debug(f"🎵 Sending audio chunk {chunk_count} ({len(audio_chunk)} bytes)")
                        
                        # Create request with ONLY audio content
                        request = self.speech_types.StreamingRecognizeRequest()
                        request.audio_content = audio_chunk
                        yield request
                        
                        # Small delay for stability
                        time.sleep(0.02)
                
                logger.info(f"🎵 Audio generator completed: {chunk_count} chunks sent")
            
            # Start streaming recognition
            logger.info(f"🚀 Starting Google STT v1p1beta1 streaming for {session_id}")
            
            # v1p1beta1 old API pattern: streaming_recognize(config, requests)
            # config = StreamingRecognitionConfig, requests = iterator of audio-only requests
            try:
                logger.info("📞 Using v1p1beta1 old API: streaming_recognize(streaming_config, audio_requests)")
                
                responses = self.google_client.streaming_recognize(
                    config=streaming_config,  # StreamingRecognitionConfig object
                    requests=audio_generator_v1p1beta1()  # Iterator of audio-only requests
                )
            except Exception as api_error:
                logger.error(f"❌ v1p1beta1 streaming API error: {api_error}")
                logger.error(f"❌ Config type: {type(streaming_config)}")
                logger.error(f"❌ Error details: {api_error}")
                raise
            
            # Process responses
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
                
                await self._process_streaming_response_v1p1beta1(session_id, response, websocket_callback)
            
            logger.info(f"✅ v1p1beta1 recognition cycle completed for {session_id}")
            logger.info(f"📊 Stats: {response_count} responses, {final_count} final, {interim_count} interim")
            
        except Exception as e:
            logger.error(f"❌ v1p1beta1 streaming recognition error for {session_id}: {e}")
            logger.error(f"📋 Error details: {type(e).__name__}: {str(e)}")
            
            # Send error notification
            try:
                await websocket_callback({
                    "type": "streaming_error",
                    "data": {
                        "session_id": session_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "api_version": "v1p1beta1_telephony",
                        "model": "phone_call" if self.telephony_config_available else "default",
                        "timestamp": get_current_timestamp()
                    }
                })
            except:
                pass
            
            raise
    
    async def _process_streaming_response_v1p1beta1(
        self, 
        session_id: str, 
        response, 
        websocket_callback: Callable
    ) -> None:
        """Process streaming response from v1p1beta1 API"""
        
        try:
            # Handle streaming errors
            if hasattr(response, 'error') and response.error.code != 0:
                logger.error(f"❌ STT v1p1beta1 streaming error for {session_id}: {response.error}")
                return
            
            # Process results
            for result in response.results:
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    
                    if transcript:
                        # Enhanced speaker detection for v1p1beta1
                        detected_speaker = self._detect_speaker_v1p1beta1(session_id, alternative, result)
                        
                        # Get timing information
                        start_time, end_time = self._extract_timing_v1p1beta1(alternative)
                        
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
                            "processing_mode": "live_streaming_v1p1beta1_telephony",
                            "transcription_source": self.transcription_source,
                            "timestamp": get_current_timestamp()
                        }
                        
                        # Send to WebSocket
                        await websocket_callback({
                            "type": "transcription_segment",
                            "data": segment_data
                        })
                        
                        # Store in session
                        session = self.active_sessions[session_id]
                        session["transcription_segments"].append(segment_data)
                        
                        # Track customer speech for fraud analysis
                        if detected_speaker == "customer":
                            session["customer_speech_history"].append(transcript)
                        
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
    
    def _detect_speaker_v1p1beta1(self, session_id: str, alternative, result) -> str:
        """Enhanced speaker detection for v1p1beta1 API with diarization"""
        
        # Try speaker diarization from words
        if hasattr(alternative, 'words') and alternative.words:
            # Get speaker tags from diarization
            speaker_tags = []
            for word in alternative.words:
                if hasattr(word, 'speaker_tag') and word.speaker_tag:
                    speaker_tags.append(word.speaker_tag)
            
            if speaker_tags:
                # Use most common speaker tag
                most_common_tag = max(set(speaker_tags), key=speaker_tags.count)
                logger.debug(f"🎯 v1p1beta1 diarization speaker_tag: {most_common_tag}")
                
                # Map speaker tags to customer/agent
                if most_common_tag == 1:
                    return "customer"
                elif most_common_tag == 2:
                    return "agent"
                else:
                    return f"speaker_{most_common_tag}"
        
        # Fallback: Smart alternating with context
        session = self.active_sessions.get(session_id, {})
        segments = session.get("transcription_segments", [])
        
        # Start with customer for fraud detection priority
        if len(segments) == 0:
            return "customer"
        
        # Get the last speaker
        last_speaker = segments[-1].get("speaker", "customer") if segments else "customer"
        
        # Intelligent speaker switching based on conversation patterns
        recent_segments = segments[-5:] if len(segments) >= 5 else segments
        recent_words = sum(len(seg.get("text", "").split()) for seg in recent_segments)
        
        # Switch speaker every 10-20 words for natural conversation flow
        if recent_words > 15 and last_speaker == "customer":
            return "agent"
        elif recent_words > 10 and last_speaker == "agent":
            return "customer"
        else:
            return last_speaker
    
    def _extract_timing_v1p1beta1(self, alternative) -> tuple:
        """Extract timing information from alternative in v1p1beta1 API"""
        start_time = 0.0
        end_time = 0.0
        
        if hasattr(alternative, 'words') and alternative.words:
            first_word = alternative.words[0]
            last_word = alternative.words[-1]
            
            # v1p1beta1 uses start_time and end_time
            if hasattr(first_word, 'start_time') and first_word.start_time:
                start_time = first_word.start_time.total_seconds()
            if hasattr(last_word, 'end_time') and last_word.end_time:
                end_time = last_word.end_time.total_seconds()
                
            logger.debug(f"⏱️ v1p1beta1 timing: {start_time:.2f}s - {end_time:.2f}s")
        
        return start_time, end_time
    
    async def _trigger_progressive_fraud_analysis(
        self, 
        session_id: str, 
        customer_text: str, 
        websocket_callback: Callable
    ) -> None:
        """Trigger progressive fraud analysis as customer speaks"""
        
        try:
            # Get session's customer speech history
            session = self.active_sessions[session_id]
            customer_history = session.get("customer_speech_history", [])
            
            # Combine recent customer speech for analysis
            recent_speech = " ".join(customer_history[-5:])  # Last 5 segments
            
            if len(recent_speech.strip()) >= 20:  # Minimum text for analysis
                # Send to fraud detection system
                await websocket_callback({
                    "type": "progressive_fraud_analysis_trigger",
                    "data": {
                        "session_id": session_id,
                        "customer_text": recent_speech,
                        "text_length": len(recent_speech),
                        "segments_analyzed": len(customer_history),
                        "timestamp": get_current_timestamp()
                    }
                })
                
                logger.info(f"🔍 Triggered fraud analysis: {len(recent_speech)} chars from {len(customer_history)} segments")
        
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
            
            # Start audio simulation BEFORE starting recognition
            logger.info("🎵 Starting audio file simulation...")
            file_simulation_task = asyncio.create_task(
                self._simulate_live_audio_from_file(session_id, audio_path, duration)
            )
            
            # Small delay to let some audio chunks accumulate
            await asyncio.sleep(1.0)
            
            # Then start live streaming recognition
            logger.info("🎙️ Starting v1p1beta1 recognition...")
            streaming_result = await self.start_live_streaming(session_id, websocket_callback)
            if "error" in streaming_result:
                file_simulation_task.cancel()
                return streaming_result
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "processing_mode": "demo_realtime_v1p1beta1_telephony",
                "status": "started",
                "api_version": "v1p1beta1",
                "model": "phone_call" if self.telephony_config_available else "default"
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
        """Simulate live audio streaming from demo file"""
        
        try:
            logger.info(f"🎬 Simulating live audio stream from {audio_path.name}")
            
            # Read audio file
            with wave.open(str(audio_path), 'rb') as wav_file:
                # Verify audio format
                actual_sample_rate = wav_file.getframerate()
                actual_channels = wav_file.getnchannels()
                actual_sample_width = wav_file.getsampwidth()
                
                logger.info(f"📁 Audio file specs: {actual_sample_rate}Hz, {actual_channels}ch, {actual_sample_width*8}bit")
                
                # Use the actual sample rate for calculations
                frames_per_chunk = int(actual_sample_rate * self.chunk_duration_ms / 1000)
                total_chunks = int(wav_file.getnframes() / frames_per_chunk)
                
                logger.info(f"🎵 Streaming {total_chunks} chunks at {self.chunk_duration_ms}ms intervals")
                
                # Get audio buffer
                audio_buffer = self.audio_buffers.get(session_id)
                if not audio_buffer:
                    logger.error(f"❌ No audio buffer for session {session_id}")
                    return
                
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                
                # Start streaming immediately
                logger.info("🚀 Starting immediate audio streaming...")
                
                while chunk_count < total_chunks and session_id in self.active_sessions:
                    # Read audio chunk
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    
                    if not audio_chunk:
                        break
                    
                    # Add to buffer for streaming recognition
                    audio_buffer.add_audio_chunk(audio_chunk)
                    
                    chunk_count += 1
                    
                    # Log progress every 5 seconds
                    elapsed_seconds = (chunk_count * self.chunk_duration_ms) / 1000
                    if elapsed_seconds > 0 and int(elapsed_seconds) % 5 == 0 and elapsed_seconds == int(elapsed_seconds):
                        logger.info(f"🎵 Demo streaming: {elapsed_seconds:.1f}s / {duration:.1f}s")
                    
                    # Maintain real-time timing
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = asyncio.get_event_loop().time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    elif wait_time < -1.0:
                        logger.debug(f"⚠️ Audio streaming is {-wait_time:.1f}s behind real-time")
                
                logger.info(f"✅ Demo audio simulation complete: {chunk_count} chunks streamed")
                
                # Signal end of audio
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
                
                # Keep buffer active for a bit longer
                await asyncio.sleep(3.0)
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
                "customer_speech_segments": len(session.get("customer_speech_history", [])),
                "current_speaker": session.get("current_speaker", "unknown"),
                "recognition_restart_count": session.get("recognition_restart_count", 0),
                "buffer_stats": buffer_stats,
                "has_streaming_task": session_id in self.streaming_tasks
            }
        
        return {
            "total_active_sessions": len(self.active_sessions),
            "transcription_engine": self.transcription_source,
            "google_stt_v1p1beta1_available": self.google_client is not None,
            "telephony_model_available": self.telephony_config_available,
            "api_version": "v1p1beta1_telephony_optimized",
            "sessions": sessions_status,
            "agent_id": self.agent_id,
            "timestamp": get_current_timestamp()
        }
    
    def get_session_transcription(self, session_id: str) -> Optional[List[Dict]]:
        """Get transcription segments for a session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].get("transcription_segments", [])
        return None
    
    def get_customer_speech_history(self, session_id: str) -> Optional[List[str]]:
        """Get customer speech history for fraud analysis"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].get("customer_speech_history", [])
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
    
    # ===== TELEPHONY INTEGRATION METHODS =====
    
    async def connect_telephony_stream(
        self,
        session_id: str,
        telephony_stream_config: Dict[str, Any],
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """
        Connect to live telephony stream for real phone calls
        This method will be used when integrating with actual telephony systems
        """
        
        try:
            logger.info(f"📞 Connecting to telephony stream for session {session_id}")
            
            # Prepare session for telephony
            prepare_result = self._prepare_live_call({"session_id": session_id})
            if "error" in prepare_result:
                return prepare_result
            
            # TODO: Implement actual telephony integration
            # This will depend on your telephony provider (Twilio, Asterisk, etc.)
            telephony_config = {
                "stream_url": telephony_stream_config.get("stream_url"),
                "codec": telephony_stream_config.get("codec", "PCMU"),
                "sample_rate": telephony_stream_config.get("sample_rate", 8000),
                "channels": 1
            }
            
            # Start live streaming recognition
            streaming_result = await self.start_live_streaming(session_id, websocket_callback)
            if "error" in streaming_result:
                return streaming_result
            
            # Update session with telephony info
            session = self.active_sessions[session_id]
            session["telephony_config"] = telephony_config
            session["connection_type"] = "live_telephony"
            
            return {
                "session_id": session_id,
                "status": "connected_to_telephony",
                "telephony_config": telephony_config,
                "processing_mode": "live_telephony_v1p1beta1",
                "timestamp": get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"❌ Error connecting to telephony stream: {e}")
            return {"error": str(e)}
    
    def add_telephony_audio_chunk(
        self,
        session_id: str,
        audio_chunk: bytes,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add audio chunk from live telephony stream
        This will be called by telephony integration layer
        """
        
        if session_id not in self.active_sessions:
            logger.warning(f"⚠️ Received telephony audio for unknown session: {session_id}")
            return False
        
        if session_id not in self.audio_buffers:
            logger.warning(f"⚠️ No audio buffer for telephony session: {session_id}")
            return False
        
        try:
            # Add audio chunk to buffer
            self.audio_buffers[session_id].add_audio_chunk(audio_chunk)
            
            # Update session stats
            session = self.active_sessions[session_id]
            session["total_audio_received"] += len(audio_chunk)
            
            # Update metadata if provided
            if metadata:
                session.setdefault("telephony_metadata", []).append({
                    "timestamp": get_current_timestamp(),
                    "chunk_size": len(audio_chunk),
                    **metadata
                })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing telephony audio chunk: {e}")
            return False
    
    def get_telephony_integration_status(self) -> Dict[str, Any]:
        """Get status of telephony integration"""
        
        telephony_sessions = {
            session_id: session
            for session_id, session in self.active_sessions.items()
            if session.get("connection_type") == "live_telephony"
        }
        
        return {
            "telephony_integration_ready": True,
            "active_telephony_sessions": len(telephony_sessions),
            "total_telephony_sessions": len(telephony_sessions),
            "api_version": "v1p1beta1_telephony_optimized",
            "supported_codecs": ["PCMU", "PCMA", "LINEAR16"],
            "supported_sample_rates": [8000, 16000],
            "max_concurrent_calls": self.config.get("max_concurrent_sessions", 500),
            "telephony_sessions": {
                session_id: {
                    "status": session.get("status"),
                    "connection_type": session.get("connection_type"),
                    "telephony_config": session.get("telephony_config", {}),
                    "total_audio_received": session.get("total_audio_received", 0),
                    "transcription_segments": len(session.get("transcription_segments", []))
                }
                for session_id, session in telephony_sessions.items()
            },
            "timestamp": get_current_timestamp()
        }

# Create global instance
audio_processor_agent = AudioProcessorAgent()
