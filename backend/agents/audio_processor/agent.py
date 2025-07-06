# backend/agents/audio_processor/agent.py - FIXED FOR LARGE FILES AND SPEAKER DIARIZATION
"""
Audio Processing Agent for LIVE PHONE CALLS
FIXED: Large file handling, proper speaker diarization, and streaming reliability
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
        self.buffer = queue.Queue(maxsize=200)  # Increased buffer size for large files
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
                
                if self.total_chunks_added % 50 == 0:
                    logger.debug(f"📊 Audio buffer: {self.total_chunks_added} chunks added, queue size: {self.buffer.qsize()}")
                    
            except queue.Full:
                logger.warning("⚠️ Audio buffer full, dropping chunk to maintain real-time performance")
    
    def get_audio_chunks(self):
        """Generator for audio chunks with improved flow control"""
        logger.info("🎵 Audio chunk generator started for live streaming")
        
        consecutive_empty_reads = 0
        max_empty_reads = 100  # Increased for longer files (10 seconds of no data)
        
        while self.is_active:
            try:
                chunk = self.buffer.get(timeout=0.1)
                
                if chunk and len(chunk) > 0:
                    self.total_chunks_yielded += 1
                    consecutive_empty_reads = 0
                    
                    if self.total_chunks_yielded <= 5 or self.total_chunks_yielded % 100 == 0:
                        logger.debug(f"🎵 Yielding chunk {self.total_chunks_yielded} (size: {len(chunk)} bytes)")
                    
                    yield chunk
                else:
                    consecutive_empty_reads += 1
                    
            except queue.Empty:
                consecutive_empty_reads += 1
                
                # Check if we should stop due to inactivity
                if time.time() - self.last_activity > 10.0:  # 10 seconds of no new data
                    logger.info(f"🎵 Audio generator stopping: no new data for 10s")
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
    Live Audio Processor for Real-time Phone Calls
    FIXED: Large file handling and speaker diarization
    """
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Live Telephony Audio Processor (v1p1beta1 FIXED)"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, LiveAudioBuffer] = {}
        
        # Telephony-optimized settings
        self.chunk_duration_ms = 200  # 200ms chunks for responsiveness
        self.sample_rate = 16000  # Standard telephony rate
        self.channels = 1  # Mono audio
        self.chunk_size = int(self.sample_rate * self.channels * 2 * self.chunk_duration_ms / 1000)
        
        # Speaker tracking improvements
        self.speaker_states: Dict[str, str] = {}
        self.speaker_confidence: Dict[str, float] = {}
        self.conversation_context: Dict[str, List[Dict]] = {}
        
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
            "max_streaming_duration": 600,  # Increased to 10 minutes for large files
            "restart_recognition_timeout": 300,  # 5 minutes for stability
            "large_file_chunk_overlap": 0.1,  # 10% overlap for large files
            "enhanced_diarization": True
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
            self.transcription_source = "google_stt_v1p1beta1_telephony_fixed"
            
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
                    # Enhanced speaker diarization config
                    diarization_config=speech.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,  # Force 2 speakers
                        max_speaker_count=2   # Limit to 2 speakers
                    )
                )
                logger.info("✅ v1p1beta1: phone_call model with enhanced diarization configured successfully")
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
    
    # ... [keeping all existing process methods the same] ...
    
    async def _live_streaming_recognition_v1p1beta1(self, session_id: str) -> None:
        """
        v1p1beta1 API streaming recognition with FIXED large file and speaker handling
        """
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        
        try:
            logger.info(f"🎙️ Starting v1p1beta1 recognition cycle for {session_id}")
            
            # Create ENHANCED telephony-optimized recognition config
            if self.telephony_config_available:
                recognition_config = self.speech_types.RecognitionConfig(
                    encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB",
                    model="phone_call",  # Optimal for telephony
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    # ENHANCED speaker diarization for better separation
                    diarization_config=self.speech_types.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,  # FORCE 2 speakers
                        max_speaker_count=2,  # LIMIT to 2 speakers  
                        speaker_tag=1  # Enable speaker tagging
                    ),
                    # Enhanced speech adaptation for banking/fraud terms
                    speech_contexts=[
                        self.speech_types.SpeechContext(
                            phrases=[
                                # Banking/Service phrases (typically agent)
                                "good morning", "good afternoon", "customer service", "this is", "how can I help",
                                "can I assist", "security purposes", "verify your identity", "thank you for calling",
                                "HSBC customer service", "for security", "can you confirm", "I need to",
                                
                                # Customer request phrases  
                                "I need to", "I want to", "help me", "I have a problem", "urgent",
                                "transfer money", "send money", "investment", "emergency", "quickly",
                                "my boyfriend", "my girlfriend", "overseas", "stuck", "stranded",
                                "guaranteed returns", "investment opportunity", "police called", "investigation",
                                
                                # Common fraud terms
                                "romance", "military", "deployed", "medical emergency", "customs",
                                "margin call", "trading platform", "bitcoin", "cryptocurrency",
                                "government official", "tax office", "arrest warrant", "legal action"
                            ],
                            boost=20  # Increased boost for better recognition
                        )
                    ],
                    # Additional settings for large files
                    enable_spoken_punctuation=True,
                    enable_spoken_emojis=False,
                    use_enhanced=True  # Use enhanced models
                )
                logger.info("📞 Using phone_call model with ENHANCED diarization (forced 2 speakers)")
            else:
                # Fallback configuration with enhanced diarization
                recognition_config = self.speech_types.RecognitionConfig(
                    encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB",
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    diarization_config=self.speech_types.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,  # FORCE 2 speakers
                        max_speaker_count=2   # LIMIT to 2 speakers
                    ),
                    use_enhanced=True
                )
                logger.info("📞 Using default model with ENHANCED diarization (forced 2 speakers)")
            
            # Create streaming config for v1p1beta1 with enhanced settings
            streaming_config = self.speech_types.StreamingRecognitionConfig(
                config=recognition_config,
                interim_results=True,
                single_utterance=False
            )
            
            # Create the request generator - ENHANCED for large files
            def audio_generator_v1p1beta1():
                chunk_count = 0
                total_bytes = 0
                logger.info("📞 Audio generator: Enhanced for large files with speaker tracking")
                
                for audio_chunk in audio_buffer.get_audio_chunks():
                    if session_id not in self.active_sessions:
                        logger.info(f"🛑 Audio generator stopping for {session_id}")
                        break
                    
                    if audio_chunk and len(audio_chunk) > 0:
                        chunk_count += 1
                        total_bytes += len(audio_chunk)
                        
                        if chunk_count <= 5 or chunk_count % 200 == 0:  # Log every 200 chunks for large files
                            logger.debug(f"🎵 Sending audio chunk {chunk_count} ({len(audio_chunk)} bytes, total: {total_bytes} bytes)")
                        
                        # Create request with ONLY audio content
                        request = self.speech_types.StreamingRecognizeRequest()
                        request.audio_content = audio_chunk
                        yield request
                        
                        # Smaller delay for large files
                        time.sleep(0.01)  # Reduced delay
                
                logger.info(f"🎵 Audio generator completed: {chunk_count} chunks sent, {total_bytes} total bytes")
            
            # Start streaming recognition
            logger.info(f"🚀 Starting Google STT v1p1beta1 streaming for {session_id}")
            
            try:
                responses = self.google_client.streaming_recognize(
                    config=streaming_config,
                    requests=audio_generator_v1p1beta1()
                )
            except Exception as api_error:
                logger.error(f"❌ v1p1beta1 streaming API error: {api_error}")
                raise
            
            # Process responses with ENHANCED speaker tracking
            response_count = 0
            interim_count = 0
            final_count = 0
            last_speaker = None
            speaker_transition_count = 0
            
            for response in responses:
                if session_id not in self.active_sessions:
                    logger.info(f"🛑 Session {session_id} ended, stopping recognition")
                    break
                
                response_count += 1
                
                # Enhanced response processing
                for result in response.results:
                    if result.is_final:
                        final_count += 1
                    else:
                        interim_count += 1
                
                if response_count % 50 == 0:  # Increased logging interval for large files
                    logger.debug(f"📨 Processed {response_count} responses for {session_id} (Final: {final_count}, Interim: {interim_count})")
                
                await self._process_streaming_response_v1p1beta1_enhanced(session_id, response, websocket_callback)
            
            logger.info(f"✅ v1p1beta1 recognition cycle completed for {session_id}")
            logger.info(f"📊 Stats: {response_count} responses, {final_count} final, {interim_count} interim")
            
        except Exception as e:
            logger.error(f"❌ v1p1beta1 streaming recognition error for {session_id}: {e}")
            
            # Send error notification
            try:
                await websocket_callback({
                    "type": "streaming_error",
                    "data": {
                        "session_id": session_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "api_version": "v1p1beta1_telephony_enhanced",
                        "timestamp": get_current_timestamp()
                    }
                })
            except:
                pass
            
            raise
    
    async def _process_streaming_response_v1p1beta1_enhanced(
        self, 
        session_id: str, 
        response, 
        websocket_callback: Callable
    ) -> None:
        """ENHANCED response processing with improved speaker diarization"""
        
        try:
            # Handle streaming errors
            if hasattr(response, 'error') and response.error.code != 0:
                logger.error(f"❌ STT v1p1beta1 streaming error for {session_id}: {response.error}")
                return
            
            # Process results with enhanced speaker detection
            for result in response.results:
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    
                    if transcript:
                        # ENHANCED speaker detection
                        detected_speaker = self._detect_speaker_v1p1beta1_enhanced(session_id, alternative, result)
                        
                        # Get timing information
                        start_time, end_time = self._extract_timing_v1p1beta1(alternative)
                        
                        session = self.active_sessions[session_id]
                        
                        if result.is_final:
                            # FINAL RESULT: Add as new segment
                            segment_data = {
                                "session_id": session_id,
                                "speaker": detected_speaker,
                                "text": transcript,
                                "confidence": getattr(alternative, 'confidence', 0.9),
                                "is_final": True,
                                "start": start_time,
                                "end": end_time,
                                "duration": end_time - start_time,
                                "processing_mode": "live_streaming_v1p1beta1_enhanced",
                                "transcription_source": self.transcription_source,
                                "timestamp": get_current_timestamp()
                            }
                            
                            # Send final result to WebSocket
                            await websocket_callback({
                                "type": "transcription_segment",
                                "data": segment_data
                            })
                            
                            # Store final result in session
                            session["transcription_segments"].append(segment_data)
                            
                            # Track customer speech for fraud analysis (FINAL ONLY)
                            if detected_speaker == "customer":
                                session["customer_speech_history"].append(transcript)
                                
                                # Trigger fraud analysis for final customer speech
                                await self._trigger_progressive_fraud_analysis(
                                    session_id, transcript, websocket_callback
                                )
                            
                            # Update conversation context
                            if session_id not in self.conversation_context:
                                self.conversation_context[session_id] = []
                            self.conversation_context[session_id].append({
                                "speaker": detected_speaker,
                                "text": transcript,
                                "timestamp": time.time()
                            })
                            
                            # Keep only last 10 conversation turns for context
                            if len(self.conversation_context[session_id]) > 10:
                                self.conversation_context[session_id] = self.conversation_context[session_id][-10:]
                            
                            logger.info(f"📝 FINAL: {detected_speaker} - '{transcript[:50]}...'")
                            
                        else:
                            # INTERIM RESULT: Handle properly
                            if len(transcript) > 8:
                                
                                interim_segment_data = {
                                    "session_id": session_id,
                                    "speaker": detected_speaker,
                                    "text": transcript,
                                    "confidence": getattr(alternative, 'confidence', 0.9),
                                    "is_final": False,
                                    "start": start_time,
                                    "end": end_time,
                                    "duration": end_time - start_time,
                                    "processing_mode": "live_streaming_v1p1beta1_enhanced",
                                    "transcription_source": self.transcription_source,
                                    "timestamp": get_current_timestamp(),
                                    "is_interim_update": True
                                }
                                
                                # Check if we should update or create new interim
                                current_interim = session.get("current_interim_segment")
                                
                                if (current_interim and 
                                    current_interim.get("speaker") == detected_speaker and
                                    len(transcript) > len(current_interim.get("text", ""))):
                                    
                                    await websocket_callback({
                                        "type": "transcription_interim_update",
                                        "data": interim_segment_data
                                    })
                                    
                                    logger.debug(f"📝 interim UPDATE: {detected_speaker} - '{transcript[:30]}...'")
                                    
                                else:
                                    await websocket_callback({
                                        "type": "transcription_interim_new",
                                        "data": interim_segment_data
                                    })
                                    
                                    logger.debug(f"📝 interim NEW: {detected_speaker} - '{transcript[:30]}...'")
                                
                                # Store current interim state
                                session["current_interim_segment"] = interim_segment_data
        
        except Exception as e:
            logger.error(f"❌ Error processing enhanced streaming response: {e}")
    
    def _detect_speaker_v1p1beta1_enhanced(self, session_id: str, alternative, result) -> str:
        """ENHANCED speaker detection with better logic and conversation context"""
        
        # First, try Google's speaker diarization
        if hasattr(alternative, 'words') and alternative.words:
            speaker_tags = []
            confident_tags = []
            
            for word in alternative.words:
                if hasattr(word, 'speaker_tag') and word.speaker_tag is not None:
                    speaker_tags.append(word.speaker_tag)
                    
                    # Check word confidence for better speaker assignment
                    if hasattr(word, 'confidence') and word.confidence > 0.8:
                        confident_tags.append(word.speaker_tag)
            
            if confident_tags:
                # Use confident tags if available
                most_common_tag = max(set(confident_tags), key=confident_tags.count)
                unique_tags = set(confident_tags)
                
                logger.debug(f"🎯 DIARIZATION SUCCESS: confident_tags={unique_tags}, most_common={most_common_tag}")
                
                # Map Google's speaker tags with enhanced logic
                if most_common_tag == 1:
                    return "customer"  # First speaker = customer (calls in)
                elif most_common_tag == 2:
                    return "agent"     # Second speaker = agent (answers)
                    
            elif speaker_tags:
                # Fallback to all tags if no confident ones
                most_common_tag = max(set(speaker_tags), key=speaker_tags.count)
                unique_tags = set(speaker_tags)
                
                logger.debug(f"🎯 DIARIZATION BASIC: speaker_tags={unique_tags}, most_common={most_common_tag}")
                
                if most_common_tag == 1:
                    return "customer"
                elif most_common_tag == 2:
                    return "agent"
        
        # CONTENT-BASED DETECTION (primary method when Google diarization fails)
        current_text = getattr(alternative, 'transcript', '').lower()
        
        # Strong agent indicators (they answer the phone)
        agent_phrases = [
            'good morning', 'good afternoon', 'customer service', 'customer services',
            'this is', 'speaking', 'hsbc', 'how can i help', 'how can i assist',
            'can i assist', 'thank you for calling', 'for security', 'i need to verify',
            'can you confirm', 'thank you', 'i\'ll help you', 'let me check',
            'i understand', 'i can see', 'i\'ll transfer you', 'one moment please',
            'can i take your', 'i can help with that'
        ]
        
        # Strong customer indicators (they make requests)
        customer_phrases = [
            'i need to', 'i want to', 'i would like', 'help me', 'i have a problem',
            'urgent', 'quickly', 'emergency', 'my boyfriend', 'my girlfriend', 'my partner',
            'transfer money', 'send money', 'investment', 'guaranteed returns',
            'police called', 'investigation', 'stuck overseas', 'medical emergency',
            'hi,', 'hello,', 'yes,', 'it\'s for', 'it\'s mrs', 'it\'s mr'
        ]
        
        # Check for strong content indicators FIRST
        agent_matches = [phrase for phrase in agent_phrases if phrase in current_text]
        customer_matches = [phrase for phrase in customer_phrases if phrase in current_text]
        
        if agent_matches and not customer_matches:
            logger.info(f"🎯 CONTENT: Agent phrase detected -> agent ('{agent_matches[0]}')")
            return "agent"
        elif customer_matches and not agent_matches:
            logger.info(f"🎯 CONTENT: Customer phrase detected -> customer ('{customer_matches[0]}')")
            return "customer"
        elif agent_matches and customer_matches:
            # Both detected, use stronger signal
            if len(agent_matches) > len(customer_matches):
                logger.info(f"🎯 CONTENT: Stronger agent signal -> agent ({len(agent_matches)} vs {len(customer_matches)})")
                return "agent"
            else:
                logger.info(f"🎯 CONTENT: Stronger customer signal -> customer ({len(customer_matches)} vs {len(agent_matches)})")
                return "customer"
        
        # CONVERSATION FLOW ANALYSIS
        session = self.active_sessions.get(session_id, {})
        segments = session.get("transcription_segments", [])
        conversation_context = self.conversation_context.get(session_id, [])
        
        # For very first utterance, use smarter detection
        if len(segments) == 0 and len(conversation_context) == 0:
            # First utterance content analysis
            if any(phrase in current_text for phrase in ['good morning', 'good afternoon', 'customer service', 'this is']):
                logger.info("🎯 FIRST: Agent greeting detected")
                return "agent"
            elif any(phrase in current_text for phrase in ['hi', 'hello', 'i need', 'help me']):
                logger.info("🎯 FIRST: Customer opening detected")
                return "customer"
            else:
                # Default: customer calls first
                logger.info("🎯 FIRST: Default to customer (caller)")
                return "customer"
        
        # Natural conversation flow with enhanced logic
        if conversation_context:
            recent_context = conversation_context[-3:]  # Last 3 turns
            last_speaker = recent_context[-1]["speaker"] if recent_context else "customer"
            
            # Count recent turns by each speaker
            agent_turns = sum(1 for turn in recent_context if turn["speaker"] == "agent")
            customer_turns = sum(1 for turn in recent_context if turn["speaker"] == "customer")
            
            # Smart alternation with bias correction
            if agent_turns > customer_turns + 1:
                logger.info(f"🎯 FLOW: Too many agent turns -> customer")
                return "customer"
            elif customer_turns > agent_turns + 1:
                logger.info(f"🎯 FLOW: Too many customer turns -> agent")
                return "agent"
            else:
                # Natural alternation
                next_speaker = "agent" if last_speaker == "customer" else "customer"
                logger.info(f"🎯 FLOW: Natural alternation {last_speaker} -> {next_speaker}")
                return next_speaker
        
        # Final fallback: check if this looks like a continuation
        if segments:
            last_speaker = segments[-1].get("speaker", "customer")
            # If text is very short, might be continuation of same speaker
            if len(current_text.split()) < 3:
                logger.info(f"🎯 CONTINUATION: Short text -> {last_speaker}")
                return last_speaker
            else:
                # Longer text, likely speaker change
                next_speaker = "agent" if last_speaker == "customer" else "customer"
                logger.info(f"🎯 ALTERNATION: Longer text {last_speaker} -> {next_speaker}")
                return next_speaker
        
        # Ultimate fallback
        logger.info("🎯 ULTIMATE FALLBACK: customer")
        return "customer"
    
    # ... [rest of the methods remain the same as in the original file] ...
    
    async def _simulate_live_audio_from_file(
        self,
        session_id: str,
        audio_path: Path,
        duration: float
    ) -> None:
        """Simulate live audio streaming from demo file - ENHANCED for large files"""
        
        try:
            logger.info(f"🎬 Simulating live audio stream from {audio_path.name} (ENHANCED for large files)")
            
            # Read audio file
            with wave.open(str(audio_path), 'rb') as wav_file:
                # Verify audio format
                actual_sample_rate = wav_file.getframerate()
                actual_channels = wav_file.getnchannels()
                actual_sample_width = wav_file.getsampwidth()
                
                logger.info(f"📁 Audio file specs: {actual_sample_rate}Hz, {actual_channels}ch, {actual_sample_width*8}bit")
                
                # Enhanced chunking for large files
                frames_per_chunk = int(actual_sample_rate * self.chunk_duration_ms / 1000)
                total_frames = wav_file.getnframes()
                total_chunks = int(total_frames / frames_per_chunk)
                
                # Calculate file size for memory management
                file_size_mb = (total_frames * actual_channels * actual_sample_width) / (1024 * 1024)
                
                logger.info(f"🎵 Large file processing: {file_size_mb:.1f}MB, {total_chunks} chunks, {duration:.1f}s")
                
                # Get audio buffer
                audio_buffer = self.audio_buffers.get(session_id)
                if not audio_buffer:
                    logger.error(f"❌ No audio buffer for session {session_id}")
                    return
                
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                bytes_processed = 0
                
                # Start streaming with enhanced timing for large files
                logger.info("🚀 Starting enhanced audio streaming for large file...")
                
                while chunk_count < total_chunks and session_id in self.active_sessions:
                    # Read audio chunk
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    
                    if not audio_chunk:
                        logger.warning(f"⚠️ No more audio data at chunk {chunk_count}/{total_chunks}")
                        break
                    
                    # Add to buffer for streaming recognition
                    audio_buffer.add_audio_chunk(audio_chunk)
                    
                    chunk_count += 1
                    bytes_processed += len(audio_chunk)
                    
                    # Enhanced progress logging for large files
                    elapsed_seconds = (chunk_count * self.chunk_duration_ms) / 1000
                    if elapsed_seconds > 0 and (
                        int(elapsed_seconds) % 10 == 0 and elapsed_seconds == int(elapsed_seconds) or
                        chunk_count % 500 == 0  # Every 100 seconds of audio
                    ):
                        progress_percent = (chunk_count / total_chunks) * 100
                        mb_processed = bytes_processed / (1024 * 1024)
                        logger.info(f"🎵 Large file progress: {elapsed_seconds:.1f}s / {duration:.1f}s ({progress_percent:.1f}%, {mb_processed:.1f}MB)")
                    
                    # Enhanced timing control for large files
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = asyncio.get_event_loop().time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        # Use smaller sleep intervals for large files
                        await asyncio.sleep(min(wait_time, 0.05))  # Max 50ms sleep
                    elif wait_time < -2.0:  # Allow more drift for large files
                        logger.debug(f"⚠️ Large file streaming is {-wait_time:.1f}s behind real-time")
                
                logger.info(f"✅ Large file simulation complete: {chunk_count} chunks streamed, {bytes_processed/(1024*1024):.1f}MB processed")
                
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
                                "file_size_mb": file_size_mb,
                                "bytes_processed": bytes_processed,
                                "timestamp": get_current_timestamp()
                            }
                        })
                
                # Keep buffer active longer for large files
                await asyncio.sleep(5.0)
                audio_buffer.stop()
                
        except Exception as e:
            logger.error(f"❌ Error simulating large audio file: {e}")
    
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
            "api_version": "v1p1beta1_telephony_enhanced",
            "telephony_model_available": self.telephony_config_available,
            "enhancements": [
                "Large file support (10+ minutes)",
                "Enhanced speaker diarization (forced 2 speakers)",
                "Conversation context tracking",
                "Improved pattern matching",
                "Better memory management"
            ],
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
        
        # Initialize session with enhanced tracking
        self.active_sessions[session_id] = {
            "start_time": datetime.now(),
            "status": "ready",
            "total_audio_received": 0,
            "transcription_segments": [],
            "current_speaker": "customer",
            "recognition_restart_count": 0,
            "customer_speech_history": [],
            "speaker_confidence_history": [],
            "conversation_turns": 0
        }
        
        # Initialize enhanced audio buffer
        self.audio_buffers[session_id] = LiveAudioBuffer(self.chunk_size)
        self.audio_buffers[session_id].start()
        self.speaker_states[session_id] = "customer"
        self.speaker_confidence[session_id] = 0.5
        self.conversation_context[session_id] = []
        
        logger.info(f"✅ Enhanced live call session {session_id} prepared and ready")
        
        return {
            "session_id": session_id,
            "status": "ready_for_live_audio",
            "chunk_size_bytes": self.chunk_size,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "buffer_active": True,
            "api_version": "v1p1beta1_telephony_enhanced",
            "enhancements_enabled": True
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
        
        # Add to enhanced buffer
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
            "buffer_size": self.audio_buffers[session_id].buffer.qsize(),
            "enhancement": "enhanced_processing_active"
        }
    
    async def start_live_streaming(
        self, 
        session_id: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Start live streaming recognition using enhanced v1p1beta1 API"""
        
        try:
            logger.info(f"📞 Starting enhanced live streaming recognition for session {session_id}")
            
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
            
            # Start enhanced streaming task with restart capability
            streaming_task = asyncio.create_task(
                self._live_streaming_with_restart_v1p1beta1_enhanced(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            # Send confirmation
            await websocket_callback({
                "type": "live_streaming_started",
                "data": {
                    "session_id": session_id,
                    "processing_mode": "live_telephony_streaming_v1p1beta1_enhanced",
                    "transcription_engine": self.transcription_source,
                    "sample_rate": self.sample_rate,
                    "chunk_duration_ms": self.chunk_duration_ms,
                    "api_version": "v1p1beta1_telephony_enhanced",
                    "model": "phone_call" if self.telephony_config_available else "default",
                    "speaker_diarization": "enhanced_forced_2_speakers",
                    "large_file_support": True,
                    "conversation_context": True,
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {
                "session_id": session_id,
                "status": "streaming",
                "processing_mode": "live_telephony_streaming_v1p1beta1_enhanced"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting enhanced live streaming: {e}")
            return {"error": str(e)}
    
    async def _live_streaming_with_restart_v1p1beta1_enhanced(self, session_id: str) -> None:
        """Enhanced live streaming with automatic restart for long calls"""
        session = self.active_sessions[session_id]
        restart_interval = self.config.get("restart_recognition_timeout", 300)  # 5 minutes
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            try:
                logger.info(f"🔄 Starting enhanced recognition cycle for {session_id}")
                
                # Run recognition for the specified interval
                await asyncio.wait_for(
                    self._live_streaming_recognition_v1p1beta1(session_id),
                    timeout=restart_interval
                )
                
            except asyncio.TimeoutError:
                # Normal restart due to timeout
                session["recognition_restart_count"] += 1
                logger.info(f"🔄 Restarting enhanced recognition for {session_id} (restart #{session['recognition_restart_count']})")
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"❌ Enhanced recognition error for {session_id}: {e}")
                session["recognition_restart_count"] += 1
                await asyncio.sleep(2.0)
                
                if session["recognition_restart_count"] > 5:  # Increased tolerance for large files
                    logger.error(f"❌ Too many restart attempts for {session_id}, stopping")
                    break
    
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
                        "enhancement": "enhanced_speaker_detection_active",
                        "timestamp": get_current_timestamp()
                    }
                })
                
                logger.info(f"🔍 Triggered fraud analysis: {len(recent_speech)} chars from {len(customer_history)} segments")
        
        except Exception as e:
            logger.error(f"❌ Error triggering fraud analysis: {e}")
    
    async def stop_live_streaming(self, session_id: str) -> Dict[str, Any]:
        """Stop live streaming for a session with proper cleanup"""
        try:
            logger.info(f"🛑 Stopping enhanced live streaming for {session_id}")
            
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
                    logger.info(f"✅ Enhanced streaming task cancelled for {session_id}")
                del self.streaming_tasks[session_id]
            
            # Enhanced cleanup
            del self.active_sessions[session_id]
            
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            
            if session_id in self.speaker_states:
                del self.speaker_states[session_id]
                
            if session_id in self.speaker_confidence:
                del self.speaker_confidence[session_id]
                
            if session_id in self.conversation_context:
                del self.conversation_context[session_id]
            
            logger.info(f"✅ Enhanced live streaming stopped and cleaned up for {session_id}")
            
            return {
                "session_id": session_id,
                "status": "stopped",
                "cleanup_complete": True,
                "enhancements_cleaned": True,
                "timestamp": get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping enhanced live streaming for {session_id}: {e}")
            return {"error": str(e)}
    
    # ===== DEMO FILE PROCESSING METHODS =====
    
    async def start_realtime_processing(
        self,
        session_id: str,
        audio_filename: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Start real-time processing of demo audio file with enhancements"""
        
        try:
            logger.info(f"🎬 Starting enhanced real-time demo processing: {audio_filename}")
            
            # Prepare session for live call
            prepare_result = self._prepare_live_call({"session_id": session_id})
            if "error" in prepare_result:
                return prepare_result
            
            # Load and validate audio file
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            if not audio_path.exists():
                return {"error": f"Audio file not found: {audio_filename}"}
            
            # Get enhanced audio file info
            try:
                with wave.open(str(audio_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    file_size = audio_path.stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    
                logger.info(f"📁 Enhanced demo file: {duration:.1f}s, {sample_rate}Hz, {file_size_mb:.1f}MB")
                
                # Check if this is a large file
                if file_size_mb > 10:
                    logger.info(f"🗂️ Large file detected: {file_size_mb:.1f}MB - using enhanced processing")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not read WAV info: {e}")
                duration = 30.0  # Default estimate
                file_size_mb = 1.0
            
            # Start enhanced audio simulation BEFORE starting recognition
            logger.info("🎵 Starting enhanced audio file simulation...")
            file_simulation_task = asyncio.create_task(
                self._simulate_live_audio_from_file(session_id, audio_path, duration)
            )
            
            # Longer delay for large files to let audio accumulate
            wait_time = 2.0 if file_size_mb > 10 else 1.0
            await asyncio.sleep(wait_time)
            
            # Then start enhanced live streaming recognition
            logger.info("🎙️ Starting enhanced v1p1beta1 recognition...")
            streaming_result = await self.start_live_streaming(session_id, websocket_callback)
            if "error" in streaming_result:
                file_simulation_task.cancel()
                return streaming_result
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "file_size_mb": file_size_mb,
                "processing_mode": "demo_realtime_v1p1beta1_enhanced",
                "status": "started",
                "api_version": "v1p1beta1_enhanced",
                "model": "phone_call" if self.telephony_config_available else "default",
                "large_file_optimized": file_size_mb > 10,
                "speaker_diarization": "enhanced_forced_2_speakers"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting enhanced realtime processing: {e}")
            return {"error": str(e)}
            
            # Cleanup on error
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            if session_id in self.speaker_states:
                del self.speaker_states[session_id]
    
    async def stop_realtime_processing(self, session_id: str) -> Dict[str, Any]:
        """Stop real-time processing"""
        return await self.stop_live_streaming(session_id)
    
    # ===== STATUS AND MANAGEMENT METHODS =====
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all active sessions with enhancements"""
        
        sessions_status = {}
        for session_id, session in self.active_sessions.items():
            
            sessions_status[session_id] = {
                "status": session.get("status", "unknown"),
                "start_time": session.get("start_time", "").isoformat() if session.get("start_time") else "",
                "total_audio_received": session.get("total_audio_received", 0),
                "transcription_segments": len(session.get("transcription_segments", [])),
                "customer_speech_segments": len(session.get("customer_speech_history", [])),
                "current_speaker": session.get("current_speaker", "unknown"),
                "recognition_restart_count": session.get("recognition_restart_count", 0),
                "conversation_turns": session.get("conversation_turns", 0),
                "has_streaming_task": session_id in self.streaming_tasks,
                "speaker_confidence": self.speaker_confidence.get(session_id, 0.0),
                "conversation_context_length": len(self.conversation_context.get(session_id, [])),
                "enhancement_status": "enhanced_processing_active"
            }
        
        return {
            "total_active_sessions": len(self.active_sessions),
            "transcription_engine": self.transcription_source,
            "google_stt_v1p1beta1_available": self.google_client is not None,
            "telephony_model_available": self.telephony_config_available,
            "api_version": "v1p1beta1_telephony_enhanced",
            "enhancements": [
                "Large file support (10+ minutes)",
                "Enhanced speaker diarization (forced 2 speakers)",
                "Conversation context tracking",
                "Improved pattern matching",
                "Better memory management",
                "Enhanced timing control"
            ],
            "sessions": sessions_status,
            "agent_id": self.agent_id,
            "timestamp": get_current_timestamp()
        }

# Create global instance
audio_processor_agent = AudioProcessorAgent()
