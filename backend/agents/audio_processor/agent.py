# backend/agents/audio_processor/agent.py - COMPLETE SIMPLIFIED VERSION
"""
Audio Processing Agent - COMPLETE SIMPLIFIED to rely only on Google STT diarization
Removed: Complex fallback logic, pattern matching, conversation flow
Focus: Google STT forced 2-speaker diarization only
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
    """Thread-safe audio buffer for live streaming"""
    
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
        """Generator for audio chunks"""
        logger.info("🎵 Audio chunk generator started")
        
        consecutive_empty_reads = 0
        max_empty_reads = 100  # 10 seconds of no data
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            "total_chunks_added": self.total_chunks_added,
            "total_chunks_yielded": self.total_chunks_yielded,
            "buffer_size": self.buffer.qsize(),
            "is_active": self.is_active,
            "last_activity": self.last_activity
        }

class AudioProcessorAgent(BaseAgent):
    """
    SIMPLIFIED Audio Processor - Google STT Diarization Only
    """
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Simplified Audio Processor (Google STT Only)"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, LiveAudioBuffer] = {}
        
        # Audio settings
        self.chunk_duration_ms = 200  # 200ms chunks
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = int(self.sample_rate * self.channels * 2 * self.chunk_duration_ms / 1000)
        
        # Initialize Google STT
        self._initialize_google_stt()
              
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"📞 {self.agent_name} initialized (SIMPLIFIED)")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        settings = get_settings()
        
        return {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio",
            "enable_interim_results": True,
            "single_utterance": False,
            "max_streaming_duration": 600,
            "restart_recognition_timeout": 300
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
            "api_version": "v1p1beta1_simplified",
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
            "recognition_restart_count": 0,
            "customer_speech_history": []
        }
        
        # Initialize audio buffer
        self.audio_buffers[session_id] = LiveAudioBuffer(self.chunk_size)
        self.audio_buffers[session_id].start()
        
        logger.info(f"✅ Live call session {session_id} prepared (SIMPLIFIED)")
        
        return {
            "session_id": session_id,
            "status": "ready_for_live_audio",
            "chunk_size_bytes": self.chunk_size,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "buffer_active": True,
            "api_version": "v1p1beta1_simplified"
        }
    
    def _add_audio_chunk(self, data: Dict) -> Dict[str, Any]:
        """Add audio chunk from telephony system"""
        session_id = data.get('session_id')
        audio_data = data.get('audio_data')
        
        if not session_id or session_id not in self.active_sessions:
            return {"error": "Invalid or unknown session"}
        
        if not audio_data:
            return {"error": "No audio data provided"}
        
        # Add to buffer
        self.audio_buffers[session_id].add_audio_chunk(audio_data)
        
        # Update session stats
        session = self.active_sessions[session_id]
        session["total_audio_received"] += len(audio_data)
        
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
        """Start live streaming recognition"""
        
        try:
            logger.info(f"📞 Starting SIMPLIFIED live streaming for session {session_id}")
            
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not prepared")
            
            if not self.google_client:
                raise Exception("Google Speech-to-Text not available")
            
            # Get audio buffer
            audio_buffer = self.audio_buffers[session_id]
            if not audio_buffer.is_active:
                audio_buffer.start()
            
            # Update session status
            session = self.active_sessions[session_id]
            session["status"] = "streaming"
            session["websocket_callback"] = websocket_callback
            
            # Start streaming task
            streaming_task = asyncio.create_task(
                self._live_streaming_with_restart(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            # Send confirmation
            await websocket_callback({
                "type": "live_streaming_started",
                "data": {
                    "session_id": session_id,
                    "processing_mode": "simplified_google_stt_only",
                    "transcription_engine": self.transcription_source,
                    "sample_rate": self.sample_rate,
                    "api_version": "v1p1beta1_simplified",
                    "model": "phone_call" if self.telephony_config_available else "default",
                    "speaker_diarization": "forced_2_speakers_only",
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {
                "session_id": session_id,
                "status": "streaming",
                "processing_mode": "simplified_google_stt_only"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting simplified streaming: {e}")
            return {"error": str(e)}
    
    async def _live_streaming_with_restart(self, session_id: str) -> None:
        """Live streaming with automatic restart"""
        session = self.active_sessions[session_id]
        restart_interval = self.config.get("restart_recognition_timeout", 300)  # 5 minutes
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            try:
                logger.info(f"🔄 Starting recognition cycle for {session_id}")
                
                # Run recognition for the specified interval
                await asyncio.wait_for(
                    self._live_streaming_recognition(session_id),
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
                
                if session["recognition_restart_count"] > 5:
                    logger.error(f"❌ Too many restart attempts for {session_id}, stopping")
                    break
    
    async def _live_streaming_recognition(self, session_id: str) -> None:
        """
        SIMPLIFIED v1p1beta1 streaming recognition - Google STT diarization ONLY
        """
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        
        try:
            logger.info(f"🎙️ Starting SIMPLIFIED v1p1beta1 recognition for {session_id}")
            
            # Create SIMPLIFIED recognition config - ONLY Google STT diarization
            if self.telephony_config_available:
                recognition_config = self.speech_types.RecognitionConfig(
                    encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB",
                    model="phone_call",
                    enable_automatic_punctuation=True,
                    enable_word_confidence=True,
                    enable_word_time_offsets=True,
                    # FORCED 2-speaker diarization - NO FALLBACK
                    diarization_config=self.speech_types.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,  # FORCE 2 speakers
                        max_speaker_count=2,  # LIMIT to 2 speakers
                        speaker_tag=1         # Enable speaker tagging
                    ),
                    use_enhanced=True
                )
                logger.info("📞 Using phone_call model with FORCED 2-speaker diarization (NO FALLBACK)")
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
            
            # SIMPLIFIED audio generator - NO DELAYS
            def audio_generator():
                chunk_count = 0
                logger.info("📞 SIMPLIFIED audio generator: NO delays, direct streaming")
                
                for audio_chunk in audio_buffer.get_audio_chunks():
                    if session_id not in self.active_sessions:
                        break
                    
                    if audio_chunk and len(audio_chunk) > 0:
                        chunk_count += 1
                        
                        # Create request with ONLY audio content
                        request = self.speech_types.StreamingRecognizeRequest()
                        request.audio_content = audio_chunk
                        yield request
                        
                        # NO DELAY - direct streaming for immediate response
                
                logger.info(f"🎵 SIMPLIFIED audio generator completed: {chunk_count} chunks")
            
            # Start streaming recognition
            logger.info(f"🚀 Starting SIMPLIFIED Google STT streaming for {session_id}")
            
            responses = self.google_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )
            
            # Process responses - SIMPLIFIED
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
            
            logger.info(f"✅ SIMPLIFIED recognition completed for {session_id}")
            logger.info(f"📊 Stats: {response_count} responses, {final_count} final, {interim_count} interim")
            
        except Exception as e:
            logger.error(f"❌ SIMPLIFIED streaming error for {session_id}: {e}")
            
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
    
    async def _process_streaming_response_simplified(
        self, 
        session_id: str, 
        response, 
        websocket_callback: Callable
    ) -> None:
        """SIMPLIFIED response processing - Google STT diarization ONLY"""
        
        try:
            # Handle streaming errors
            if hasattr(response, 'error') and response.error.code != 0:
                logger.error(f"❌ STT streaming error for {session_id}: {response.error}")
                return
            
            # Process results with SIMPLIFIED speaker detection
            for result in response.results:
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    
                    if transcript:
                        # SIMPLIFIED speaker detection - ONLY Google STT
                        detected_speaker = self._get_google_speaker_only(alternative)
                        
                        # Get timing information
                        start_time, end_time = self._extract_timing(alternative)
                        
                        session = self.active_sessions[session_id]
                        
                        if result.is_final:
                            # FINAL RESULT
                            segment_data = {
                                "session_id": session_id,
                                "speaker": detected_speaker,
                                "text": transcript,
                                "confidence": getattr(alternative, 'confidence', 0.9),
                                "is_final": True,
                                "start": start_time,
                                "end": end_time,
                                "duration": end_time - start_time,
                                "processing_mode": "simplified_google_stt_only",
                                "transcription_source": self.transcription_source,
                                "timestamp": get_current_timestamp()
                            }
                            
                            # Send final result - NO DELAY
                            await websocket_callback({
                                "type": "transcription_segment",
                                "data": segment_data
                            })
                            
                            # Store final result
                            session["transcription_segments"].append(segment_data)
                            
                            # Track customer speech for fraud analysis
                            if detected_speaker == "customer":
                                session["customer_speech_history"].append(transcript)
                                
                                # Trigger fraud analysis - NO DELAY
                                await self._trigger_fraud_analysis(
                                    session_id, transcript, websocket_callback
                                )
                            
                            logger.info(f"📝 FINAL: {detected_speaker} - '{transcript[:50]}...'")
                            
                        else:
                            # INTERIM RESULT
                            if len(transcript) > 8:
                                
                                interim_data = {
                                    "session_id": session_id,
                                    "speaker": detected_speaker,
                                    "text": transcript,
                                    "confidence": getattr(alternative, 'confidence', 0.9),
                                    "is_final": False,
                                    "start": start_time,
                                    "end": end_time,
                                    "duration": end_time - start_time,
                                    "processing_mode": "simplified_google_stt_only",
                                    "transcription_source": self.transcription_source,
                                    "timestamp": get_current_timestamp(),
                                    "is_interim_update": True
                                }
                                
                                # Send interim result - NO DELAY
                                current_interim = session.get("current_interim_segment")
                                
                                if (current_interim and 
                                    current_interim.get("speaker") == detected_speaker and
                                    len(transcript) > len(current_interim.get("text", ""))):
                                    
                                    await websocket_callback({
                                        "type": "transcription_interim_update",
                                        "data": interim_data
                                    })
                                    
                                else:
                                    await websocket_callback({
                                        "type": "transcription_interim_new",
                                        "data": interim_data
                                    })
                                
                                # Store current interim state
                                session["current_interim_segment"] = interim_data
        
        except Exception as e:
            logger.error(f"❌ Error processing SIMPLIFIED response: {e}")
    
    def _get_google_speaker_only(self, alternative) -> str:
        """SIMPLIFIED speaker detection - ONLY Google STT diarization, NO FALLBACK"""
        
        # ONLY use Google's speaker diarization - NO FALLBACK LOGIC
        if hasattr(alternative, 'words') and alternative.words:
            speaker_tags = []
            
            for word in alternative.words:
                if hasattr(word, 'speaker_tag') and word.speaker_tag is not None:
                    speaker_tags.append(word.speaker_tag)
            
            if speaker_tags:
                # Use most common speaker tag
                most_common_tag = max(set(speaker_tags), key=speaker_tags.count)
                
                logger.debug(f"🎯 GOOGLE STT ONLY: speaker_tag={most_common_tag}")
                
                # Map speaker tags: 1 = customer, 2 = agent
                if most_common_tag == 1:
                    return "customer"
                elif most_common_tag == 2:
                    return "agent"
                else:
                    return f"speaker_{most_common_tag}"
        
        # If Google STT provides no speaker tags, return unknown
        # NO FALLBACK LOGIC - let Google STT handle it
        logger.debug("🎯 GOOGLE STT: No speaker tags provided")
        return "unknown"
    
    def _extract_timing(self, alternative) -> tuple:
        """Extract timing information"""
        start_time = 0.0
        end_time = 0.0
        
        if hasattr(alternative, 'words') and alternative.words:
            first_word = alternative.words[0]
            last_word = alternative.words[-1]
            
            if hasattr(first_word, 'start_time') and first_word.start_time:
                start_time = first_word.start_time.total_seconds()
            if hasattr(last_word, 'end_time') and last_word.end_time:
                end_time = last_word.end_time.total_seconds()
        
        return start_time, end_time
    
    async def _trigger_fraud_analysis(
        self, 
        session_id: str, 
        customer_text: str, 
        websocket_callback: Callable
    ) -> None:
        """Trigger fraud analysis - NO DELAY"""
        
        try:
            session = self.active_sessions[session_id]
            customer_history = session.get("customer_speech_history", [])
            
            # Combine recent customer speech
            recent_speech = " ".join(customer_history[-5:])
            
            if len(recent_speech.strip()) >= 20:
                # Send to fraud detection - NO DELAY
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
                
                logger.info(f"🔍 Fraud analysis triggered: {len(recent_speech)} chars")
        
        except Exception as e:
            logger.error(f"❌ Error triggering fraud analysis: {e}")
    
    async def stop_live_streaming(self, session_id: str) -> Dict[str, Any]:
        """Stop live streaming"""
        try:
            logger.info(f"🛑 Stopping SIMPLIFIED streaming for {session_id}")
            
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
                    logger.info(f"✅ SIMPLIFIED streaming task cancelled for {session_id}")
                del self.streaming_tasks[session_id]
            
            # Cleanup
            del self.active_sessions[session_id]
            
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            
            logger.info(f"✅ SIMPLIFIED streaming stopped for {session_id}")
            
            return {
                "session_id": session_id,
                "status": "stopped",
                "cleanup_complete": True,
                "timestamp": get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping SIMPLIFIED streaming: {e}")
            return {"error": str(e)}
    
    # ===== DEMO FILE PROCESSING =====
    
    async def start_realtime_processing(
        self,
        session_id: str,
        audio_filename: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Start real-time processing - SIMPLIFIED"""
        
        try:
            logger.info(f"🎬 Starting SIMPLIFIED demo processing: {audio_filename}")
            
            # Prepare session
            prepare_result = self._prepare_live_call({"session_id": session_id})
            if "error" in prepare_result:
                return prepare_result
            
            # Load audio file
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            if not audio_path.exists():
                return {"error": f"Audio file not found: {audio_filename}"}
            
            # Get audio info
            try:
                with wave.open(str(audio_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    
                logger.info(f"📁 Audio file: {duration:.1f}s, {sample_rate}Hz")
            except Exception as e:
                logger.warning(f"⚠️ Could not read WAV info: {e}")
                duration = 30.0
            
            # Start audio simulation - NO DELAY
            logger.info("🎵 Starting audio simulation...")
            file_simulation_task = asyncio.create_task(
                self._simulate_audio_file(session_id, audio_path, duration)
            )
            
            # Start recognition immediately - NO DELAY
            logger.info("🎙️ Starting recognition...")
            streaming_result = await self.start_live_streaming(session_id, websocket_callback)
            if "error" in streaming_result:
                file_simulation_task.cancel()
                return streaming_result
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "processing_mode": "simplified_google_stt_only",
                "status": "started",
                "api_version": "v1p1beta1_simplified",
                "model": "phone_call" if self.telephony_config_available else "default",
                "speaker_diarization": "forced_2_speakers_google_only"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting SIMPLIFIED processing: {e}")
            return {"error": str(e)}
            
            # Cleanup on error
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
    
    async def _simulate_audio_file(
        self,
        session_id: str,
        audio_path: Path,
        duration: float
    ) -> None:
        """Simulate audio file - SIMPLIFIED with NO DELAYS but PROPER TIMING"""
        
        try:
            logger.info(f"🎬 SIMPLIFIED audio simulation: {audio_path.name}")
            
            # Read audio file
            with wave.open(str(audio_path), 'rb') as wav_file:
                actual_sample_rate = wav_file.getframerate()
                actual_channels = wav_file.getnchannels()
                actual_sample_width = wav_file.getsampwidth()
                
                logger.info(f"📁 Audio specs: {actual_sample_rate}Hz, {actual_channels}ch, {actual_sample_width*8}bit")
                
                # Calculate chunks based on ACTUAL sample rate
                frames_per_chunk = int(actual_sample_rate * self.chunk_duration_ms / 1000)
                total_frames = wav_file.getnframes()
                total_chunks = int(total_frames / frames_per_chunk)
                
                logger.info(f"🎵 Streaming {total_chunks} chunks (frames_per_chunk: {frames_per_chunk})")
                
                # Get audio buffer
                audio_buffer = self.audio_buffers.get(session_id)
                if not audio_buffer:
                    logger.error(f"❌ No audio buffer for session {session_id}")
                    return
                
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                
                # Start streaming with PROPER timing
                logger.info("🚀 Starting audio streaming with proper timing...")
                
                while chunk_count < total_chunks and session_id in self.active_sessions:
                    # Read audio chunk
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    
                    if not audio_chunk:
                        logger.warning(f"⚠️ No audio data at chunk {chunk_count}/{total_chunks}")
                        break
                    
                    # Add to buffer IMMEDIATELY
                    audio_buffer.add_audio_chunk(audio_chunk)
                    
                    chunk_count += 1
                    
                    # Progress logging every 25 chunks (5 seconds at 200ms)
                    if chunk_count % 25 == 0:
                        elapsed_seconds = (chunk_count * self.chunk_duration_ms) / 1000
                        logger.info(f"🎵 Progress: {elapsed_seconds:.1f}s / {duration:.1f}s ({chunk_count}/{total_chunks})")
                    
                    # PROPER timing control - maintain real-time pace
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = asyncio.get_event_loop().time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        # Sleep to maintain real-time pace - NOT too small
                        await asyncio.sleep(min(wait_time, 0.2))  # Max 200ms sleep = chunk duration
                    elif wait_time < -1.0:
                        logger.debug(f"⚠️ Audio simulation running {-wait_time:.1f}s behind")
                
                logger.info(f"✅ Audio simulation complete: {chunk_count} chunks streamed")
                
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
                
                # Keep buffer active for a bit longer
                await asyncio.sleep(3.0)
                audio_buffer.stop()
                
        except Exception as e:
            logger.error(f"❌ Error simulating audio: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    async def stop_realtime_processing(self, session_id: str) -> Dict[str, Any]:
        """Stop real-time processing"""
        return await self.stop_live_streaming(session_id)
    
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
                "recognition_restart_count": session.get("recognition_restart_count", 0),
                "has_streaming_task": session_id in self.streaming_tasks,
                "buffer_stats": buffer_stats
            }
        
        return {
            "total_active_sessions": len(self.active_sessions),
            "transcription_engine": self.transcription_source,
            "google_stt_available": self.google_client is not None,
            "telephony_model_available": self.telephony_config_available,
            "api_version": "v1p1beta1_simplified",
            "processing_mode": "google_stt_diarization_only",
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
        """Connect to live telephony stream for real phone calls"""
        
        try:
            logger.info(f"📞 Connecting to telephony stream for session {session_id}")
            
            # Prepare session for telephony
            prepare_result = self._prepare_live_call({"session_id": session_id})
            if "error" in prepare_result:
                return prepare_result
            
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
                "processing_mode": "simplified_telephony",
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
        """Add audio chunk from live telephony stream"""
        
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
            "api_version": "v1p1beta1_simplified",
            "processing_mode": "simplified_google_stt_only",
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "status": "active",
            "transcription_source": self.transcription_source,
            "google_stt_available": self.google_client is not None,
            "telephony_model_available": self.telephony_config_available,
            "api_version": "v1p1beta1_simplified",
            "processing_mode": "simplified_google_stt_only",
            "active_sessions": len(self.active_sessions),
            "capabilities": [cap.value for cap in self.capabilities],
            "config": {
                "sample_rate": self.sample_rate,
                "chunk_duration_ms": self.chunk_duration_ms,
                "channels": self.channels,
                "forced_2_speaker_diarization": True,
                "no_fallback_logic": True
            },
            "metrics": self.metrics,
            "timestamp": get_current_timestamp()
        }

# Create global instance
audio_processor_agent = AudioProcessorAgent()
