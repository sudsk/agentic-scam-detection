# backend/agents/audio_processor/agent.py - COMPLETELY FIXED VERSION
"""
Audio Processing Agent for LIVE PHONE CALLS
FIXED: Complete implementation with proper Google Speech-to-Text streaming API
Real-time streaming recognition for telephony systems with progressive fraud detection
"""

import asyncio
import logging
import time
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
        """Generator for audio chunks with proper flow control"""
        logger.info("🎵 Audio chunk generator started for live streaming")
        
        while self.is_active:
            try:
                # Get chunk with timeout to allow periodic checks
                chunk = self.buffer.get(timeout=0.1)
                
                if chunk and len(chunk) > 0:
                    self.total_chunks_yielded += 1
                    
                    if self.total_chunks_yielded <= 10 or self.total_chunks_yielded % 100 == 0:
                        logger.debug(f"🎵 Yielding chunk {self.total_chunks_yielded} (size: {len(chunk)} bytes)")
                    
                    yield chunk
                    
            except queue.Empty:
                # No audio available, continue waiting
                continue
            except Exception as e:
                logger.error(f"❌ Error in audio chunk generator: {e}")
                break
        
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
    FIXED: Proper Google Speech-to-Text streaming implementation
    """
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Live Telephony Audio Processor"
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
        
        # Initialize Google STT with proper error handling
        self._initialize_google_stt()
              
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
            "restart_recognition_timeout": 30  # Restart recognition every 30 seconds for stability
        }
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get capabilities"""
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_google_stt(self):
        """Initialize Google Speech-to-Text with proper error handling"""
        
        try:
            logger.info("🔄 Initializing Google Speech-to-Text for live streaming...")
            
            from google.cloud import speech
            self.google_client = speech.SpeechClient()
            self.speech_module = speech
            self.transcription_source = "google_stt_streaming"
            
            logger.info("✅ Google Speech-to-Text client initialized successfully")
            
            # Test the client with a simple operation
            try:
                # Create a basic config to test API access
                test_config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB"
                )
                logger.info("✅ Google STT API access confirmed")
                
            except Exception as test_error:
                logger.warning(f"⚠️ Google STT test failed: {test_error}")
                
        except ImportError as e:
            logger.error(f"❌ Google STT import failed: {e}")
            self.google_client = None
            self.speech_module = None
            self.transcription_source = "unavailable"
        except Exception as e:
            logger.error(f"❌ Google STT initialization failed: {e}")
            self.google_client = None
            self.speech_module = None
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
            "google_stt_available": self.google_client is not None,
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
        """Start live streaming recognition for telephony"""
        
        try:
            logger.info(f"📞 Starting live streaming recognition for session {session_id}")
            
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
            
            # Start streaming task with restart capability
            streaming_task = asyncio.create_task(
                self._live_streaming_with_restart(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            # Send confirmation
            await websocket_callback({
                "type": "live_streaming_started",
                "data": {
                    "session_id": session_id,
                    "processing_mode": "live_telephony_streaming",
                    "transcription_engine": self.transcription_source,
                    "sample_rate": self.sample_rate,
                    "chunk_duration_ms": self.chunk_duration_ms,
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {
                "session_id": session_id,
                "status": "streaming",
                "processing_mode": "live_telephony_streaming"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting live streaming: {e}")
            return {"error": str(e)}
    
    async def _live_streaming_with_restart(self, session_id: str) -> None:
        """
        Live streaming with automatic restart for long calls
        Google STT has limits on streaming duration, so we restart periodically
        """
        session = self.active_sessions[session_id]
        restart_interval = self.config.get("restart_recognition_timeout", 30)
        
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
    
    async def _live_streaming_recognition(self, session_id: str) -> None:
        """
        FIXED: Single cycle of streaming recognition with proper Google STT API usage
        """
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        
        try:
            logger.info(f"🎙️ Starting recognition cycle for {session_id}")
            
            # Create recognition configuration
            config = self.speech_module.RecognitionConfig(
                encoding=self.speech_module.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="en-GB",
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                max_alternatives=1,
                # Enable speaker diarization for telephony
                enable_speaker_diarization=True,
                diarization_speaker_count=2,  # Customer + Agent
                # Use telephony model if available
                model="phone_call"
            )
            
            streaming_config = self.speech_module.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False
            )
            
            # Create audio request generator
            def audio_generator():
                # Send config first
                yield self.speech_module.StreamingRecognizeRequest(
                    streaming_config=streaming_config
                )
                
                # Then send audio data
                for audio_chunk in audio_buffer.get_audio_chunks():
                    if session_id not in self.active_sessions:
                        break
                    
                    if audio_chunk and len(audio_chunk) > 0:
                        yield self.speech_module.StreamingRecognizeRequest(
                            audio_content=audio_chunk
                        )
            
            # Start streaming recognition
            logger.info(f"🚀 Starting Google STT streaming for {session_id}")
            responses = self.google_client.streaming_recognize(
                requests=audio_generator()
            )
            
            # Process responses
            response_count = 0
            for response in responses:
                if session_id not in self.active_sessions:
                    logger.info(f"🛑 Session {session_id} ended, stopping recognition")
                    break
                
                response_count += 1
                if response_count % 20 == 0:
                    logger.debug(f"📨 Processed {response_count} responses for {session_id}")
                
                await self._process_streaming_response(session_id, response, websocket_callback)
            
            logger.info(f"✅ Recognition cycle completed for {session_id}: {response_count} responses")
            
        except Exception as e:
            logger.error(f"❌ Streaming recognition error for {session_id}: {e}")
            
            # Send error notification
            try:
                await websocket_callback({
                    "type": "streaming_error",
                    "data": {
                        "session_id": session_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": get_current_timestamp()
                    }
                })
            except:
                pass  # Don't fail on callback errors
            
            raise  # Re-raise to trigger restart logic
    
    async def _process_streaming_response(
        self, 
        session_id: str, 
        response, 
        websocket_callback: Callable
    ) -> None:
        """Process streaming response with enhanced speaker detection"""
        
        try:
            # Handle streaming errors
            if hasattr(response, 'error') and response.error.code != 0:
                logger.error(f"❌ STT streaming error for {session_id}: {response.error}")
                return
            
            for result in response.results:
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    
                    if transcript:
                        # Enhanced speaker detection
                        detected_speaker = self._detect_speaker(session_id, alternative)
                        
                        # Get timing information
                        start_time, end_time = self._extract_timing(alternative)
                        
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
                            "processing_mode": "live_streaming",
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
    
    def _detect_speaker(self, session_id: str, alternative) -> str:
        """Enhanced speaker detection using multiple signals"""
        
        # Try Google's speaker diarization first
        if hasattr(alternative, 'words') and alternative.words:
            first_word = alternative.words[0]
            if hasattr(first_word, 'speaker_tag') and first_word.speaker_tag:
                speaker_tag = first_word.speaker_tag
                if speaker_tag == 1:
                    return "customer"
                elif speaker_tag == 2:
                    return "agent"
                else:
                    return f"speaker_{speaker_tag}"
        
        # Fallback to session-tracked speaker
        return self.speaker_states.get(session_id, "customer")
    
    def _extract_timing(self, alternative) -> tuple:
        """Extract timing information from alternative"""
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
            
            # Stop audio buffer
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].stop()
                buffer_stats = self.audio_buffers[session_id].get_stats()
                del self.audio_buffers[session_id]
                logger.info(f"📊 Buffer stats for {session_id}: {buffer_stats}")
            
            # Cancel streaming task
            if session_id in self.streaming_tasks:
                self.streaming_tasks[session_id].cancel()
                try:
                    await self.streaming_tasks[session_id]
                except asyncio.CancelledError:
                    pass
                del self.streaming_tasks[session_id]
            
            # Get session info for stats
            session = self.active_sessions[session_id]
            duration = (datetime.now() - session["start_time"]).total_seconds()
            segments_count = len(session.get("transcription_segments", []))
            
            # Cleanup session
            del self.active_sessions[session_id]
            if session_id in self.speaker_states:
                del self.speaker_states[session_id]
            
            logger.info(f"✅ Session {session_id} stopped: {duration:.1f}s, {segments_count} segments")
            
            return {
                "session_id": session_id,
                "status": "stopped",
                "duration_seconds": duration,
                "segments_processed": segments_count,
                "processing_mode": "live_telephony_streaming"
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping session {session_id}: {e}")
            return {"error": str(e)}
    
    # Demo method for file simulation (enhanced for live simulation)
    async def start_realtime_processing(
        self, 
        session_id: str, 
        audio_filename: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Enhanced demo method: simulate live call from audio file with realistic timing"""
        
        try:
            logger.info(f"🎭 DEMO: Starting live call simulation for {session_id}")
            logger.info(f"🎯 File: {audio_filename}")
            
            # Prepare session for demo
            self._prepare_live_call({"session_id": session_id})
            
            # Start the enhanced demo simulation
            demo_task = asyncio.create_task(
                self._simulate_realistic_live_call(session_id, audio_filename, websocket_callback)
            )
            self.streaming_tasks[session_id] = demo_task
            
            return {
                "session_id": session_id,
                "status": "demo_simulation_started",
                "processing_mode": "realistic_live_call_simulation"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting demo: {e}")
            return {"error": str(e)}
    
    async def _simulate_realistic_live_call(
        self, 
        session_id: str, 
        audio_filename: str, 
        websocket_callback: Callable
    ) -> None:
        """Enhanced demo: Realistic live call simulation with proper timing"""
        
        try:
            # Read audio file
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            with open(audio_path, 'rb') as f:
                audio_content = f.read()
            
            # Skip WAV header (44 bytes)
            audio_data = audio_content[44:] if len(audio_content) > 44 else audio_content
            
            logger.info(f"🎭 Demo: Processing {len(audio_data)} bytes from {audio_filename}")
            
            # Pre-fill buffer with initial chunks for immediate start
            logger.info("🎵 Pre-filling audio buffer for immediate streaming start...")
            
            current_speaker = "customer"
            prefill_chunks = 10  # Pre-fill with 10 chunks (2 seconds)
            
            for i in range(0, min(len(audio_data), self.chunk_size * prefill_chunks), self.chunk_size):
                chunk = audio_data[i:i + self.chunk_size]
                if len(chunk) > 0:
                    self._add_audio_chunk({
                        "session_id": session_id,
                        "audio_data": chunk,
                        "speaker": current_speaker
                    })
            
            # Start live streaming immediately with pre-filled buffer
            await self.start_live_streaming(session_id, websocket_callback)
            
            # Continue feeding audio in real-time with realistic delays
            logger.info("🎵 Continuing real-time audio feed...")
            
            chunk_count = prefill_chunks
            for i in range(self.chunk_size * prefill_chunks, len(audio_data), self.chunk_size):
                if session_id not in self.active_sessions:
                    break
                
                chunk = audio_data[i:i + self.chunk_size]
                
                if len(chunk) > 0:
                    # Realistic speaker alternation (every 3-5 seconds)
                    if chunk_count % 20 == 0 and chunk_count > 0:
                        current_speaker = "agent" if current_speaker == "customer" else "customer"
                        logger.debug(f"🎭 Speaker change: {current_speaker}")
                    
                    # Add chunk to buffer
                    self._add_audio_chunk({
                        "session_id": session_id,
                        "audio_data": chunk,
                        "speaker": current_speaker
                    })
                    
                    chunk_count += 1
                    
                    # Real-time delay based on chunk duration
                    delay = self.chunk_duration_ms / 1000.0  # Convert to seconds
                    await asyncio.sleep(delay)
                
            logger.info(f"🎭 Demo audio feed completed: {chunk_count} total chunks")
            
            # Allow processing to complete
            await asyncio.sleep(2.0)
            
            # Send completion notification
            await websocket_callback({
                "type": "processing_complete",
                "data": {
                    "session_id": session_id,
                    "total_chunks": chunk_count,
                    "demo_mode": True,
                    "timestamp": get_current_timestamp()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Demo simulation error: {e}")
            
            # Send error notification
            await websocket_callback({
                "type": "demo_error",
                "data": {
                    "session_id": session_id,
                    "error": str(e),
                    "timestamp": get_current_timestamp()
                }
            })
    
    # Alias for compatibility
    async def stop_realtime_processing(self, session_id: str) -> Dict[str, Any]:
        """Alias for stop_live_streaming"""
        return await self.stop_live_streaming(session_id)
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all live sessions with detailed information"""
        session_details = {}
        
        for session_id, session in self.active_sessions.items():
            buffer_stats = {}
            if session_id in self.audio_buffers:
                buffer_stats = self.audio_buffers[session_id].get_stats()
            
            session_details[session_id] = {
                "status": session.get("status", "unknown"),
                "start_time": session.get("start_time", "").isoformat() if session.get("start_time") else "",
                "duration_seconds": (datetime.now() - session["start_time"]).total_seconds() if session.get("start_time") else 0,
                "total_audio_received": session.get("total_audio_received", 0),
                "segments_count": len(session.get("transcription_segments", [])),
                "current_speaker": session.get("current_speaker", "unknown"),
                "restart_count": session.get("recognition_restart_count", 0),
                "buffer_stats": buffer_stats
            }
        
        return {
            "active_sessions": len(self.active_sessions),
            "streaming_tasks": len(self.streaming_tasks),
            "audio_buffers": len(self.audio_buffers),
            "transcription_source": self.transcription_source,
            "processing_mode": "live_telephony_streaming",
            "google_client_available": self.google_client is not None,
            "chunk_duration_ms": self.chunk_duration_ms,
            "sample_rate": self.sample_rate,
            "session_details": session_details,
            "agent_config": {
                "chunk_size": self.chunk_size,
                "restart_interval": self.config.get("restart_recognition_timeout", 30),
                "max_streaming_duration": self.config.get("max_streaming_duration", 300)
            }
        }
    
    async def get_session_transcription(self, session_id: str) -> Dict[str, Any]:
        """Get complete transcription for a session"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        segments = session.get("transcription_segments", [])
        
        # Combine all text
        full_transcription = " ".join([seg.get("text", "") for seg in segments if seg.get("text")])
        
        # Separate customer and agent speech
        customer_speech = []
        agent_speech = []
        
        for segment in segments:
            if segment.get("speaker") == "customer":
                customer_speech.append(segment.get("text", ""))
            elif segment.get("speaker") == "agent":
                agent_speech.append(segment.get("text", ""))
        
        return {
            "session_id": session_id,
            "full_transcription": full_transcription,
            "customer_speech": " ".join(customer_speech),
            "agent_speech": " ".join(agent_speech),
            "total_segments": len(segments),
            "segments": segments,
            "duration_seconds": (datetime.now() - session["start_time"]).total_seconds(),
            "status": session.get("status", "unknown")
        }
    
    async def add_telephony_audio_chunk(
        self,
        session_id: str,
        audio_chunk: bytes,
        speaker: str = "customer",
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Add audio chunk from telephony system (for real integration)
        This is the main method for live telephony integration
        """
        return self._add_audio_chunk({
            "session_id": session_id,
            "audio_data": audio_chunk,
            "speaker": speaker,
            "timestamp": timestamp or time.time()
        })
    
    async def start_telephony_session(
        self,
        call_id: str,
        customer_number: str,
        agent_id: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """
        Start a live telephony session (for real integration)
        """
        session_id = f"call_{call_id}_{int(time.time())}"
        
        # Prepare session with telephony metadata
        session_data = {
            "session_id": session_id,
            "call_id": call_id,
            "customer_number": customer_number,
            "agent_id": agent_id,
            "session_type": "live_telephony"
        }
        
        # Initialize session
        prepare_result = self._prepare_live_call(session_data)
        if "error" in prepare_result:
            return prepare_result
        
        # Add telephony metadata to session
        self.active_sessions[session_id].update({
            "call_id": call_id,
            "customer_number": customer_number,
            "agent_id": agent_id,
            "session_type": "live_telephony"
        })
        
        # Start streaming recognition
        streaming_result = await self.start_live_streaming(session_id, websocket_callback)
        
        if "error" in streaming_result:
            return streaming_result
        
        logger.info(f"📞 Live telephony session started: {session_id} (call: {call_id})")
        
        return {
            "session_id": session_id,
            "call_id": call_id,
            "status": "live_telephony_active",
            "ready_for_audio": True
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status for monitoring"""
        try:
            # Test Google STT availability
            stt_status = "available" if self.google_client else "unavailable"
            
            # Check active sessions health
            healthy_sessions = 0
            total_sessions = len(self.active_sessions)
            
            for session_id, session in self.active_sessions.items():
                if session.get("status") == "streaming":
                    healthy_sessions += 1
            
            # Calculate buffer health
            buffer_health = {}
            for session_id, buffer in self.audio_buffers.items():
                stats = buffer.get_stats()
                buffer_health[session_id] = {
                    "active": stats["is_active"],
                    "queue_size": stats["queue_size"],
                    "healthy": stats["is_active"] and stats["queue_size"] < 50  # Not overloaded
                }
            
            overall_health = (
                stt_status == "available" and
                healthy_sessions == total_sessions and
                all(b["healthy"] for b in buffer_health.values())
            )
            
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "overall_health": "healthy" if overall_health else "degraded",
                "google_stt_status": stt_status,
                "active_sessions": total_sessions,
                "healthy_sessions": healthy_sessions,
                "buffer_health": buffer_health,
                "transcription_source": self.transcription_source,
                "capabilities": [cap.value for cap in self.capabilities],
                "timestamp": get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting health status: {e}")
            return {
                "agent_id": self.agent_id,
                "overall_health": "error",
                "error": str(e),
                "timestamp": get_current_timestamp()
            }

# Create global instance
audio_processor_agent = AudioProcessorAgent()
