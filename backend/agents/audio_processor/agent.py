# backend/agents/audio_processor/agent.py - FIXED FOR REAL-TIME STREAMING
"""
Audio Processing Agent for LIVE PHONE CALLS
FIXED: Proper real-time audio streaming to avoid Google STT timeout
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

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings
from ...utils import get_current_timestamp

logger = logging.getLogger(__name__)

class LiveAudioBuffer:
    """Thread-safe audio buffer for live streaming"""
    
    def __init__(self, chunk_size: int = 1600):  # 100ms at 16kHz
        self.chunk_size = chunk_size
        self.buffer = queue.Queue()
        self.is_active = False
        
    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk from telephony system"""
        if self.is_active:
            self.buffer.put(audio_data)
    
    def get_audio_chunks(self):
        """Generator for audio chunks with proper timing"""
        logger.info("🎵 Audio chunk generator started")
        chunk_count = 0
        empty_iterations = 0
        
        while self.is_active:
            try:
                # Try to get chunk with short timeout
                chunk = self.buffer.get(timeout=0.05)  # 50ms timeout
                
                if chunk and len(chunk) > 0:
                    chunk_count += 1
                    empty_iterations = 0
                    
                    if chunk_count <= 5 or chunk_count % 50 == 0:
                        logger.info(f"🎵 Yielding audio chunk {chunk_count} (size: {len(chunk)} bytes)")
                    
                    yield chunk
                else:
                    empty_iterations += 1
                    
            except queue.Empty:
                empty_iterations += 1
                # For real-time streaming, we need to send silence or wait
                # Don't let too much time pass without sending audio
                if empty_iterations > 20:  # 1 second without audio
                    logger.warning(f"⏳ No audio for 1s, sending silence to keep stream alive")
                    # Send silence (zeros) to keep the stream alive
                    silence = b'\x00' * self.chunk_size
                    yield silence
                    empty_iterations = 0
                continue
            except Exception as e:
                logger.error(f"❌ Error in audio chunk generator: {e}")
                break
        
        logger.info(f"🎵 Audio chunk generator completed: {chunk_count} total chunks")
    
    def start(self):
        self.is_active = True
        
    def stop(self):
        self.is_active = False

class AudioProcessorAgent(BaseAgent):
    """
    Live Audio Processor for Real-time Phone Calls
    FIXED: Proper real-time streaming to avoid Google STT timeouts
    """
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Live Telephony Audio Processor"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, LiveAudioBuffer] = {}
        
        # Live streaming settings
        self.chunk_duration_ms = 100  # 100ms chunks
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000) * 2  # 16-bit
        
        # Speaker tracking
        self.speaker_states: Dict[str, str] = {}  # session_id -> current_speaker
        
        # Initialize Google STT
        self._initialize_google_stt()
              
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"📞 {self.agent_name} ready for live calls")
        logger.info(f"🎙️ Chunk size: {self.chunk_duration_ms}ms ({self.chunk_size} bytes)")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        settings = get_settings()
        
        base_config = {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio",
            "enable_interim_results": True,
            "single_utterance": False,
            "max_streaming_duration": 300  # 5 minutes max per stream
        }
        
        return base_config
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get capabilities"""
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_google_stt(self):
        """Initialize Google Speech-to-Text for streaming"""
        
        try:
            logger.info("🔄 Initializing Google STT for live streaming...")
            
            # Try the standard speech library first
            try:
                from google.cloud import speech
                self.google_client = speech.SpeechClient()
                self.speech_module = speech
                self.transcription_source = "google_stt_v1_streaming"
                logger.info("✅ Using Google STT v1")
            except ImportError:
                # Fallback to v1p1beta1 if available
                logger.info("🔄 Trying v1p1beta1...")
                from google.cloud import speech_v1p1beta1 as speech
                self.google_client = speech.SpeechClient()
                self.speech_module = speech  
                self.transcription_source = "google_stt_v1p1beta1_streaming"
                logger.info("✅ Using Google STT v1p1beta1")
            
            logger.info("✅ Google STT live streaming client ready!")
                        
        except ImportError as e:
            logger.error(f"❌ Google STT import failed: {e}")
            self.google_client = None
            self.speech_module = None
            self.transcription_source = "unavailable"
        except Exception as e:
            logger.error(f"❌ Google STT init failed: {e}")
            self.google_client = None
            self.speech_module = None
            self.transcription_source = "failed"
        
        logger.info(f"🎯 Transcription source: {self.transcription_source}")
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Main processing method"""
        if isinstance(input_data, dict):
            action = input_data.get('action', 'status')
            if action == 'start_live_call':
                return self._prepare_live_call(input_data)
            elif action == 'add_audio_chunk':
                return self._add_audio_chunk(input_data)
        
        return {
            "agent_type": self.agent_type,
            "status": "ready_for_live_calls",
            "transcription_source": self.transcription_source,
            "active_calls": len(self.active_sessions),
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _prepare_live_call(self, data: Dict) -> Dict[str, Any]:
        """Prepare for a live call"""
        session_id = data.get('session_id')
        if not session_id:
            return {"error": "session_id required"}
        
        if session_id in self.active_sessions:
            return {"error": f"Session {session_id} already active"}
        
        # Initialize session
        self.active_sessions[session_id] = {
            "start_time": datetime.now(),
            "status": "ready",
            "total_audio_received": 0,
            "transcription_segments": [],
            "current_speaker": "customer"
        }
        
        # Initialize audio buffer
        self.audio_buffers[session_id] = LiveAudioBuffer(self.chunk_size)
        self.audio_buffers[session_id].start()
        self.speaker_states[session_id] = "customer"
        
        logger.info(f"✅ Session {session_id} prepared with active audio buffer")
        
        return {
            "session_id": session_id,
            "status": "ready_for_live_audio",
            "chunk_size_bytes": self.chunk_size,
            "sample_rate": self.sample_rate,
            "audio_buffer_active": True
        }
    
    def _add_audio_chunk(self, data: Dict) -> Dict[str, Any]:
        """Add audio chunk from telephony system"""
        session_id = data.get('session_id')
        audio_data = data.get('audio_data')
        speaker = data.get('speaker', 'customer')
        
        if not session_id or session_id not in self.active_sessions:
            return {"error": "Invalid session"}
        
        if not audio_data:
            return {"error": "No audio data"}
        
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
            "total_bytes": session["total_audio_received"]
        }
    
    async def start_live_streaming(
        self, 
        session_id: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Start live streaming recognition for a phone call"""
        
        try:
            logger.info(f"📞 STARTING LIVE STREAMING")
            logger.info(f"🎯 Session: {session_id}")
            
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not prepared")
            
            if not self.google_client:
                raise Exception("Google Speech-to-Text not available")
            
            # Update session status
            session = self.active_sessions[session_id]
            session["status"] = "streaming"
            session["websocket_callback"] = websocket_callback
            
            # Start streaming task
            streaming_task = asyncio.create_task(
                self._live_streaming_recognition(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            # Send confirmation
            await websocket_callback({
                "type": "live_streaming_started",
                "data": {
                    "session_id": session_id,
                    "processing_mode": "live_telephony_streaming",
                    "transcription_engine": self.transcription_source,
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
    
    async def _live_streaming_recognition(self, session_id: str) -> None:
        """Perform live streaming recognition with proper timing"""
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        
        try:
            logger.info(f"🎙️ Starting live streaming recognition for {session_id}")
            
            # Configure streaming recognition
            config = self.speech_module.RecognitionConfig(
                encoding=self.speech_module.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="en-GB",
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                max_alternatives=1,
                model="phone_call" if hasattr(self.speech_module.RecognitionConfig, 'model') else None
            )
            
            streaming_config = self.speech_module.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False
            )
            
            logger.info(f"✅ Streaming config created successfully")
            
            # Create request generator with proper timing
            def request_generator():
                logger.info("🔄 Starting request generator...")
                
                # Send config first
                yield self.speech_module.StreamingRecognizeRequest(
                    streaming_config=streaming_config
                )
                logger.info("✅ Config sent")
                
                # Now stream audio with proper timing
                chunk_count = 0
                last_chunk_time = time.time()
                
                for audio_chunk in audio_buffer.get_audio_chunks():
                    if session_id not in self.active_sessions:
                        logger.info("🛑 Session ended, stopping audio stream")
                        break
                    
                    # Ensure we're sending audio at roughly real-time rate
                    current_time = time.time()
                    time_since_last = current_time - last_chunk_time
                    
                    # If we're sending too fast, slow down
                    if time_since_last < 0.09:  # Less than 90ms
                        time.sleep(0.1 - time_since_last)
                    
                    chunk_count += 1
                    if chunk_count % 50 == 0:
                        logger.debug(f"🎵 Streaming chunk {chunk_count}")
                    
                    yield self.speech_module.StreamingRecognizeRequest(
                        audio_content=audio_chunk
                    )
                    
                    last_chunk_time = time.time()
                
                logger.info(f"🏁 Request generator completed: {chunk_count} chunks")
            
            # Start streaming recognition
            logger.info("🚀 Starting streaming recognition...")
            responses = self.google_client.streaming_recognize(
                requests=request_generator()
            )
            
            # Process responses
            response_count = 0
            for response in responses:
                response_count += 1
                
                if session_id not in self.active_sessions:
                    logger.info("🛑 Session stopped, breaking response loop")
                    break
                
                await self._process_streaming_response(session_id, response, websocket_callback)
            
            logger.info(f"✅ Streaming completed: {response_count} responses processed")
            
        except asyncio.CancelledError:
            logger.info(f"🛑 Live streaming cancelled for {session_id}")
        except Exception as e:
            logger.error(f"❌ Error in live streaming: {e}")
            await websocket_callback({
                "type": "streaming_error",
                "data": {
                    "session_id": session_id,
                    "error": str(e),
                    "timestamp": get_current_timestamp()
                }
            })
        finally:
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].stop()
    
    async def _process_streaming_response(
        self, 
        session_id: str, 
        response, 
        websocket_callback: Callable
    ) -> None:
        """Process individual streaming response"""
        
        try:
            if response.error:
                logger.error(f"❌ Streaming error: {response.error}")
                return
            
            for result in response.results:
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    
                    if transcript:
                        # Use current speaker state
                        detected_speaker = self.speaker_states.get(session_id, "customer")
                        
                        # Create segment data
                        segment_data = {
                            "session_id": session_id,
                            "speaker": detected_speaker,
                            "text": transcript,
                            "confidence": alternative.confidence,
                            "is_final": result.is_final,
                            "processing_mode": "live_streaming",
                            "transcription_source": self.transcription_source,
                            "timestamp": get_current_timestamp()
                        }
                        
                        # Send segment
                        await websocket_callback({
                            "type": "transcription_segment",
                            "data": segment_data
                        })
                        
                        # Trigger fraud analysis for customer speech
                        if result.is_final and detected_speaker == "customer":
                            await self._trigger_progressive_fraud_analysis(
                                session_id, transcript, websocket_callback
                            )
                        
                        # Log the segment
                        status = "FINAL" if result.is_final else "interim"
                        logger.info(f"📝 {status}: {detected_speaker} - '{transcript[:50]}...'")
        
        except Exception as e:
            logger.error(f"❌ Error processing streaming response: {e}")
    
    async def _trigger_progressive_fraud_analysis(
        self, 
        session_id: str, 
        customer_text: str, 
        websocket_callback: Callable
    ) -> None:
        """Trigger progressive fraud analysis as customer speaks"""
        
        try:
            session = self.active_sessions[session_id]
            if "customer_speech_history" not in session:
                session["customer_speech_history"] = []
            
            session["customer_speech_history"].append(customer_text)
            
            # Combine recent customer speech
            recent_speech = " ".join(session["customer_speech_history"][-5:])
            
            if len(recent_speech.strip()) >= 20:
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
    
    # Demo method with improved real-time simulation
    async def start_realtime_processing(
        self, 
        session_id: str, 
        audio_filename: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Demo method: simulate live call with proper real-time streaming"""
        
        try:
            logger.info(f"🎭 DEMO: Starting live call simulation")
            logger.info(f"🎯 Session: {session_id}, File: {audio_filename}")
            
            # Prepare session
            self._prepare_live_call({"session_id": session_id})
            
            # Start the demo simulation with real-time feeding
            demo_task = asyncio.create_task(
                self._simulate_live_call_realtime(session_id, audio_filename, websocket_callback)
            )
            self.streaming_tasks[session_id] = demo_task
            
            return {
                "session_id": session_id,
                "status": "demo_simulation_started",
                "processing_mode": "demo_live_call_realtime"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting demo: {e}")
            return {"error": str(e)}
    
    async def _simulate_live_call_realtime(
        self, 
        session_id: str, 
        audio_filename: str, 
        websocket_callback: Callable
    ) -> None:
        """Demo: Simulate live call with real-time audio feeding"""
        
        try:
            # Read audio file
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            with open(audio_path, 'rb') as f:
                audio_content = f.read()
            
            # Skip WAV header
            audio_data = audio_content[44:] if len(audio_content) > 44 else audio_content
            
            logger.info(f"🎭 Demo: Loaded {len(audio_data)} bytes of audio")
            
            # Start audio feeder task that simulates real-time
            feeder_task = asyncio.create_task(
                self._feed_audio_realtime(session_id, audio_data)
            )
            
            # Start live streaming
            await self.start_live_streaming(session_id, websocket_callback)
            
            # Wait for feeder to complete
            await feeder_task
            
            logger.info(f"🎭 Demo completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Demo simulation error: {e}")
            await websocket_callback({
                "type": "demo_error",
                "data": {
                    "session_id": session_id,
                    "error": str(e),
                    "timestamp": get_current_timestamp()
                }
            })
    
    async def _feed_audio_realtime(self, session_id: str, audio_data: bytes) -> None:
        """Feed audio chunks at real-time rate"""
        
        try:
            current_speaker = "customer"
            chunk_count = 0
            start_time = time.time()
            
            # Feed chunks at real-time rate (100ms per chunk)
            for i in range(0, len(audio_data), self.chunk_size):
                if session_id not in self.active_sessions:
                    break
                
                chunk = audio_data[i:i + self.chunk_size]
                if len(chunk) > 0:
                    # Pad if necessary
                    if len(chunk) < self.chunk_size:
                        chunk = chunk + b'\x00' * (self.chunk_size - len(chunk))
                    
                    # Switch speakers occasionally
                    if chunk_count > 0 and chunk_count % 150 == 0:  # Every 15 seconds
                        current_speaker = "agent" if current_speaker == "customer" else "customer"
                    
                    # Add chunk
                    self._add_audio_chunk({
                        "session_id": session_id,
                        "audio_data": chunk,
                        "speaker": current_speaker
                    })
                    
                    chunk_count += 1
                    
                    # Calculate timing
                    expected_time = start_time + (chunk_count * 0.1)  # 100ms per chunk
                    current_time = time.time()
                    
                    # Sleep to maintain real-time rate
                    if current_time < expected_time:
                        await asyncio.sleep(expected_time - current_time)
                    
                    if chunk_count % 100 == 0:
                        logger.debug(f"🎵 Fed {chunk_count} chunks ({chunk_count * 0.1:.1f}s)")
            
            logger.info(f"✅ Audio feeding completed: {chunk_count} chunks in {time.time() - start_time:.1f}s")
            
            # Wait a bit before stopping
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Error feeding audio: {e}")
    
    async def stop_live_streaming(self, session_id: str) -> Dict[str, Any]:
        """Stop live streaming for a session"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            # Stop audio buffer
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].stop()
                del self.audio_buffers[session_id]
            
            # Cancel streaming task
            if session_id in self.streaming_tasks:
                self.streaming_tasks[session_id].cancel()
                del self.streaming_tasks[session_id]
            
            # Get session info
            session = self.active_sessions[session_id]
            duration = (datetime.now() - session["start_time"]).total_seconds()
            
            # Cleanup
            del self.active_sessions[session_id]
            if session_id in self.speaker_states:
                del self.speaker_states[session_id]
            
            logger.info(f"🛑 Stopped live streaming for {session_id}")
            
            return {
                "session_id": session_id,
                "status": "stopped",
                "duration": duration,
                "processing_mode": "live_telephony_streaming"
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping session: {e}")
            return {"error": str(e)}
    
    async def stop_realtime_processing(self, session_id: str) -> Dict[str, Any]:
        """Alias for stop_live_streaming"""
        return await self.stop_live_streaming(session_id)
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all live sessions"""
        return {
            "active_sessions": len(self.active_sessions),
            "streaming_tasks": len(self.streaming_tasks),
            "audio_buffers": len(self.audio_buffers),
            "transcription_source": self.transcription_source,
            "processing_mode": "live_telephony_streaming",
            "google_client_available": self.google_client is not None,
            "chunk_duration_ms": self.chunk_duration_ms,
            "sample_rate": self.sample_rate
        }

# Create global instance
audio_processor_agent = AudioProcessorAgent()
