# backend/agents/audio_processor/agent.py - FIXED GOOGLE STT STREAMING
"""
Audio Processing Agent for LIVE PHONE CALLS
FIXED: Correct Google Speech-to-Text streaming API usage
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
        """Generator for audio chunks - yields actual audio data"""
        while self.is_active:
            try:
                chunk = self.buffer.get(timeout=0.1)
                if chunk and len(chunk) > 0:  # Only yield non-empty chunks
                    yield chunk
            except queue.Empty:
                # No audio available, continue waiting
                continue
            except Exception as e:
                logger.error(f"❌ Error getting audio chunk: {e}")
                break
    
    def start(self):
        self.is_active = True
        
    def stop(self):
        self.is_active = False

class AudioProcessorAgent(BaseAgent):
    """
    Live Audio Processor for Real-time Phone Calls
    FIXED: Uses correct Google Speech-to-Text streaming API
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
            "audio_base_path": "data/sample_audio",  # For demo files
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
            "current_speaker": "customer"  # Default
        }
        
        # Initialize audio buffer
        self.audio_buffers[session_id] = LiveAudioBuffer(self.chunk_size)
        self.speaker_states[session_id] = "customer"
        
        return {
            "session_id": session_id,
            "status": "ready_for_live_audio",
            "chunk_size_bytes": self.chunk_size,
            "sample_rate": self.sample_rate
        }
    
    def _add_audio_chunk(self, data: Dict) -> Dict[str, Any]:
        """Add audio chunk from telephony system"""
        session_id = data.get('session_id')
        audio_data = data.get('audio_data')  # bytes
        speaker = data.get('speaker', 'customer')  # customer or agent
        
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
            
            # IMPORTANT: Start audio buffer BEFORE creating request generator
            self.audio_buffers[session_id].start()
            logger.info("✅ Audio buffer started and ready")
            
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
        """
        FIXED: Perform live streaming recognition with correct Google STT API usage
        """
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        
        try:
            logger.info(f"🎙️ Starting live streaming recognition for {session_id}")
            
            # FIXED: Configure streaming recognition with correct speaker diarization
            # Based on Google documentation: use diarization_config with SpeakerDiarizationConfig
            try:
                # Create speaker diarization config if available
                diarization_config = None
                if hasattr(self.speech_module, 'SpeakerDiarizationConfig'):
                    diarization_config = self.speech_module.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,  # Customer + Agent
                        max_speaker_count=2
                    )
                    logger.info("✅ Speaker diarization config created")
                else:
                    logger.warning("⚠️ SpeakerDiarizationConfig not available, skipping diarization")
                
                # Create main recognition config with robust field handling
                config_params = {
                    "encoding": self.speech_module.RecognitionConfig.AudioEncoding.LINEAR16,
                    "sample_rate_hertz": self.sample_rate,
                    "language_code": "en-GB",
                    "enable_automatic_punctuation": True,
                    "enable_word_confidence": True,
                    "max_alternatives": 1
                }
                
                # Add diarization config if available
                if diarization_config:
                    config_params["diarization_config"] = diarization_config
                
                # Add model if supported (for phone call optimization)
                if hasattr(self.speech_module.RecognitionConfig, 'model'):
                    # Try different model field names
                    try:
                        config_params["model"] = "phone_call"  # Optimized for telephony
                        logger.info("✅ Using phone_call model for telephony optimization")
                    except:
                        try:
                            config_params["model"] = "telephony"  # Alternative name
                            logger.info("✅ Using telephony model")
                        except:
                            logger.info("ℹ️ Using default model (phone_call/telephony not available)")
                
                config = self.speech_module.RecognitionConfig(**config_params)
                logger.info("✅ RecognitionConfig created successfully")
                
            except Exception as config_error:
                logger.error(f"❌ Error creating RecognitionConfig: {config_error}")
                # Fallback to minimal config
                logger.info("🔄 Falling back to minimal configuration...")
                config = self.speech_module.RecognitionConfig(
                    encoding=self.speech_module.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code="en-GB"
                )
                logger.info("✅ Minimal RecognitionConfig created as fallback")
            
            streaming_config = self.speech_module.StreamingRecognitionConfig(
                config=config,
                interim_results=True,  # Get interim results for real-time feedback
                single_utterance=False  # Continuous conversation
            )
            
            logger.info(f"✅ Streaming config created successfully")
            
            # FIXED: Create streaming request generator that sends config exactly once
            # The error shows multiple streaming_config requests are being sent
            def request_generator():
                logger.info("🔄 Starting request generator...")
                config_sent = False
                
                # Send the configuration request exactly once
                if not config_sent:
                    logger.info("✅ Sending streaming config (one time only)")
                    yield self.speech_module.StreamingRecognizeRequest(
                        streaming_config=streaming_config
                    )
                    config_sent = True
                    logger.info("✅ Config sent, now streaming audio...")
                
                # Now stream audio content only
                chunk_count = 0
                try:
                    for audio_chunk in audio_buffer.get_audio_chunks():
                        if session_id not in self.active_sessions:
                            logger.info("🛑 Session ended, stopping audio stream")
                            break
                        
                        # Validate we have audio data
                        if audio_chunk and len(audio_chunk) > 0:
                            chunk_count += 1
                            if chunk_count % 50 == 0:  # Log every 50 chunks (5 seconds)
                                logger.debug(f"🎵 Streaming audio chunk {chunk_count}")
                            
                            # Send ONLY audio_content, no config
                            yield self.speech_module.StreamingRecognizeRequest(
                                audio_content=audio_chunk
                            )
                        else:
                            logger.debug("⚠️ Skipping empty audio chunk")
                            
                except Exception as e:
                    logger.error(f"❌ Error in request generator: {e}")
                finally:
                    logger.info(f"🏁 Request generator completed: {chunk_count} audio chunks sent")
            
            logger.info("🚀 Starting streaming recognition with CORRECT method signature...")
            
            # MULTIPLE ATTEMPTS: Try different method signatures to handle various Google STT versions
            try:
                # Method 1: Named parameters (newer versions)
                logger.info("🔧 Attempting method 1: named parameters...")
                
                # CRITICAL FIX: When using config parameter, don't send streaming_config in requests
                def audio_only_generator():
                    logger.info("🔄 Generating audio-only requests (config passed separately)...")
                    chunk_count = 0
                    try:
                        for audio_chunk in audio_buffer.get_audio_chunks():
                            if session_id not in self.active_sessions:
                                logger.info("🛑 Session ended, stopping audio stream")
                                break
                            
                            if audio_chunk and len(audio_chunk) > 0:
                                chunk_count += 1
                                if chunk_count % 50 == 0:
                                    logger.debug(f"🎵 Streaming audio chunk {chunk_count}")
                                
                                # Send ONLY audio_content when config is passed as parameter
                                yield self.speech_module.StreamingRecognizeRequest(
                                    audio_content=audio_chunk
                                )
                            
                    except Exception as e:
                        logger.error(f"❌ Error in audio generator: {e}")
                    finally:
                        logger.info(f"🏁 Audio generator completed: {chunk_count} chunks sent")
                
                # Call streaming_recognize with config as parameter and audio-only requests
                responses = self.google_client.streaming_recognize(
                    config=streaming_config,
                    requests=audio_only_generator()
                )
                logger.info("✅ Method 1 successful: streaming recognition started")
                
            except (TypeError, AttributeError) as e1:
                logger.warning(f"⚠️ Method 1 failed ({e1}), trying method 2...")
                
                try:
                    # Method 2: Positional parameters (some versions)  
                    logger.info("🔧 Attempting method 2: positional parameters...")
                    responses = self.google_client.streaming_recognize(
                        streaming_config, audio_only_generator()
                    )
                    logger.info("✅ Method 2 successful: streaming recognition started")
                    
                except (TypeError, AttributeError) as e2:
                    logger.warning(f"⚠️ Method 2 failed ({e2}), trying method 3...")
                    
                    try:
                        # Method 3: Traditional approach with config in first request
                        logger.info("🔧 Attempting method 3: config in first request...")
                        responses = self.google_client.streaming_recognize(
                            requests=request_generator()
                        )
                        logger.info("✅ Method 3 successful: streaming recognition started")
                        
                    except (TypeError, AttributeError) as e3:
                        logger.error(f"❌ All streaming methods failed:")
                        logger.error(f"  Method 1: {e1}")
                        logger.error(f"  Method 2: {e2}")
                        logger.error(f"  Method 3: {e3}")
                        
                        # Debug: Show available methods
                        try:
                            client_methods = [method for method in dir(self.google_client) if 'streaming' in method.lower()]
                            logger.error(f"🔍 Available streaming methods: {client_methods}")
                            
                            # Try to inspect the method signature
                            import inspect
                            if hasattr(self.google_client, 'streaming_recognize'):
                                sig = inspect.signature(self.google_client.streaming_recognize)
                                logger.error(f"🔍 Method signature: {sig}")
                                
                        except Exception as debug_e:
                            logger.error(f"🔍 Debug inspection failed: {debug_e}")
                        
                        raise Exception(f"All streaming recognition methods failed: {e1}, {e2}, {e3}")
            
            # Process streaming responses with robust error handling
            response_count = 0
            try:
                for response in responses:
                    response_count += 1
                    if response_count % 10 == 0:  # Less frequent logging
                        logger.debug(f"📨 Processed {response_count} responses")
                        
                    # Check if session is still active
                    if session_id not in self.active_sessions:
                        logger.info("🛑 Session stopped, breaking response loop")
                        break
                    
                    # Process the response
                    await self._process_streaming_response(session_id, response, websocket_callback)
                    
                logger.info(f"✅ Streaming completed successfully: {response_count} responses processed")
                
            except Exception as response_error:
                logger.error(f"❌ Error processing responses: {response_error}")
                raise response_error
            
        except asyncio.CancelledError:
            logger.info(f"🛑 Live streaming cancelled for {session_id}")
        except Exception as e:
            logger.error(f"❌ Error in live streaming: {e}")
            logger.error(f"🔧 Error type: {type(e)}")
            logger.error(f"🔧 Error args: {e.args}")
            
            # More detailed error information
            import traceback
            logger.error(f"🔧 Full traceback: {traceback.format_exc()}")
            
            await websocket_callback({
                "type": "streaming_error",
                "data": {
                    "session_id": session_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": get_current_timestamp()
                }
            })
        finally:
            # Cleanup
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
                        # IMPROVED: Use Google's speaker diarization if available
                        detected_speaker = "customer"  # Default
                        
                        # Check if speaker diarization data is available
                        if hasattr(alternative, 'words') and alternative.words:
                            # Use the speaker tag from the first word as representative
                            first_word = alternative.words[0]
                            if hasattr(first_word, 'speaker_tag') and first_word.speaker_tag:
                                # Map speaker tags to meaningful names
                                # Usually speaker_tag 1 = first speaker, 2 = second speaker
                                speaker_tag = first_word.speaker_tag
                                if speaker_tag == 1:
                                    detected_speaker = "customer"  # Assume first speaker is customer
                                elif speaker_tag == 2:
                                    detected_speaker = "agent"     # Second speaker is agent
                                else:
                                    detected_speaker = f"speaker_{speaker_tag}"  # Multiple speakers
                                
                                logger.debug(f"🎤 Detected speaker tag {speaker_tag} -> {detected_speaker}")
                        else:
                            # Fallback to telephony system's speaker detection
                            detected_speaker = self.speaker_states.get(session_id, "customer")
                        
                        # Create segment data with improved timing
                        start_time = 0.0
                        end_time = 0.0
                        
                        if hasattr(alternative, 'words') and alternative.words:
                            # Get timing from word-level timestamps
                            first_word = alternative.words[0]
                            last_word = alternative.words[-1]
                            
                            if hasattr(first_word, 'start_time') and first_word.start_time:
                                start_time = first_word.start_time.total_seconds()
                            if hasattr(last_word, 'end_time') and last_word.end_time:
                                end_time = last_word.end_time.total_seconds()
                        
                        segment_data = {
                            "session_id": session_id,
                            "speaker": detected_speaker,
                            "text": transcript,
                            "confidence": alternative.confidence,
                            "is_final": result.is_final,
                            "start": start_time,
                            "end": end_time,
                            "duration": end_time - start_time,
                            "processing_mode": "live_streaming",
                            "transcription_source": self.transcription_source,
                            "timestamp": get_current_timestamp()
                        }
                        
                        # Send segment (both interim and final)
                        await websocket_callback({
                            "type": "transcription_segment",
                            "data": segment_data
                        })
                        
                        # If final, trigger fraud analysis for customer speech
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
            # Add to session's customer speech history
            session = self.active_sessions[session_id]
            if "customer_speech_history" not in session:
                session["customer_speech_history"] = []
            
            session["customer_speech_history"].append(customer_text)
            
            # Combine recent customer speech for analysis
            recent_speech = " ".join(session["customer_speech_history"][-5:])  # Last 5 segments
            
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
    
    # Demo method for file simulation (temporary)
    async def start_realtime_processing(
        self, 
        session_id: str, 
        audio_filename: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Demo method: simulate live call from audio file"""
        
        try:
            logger.info(f"🎭 DEMO: Simulating live call from file")
            logger.info(f"🎯 Session: {session_id}, File: {audio_filename}")
            
            # Prepare session for demo
            self._prepare_live_call({"session_id": session_id})
            
            # Start demo streaming simulation
            demo_task = asyncio.create_task(
                self._simulate_live_call_from_file(session_id, audio_filename, websocket_callback)
            )
            self.streaming_tasks[session_id] = demo_task
            
            return {
                "session_id": session_id,
                "status": "demo_simulation_started",
                "processing_mode": "demo_live_call_simulation"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting demo: {e}")
            return {"error": str(e)}
    
    async def _simulate_live_call_from_file(
        self, 
        session_id: str, 
        audio_filename: str, 
        websocket_callback: Callable
    ) -> None:
        """Demo: Simulate live call by streaming audio file in chunks"""
        
        try:
            # Read audio file
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            with open(audio_path, 'rb') as f:
                audio_content = f.read()
            
            # Skip WAV header (44 bytes)
            audio_data = audio_content[44:] if len(audio_content) > 44 else audio_content
            
            # Start live streaming
            await self.start_live_streaming(session_id, websocket_callback)
            
            # Simulate streaming chunks
            current_speaker = "customer"
            chunk_count = 0
            
            for i in range(0, len(audio_data), self.chunk_size):
                if session_id not in self.active_sessions:
                    break
                
                chunk = audio_data[i:i + self.chunk_size]
                
                # Alternate speaker every few chunks (demo)
                if chunk_count % 10 == 0 and chunk_count > 0:
                    current_speaker = "agent" if current_speaker == "customer" else "customer"
                
                # Add chunk to buffer
                self._add_audio_chunk({
                    "session_id": session_id,
                    "audio_data": chunk,
                    "speaker": current_speaker
                })
                
                chunk_count += 1
                
                # Wait for real-time simulation
                await asyncio.sleep(self.chunk_duration_ms / 1000.0)
            
            logger.info(f"🎭 Demo simulation completed: {chunk_count} chunks")
            
        except Exception as e:
            logger.error(f"❌ Demo simulation error: {e}")
    
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
    
    # Alias for compatibility
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
