# backend/agents/audio_processor/agent.py - FINAL FIXED VERSION
"""
FINAL FIXED Audio Processing Agent for Google Speech-to-Text v1p1beta1
- Independent audio playback and transcription
- Fixed audio generator completion issue
- Better speaker detection with channel analysis
- Proper session cleanup
- Fixed Google STT timeout issues
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
import numpy as np

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings
from ...utils import get_current_timestamp

logger = logging.getLogger(__name__)

class AudioBuffer:
    """Enhanced audio buffer for streaming with better management"""
    
    def __init__(self, chunk_size: int = 3200):
        self.chunk_size = chunk_size
        self.buffer = queue.Queue(maxsize=1000)  # Increased buffer size
        self.is_active = False
        self.total_chunks_added = 0
        self.total_chunks_consumed = 0
        
    def add_chunk(self, audio_data: bytes):
        if self.is_active and len(audio_data) > 0:
            try:
                self.buffer.put(audio_data, block=False)
                self.total_chunks_added += 1
            except queue.Full:
                try:
                    # Remove oldest chunk and add new one
                    self.buffer.get_nowait()
                    self.buffer.put(audio_data, block=False)
                    self.total_chunks_added += 1
                except queue.Empty:
                    pass
 
    def get_chunks(self):
        """Generator for audio chunks with better timeout handling"""
        empty_count = 0
        max_empty = 100  # Increased timeout for long audio
        consecutive_empty = 0
        
        while self.is_active:
            try:
                chunk = self.buffer.get(timeout=0.1)
                if chunk and len(chunk) > 0:
                    empty_count = 0
                    consecutive_empty = 0
                    self.total_chunks_consumed += 1
                    yield chunk
                else:
                    empty_count += 1
                    consecutive_empty += 1
            except queue.Empty:
                empty_count += 1
                consecutive_empty += 1
                
                # Break if too many consecutive empty gets
                if consecutive_empty >= max_empty:
                    logger.info(f"🎵 Audio buffer: Breaking after {consecutive_empty} empty reads")
                    break
                    
                continue
        
        logger.info(f"🎵 Audio buffer complete: {self.total_chunks_consumed}/{self.total_chunks_added} chunks processed")
    
    def start(self):
        self.is_active = True
        self.total_chunks_added = 0
        self.total_chunks_consumed = 0
        
    def stop(self):
        self.is_active = False
        
    def get_stats(self):
        return {
            "active": self.is_active,
            "queue_size": self.buffer.qsize(),
            "total_added": self.total_chunks_added,
            "total_consumed": self.total_chunks_consumed
        }

class StereoSpeakerTracker:
    """ENHANCED Stereo speaker tracker with better channel analysis"""
    
    def __init__(self):
        self.speaker_history = []
        self.current_speaker = "agent"  # Agent always starts
        self.turn_count = {"agent": 0, "customer": 0}
        self.segment_count = 0
        self.channel_energy_history = []
        
        # FIXED MAPPING: Agent = Left Channel, Customer = Right Channel
        self.speaker_channel_mapping = {
            "agent": "left",      # Agent is ALWAYS on left channel
            "customer": "right"   # Customer is ALWAYS on right channel
        }
        
        # Energy thresholds for better detection
        self.energy_threshold = 2.0  # Increased threshold
        self.min_energy_diff = 1000000  # Minimum energy difference to consider
        
        logger.info(f"🎯 FIXED Channel mapping: Agent=LEFT, Customer=RIGHT")
        
    def add_channel_data(self, left_channel: np.ndarray, right_channel: np.ndarray):
        """Store channel energy data for speaker detection"""
        left_energy = np.sum(left_channel.astype(np.float64) ** 2)  # Use float64 for precision
        right_energy = np.sum(right_channel.astype(np.float64) ** 2)
        
        # Only store data if there's significant energy in at least one channel
        if left_energy > self.min_energy_diff or right_energy > self.min_energy_diff:
            self.channel_energy_history.append({
                "left_energy": left_energy,
                "right_energy": right_energy,
                "dominant_channel": "left" if left_energy > right_energy else "right",
                "energy_ratio": left_energy / (right_energy + 1e-10),
                "total_energy": left_energy + right_energy
            })
        
        # Keep only recent history (last 10 chunks for responsive detection)
        if len(self.channel_energy_history) > 10:
            self.channel_energy_history.pop(0)
    
    def detect_speaker(self, google_speaker_tag: Optional[int], transcript: str) -> str:
        """ENHANCED channel-based speaker detection with better logic"""
        
        self.segment_count += 1
        
        # Use recent channel energy to determine active speaker
        if self.channel_energy_history:
            # Look at last 5 chunks for more stable detection
            recent_data = self.channel_energy_history[-5:]
            
            if recent_data:
                # Calculate weighted average (more weight to recent chunks)
                weights = [1, 1, 2, 3, 4]  # More weight to recent data
                total_weight = sum(weights[-len(recent_data):])
                
                weighted_left = sum(d["left_energy"] * w for d, w in zip(recent_data, weights[-len(recent_data):]))
                weighted_right = sum(d["right_energy"] * w for d, w in zip(recent_data, weights[-len(recent_data):]))
                
                avg_left = weighted_left / total_weight
                avg_right = weighted_right / total_weight
                
                # Determine speaker based on energy with higher threshold
                if avg_left > avg_right * self.energy_threshold:  # Left channel dominant
                    detected_speaker = "agent"  # Agent is on left
                elif avg_right > avg_left * self.energy_threshold:  # Right channel dominant
                    detected_speaker = "customer"  # Customer is on right
                else:
                    # Energy levels too close - keep current speaker to avoid flickering
                    detected_speaker = self.current_speaker
                
                # Update speaker if changed
                if detected_speaker != self.current_speaker:
                    logger.info(f"🔄 Channel switch: {self.current_speaker} → {detected_speaker} (L:{avg_left:.0f}, R:{avg_right:.0f})")
                    self.current_speaker = detected_speaker
                    self.turn_count[detected_speaker] += 1
            else:
                detected_speaker = self.current_speaker
        else:
            # No channel data yet - agent starts first
            detected_speaker = "agent" if self.segment_count == 1 else self.current_speaker
        
        self.speaker_history.append(detected_speaker)
        return detected_speaker
    
    def get_stats(self) -> Dict:
        recent_energy = self.channel_energy_history[-3:] if self.channel_energy_history else []
        return {
            "current_speaker": self.current_speaker,
            "turn_count": self.turn_count.copy(),
            "total_segments": len(self.speaker_history),
            "channel_mapping": self.speaker_channel_mapping.copy(),
            "recent_channel_energy": recent_energy,
            "energy_threshold": self.energy_threshold
        }

class AudioProcessorAgent(BaseAgent):
    """FINAL FIXED Stereo Audio Processor for Real-time Phone Calls"""
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Final Fixed Stereo Audio Processor (v1p1beta1)"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, AudioBuffer] = {}
        self.speaker_trackers: Dict[str, StereoSpeakerTracker] = {}
        
        # Audio settings - STEREO ONLY
        self.sample_rate = 16000
        self.channels = 2  # FIXED: Only stereo supported
        self.chunk_duration_ms = 200
        self.chunk_size = int(self.sample_rate * 2 * self.chunk_duration_ms / 1000)
        
        self._initialize_google_stt()
        agent_registry.register_agent(self)
        
        logger.info(f"🎵 {self.agent_name} initialized - STEREO ONLY")
        logger.info(f"🎯 FIXED MAPPING: Agent=LEFT channel, Customer=RIGHT channel")
    
    def _get_default_config(self) -> Dict[str, Any]:
        settings = get_settings()
        return {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio",
            "enable_interim_results": True,
            "max_streaming_duration": 300,
            "restart_timeout": 60  # Reduced to avoid timeout issues
        }
    
    def _get_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_google_stt(self):
        """Initialize Google Speech-to-Text v1p1beta1"""
        try:
            from google.cloud import speech_v1p1beta1 as speech
            self.google_client = speech.SpeechClient()
            self.speech_types = speech
            logger.info("✅ Google Speech-to-Text v1p1beta1 initialized")
        except ImportError as e:
            logger.error(f"❌ Google STT import failed: {e}")
            self.google_client = None
            self.speech_types = None
        except Exception as e:
            logger.error(f"❌ Google STT initialization failed: {e}")
            self.google_client = None
            self.speech_types = None
    
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
                    asyncio.create_task(self.stop_streaming(session_id))
        
        return {
            "agent_type": self.agent_type,
            "status": "ready",
            "active_sessions": len(self.active_sessions),
            "google_stt_available": self.google_client is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    def _prepare_live_call(self, data: Dict) -> Dict[str, Any]:
        """Prepare for live call processing"""
        session_id = data.get('session_id')
        if not session_id:
            return {"error": "session_id required"}
        
        if session_id in self.active_sessions:
            asyncio.create_task(self.stop_streaming(session_id))
        
        self.active_sessions[session_id] = {
            "start_time": datetime.now(),
            "status": "ready",
            "transcription_segments": [],
            "restart_count": 0,
            "audio_completed": False
        }
        
        self.audio_buffers[session_id] = AudioBuffer(self.chunk_size)
        
        logger.info(f"✅ Session {session_id} prepared")
        
        return {
            "session_id": session_id,
            "status": "ready",
            "chunk_size": self.chunk_size,
            "sample_rate": self.sample_rate
        }
    
    def _add_audio_chunk(self, data: Dict) -> Dict[str, Any]:
        """Add audio chunk to buffer"""
        session_id = data.get('session_id')
        audio_data = data.get('audio_data')
        
        if not session_id or session_id not in self.active_sessions:
            return {"error": "Invalid session"}
        
        if not audio_data:
            return {"error": "No audio data"}
        
        self.audio_buffers[session_id].add_chunk(audio_data)
        return {"status": "chunk_added", "size": len(audio_data)}
    
    async def start_realtime_processing(self, session_id: str, audio_filename: str, websocket_callback: Callable) -> Dict[str, Any]:
        """Start real-time STEREO processing with INDEPENDENT audio and transcription"""
        try:
            logger.info(f"🎵 Starting INDEPENDENT STEREO processing: {audio_filename}")
            
            prepare_result = self._prepare_live_call({"session_id": session_id})
            if "error" in prepare_result:
                return prepare_result
            
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            if not audio_path.exists():
                return {"error": f"Audio file not found: {audio_filename}"}
            
            with wave.open(str(audio_path), 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                duration = frames / sample_rate
                
                logger.info(f"📁 Audio: {duration:.1f}s, {sample_rate}Hz, {channels}ch")
            
            # ENFORCE STEREO ONLY
            if channels != 2:
                return {"error": f"Only stereo audio supported. File has {channels} channels."}
            
            # Always use stereo tracker with fixed mapping
            self.speaker_trackers[session_id] = StereoSpeakerTracker()
            processing_mode = "stereo_independent"
            logger.info("🎯 STEREO: Agent=LEFT, Customer=RIGHT (INDEPENDENT PROCESSING)")
            
            # Start transcription FIRST (with small delay)
            await asyncio.sleep(0.3)
            transcription_task = asyncio.create_task(
                self._start_transcription_independent(session_id, websocket_callback)
            )
            self.active_sessions[session_id]["transcription_task"] = transcription_task
            
            # Start audio processing AFTER transcription is ready
            await asyncio.sleep(0.5)
            audio_task = asyncio.create_task(
                self._process_stereo_audio_independent(session_id, audio_path, websocket_callback)
            )
            self.active_sessions[session_id]["audio_task"] = audio_task
            self.active_sessions[session_id]["audio_type"] = "stereo"
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "channels": channels,
                "processing_mode": processing_mode,
                "channel_mapping": "Agent=LEFT, Customer=RIGHT",
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting processing: {e}")
            return {"error": str(e)}
    
    async def _process_stereo_audio_independent(self, session_id: str, audio_path: Path, websocket_callback: Callable):
        """INDEPENDENT stereo audio processing that doesn't block transcription"""
        try:
            logger.info(f"🎵 Starting INDEPENDENT STEREO audio processing for {session_id}")
            
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)
                audio_buffer = self.audio_buffers[session_id]
                speaker_tracker = self.speaker_trackers[session_id]
                
                if not audio_buffer.is_active:
                    audio_buffer.start()
                
                chunk_count = 0
                start_time = time.time()
                total_frames = wav_file.getnframes()
                total_chunks = total_frames // frames_per_chunk
                
                logger.info(f"🎵 Processing {total_chunks} stereo chunks independently")
                
                while chunk_count < total_chunks:
                    # Read stereo frames
                    stereo_chunk = wav_file.readframes(frames_per_chunk)
                    if not stereo_chunk:
                        logger.info(f"📁 Reached end of audio file at chunk {chunk_count}")
                        break
                    
                    # Convert to numpy array for channel processing
                    stereo_array = np.frombuffer(stereo_chunk, dtype=np.int16)
                    left_channel = stereo_array[0::2]   # Agent channel
                    right_channel = stereo_array[1::2]  # Customer channel
                    
                    # Add channel data to tracker for speaker detection
                    speaker_tracker.add_channel_data(left_channel, right_channel)
                    
                    # Mix channels to mono for STT (preserving both speakers)
                    mono_array = ((left_channel.astype(np.int32) + right_channel.astype(np.int32)) // 2).astype(np.int16)
                    mono_chunk = mono_array.tobytes()
                    
                    # Add to buffer for STT processing
                    audio_buffer.add_chunk(mono_chunk)
                    chunk_count += 1
                    
                    # REAL-TIME timing (crucial for Google STT)
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = time.time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    
                    # Progress logging every 50 chunks (10 seconds)
                    if chunk_count % 50 == 0:
                        progress = (chunk_count / total_chunks) * 100 if total_chunks > 0 else 0
                        elapsed = chunk_count * self.chunk_duration_ms / 1000
                        logger.info(f"🎵 Audio progress: {progress:.1f}% ({elapsed:.1f}s) - {chunk_count}/{total_chunks}")
                        
            # Mark audio as completed
            self.active_sessions[session_id]["audio_completed"] = True
            logger.info(f"✅ INDEPENDENT audio processing complete: {chunk_count}/{total_chunks} chunks processed")
            
            # Wait for transcription to finish processing remaining chunks
            await asyncio.sleep(10.0)
            
            # Stop the buffer
            audio_buffer.stop()
            logger.info(f"🛑 Audio buffer stopped after completion")
            
        except Exception as e:
            logger.error(f"❌ Error in independent stereo processing: {e}")
            # Mark as completed even on error
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["audio_completed"] = True
            raise
    
    async def _start_transcription_independent(self, session_id: str, websocket_callback: Callable):
        """Start INDEPENDENT transcription processing"""
        try:
            logger.info(f"🎙️ Starting INDEPENDENT transcription for {session_id}")
            
            streaming_result = await self.start_streaming(session_id, websocket_callback)
            if "error" in streaming_result:
                logger.error(f"❌ Failed to start transcription: {streaming_result['error']}")
                await websocket_callback({
                    "type": "transcription_error", 
                    "data": {
                        "session_id": session_id,
                        "error": streaming_result['error'],
                        "timestamp": get_current_timestamp()
                    }
                })
            
        except Exception as e:
            logger.error(f"❌ Error in independent transcription: {e}")
    
    async def start_streaming(self, session_id: str, websocket_callback: Callable) -> Dict[str, Any]:
        """Start streaming recognition for STEREO audio"""
        try:
            logger.info(f"🎙️ Starting STEREO streaming recognition for {session_id}")
            
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not prepared")
            
            if not self.google_client:
                raise Exception("Google Speech-to-Text not available")
            
            audio_buffer = self.audio_buffers[session_id]
            if not audio_buffer.is_active:
                audio_buffer.start()
                logger.info(f"✅ Audio buffer started for {session_id}")
            
            session = self.active_sessions[session_id]
            session["status"] = "streaming"
            session["websocket_callback"] = websocket_callback
            
            streaming_task = asyncio.create_task(
                self._streaming_with_restart_fixed(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            logger.info(f"✅ STEREO streaming task created for {session_id}")
            
            await websocket_callback({
                "type": "streaming_started",
                "data": {
                    "session_id": session_id,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "model": "phone_call",
                    "api_version": "v1p1beta1",
                    "channel_mapping": "Agent=LEFT, Customer=RIGHT",
                    "timestamp": get_current_timestamp()
                }
            })
            
            logger.info(f"✅ STEREO streaming started confirmation sent for {session_id}")
            
            return {"session_id": session_id, "status": "streaming"}
            
        except Exception as e:
            logger.error(f"❌ Error starting streaming: {e}")
            return {"error": str(e)}
    
    async def _streaming_with_restart_fixed(self, session_id: str):
        """FIXED streaming with restart that doesn't cause timeouts"""
        session = self.active_sessions[session_id]
        restart_interval = 50  # Shorter restart interval to avoid timeouts
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            try:
                await asyncio.wait_for(
                    self._do_streaming_recognition_fixed(session_id),
                    timeout=restart_interval
                )
                
                # Check if audio is completed
                if session.get("audio_completed", False):
                    logger.info(f"🎵 Audio completed, ending streaming for {session_id}")
                    break
                    
            except asyncio.TimeoutError:
                session["restart_count"] += 1
                logger.info(f"🔄 Restarting recognition for {session_id} (restart #{session['restart_count']})")
                
                # Don't restart too many times
                if session["restart_count"] > 10:
                    logger.warning(f"⚠️ Too many restarts for {session_id}, stopping")
                    break
                    
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Recognition error: {e}")
                session["restart_count"] += 1
                if session["restart_count"] > 5:
                    break
                await asyncio.sleep(1.0)
    
    async def _do_streaming_recognition_fixed(self, session_id: str):
        """FIXED streaming recognition that handles audio properly"""
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        speaker_tracker = self.speaker_trackers[session_id]
        
        logger.info(f"🎙️ Starting FIXED Google STT streaming for {session_id}")
        
        # OPTIMIZED recognition config
        recognition_config = self.speech_types.RecognitionConfig(
            encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code="en-GB",
            model="phone_call",
            use_enhanced=True,   
            enable_automatic_punctuation=True,
            enable_word_confidence=True,
            enable_word_time_offsets=True,
            # Simplified diarization
            diarization_config=self.speech_types.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,
                max_speaker_count=2
            )
        )
        
        streaming_config = self.speech_types.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
            single_utterance=False
        )
        
        # FIXED audio request generator
        def audio_generator():
            logger.info(f"🎵 FIXED audio generator starting for {session_id}")
            chunk_count = 0
            last_chunk_time = time.time()
            
            for chunk in audio_buffer.get_chunks():
                if session_id not in self.active_sessions:
                    logger.info(f"🛑 Session {session_id} ended, stopping audio generator")
                    break
                
                chunk_count += 1
                current_time = time.time()
                
                # Log timing information
                if chunk_count <= 5 or chunk_count % 100 == 0:
                    time_since_last = current_time - last_chunk_time
                    logger.info(f"🎵 Chunk {chunk_count}: {len(chunk)} bytes, {time_since_last:.3f}s since last")
                
                last_chunk_time = current_time
                
                request = self.speech_types.StreamingRecognizeRequest()
                request.audio_content = chunk
                yield request
            
            logger.info(f"🎵 FIXED audio generator completed: {chunk_count} chunks sent")
        
        # Start streaming recognition
        try:
            logger.info(f"🚀 Starting Google STT v1p1beta1 for {session_id}")
            
            responses = self.google_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )
            
            response_count = 0
            last_final_text = ""
            
            for response in responses:
                if session_id not in self.active_sessions:
                    logger.info(f"🛑 Session {session_id} ended, stopping response processing")
                    break
                
                response_count += 1
                if response_count <= 5 or response_count % 20 == 0:
                    logger.info(f"📨 Processing response {response_count} for {session_id}")
                
                # Process with stereo speaker detection
                processed = await self._process_stereo_response_fixed(
                    session_id, response, speaker_tracker, websocket_callback, last_final_text
                )
                
                if processed and processed.get("is_final"):
                    last_final_text = processed.get("text", "")
                
            logger.info(f"✅ Google STT streaming completed for {session_id}: {response_count} responses processed")
                
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.warning(f"⚠️ Google STT timeout for {session_id}: {e}")
            else:
                logger.error(f"❌ Streaming error for {session_id}: {e}")
            
            # Send error to client
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
    
    async def _process_stereo_response_fixed(
        self, 
        session_id: str, 
        response, 
        speaker_tracker: StereoSpeakerTracker, 
        websocket_callback: Callable,
        last_final_text: str
    ) -> Optional[Dict]:
        """FIXED response processing with better deduplication"""
        try:
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                
                if not transcript:
                    continue
                
                # Better deduplication
                if result.is_final and last_final_text:
                    if transcript == last_final_text:
                        logger.debug(f"🚫 Skipping exact duplicate: '{transcript[:30]}...'")
                        continue
                    
                    # Skip if it's a subset of the last final text
                    if len(transcript) < len(last_final_text) and transcript in last_final_text:
                        logger.debug(f"🚫 Skipping subset: '{transcript[:30]}...'")
                        continue
                
                # Extract Google speaker tag
                google_speaker_tag = None
                if hasattr(alternative, 'words') and alternative.words:
                    tags = [w.speaker_tag for w in alternative.words 
                           if hasattr(w, 'speaker_tag') and w.speaker_tag is not None]
                    if tags:
                        google_speaker_tag = max(set(tags), key=tags.count)
                
                # Use STEREO channel-based speaker detection
                detected_speaker = speaker_tracker.detect_speaker(google_speaker_tag, transcript)
                
                # Extract timing
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
                    "speaker_tag": google_speaker_tag,
                    "channel_detection": True,
                    "timestamp": get_current_timestamp()
                }
                
                if result.is_final:
                    # Store final segment
                    session = self.active_sessions[session_id]
                    session["transcription_segments"].append(segment_data)
                    
                    await websocket_callback({
                        "type": "transcription_segment",
                        "data": segment_data
                    })
                    
                    logger.info(f"📝 FINAL [{detected_speaker.upper()}]: '{transcript[:50]}...'")
                    
                    # Trigger fraud analysis for customer speech
                    if detected_speaker == "customer":
                        await self._trigger_fraud_analysis(
                            session_id, transcript, websocket_callback
                        )
                    
                    return segment_data
                else:
                    # Send interim results for longer text
                    if len(transcript) > 8:  # Only send substantial interim results
                        await websocket_callback({
                            "type": "transcription_interim",
                            "data": segment_data
                        })
        
        except Exception as e:
            logger.error(f"❌ Error processing stereo response: {e}")
        
        return None
    
    def _extract_timing(self, alternative) -> tuple:
        """Extract timing from alternative"""
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
    
    async def _trigger_fraud_analysis(self, session_id: str, customer_text: str, websocket_callback: Callable):
        """Trigger fraud analysis for customer speech"""
        try:
            await websocket_callback({
                "type": "progressive_fraud_analysis_trigger",
                "data": {
                    "session_id": session_id,
                    "customer_text": customer_text,
                    "timestamp": get_current_timestamp()
                }
            })
        except Exception as e:
            logger.error(f"❌ Error triggering fraud analysis: {e}")
    
    async def stop_streaming(self, session_id: str) -> Dict[str, Any]:
        """FIXED stop streaming with proper cleanup"""
        try:
            logger.info(f"🛑 STOP streaming for session {session_id}")
            
            if session_id not in self.active_sessions:
                logger.warning(f"⚠️ Session {session_id} not found")
                return {"error": "Session not found"}
            
            # Mark session as stopping
            session = self.active_sessions[session_id]
            session["status"] = "stopping"
            
            # Stop audio buffer immediately
            if session_id in self.audio_buffers:
                buffer = self.audio_buffers[session_id]
                buffer.stop()
                logger.info(f"🛑 Audio buffer stopped for {session_id}: {buffer.get_stats()}")
            
            # Cancel streaming task
            if session_id in self.streaming_tasks:
                task = self.streaming_tasks[session_id]
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        logger.info(f"✅ Streaming task cancelled for {session_id}")
                del self.streaming_tasks[session_id]
            
            # Cancel audio task
            if "audio_task" in session:
                audio_task = session["audio_task"]
                if not audio_task.done():
                    audio_task.cancel()
                    try:
                        await asyncio.wait_for(audio_task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        logger.info(f"✅ Audio task cancelled for {session_id}")
            
            # Cancel transcription task
            if "transcription_task" in session:
                transcription_task = session["transcription_task"]
                if not transcription_task.done():
                    transcription_task.cancel()
                    try:
                        await asyncio.wait_for(transcription_task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        logger.info(f"✅ Transcription task cancelled for {session_id}")
            
            # Clean up session data
            if session_id in self.active_sessions:
                segments_count = len(session.get("transcription_segments", []))
                logger.info(f"📊 Session {session_id} processed {segments_count} segments")
                del self.active_sessions[session_id]
                
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
                
            if session_id in self.speaker_trackers:
                tracker_stats = self.speaker_trackers[session_id].get_stats()
                logger.info(f"📊 Speaker stats for {session_id}: {tracker_stats}")
                del self.speaker_trackers[session_id]
            
            logger.info(f"✅ COMPLETE cleanup for {session_id}")
            
            return {
                "session_id": session_id, 
                "status": "stopped", 
                "cleanup": "complete"
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping streaming for {session_id}: {e}")
            
            # Force cleanup even on error
            try:
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                if session_id in self.audio_buffers:
                    del self.audio_buffers[session_id]
                if session_id in self.speaker_trackers:
                    del self.speaker_trackers[session_id]
                if session_id in self.streaming_tasks:
                    del self.streaming_tasks[session_id]
            except:
                pass
            
            return {"error": str(e)}
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all sessions with detailed information"""
        sessions_status = {}
        for session_id, session in self.active_sessions.items():
            speaker_stats = {}
            buffer_stats = {}
            
            if session_id in self.speaker_trackers:
                speaker_stats = self.speaker_trackers[session_id].get_stats()
                
            if session_id in self.audio_buffers:
                buffer_stats = self.audio_buffers[session_id].get_stats()
            
            sessions_status[session_id] = {
                "status": session.get("status"),
                "start_time": session.get("start_time", "").isoformat() if session.get("start_time") else "",
                "transcription_segments": len(session.get("transcription_segments", [])),
                "restart_count": session.get("restart_count", 0),
                "audio_completed": session.get("audio_completed", False),
                "speaker_stats": speaker_stats,
                "buffer_stats": buffer_stats,
                "audio_type": "stereo_independent"
            }
        
        return {
            "active_sessions": len(self.active_sessions),
            "google_stt_available": self.google_client is not None,
            "processing_mode": "stereo_only_independent",
            "channel_mapping": "Agent=LEFT, Customer=RIGHT",
            "sessions": sessions_status,
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
            segments = self.active_sessions[session_id].get("transcription_segments", [])
            return [seg["text"] for seg in segments if seg.get("speaker") == "customer"]
        return None
    
    async def cleanup_completed_sessions(self) -> int:
        """Clean up sessions that have completed audio processing"""
        cleaned_count = 0
        completed_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if session.get("audio_completed", False) and session.get("status") != "streaming":
                completed_sessions.append(session_id)
        
        for session_id in completed_sessions:
            try:
                await self.stop_streaming(session_id)
                cleaned_count += 1
            except Exception as e:
                logger.error(f"❌ Error cleaning up session {session_id}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} completed sessions")
        
        return cleaned_count

# Create global instance
audio_processor_agent = AudioProcessorAgent()
