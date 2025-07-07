# backend/agents/audio_processor/agent.py - IMPROVED VERSION
"""
Improved Audio Processing Agent for Google Speech-to-Text v1p1beta1
- Simplified speaker diarization logic
- Better error handling for long audio files
- Cleaner code with reduced logging
- Support for both mono and stereo audio
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
    """Simplified audio buffer for streaming"""
    
    def __init__(self, chunk_size: int = 3200):
        self.chunk_size = chunk_size
        self.buffer = queue.Queue(maxsize=500)
        self.is_active = False
        
    def add_chunk(self, audio_data: bytes):
        if self.is_active and len(audio_data) > 0:
            try:
                self.buffer.put(audio_data, block=False)
            except queue.Full:
                # Drop oldest chunk if buffer is full
                try:
                    self.buffer.get_nowait()
                    self.buffer.put(audio_data, block=False)
                except queue.Empty:
                    pass
 
    def get_chunks(self):
        """Generator for audio chunks"""
        empty_count = 0
        max_empty = 25  # 2.5 seconds of no data
        
        while self.is_active and empty_count < max_empty:
            try:
                chunk = self.buffer.get(timeout=0.1)
                if chunk and len(chunk) > 0:
                    empty_count = 0
                    yield chunk
                else:
                    empty_count += 1
            except queue.Empty:
                empty_count += 1
                continue
    
    def start(self):
        self.is_active = True
        
    def stop(self):
        self.is_active = False

class SpeakerTracker:
    """Simple speaker tracking for agent-first conversations"""
    
    def __init__(self):
        self.speaker_history = []
        self.current_speaker = "agent"  # Agent always starts
        self.turn_count = {"agent": 0, "customer": 0}
        self.last_speaker_tag = None
        
    def detect_speaker(self, google_speaker_tag: Optional[int], transcript: str) -> str:
        """Detect speaker using simple logic"""
        
        # If this is the first segment, it's always the agent
        if len(self.speaker_history) == 0:
            self.current_speaker = "agent"
            self.turn_count["agent"] += 1
            self.speaker_history.append("agent")
            return "agent"
        
        # Use Google's speaker tag if available and reliable
        if google_speaker_tag is not None:
            if self.last_speaker_tag is None:
                # First time we get a tag - assume it's the current speaker
                self.last_speaker_tag = google_speaker_tag
                detected_speaker = self.current_speaker
            elif google_speaker_tag != self.last_speaker_tag:
                # Speaker tag changed - switch speakers
                detected_speaker = "customer" if self.current_speaker == "agent" else "agent"
                self.current_speaker = detected_speaker
                self.turn_count[detected_speaker] += 1
                self.last_speaker_tag = google_speaker_tag
            else:
                # Same speaker tag - continue with current speaker
                detected_speaker = self.current_speaker
        else:
            # No speaker tag - use simple alternation for very long segments
            # But prefer to keep the same speaker for continuity
            detected_speaker = self.current_speaker
        
        self.speaker_history.append(detected_speaker)
        return detected_speaker
    
    def get_stats(self) -> Dict:
        return {
            "current_speaker": self.current_speaker,
            "turn_count": self.turn_count.copy(),
            "total_segments": len(self.speaker_history)
        }

class AudioProcessorAgent(BaseAgent):
    """Improved Audio Processor for Real-time Phone Calls"""
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Improved Audio Processor (v1p1beta1)"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, AudioBuffer] = {}
        self.speaker_trackers: Dict[str, SpeakerTracker] = {}
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration_ms = 200
        self.chunk_size = int(self.sample_rate * 2 * self.chunk_duration_ms / 1000)
        
        self._initialize_google_stt()
        agent_registry.register_agent(self)
        
        logger.info(f"🎵 {self.agent_name} initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        settings = get_settings()
        return {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio",
            "enable_interim_results": True,
            "max_streaming_duration": 300,
            "restart_timeout": 240
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
        
        # Clean up existing session if any
        if session_id in self.active_sessions:
            asyncio.create_task(self.stop_streaming(session_id))
        
        # Initialize session
        self.active_sessions[session_id] = {
            "start_time": datetime.now(),
            "status": "ready",
            "transcription_segments": [],
            "restart_count": 0
        }
        
        # Initialize components
        self.audio_buffers[session_id] = AudioBuffer(self.chunk_size)
        self.speaker_trackers[session_id] = SpeakerTracker()
        
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
    
    async def start_streaming(self, session_id: str, websocket_callback: Callable) -> Dict[str, Any]:
        """Start streaming recognition"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not prepared")
            
            if not self.google_client:
                raise Exception("Google Speech-to-Text not available")
            
            # Start audio buffer
            audio_buffer = self.audio_buffers[session_id]
            audio_buffer.start()
            
            # Update session
            session = self.active_sessions[session_id]
            session["status"] = "streaming"
            session["websocket_callback"] = websocket_callback
            
            # Start streaming task
            streaming_task = asyncio.create_task(
                self._streaming_with_restart(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            # Send confirmation
            await websocket_callback({
                "type": "streaming_started",
                "data": {
                    "session_id": session_id,
                    "sample_rate": self.sample_rate,
                    "model": "phone_call",
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {"session_id": session_id, "status": "streaming"}
            
        except Exception as e:
            logger.error(f"❌ Error starting streaming: {e}")
            return {"error": str(e)}
    
    async def _streaming_with_restart(self, session_id: str):
        """Streaming with automatic restart for long audio"""
        session = self.active_sessions[session_id]
        restart_interval = 240  # 4 minutes
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            try:
                await asyncio.wait_for(
                    self._do_streaming_recognition(session_id),
                    timeout=restart_interval
                )
            except asyncio.TimeoutError:
                session["restart_count"] += 1
                logger.info(f"🔄 Restarting recognition for {session_id}")
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"❌ Recognition error: {e}")
                session["restart_count"] += 1
                if session["restart_count"] > 3:
                    break
                await asyncio.sleep(2.0)
    
    async def _do_streaming_recognition(self, session_id: str):
        """Core streaming recognition logic"""
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        speaker_tracker = self.speaker_trackers[session_id]
        
        # Create recognition config
        recognition_config = self.speech_types.RecognitionConfig(
            encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code="en-GB",
            model="phone_call",
            enable_automatic_punctuation=True,
            enable_word_confidence=True,
            enable_word_time_offsets=True,
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
        
        # Audio request generator
        def audio_generator():
            for chunk in audio_buffer.get_chunks():
                if session_id not in self.active_sessions:
                    break
                request = self.speech_types.StreamingRecognizeRequest()
                request.audio_content = chunk
                yield request
        
        # Start streaming
        try:
            responses = self.google_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )
            
            # Process responses
            for response in responses:
                if session_id not in self.active_sessions:
                    break
                
                await self._process_response(
                    session_id, response, speaker_tracker, websocket_callback
                )
                
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            raise
    
    async def _process_response(self, session_id: str, response, speaker_tracker: SpeakerTracker, websocket_callback: Callable):
        """Process streaming response"""
        try:
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                
                if not transcript:
                    continue
                
                # Extract speaker tag from words
                speaker_tag = None
                if hasattr(alternative, 'words') and alternative.words:
                    # Get the most common speaker tag in this result
                    tags = [w.speaker_tag for w in alternative.words 
                           if hasattr(w, 'speaker_tag') and w.speaker_tag is not None]
                    if tags:
                        speaker_tag = max(set(tags), key=tags.count)
                
                # Detect speaker
                detected_speaker = speaker_tracker.detect_speaker(speaker_tag, transcript)
                
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
                    "speaker_tag": speaker_tag,
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
                    
                    # Trigger fraud analysis for customer speech
                    if detected_speaker == "customer":
                        await self._trigger_fraud_analysis(
                            session_id, transcript, websocket_callback
                        )
                else:
                    # Send interim result
                    await websocket_callback({
                        "type": "transcription_interim",
                        "data": segment_data
                    })
                    
        except Exception as e:
            logger.error(f"❌ Error processing response: {e}")
    
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
    
    # === DEMO FILE PROCESSING ===
    
    async def start_realtime_processing(self, session_id: str, audio_filename: str, websocket_callback: Callable) -> Dict[str, Any]:
        """Start real-time processing of demo audio file"""
        try:
            # Prepare session
            prepare_result = self._prepare_live_call({"session_id": session_id})
            if "error" in prepare_result:
                return prepare_result
            
            # Load audio file
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            if not audio_path.exists():
                return {"error": f"Audio file not found: {audio_filename}"}
            
            # Check if it's stereo and handle accordingly
            with wave.open(str(audio_path), 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                duration = frames / sample_rate
                
                logger.info(f"📁 Audio: {duration:.1f}s, {sample_rate}Hz, {channels}ch")
                
                # Handle stereo audio
                if channels == 2:
                    await self._process_stereo_audio(session_id, audio_path, websocket_callback)
                else:
                    await self._process_mono_audio(session_id, audio_path, websocket_callback)
            
            # Start streaming recognition
            await asyncio.sleep(0.5)  # Let buffer fill
            streaming_result = await self.start_streaming(session_id, websocket_callback)
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "channels": channels,
                "processing_mode": f"{'stereo' if channels == 2 else 'mono'}_realtime",
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting processing: {e}")
            return {"error": str(e)}
    
    async def _process_stereo_audio(self, session_id: str, audio_path: Path, websocket_callback: Callable):
        """Process stereo audio with channel-based speaker separation"""
        try:
            # Override speaker tracker for stereo
            self.speaker_trackers[session_id] = StereoSpeakerTracker()
            
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)
                audio_buffer = self.audio_buffers[session_id]
                audio_buffer.start()
                
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                
                while True:
                    # Read stereo chunk
                    stereo_chunk = wav_file.readframes(frames_per_chunk)
                    if not stereo_chunk:
                        break
                    
                    # Convert to mono by mixing channels for STT
                    # (We'll track speakers separately using channel analysis)
                    stereo_array = np.frombuffer(stereo_chunk, dtype=np.int16)
                    left_channel = stereo_array[0::2]
                    right_channel = stereo_array[1::2]
                    
                    # Mix to mono for STT (you could also use just one channel)
                    mono_array = ((left_channel.astype(np.int32) + right_channel.astype(np.int32)) // 2).astype(np.int16)
                    mono_chunk = mono_array.tobytes()
                    
                    # Store channel data for speaker detection
                    self.speaker_trackers[session_id].add_channel_data(left_channel, right_channel)
                    
                    # Add to buffer
                    audio_buffer.add_chunk(mono_chunk)
                    chunk_count += 1
                    
                    # Maintain real-time timing
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = asyncio.get_event_loop().time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        
            logger.info(f"✅ Stereo audio processing complete: {chunk_count} chunks")
            
        except Exception as e:
            logger.error(f"❌ Error processing stereo audio: {e}")
    
    async def _process_mono_audio(self, session_id: str, audio_path: Path, websocket_callback: Callable):
        """Process mono audio (existing logic)"""
        try:
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)
                audio_buffer = self.audio_buffers[session_id]
                audio_buffer.start()
                
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                
                while True:
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    if not audio_chunk:
                        break
                    
                    audio_buffer.add_chunk(audio_chunk)
                    chunk_count += 1
                    
                    # Maintain real-time timing
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = asyncio.get_event_loop().time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        
            logger.info(f"✅ Mono audio processing complete: {chunk_count} chunks")
            
        except Exception as e:
            logger.error(f"❌ Error processing mono audio: {e}")
    
    async def stop_streaming(self, session_id: str) -> Dict[str, Any]:
        """Stop streaming for a session"""
        try:
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
                    pass
                del self.streaming_tasks[session_id]
            
            # Cleanup
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            if session_id in self.speaker_trackers:
                del self.speaker_trackers[session_id]
            
            logger.info(f"✅ Streaming stopped for {session_id}")
            
            return {"session_id": session_id, "status": "stopped"}
            
        except Exception as e:
            logger.error(f"❌ Error stopping streaming: {e}")
            return {"error": str(e)}
    
    # === STATUS METHODS ===
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all sessions"""
        sessions_status = {}
        for session_id, session in self.active_sessions.items():
            speaker_stats = {}
            if session_id in self.speaker_trackers:
                speaker_stats = self.speaker_trackers[session_id].get_stats()
            
            sessions_status[session_id] = {
                "status": session.get("status"),
                "start_time": session.get("start_time", "").isoformat() if session.get("start_time") else "",
                "transcription_segments": len(session.get("transcription_segments", [])),
                "restart_count": session.get("restart_count", 0),
                "speaker_stats": speaker_stats
            }
        
        return {
            "active_sessions": len(self.active_sessions),
            "google_stt_available": self.google_client is not None,
            "sessions": sessions_status,
            "timestamp": get_current_timestamp()
        }


class StereoSpeakerTracker(SpeakerTracker):
    """Speaker tracker for stereo audio using channel analysis"""
    
    def __init__(self):
        super().__init__()
        self.channel_data = []
        
    def add_channel_data(self, left_channel: np.ndarray, right_channel: np.ndarray):
        """Store channel data for speaker detection"""
        left_energy = np.sum(left_channel.astype(np.float32) ** 2)
        right_energy = np.sum(right_channel.astype(np.float32) ** 2)
        
        self.channel_data.append({
            "left_energy": left_energy,
            "right_energy": right_energy,
            "dominant_channel": "left" if left_energy > right_energy else "right"
        })
    
    def detect_speaker(self, google_speaker_tag: Optional[int], transcript: str) -> str:
        """Detect speaker using channel energy analysis"""
        
        # For first segment, agent is always first
        if len(self.speaker_history) == 0:
            self.current_speaker = "agent"
            self.turn_count["agent"] += 1
            self.speaker_history.append("agent")
            return "agent"
        
        # Use channel data if available
        if self.channel_data:
            recent_data = self.channel_data[-min(5, len(self.channel_data)):]
            left_dominant_count = sum(1 for d in recent_data if d["dominant_channel"] == "left")
            
            # Assume: Left channel = Agent, Right channel = Customer
            if left_dominant_count > len(recent_data) // 2:
                detected_speaker = "agent"
            else:
                detected_speaker = "customer"
            
            # Track speaker changes
            if detected_speaker != self.current_speaker:
                self.current_speaker = detected_speaker
                self.turn_count[detected_speaker] += 1
            
            self.speaker_history.append(detected_speaker)
            return detected_speaker
        
        # Fallback to parent logic
        return super().detect_speaker(google_speaker_tag, transcript)


# Create global instance
audio_processor_agent = AudioProcessorAgent()
