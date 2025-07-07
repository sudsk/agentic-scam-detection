# backend/agents/audio_processor/agent.py - PERFECT VERSION
"""
PERFECT Audio Processing Agent for Google Speech-to-Text v1p1beta1
- Independent UI audio player with backend processing
- Transcription roughly synced with UI playback (1-3s lag)
- Stop button stops both UI player and backend processing
- Optimal pacing for Google STT
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
    """Audio buffer optimized for UI sync"""
    
    def __init__(self, chunk_size: int = 3200):
        self.chunk_size = chunk_size
        self.buffer = queue.Queue(maxsize=150)  # Smaller buffer for responsiveness
        self.is_active = False
        self.total_chunks_added = 0
        
    def add_chunk(self, audio_data: bytes):
        if self.is_active and len(audio_data) > 0:
            try:
                self.buffer.put(audio_data, block=False)
                self.total_chunks_added += 1
            except queue.Full:
                # Drop oldest chunk if buffer is full
                try:
                    self.buffer.get_nowait()
                    self.buffer.put(audio_data, block=False)
                    self.total_chunks_added += 1
                except queue.Empty:
                    pass
 
    def get_chunks(self):
        """Chunk generator optimized for UI sync"""
        empty_count = 0
        max_empty = 25  # 2.5 seconds timeout for responsiveness
        
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
        self.total_chunks_added = 0
        
    def stop(self):
        self.is_active = False

class UISyncStereoTracker:
    """Stereo speaker tracker optimized for UI synchronization"""
    
    def __init__(self):
        self.speaker_history = []
        self.current_speaker = "agent"
        self.turn_count = {"agent": 0, "customer": 0}
        self.segment_count = 0
        self.channel_energy_buffer = []
        
        logger.info(f"🎯 UI-Sync Channel mapping: Agent=LEFT, Customer=RIGHT")
        
    def add_channel_data(self, left_channel: np.ndarray, right_channel: np.ndarray):
        """Store channel energy for responsive detection"""
        left_energy = float(np.sum(left_channel.astype(np.float64) ** 2))
        right_energy = float(np.sum(right_channel.astype(np.float64) ** 2))
        
        # Store if there's meaningful energy
        if left_energy > 500000 or right_energy > 500000:  # Lower threshold for responsiveness
            self.channel_energy_buffer.append({
                "left": left_energy,
                "right": right_energy
            })
            
            # Keep only last 3 measurements for quick response
            if len(self.channel_energy_buffer) > 3:
                self.channel_energy_buffer.pop(0)
    
    def detect_speaker(self, google_speaker_tag: Optional[int], transcript: str) -> str:
        """Fast speaker detection for UI sync"""
        
        self.segment_count += 1
        
        # First segment is always agent
        if self.segment_count == 1:
            self.current_speaker = "agent"
            self.turn_count["agent"] += 1
            self.speaker_history.append("agent")
            return "agent"
        
        # Use channel energy if available
        if len(self.channel_energy_buffer) >= 1:  # Respond quickly
            recent = self.channel_energy_buffer[-1:]  # Just use most recent
            avg_left = recent[0]["left"]
            avg_right = recent[0]["right"]
            
            # Quick threshold-based detection
            if avg_left > avg_right * 2.5:  # Left dominance
                detected_speaker = "agent"
            elif avg_right > avg_left * 2.5:  # Right dominance
                detected_speaker = "customer"
            else:
                # Use content hints for quick decision
                detected_speaker = self._quick_content_detection(transcript)
        else:
            # Fallback to content detection
            detected_speaker = self._quick_content_detection(transcript)
        
        # Update if changed
        if detected_speaker != self.current_speaker:
            logger.info(f"🔄 Quick switch: {self.current_speaker} → {detected_speaker}")
            self.current_speaker = detected_speaker
            self.turn_count[detected_speaker] += 1
        
        self.speaker_history.append(detected_speaker)
        return detected_speaker
    
    def _quick_content_detection(self, transcript: str) -> str:
        """Quick content-based detection for UI responsiveness"""
        text = transcript.lower()
        
        # Strong agent indicators
        if any(phrase in text for phrase in ["hsbc", "customer services", "james", "mrs williams", "i can help"]):
            return "agent"
        
        # Strong customer indicators  
        if any(phrase in text for phrase in ["hi", "hello", "i need", "urgent", "turkey", "alex", "patricia"]):
            return "customer"
        
        # Default: keep current
        return self.current_speaker
    
    def get_stats(self) -> Dict:
        return {
            "current_speaker": self.current_speaker,
            "turn_count": self.turn_count.copy(),
            "total_segments": len(self.speaker_history)
        }

class AudioProcessorAgent(BaseAgent):
    """Perfect Audio Processor with UI Sync"""
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Perfect UI-Sync Audio Processor (v1p1beta1)"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, AudioBuffer] = {}
        self.speaker_trackers: Dict[str, UISyncStereoTracker] = {}
        
        # Audio settings optimized for UI sync
        self.sample_rate = 16000
        self.channels = 2
        self.chunk_duration_ms = 200
        self.chunk_size = int(self.sample_rate * 2 * self.chunk_duration_ms / 1000)
        
        # UI sync settings
        self.ui_sync_delay = 1.5  # 1.5 second delay to roughly sync with UI
        self.processing_speed_factor = 1.1  # Slightly faster than real-time
        
        self._initialize_google_stt()
        agent_registry.register_agent(self)
        
        logger.info(f"🎵 {self.agent_name} initialized - UI SYNC MODE")
        logger.info(f"🎯 UI sync delay: {self.ui_sync_delay}s, Speed factor: {self.processing_speed_factor}x")
    
    def _get_default_config(self) -> Dict[str, Any]:
        settings = get_settings()
        return {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio",
            "enable_interim_results": True,
            "max_streaming_duration": 300,
            "restart_timeout": 40  # Shorter for responsiveness
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
            "ui_sync_enabled": True,
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
            "ui_player_started": False,
            "processing_stopped": False
        }
        
        self.audio_buffers[session_id] = AudioBuffer(self.chunk_size)
        self.speaker_trackers[session_id] = UISyncStereoTracker()
        
        logger.info(f"✅ Session {session_id} prepared for UI sync")
        
        return {
            "session_id": session_id,
            "status": "ready",
            "chunk_size": self.chunk_size,
            "sample_rate": self.sample_rate,
            "ui_sync_enabled": True
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
        """Start UI-synced processing with independent audio player"""
        try:
            logger.info(f"🎵 Starting UI-SYNC processing: {audio_filename}")
            
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
            
            if channels != 2:
                return {"error": f"Only stereo audio supported. File has {channels} channels."}
            
            processing_mode = "ui_synced_independent"
            logger.info("🎯 UI-SYNC: Independent player with synced transcription")
            
            # Send UI player start signal first
            await websocket_callback({
                "type": "ui_player_start",
                "data": {
                    "session_id": session_id,
                    "audio_filename": audio_filename,
                    "duration": duration,
                    "message": "Start UI audio player independently",
                    "timestamp": get_current_timestamp()
                }
            })
            
            # Mark UI player as started
            self.active_sessions[session_id]["ui_player_started"] = True
            
            # Start transcription (with UI sync delay)
            transcription_task = asyncio.create_task(
                self._start_ui_sync_transcription(session_id, websocket_callback)
            )
            self.active_sessions[session_id]["transcription_task"] = transcription_task
            
            # Start backend audio processing (slightly delayed for UI sync)
            await asyncio.sleep(self.ui_sync_delay)
            
            audio_task = asyncio.create_task(
                self._process_audio_ui_sync(session_id, audio_path, websocket_callback)
            )
            self.active_sessions[session_id]["audio_task"] = audio_task
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "channels": channels,
                "processing_mode": processing_mode,
                "ui_sync_delay": self.ui_sync_delay,
                "channel_mapping": "Agent=LEFT, Customer=RIGHT",
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting processing: {e}")
            return {"error": str(e)}
    
    async def _process_audio_ui_sync(self, session_id: str, audio_path: Path, websocket_callback: Callable):
        """Audio processing synced with UI player"""
        try:
            logger.info(f"🎵 Starting UI-SYNC audio processing for {session_id}")
            
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)
                audio_buffer = self.audio_buffers[session_id]
                speaker_tracker = self.speaker_trackers[session_id]
                
                audio_buffer.start()
                
                chunk_count = 0
                start_time = time.time()
                total_frames = wav_file.getnframes()
                total_chunks = total_frames // frames_per_chunk
                
                logger.info(f"🎵 Processing {total_chunks} chunks with UI sync timing")
                
                while chunk_count < total_chunks:
                    # Check if processing was stopped
                    if self.active_sessions[session_id].get("processing_stopped", False):
                        logger.info(f"🛑 Processing stopped by user for {session_id}")
                        break
                    
                    # Read stereo frames
                    stereo_chunk = wav_file.readframes(frames_per_chunk)
                    if not stereo_chunk:
                        break
                    
                    # Process stereo channels
                    stereo_array = np.frombuffer(stereo_chunk, dtype=np.int16)
                    left_channel = stereo_array[0::2]
                    right_channel = stereo_array[1::2]
                    
                    # Track channel energy
                    speaker_tracker.add_channel_data(left_channel, right_channel)
                    
                    # Convert to mono for STT
                    mono_array = ((left_channel.astype(np.int32) + right_channel.astype(np.int32)) // 2).astype(np.int16)
                    mono_chunk = mono_array.tobytes()
                    
                    # Add to buffer
                    audio_buffer.add_chunk(mono_chunk)
                    chunk_count += 1
                    
                    # UI-synced timing (slightly faster than real-time)
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / (1000 * self.processing_speed_factor))
                    current_time = time.time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    
                    # Progress logging
                    if chunk_count % 100 == 0:
                        progress = (chunk_count / total_chunks) * 100
                        elapsed = chunk_count * self.chunk_duration_ms / 1000
                        logger.info(f"🎵 UI-Sync Progress: {progress:.1f}% ({elapsed:.1f}s) - {chunk_count}/{total_chunks}")
                        
                        # Send progress update to UI
                        await websocket_callback({
                            "type": "processing_progress",
                            "data": {
                                "session_id": session_id,
                                "progress": progress,
                                "chunks_processed": chunk_count,
                                "total_chunks": total_chunks,
                                "timestamp": get_current_timestamp()
                            }
                        })
                        
            logger.info(f"✅ UI-sync audio processing complete: {chunk_count}/{total_chunks} chunks")
            
            # Wait for transcription to finish
            await asyncio.sleep(3.0)
            audio_buffer.stop()
            
            # Notify UI that processing is complete
            await websocket_callback({
                "type": "processing_complete",
                "data": {
                    "session_id": session_id,
                    "total_chunks": chunk_count,
                    "message": "Backend processing complete",
                    "timestamp": get_current_timestamp()
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error in UI-sync audio processing: {e}")
            raise
    
    async def _start_ui_sync_transcription(self, session_id: str, websocket_callback: Callable):
        """Start transcription with UI synchronization"""
        try:
            logger.info(f"🎙️ Starting UI-SYNC transcription for {session_id}")
            
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
            logger.error(f"❌ Error in UI-sync transcription: {e}")
    
    async def start_streaming(self, session_id: str, websocket_callback: Callable) -> Dict[str, Any]:
        """Start streaming recognition with UI sync"""
        try:
            logger.info(f"🎙️ Starting UI-sync streaming recognition for {session_id}")
            
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not prepared")
            
            if not self.google_client:
                raise Exception("Google Speech-to-Text not available")
            
            audio_buffer = self.audio_buffers[session_id]
            audio_buffer.start()
            
            session = self.active_sessions[session_id]
            session["status"] = "streaming"
            session["websocket_callback"] = websocket_callback
            
            streaming_task = asyncio.create_task(
                self._streaming_with_restart_ui_sync(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            await websocket_callback({
                "type": "streaming_started",
                "data": {
                    "session_id": session_id,
                    "sample_rate": self.sample_rate,
                    "model": "phone_call",
                    "processing_mode": "ui_synced",
                    "ui_sync_delay": self.ui_sync_delay,
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {"session_id": session_id, "status": "streaming"}
            
        except Exception as e:
            logger.error(f"❌ Error starting streaming: {e}")
            return {"error": str(e)}
    
    async def _streaming_with_restart_ui_sync(self, session_id: str):
        """Streaming with restart optimized for UI sync"""
        session = self.active_sessions[session_id]
        restart_interval = 40  # Shorter for UI responsiveness
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            # Check if processing was stopped
            if session.get("processing_stopped", False):
                logger.info(f"🛑 Streaming stopped by user for {session_id}")
                break
                
            try:
                await asyncio.wait_for(
                    self._do_streaming_recognition_ui_sync(session_id),
                    timeout=restart_interval
                )
            except asyncio.TimeoutError:
                session["restart_count"] += 1
                logger.info(f"🔄 UI-sync restart #{session['restart_count']} for {session_id}")
                
                if session["restart_count"] > 6:  # Fewer restarts for UI responsiveness
                    logger.warning(f"Too many restarts for {session_id}")
                    break
                    
                await asyncio.sleep(0.3)  # Shorter delay
                
            except Exception as e:
                logger.error(f"❌ Recognition error: {e}")
                session["restart_count"] += 1
                if session["restart_count"] > 3:
                    break
                await asyncio.sleep(0.5)
    
    async def _do_streaming_recognition_ui_sync(self, session_id: str):
        """Core streaming recognition optimized for UI sync"""
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        speaker_tracker = self.speaker_trackers[session_id]
        
        # Optimized recognition config for UI sync
        recognition_config = self.speech_types.RecognitionConfig(
            encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code="en-GB",
            model="phone_call",
            use_enhanced=True,
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
        
        def audio_generator():
            chunk_count = 0
            for chunk in audio_buffer.get_chunks():
                if session_id not in self.active_sessions or session.get("processing_stopped", False):
                    break
                chunk_count += 1
                request = self.speech_types.StreamingRecognizeRequest()
                request.audio_content = chunk
                yield request
            logger.info(f"🎵 UI-sync audio generator sent {chunk_count} chunks")
        
        try:
            responses = self.google_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )
            
            response_count = 0
            last_final_text = ""
            
            for response in responses:
                if session_id not in self.active_sessions or session.get("processing_stopped", False):
                    break
                
                response_count += 1
                processed = await self._process_response_ui_sync(
                    session_id, response, speaker_tracker, websocket_callback, last_final_text
                )
                
                if processed and processed.get("is_final"):
                    last_final_text = processed.get("text", "")
                
            logger.info(f"✅ UI-sync streaming completed: {response_count} responses")
                
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.warning(f"⚠️ UI-sync timeout: {e}")
            else:
                logger.error(f"❌ UI-sync streaming error: {e}")
            raise
    
    async def _process_response_ui_sync(
        self, 
        session_id: str, 
        response, 
        speaker_tracker: UISyncStereoTracker, 
        websocket_callback: Callable,
        last_final_text: str
    ) -> Optional[Dict]:
        """Response processing optimized for UI sync"""
        try:
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                
                if not transcript:
                    continue
                
                # Quick deduplication for UI responsiveness
                if result.is_final and transcript == last_final_text:
                    continue
                
                # Extract Google speaker tag
                google_speaker_tag = None
                if hasattr(alternative, 'words') and alternative.words:
                    tags = [w.speaker_tag for w in alternative.words 
                           if hasattr(w, 'speaker_tag') and w.speaker_tag is not None]
                    if tags:
                        google_speaker_tag = max(set(tags), key=tags.count)
                
                # Fast speaker detection
                detected_speaker = speaker_tracker.detect_speaker(google_speaker_tag, transcript)
                
                # Create segment data
                segment_data = {
                    "session_id": session_id,
                    "speaker": detected_speaker,
                    "text": transcript,
                    "confidence": getattr(alternative, 'confidence', 0.9),
                    "is_final": result.is_final,
                    "speaker_tag": google_speaker_tag,
                    "ui_synced": True,
                    "timestamp": get_current_timestamp()
                }
                
                if result.is_final:
                    session = self.active_sessions[session_id]
                    session["transcription_segments"].append(segment_data)
                    
                    await websocket_callback({
                        "type": "transcription_segment",
                        "data": segment_data
                    })
                    
                    logger.info(f"📝 UI-SYNC [{detected_speaker.upper()}]: '{transcript[:50]}...'")
                    
                    if detected_speaker == "customer":
                        await self._trigger_fraud_analysis(session_id, transcript, websocket_callback)
                    
                    return segment_data
                else:
                    # Send interim results for longer text
                    if len(transcript) > 8:
                        await websocket_callback({
                            "type": "transcription_interim",
                            "data": segment_data
                        })
        
        except Exception as e:
            logger.error(f"❌ Error processing UI-sync response: {e}")
        
        return None
    
    async def _trigger_fraud_analysis(self, session_id: str, customer_text: str, websocket_callback: Callable):
        """Trigger fraud analysis"""
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
        """Stop both UI player and backend processing"""
        try:
            logger.info(f"🛑 STOP request for UI-sync session {session_id}")
            
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            
            # Mark as stopped immediately
            session["status"] = "stopping"
            session["processing_stopped"] = True
            
            # Send stop signal to UI player first
            websocket_callback = session.get("websocket_callback")
            if websocket_callback:
                await websocket_callback({
                    "type": "ui_player_stop",
                    "data": {
                        "session_id": session_id,
                        "message": "Stop UI audio player",
                        "timestamp": get_current_timestamp()
                    }
                })
            
            # Stop buffer immediately
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].stop()
                logger.info(f"🛑 Audio buffer stopped for {session_id}")
            
            # Cancel all tasks
            tasks_to_cancel = []
            
            if session_id in self.streaming_tasks:
                tasks_to_cancel.append(self.streaming_tasks[session_id])
                del self.streaming_tasks[session_id]
            
            if "audio_task" in session:
                tasks_to_cancel.append(session["audio_task"])
            
            if "transcription_task" in session:
                tasks_to_cancel.append(session["transcription_task"])
            
            # Cancel all tasks concurrently
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to complete
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=3.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ Some tasks didn't complete within timeout for {session_id}")
            
            # Final cleanup
            segments_count = len(session.get("transcription_segments", []))
            logger.info(f"📊 Session {session_id} processed {segments_count} segments")
            
            # Clean up session data
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            if session_id in self.speaker_trackers:
                del self.speaker_trackers[session_id]
            
            # Send final stop confirmation
            if websocket_callback:
                await websocket_callback({
                    "type": "processing_stopped",
                    "data": {
                        "session_id": session_id,
                        "segments_processed": segments_count,
                        "message": "Both UI player and backend processing stopped",
                        "timestamp": get_current_timestamp()
                    }
                })
            
            logger.info(f"✅ COMPLETE UI-sync stop for {session_id}")
            
            return {
                "session_id": session_id, 
                "status": "stopped", 
                "ui_player_stopped": True,
                "backend_stopped": True,
                "segments_processed": segments_count
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping UI-sync session {session_id}: {e}")
            
            # Force cleanup even on error
            try:
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["processing_stopped"] = True
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
        """Get status of all sessions with UI sync info"""
        sessions_status = {}
        for session_id, session in self.active_sessions.items():
            speaker_stats = {}
            if session_id in self.speaker_trackers:
                speaker_stats = self.speaker_trackers[session_id].get_stats()
            
            sessions_status[session_id] = {
                "status": session.get("status"),
                "transcription_segments": len(session.get("transcription_segments", [])),
                "restart_count": session.get("restart_count", 0),
                "ui_player_started": session.get("ui_player_started", False),
                "processing_stopped": session.get("processing_stopped", False),
                "speaker_stats": speaker_stats,
                "ui_sync_enabled": True
            }
        
        return {
            "active_sessions": len(self.active_sessions),
            "google_stt_available": self.google_client is not None,
            "processing_mode": "ui_synced_independent",
            "ui_sync_delay": self.ui_sync_delay,
            "processing_speed_factor": self.processing_speed_factor,
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
    
    async def force_stop_all_sessions(self) -> Dict[str, Any]:
        """Force stop all active sessions (useful for cleanup)"""
        stopped_sessions = []
        
        for session_id in list(self.active_sessions.keys()):
            try:
                result = await self.stop_streaming(session_id)
                if "error" not in result:
                    stopped_sessions.append(session_id)
            except Exception as e:
                logger.error(f"❌ Error force stopping {session_id}: {e}")
        
        return {
            "stopped_sessions": stopped_sessions,
            "total_stopped": len(stopped_sessions),
            "timestamp": get_current_timestamp()
        }

# Create global instance
audio_processor_agent = AudioProcessorAgent()
