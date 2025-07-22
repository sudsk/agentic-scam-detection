# backend/agents/audio_processor/agent.py - NATIVE GOOGLE STT STEREO VERSION
"""
Native Google STT Stereo Audio Processing Agent
- Uses Google STT's native multi-channel support
- Sends stereo audio directly to Google STT
- Perfect speaker separation using channel information
- No manual mixing or energy detection needed
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

from ...config.settings import get_settings
from ...utils import get_current_timestamp

logger = logging.getLogger(__name__)

class StereoAudioBuffer:
    """Audio buffer for native stereo processing"""
    
    def __init__(self, chunk_size: int = 6400):  # Stereo chunk size (2x mono)
        self.chunk_size = chunk_size
        self.buffer = queue.Queue(maxsize=150)
        self.is_active = False
        self.total_chunks_added = 0
        self.audio_completed = False
        
    def add_chunk(self, stereo_audio_data: bytes):
        """Add stereo audio chunk directly (no conversion)"""
        if self.is_active and len(stereo_audio_data) > 0:
            try:
                self.buffer.put(stereo_audio_data, block=False)
                self.total_chunks_added += 1
            except queue.Full:
                # Drop oldest chunk if buffer is full
                try:
                    self.buffer.get_nowait()
                    self.buffer.put(stereo_audio_data, block=False)
                    self.total_chunks_added += 1
                except queue.Empty:
                    pass
 
    def get_chunks(self):
        """Generator for stereo audio chunks"""
        empty_count = 0
        max_empty = 20  # Shorter timeout for better completion detection
        
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
        
        #logger.info(f"🎵 Stereo buffer generator completed after {empty_count} empty reads")
    
    def mark_completed(self):
        """Mark audio as completed"""
        self.audio_completed = True
    
    def start(self):
        self.is_active = True
        self.total_chunks_added = 0
        self.audio_completed = False
        
    def stop(self):
        self.is_active = False

class NativeStereoTracker:
    """FIXED: Simple tracker for native Google STT stereo results"""
    
    def __init__(self):
        self.segment_count = 0
        self.turn_count = {"agent": 0, "customer": 0}
        
        # FIXED: Updated channel mapping to handle different numbering schemes
        self.channel_mapping = {
            0: "customer",     # Channel 0 (LEFT) = Customer
            1: "agent",        # Channel 1 (RIGHT) = Agent
            # Handle 1-based indexing if Google uses it
            2: "customer",     # Channel 2 might be LEFT in 1-based system
            3: "agent"         # Channel 3 might be RIGHT in 1-based system
        }
        
        # Track speaker pattern for fallback
        self.last_detected_speaker = "agent"  # Agent always starts
        
        #logger.info(f"🎯 Native STT mapping: Channel 0/2 (LEFT)=Customer, Channel 1/3 (RIGHT)=Agent")
        
    def detect_speaker_from_channel(self, channel: Optional[int]) -> str:
        """FIXED: Detect speaker from Google STT channel information"""
        self.segment_count += 1
        
        # RULE 1: First segment is ALWAYS agent (banking convention)
        if self.segment_count == 1:
            detected_speaker = "agent"
            #logger.info(f"🎯 First segment: agent (channel:{channel})")
        
        # RULE 2: Use Google's channel information if available
        elif channel is not None:
            if channel in self.channel_mapping:
                detected_speaker = self.channel_mapping[channel]
                #logger.info(f"🎯 Channel {channel} mapped to: {detected_speaker}")
            else:
                # Unknown channel - use alternation logic
                detected_speaker = "customer" if self.last_detected_speaker == "agent" else "agent"
                #logger.warning(f"🎯 Unknown channel {channel}, alternating to: {detected_speaker}")
        
        # RULE 3: No channel info - use simple alternation
        else:
            # Simple alternation: agent -> customer -> agent -> customer
            detected_speaker = "customer" if self.last_detected_speaker == "agent" else "agent"
            #logger.info(f"🎯 No channel info, alternating: {self.last_detected_speaker} -> {detected_speaker}")
        
        # Update tracking
        self.turn_count[detected_speaker] += 1
        self.last_detected_speaker = detected_speaker
        
        return detected_speaker
        
    def get_stats(self) -> Dict:
        return {
            "total_segments": self.segment_count,
            "turn_count": self.turn_count.copy(),
            "channel_mapping": self.channel_mapping.copy()
        }

class AudioProcessorAgent:
    """Native Google STT Stereo Audio Processor"""
    
    def __init__(self):
        self.agent_type="audio_processor"
        self.agent_name="Native Google STT Stereo Processor (v1p1beta1)"
        self.config = self._get_default_config()

        self.active_sessions: Dict[str, Dict] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.audio_buffers: Dict[str, StereoAudioBuffer] = {}
        self.speaker_trackers: Dict[str, NativeStereoTracker] = {}
        
        # Audio settings for native stereo
        self.sample_rate = 16000
        self.channels = 2
        self.chunk_duration_ms = 200
        self.stereo_chunk_size = int(self.sample_rate * 2 * 2 * self.chunk_duration_ms / 1000)  # 2 channels, 2 bytes per sample
        
        # UI sync settings
        self.ui_sync_delay = 1.2
        self.processing_speed_factor = 1.05
        
        self._initialize_google_stt()
        
        #logger.info(f"🎵 {self.agent_name} initialized - NATIVE STEREO MODE")
        #logger.info(f"🎯 Stereo chunk size: {self.stereo_chunk_size} bytes")
    
    def _get_default_config(self) -> Dict[str, Any]:
        settings = get_settings()
        return {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio",
            "enable_interim_results": True,
            "max_streaming_duration": 300,
            "restart_timeout": 45
        }
    
    def _initialize_google_stt(self):
        """Initialize Google Speech-to-Text v1p1beta1"""
        try:
            from google.cloud import speech_v1p1beta1 as speech
            self.google_client = speech.SpeechClient()
            self.speech_types = speech
            #logger.info("✅ Google Speech-to-Text v1p1beta1 initialized for native stereo")
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
            "native_stereo_enabled": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _prepare_live_call(self, data: Dict) -> Dict[str, Any]:
        """Prepare for native stereo processing"""
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
            "processing_stopped": False,
            "audio_completed": False,
            "completion_sent": False
        }
        
        self.audio_buffers[session_id] = StereoAudioBuffer(self.stereo_chunk_size)
        self.speaker_trackers[session_id] = NativeStereoTracker()
        
        #logger.info(f"✅ Session {session_id} prepared for native stereo")
        
        return {
            "session_id": session_id,
            "status": "ready",
            "chunk_size": self.stereo_chunk_size,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "native_stereo_enabled": True
        }
    
    def _add_audio_chunk(self, data: Dict) -> Dict[str, Any]:
        """Add stereo audio chunk to buffer"""
        session_id = data.get('session_id')
        audio_data = data.get('audio_data')
        
        if not session_id or session_id not in self.active_sessions:
            return {"error": "Invalid session"}
        
        if not audio_data:
            return {"error": "No audio data"}
        
        self.audio_buffers[session_id].add_chunk(audio_data)
        return {"status": "stereo_chunk_added", "size": len(audio_data)}
    
    async def start_realtime_processing(self, session_id: str, audio_filename: str, websocket_callback: Callable) -> Dict[str, Any]:
        """Start native stereo processing with UI sync"""
        try:
            #logger.info(f"🎵 Starting NATIVE STEREO processing: {audio_filename}")
            
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
                
                #logger.info(f"📁 Audio: {duration:.1f}s, {sample_rate}Hz, {channels}ch")
            
            if channels != 2:
                return {"error": f"Only stereo audio supported. File has {channels} channels."}
            
            processing_mode = "native_google_stereo"
            #logger.info("🎯 NATIVE STEREO: Using Google STT multi-channel support")
            
            # Send UI player start signal
            await websocket_callback({
                "type": "ui_player_start",
                "data": {
                    "session_id": session_id,
                    "audio_filename": audio_filename,
                    "duration": duration,
                    "processing_mode": processing_mode,
                    "timestamp": get_current_timestamp()
                }
            })
            
            self.active_sessions[session_id]["ui_player_started"] = True
            
            # Start transcription with native stereo
            transcription_task = asyncio.create_task(
                self._start_native_stereo_transcription(session_id, websocket_callback)
            )
            self.active_sessions[session_id]["transcription_task"] = transcription_task
            
            # Start native stereo audio processing
            await asyncio.sleep(self.ui_sync_delay)
            
            audio_task = asyncio.create_task(
                self._process_native_stereo_audio(session_id, audio_path, websocket_callback)
            )
            self.active_sessions[session_id]["audio_task"] = audio_task
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "channels": channels,
                "processing_mode": processing_mode,
                "channel_mapping": "Channel 0 (LEFT)=Customer, Channel 1 (RIGHT)=Agent",
                "native_stereo": True,
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting native stereo processing: {e}")
            return {"error": str(e)}
    
    async def _process_native_stereo_audio(self, session_id: str, audio_path: Path, websocket_callback: Callable):
        """Process stereo audio natively (no conversion to mono)"""
        try:
            #logger.info(f"🎵 Starting NATIVE stereo audio processing for {session_id}")
            
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)
                audio_buffer = self.audio_buffers[session_id]
                
                audio_buffer.start()
                
                chunk_count = 0
                start_time = time.time()
                total_frames = wav_file.getnframes()
                total_chunks = total_frames // frames_per_chunk
                
                #logger.info(f"🎵 Processing {total_chunks} NATIVE stereo chunks")
                
                while chunk_count < total_chunks:
                    # Check if processing was stopped
                    if self.active_sessions[session_id].get("processing_stopped", False):
                        #logger.info(f"🛑 Native stereo processing stopped by user for {session_id}")
                        break
                    
                    # Read stereo frames DIRECTLY (no channel separation)
                    stereo_chunk = wav_file.readframes(frames_per_chunk)
                    if not stereo_chunk:
                        #logger.info(f"📁 Reached end of stereo audio file at chunk {chunk_count}")
                        break
                    
                    # Send stereo chunk directly to buffer (NO CONVERSION)
                    audio_buffer.add_chunk(stereo_chunk)
                    chunk_count += 1
                    
                    # UI-synced timing
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / (1000 * self.processing_speed_factor))
                    current_time = time.time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    
                    # Progress logging
                    if chunk_count % 100 == 0:
                        progress = (chunk_count / total_chunks) * 100
                        elapsed = chunk_count * self.chunk_duration_ms / 1000
                        #logger.info(f"🎵 Native stereo progress: {progress:.1f}% ({elapsed:.1f}s) - {chunk_count}/{total_chunks}")
                        
                        await websocket_callback({
                            "type": "processing_progress",
                            "data": {
                                "session_id": session_id,
                                "progress": progress,
                                "chunks_processed": chunk_count,
                                "total_chunks": total_chunks,
                                "processing_mode": "native_stereo",
                                "timestamp": get_current_timestamp()
                            }
                        })
                        
            # Mark audio as completed
            self.active_sessions[session_id]["audio_completed"] = True
            audio_buffer.mark_completed()
            
            #logger.info(f"✅ Native stereo audio processing complete: {chunk_count}/{total_chunks} chunks")
            
            # Wait for transcription to finish
            await asyncio.sleep(4.0)
            audio_buffer.stop()

            # Send completion only once
            if not self.active_sessions[session_id].get("completion_sent", False):
                self.active_sessions[session_id]["completion_sent"] = True

                await websocket_callback({
                    "type": "processing_complete",
                    "data": {
                        "session_id": session_id,
                        "total_chunks": chunk_count,
                        "processing_mode": "native_stereo",
                        "timestamp": get_current_timestamp()
                    }
                })
                #logger.info(f"✅ AUDIO PROCESSOR: Sent processing_complete for {session_id}")
                
        except Exception as e:
            logger.error(f"❌ Error in native stereo audio processing: {e}")
            raise
    
    async def _start_native_stereo_transcription(self, session_id: str, websocket_callback: Callable):
        """Start transcription with native stereo support"""
        try:
            #logger.info(f"🎙️ Starting NATIVE STEREO transcription for {session_id}")
            
            streaming_result = await self.start_streaming(session_id, websocket_callback)
            if "error" in streaming_result:
                logger.error(f"❌ Failed to start native stereo transcription: {streaming_result['error']}")
                await websocket_callback({
                    "type": "transcription_error", 
                    "data": {
                        "session_id": session_id,
                        "error": streaming_result['error'],
                        "timestamp": get_current_timestamp()
                    }
                })
            
        except Exception as e:
            logger.error(f"❌ Error in native stereo transcription: {e}")
    
    async def start_streaming(self, session_id: str, websocket_callback: Callable) -> Dict[str, Any]:
        """Start streaming recognition with native stereo support"""
        try:
            #logger.info(f"🎙️ Starting NATIVE STEREO streaming recognition for {session_id}")
            
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
                self._streaming_with_restart_native(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            await websocket_callback({
                "type": "streaming_started",
                "data": {
                    "session_id": session_id,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "model": "phone_call",
                    "processing_mode": "native_stereo",
                    "channel_mapping": "Channel 0=Customer, Channel 1=Agent",
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {"session_id": session_id, "status": "streaming"}
            
        except Exception as e:
            logger.error(f"❌ Error starting native stereo streaming: {e}")
            return {"error": str(e)}
    
    async def _streaming_with_restart_native(self, session_id: str):
        """Streaming with restart for native stereo"""
        session = self.active_sessions[session_id]
        restart_interval = 45
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            # Check if processing was stopped or audio completed
            if session.get("processing_stopped", False):
                #logger.info(f"🛑 Native stereo streaming stopped by user for {session_id}")
                break
                
            if session.get("audio_completed", False):
                #logger.info(f"🏁 Audio completed, ending streaming for {session_id}")

                # Send completion only once
                if not session.get("completion_sent", False):
                    session["completion_sent"] = True
                    websocket_callback = session.get("websocket_callback")
                    if websocket_callback:
                        try:
                            await websocket_callback({
                                "type": "processing_complete",
                                "data": {
                                    "session_id": session_id,
                                    "timestamp": get_current_timestamp()
                                }
                            })
                            #logger.info(f"✅ Sent processing_complete from streaming restart for {session_id}")
                        except Exception as e:
                            logger.error(f"❌ Failed to send processing_complete from restart: {e}")
                else:
                    logger.info(f"ℹ️ Completion already sent for {session_id}, skipping duplicate")
        
                # Give it a bit more time to process remaining chunks
                await asyncio.sleep(3.0)
                break
                
            try:
                await asyncio.wait_for(
                    self._do_native_stereo_recognition(session_id),
                    timeout=restart_interval
                )
            except asyncio.TimeoutError:
                session["restart_count"] += 1
                #logger.info(f"🔄 Native stereo restart #{session['restart_count']} for {session_id}")
                
                if session["restart_count"] > 5:
                    logger.warning(f"Too many restarts for {session_id}")
                    break
                    
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Native stereo recognition error: {e}")
                session["restart_count"] += 1
                if session["restart_count"] > 3:
                    break
                await asyncio.sleep(1.0)
    
    async def _do_native_stereo_recognition(self, session_id: str):
        """ENHANCED: Core recognition with NATIVE GOOGLE STT STEREO support"""
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        speaker_tracker = self.speaker_trackers[session_id]
        
        #logger.info(f"🎙️ Starting NATIVE STEREO Google STT for {session_id}")
        
        # ENHANCED: More explicit stereo configuration
        recognition_config = self.speech_types.RecognitionConfig(
            encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code="en-GB",
            
            # 🎯 ENHANCED STEREO CONFIGURATION
            audio_channel_count=2,  # Tell Google it's stereo
            enable_separate_recognition_per_channel=True,  # Process channels separately!
            
            model="phone_call",
            use_enhanced=True,
            enable_automatic_punctuation=True,
            enable_word_confidence=True,
            enable_word_time_offsets=True,
            
            # ENHANCED: More explicit diarization as backup
            diarization_config=self.speech_types.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,
                max_speaker_count=2
            ),
            
            # ENHANCED: Better speech contexts for banking
            speech_contexts=[
                self.speech_types.SpeechContext(
                    phrases=[
                        # Banking/Agent phrases (Channel 1 - RIGHT)
                        "HSBC", "customer services", "good afternoon", "good morning",
                        "how can I assist", "can I take your", "for security purposes",
                        "Mrs Williams", "Mr", "Mrs", "account details",
                        
                        # Customer phrases (Channel 0 - LEFT)  
                        "hi", "hello", "urgent", "emergency", "I need", "please help",
                        "Patricia Williams", "Alex", "Turkey", "Istanbul", "transfer",
                        "boyfriend", "girlfriend", "stuck", "hospital", "money"
                    ],
                    boost=20  # Higher boost for better recognition
                )
            ]
        )
        
        streaming_config = self.speech_types.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
            single_utterance=False
        )
        
        def stereo_audio_generator():
            """Generator that sends STEREO audio directly to Google STT"""
            chunk_count = 0
            for stereo_chunk in audio_buffer.get_chunks():
                if session_id not in self.active_sessions or session.get("processing_stopped", False):
                    break
                chunk_count += 1
                
                # Send STEREO chunk directly (no conversion!)
                request = self.speech_types.StreamingRecognizeRequest()
                request.audio_content = stereo_chunk
                yield request
                
            #logger.info(f"🎵 Native stereo generator sent {chunk_count} stereo chunks")
        
        try:
            #logger.info(f"🚀 Starting Google STT with ENHANCED NATIVE STEREO for {session_id}")
            
            responses = self.google_client.streaming_recognize(
                config=streaming_config,
                requests=stereo_audio_generator()
            )
            
            response_count = 0
            last_final_text = ""
            
            for response in responses:
                if session_id not in self.active_sessions or session.get("processing_stopped", False):
                    break
                
                response_count += 1
                
                # DEBUG: Log first few responses to understand structure
                if response_count <= 3:
                    for result in response.results:
                        if result.alternatives:
                            transcript = result.alternatives[0].transcript.strip()
                            #if transcript:
                            #    self._debug_google_response(response, transcript)
                
                processed = await self._process_native_stereo_response(
                    session_id, response, speaker_tracker, websocket_callback, last_final_text
                )
                
                if processed and processed.get("is_final"):
                    last_final_text = processed.get("text", "")
            
            #logger.info(f"✅ Native stereo streaming completed: {response_count} responses")
                
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.warning(f"⚠️ Native stereo timeout: {e}")
            else:
                logger.error(f"❌ Native stereo streaming error: {e}")
            raise
    
    async def _process_native_stereo_response(
        self, 
        session_id: str, 
        response, 
        speaker_tracker: NativeStereoTracker, 
        websocket_callback: Callable,
        last_final_text: str
    ) -> Optional[Dict]:
        """FIXED: Process native stereo response with better channel detection"""
        try:
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                
                if not transcript:
                    continue
                
                # Simple deduplication
                if result.is_final and transcript == last_final_text:
                    continue
                
                # 🎯 IMPROVED: Extract channel information from multiple sources
                channel_tag = None
                
                # Method 1: Direct channel_tag attribute
                if hasattr(result, 'channel_tag') and result.channel_tag is not None:
                    channel_tag = result.channel_tag
                    #logger.info(f"🔍 Found result.channel_tag: {channel_tag}")
                
                # Method 2: Extract from words (alternative approach)
                elif hasattr(alternative, 'words') and alternative.words:
                    word_channels = []
                    for word in alternative.words:
                        if hasattr(word, 'channel_tag') and word.channel_tag is not None:
                            word_channels.append(word.channel_tag)
                    
                    if word_channels:
                        # Use most common channel from words
                        channel_tag = max(set(word_channels), key=word_channels.count)
                        #logger.info(f"🔍 Extracted from words, channel: {channel_tag}")
                
                # Method 3: Check if result has channel attribute
                elif hasattr(result, 'channel') and result.channel is not None:
                    channel_tag = result.channel
                    #logger.info(f"🔍 Found result.channel: {channel_tag}")
                
                # Log what we found
                #if channel_tag is not None:
                #    logger.info(f"🎯 Channel detected: {channel_tag} for text: '{transcript[:30]}...'")
                #else:
                #    logger.warning(f"🎯 No channel info for text: '{transcript[:30]}...'")
                
                # Use improved speaker detection
                detected_speaker = speaker_tracker.detect_speaker_from_channel(channel_tag)
                
                # Extract timing
                start_time, end_time = self._extract_timing(alternative)
                
                # Create segment data
                segment_data = {
                    "session_id": session_id,
                    "speaker": detected_speaker,
                    "text": transcript,
                    "confidence": getattr(alternative, 'confidence', 0.9),
                    "is_final": result.is_final,
                    "channel_tag": channel_tag,
                    "native_stereo": True,
                    "start": start_time,
                    "end": end_time,
                    "timestamp": get_current_timestamp()
                }
                
                if result.is_final:
                    session = self.active_sessions[session_id]
                    session["transcription_segments"].append(segment_data)
                    
                    await websocket_callback({
                        "type": "transcription_segment",
                        "data": segment_data
                    })
                    
                    # IMPROVED: Better logging with channel info
                    channel_display = f"Ch{channel_tag}" if channel_tag is not None else "ChNone"
                    #logger.info(f"📝 NATIVE [{detected_speaker.upper()}] {channel_display}: '{transcript[:50]}...'")
                    
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
            logger.error(f"❌ Error processing native stereo response: {e}")
        
        return None

    def _debug_google_response(self, response, transcript: str):
        """Debug helper to understand Google STT response structure"""
        try:
            for result in response.results:
                #logger.info(f"🔍 DEBUG Response for '{transcript[:20]}...':")
                #logger.info(f"  - result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                
                if hasattr(result, 'channel_tag'):
                    logger.info(f"  - result.channel_tag: {result.channel_tag}")
                if hasattr(result, 'channel'):
                    logger.info(f"  - result.channel: {result.channel}")
                    
                for alt in result.alternatives:
                    logger.info(f"  - alternative attributes: {[attr for attr in dir(alt) if not attr.startswith('_')]}")
                    
                    if hasattr(alt, 'words') and alt.words:
                        for i, word in enumerate(alt.words[:3]):  # First 3 words
                            word_attrs = [attr for attr in dir(word) if not attr.startswith('_')]
                            logger.info(f"    - word[{i}] attributes: {word_attrs}")
                            if hasattr(word, 'channel_tag'):
                                logger.info(f"    - word[{i}].channel_tag: {word.channel_tag}")
                            if hasattr(word, 'channel'):
                                logger.info(f"    - word[{i}].channel: {word.channel}")
                                
        except Exception as e:
            logger.error(f"❌ Debug error: {e}")
        
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
        """Stop native stereo processing"""
        try:
            #logger.info(f"🛑 STOP native stereo session {session_id}")
            
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            session["status"] = "stopping"
            session["processing_stopped"] = True
            
            # Send stop signal to UI
            websocket_callback = session.get("websocket_callback")
            if websocket_callback:
                await websocket_callback({
                    "type": "ui_player_stop",
                    "data": {
                        "session_id": session_id,
                        "timestamp": get_current_timestamp()
                    }
                })
            
            # Stop buffer
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].stop()
            
            # Cancel tasks
            tasks_to_cancel = []
            if session_id in self.streaming_tasks:
                tasks_to_cancel.append(self.streaming_tasks[session_id])
                del self.streaming_tasks[session_id]
            if "audio_task" in session:
                tasks_to_cancel.append(session["audio_task"])
            if "transcription_task" in session:
                tasks_to_cancel.append(session["transcription_task"])
            
            # Cancel all tasks
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ Some tasks didn't complete for {session_id}")
            
            # Cleanup
            segments_count = len(session.get("transcription_segments", []))
            
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            if session_id in self.speaker_trackers:
                del self.speaker_trackers[session_id]
            
            # Final confirmation
            if websocket_callback:
                await websocket_callback({
                    "type": "processing_stopped",
                    "data": {
                        "session_id": session_id,
                        "segments_processed": segments_count,
                        "processing_mode": "native_stereo",
                        "timestamp": get_current_timestamp()
                    }
                })
            
            #logger.info(f"✅ Native stereo session {session_id} stopped completely")
            
            return {
                "session_id": session_id, 
                "status": "stopped", 
                "processing_mode": "native_stereo",
                "segments_processed": segments_count
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping native stereo session {session_id}: {e}")
            
            # Force cleanup
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
        """Get status of all sessions with native stereo info"""
        sessions_status = {}
        for session_id, session in self.active_sessions.items():
            speaker_stats = {}
            if session_id in self.speaker_trackers:
                speaker_stats = self.speaker_trackers[session_id].get_stats()
            
            sessions_status[session_id] = {
                "status": session.get("status"),
                "transcription_segments": len(session.get("transcription_segments", [])),
                "restart_count": session.get("restart_count", 0),
                "processing_stopped": session.get("processing_stopped", False),
                "audio_completed": session.get("audio_completed", False),
                "speaker_stats": speaker_stats,
                "processing_mode": "native_stereo"
            }
        
        return {
            "active_sessions": len(self.active_sessions),
            "google_stt_available": self.google_client is not None,
            "processing_mode": "native_google_stereo",
            "native_stereo_enabled": True,
            "channel_mapping": "Channel 0 (LEFT)=Customer, Channel 1 (RIGHT)=Agent",
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
