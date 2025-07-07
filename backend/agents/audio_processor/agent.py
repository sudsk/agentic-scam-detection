# backend/agents/audio_processor/agent.py - CLEAN & SIMPLE VERSION
"""
Clean & Simple Native Google STT Stereo Audio Processor
- Removes all unnecessary complexity
- Pure Google STT native stereo support
- Simple segment aggregation
- Clean session management
"""

import asyncio
import logging
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import queue
import numpy as np

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings
from ...utils import get_current_timestamp

logger = logging.getLogger(__name__)

class AudioBuffer:
    """Simple stereo audio buffer"""
    
    def __init__(self):
        self.buffer = queue.Queue(maxsize=100)
        self.is_active = False
        
    def add_chunk(self, stereo_data: bytes):
        if self.is_active and stereo_data:
            try:
                self.buffer.put(stereo_data, block=False)
            except queue.Full:
                try:
                    self.buffer.get_nowait()
                    self.buffer.put(stereo_data, block=False)
                except queue.Empty:
                    pass
 
    def get_chunks(self):
        empty_count = 0
        while self.is_active and empty_count < 15:
            try:
                chunk = self.buffer.get(timeout=0.1)
                if chunk:
                    empty_count = 0
                    yield chunk
                else:
                    empty_count += 1
            except queue.Empty:
                empty_count += 1
    
    def start(self):
        self.is_active = True
        
    def stop(self):
        self.is_active = False

class SegmentAggregator:
    """Combines fragments into complete sentences"""
    
    def __init__(self):
        self.current_speaker = None
        self.accumulated_text = []
        self.last_time = 0
        
    def add_segment(self, speaker: str, text: str, is_final: bool) -> Optional[str]:
        current_time = time.time()
        
        # If speaker changed or too much time passed, flush
        if (self.current_speaker and speaker != self.current_speaker) or \
           (current_time - self.last_time > 2.5):
            result = self._flush()
            self.current_speaker = speaker
            self.accumulated_text = [text]
            self.last_time = current_time
            return result
        else:
            # Same speaker, accumulate
            self.current_speaker = speaker
            self.accumulated_text.append(text)
            self.last_time = current_time
            
            if is_final and self.accumulated_text:
                return self._flush()
                
        return None
    
    def _flush(self) -> Optional[str]:
        if not self.accumulated_text or not self.current_speaker:
            return None
            
        # Join and clean up text
        result = " ".join(self.accumulated_text).strip()
        result = " ".join(result.split())  # Remove extra spaces
        
        self.accumulated_text = []
        return result if result else None

class AudioProcessorAgent(BaseAgent):
    """Clean & Simple Native Stereo Audio Processor"""
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Clean Native Stereo Processor"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.audio_buffers: Dict[str, AudioBuffer] = {}
        self.aggregators: Dict[str, SegmentAggregator] = {}
        
        # Simple settings
        self.sample_rate = 16000
        self.chunk_duration_ms = 200
        
        # Channel mapping (corrected)
        self.channel_mapping = {
            0: "customer",  # LEFT = Customer
            1: "agent"      # RIGHT = Agent  
        }
        
        self._initialize_google_stt()
        agent_registry.register_agent(self)
        
        logger.info("🎵 Clean Native Stereo Processor initialized")
        logger.info("🎯 Channel mapping: LEFT=Customer, RIGHT=Agent")
    
    def _get_default_config(self) -> Dict[str, Any]:
        settings = get_settings()
        return {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio"
        }
    
    def _get_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_google_stt(self):
        try:
            from google.cloud import speech_v1p1beta1 as speech
            self.google_client = speech.SpeechClient()
            self.speech_types = speech
            logger.info("✅ Google STT initialized")
        except Exception as e:
            logger.error(f"❌ Google STT failed: {e}")
            self.google_client = None
            self.speech_types = None
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        if isinstance(input_data, dict):
            action = input_data.get('action', 'status')
            if action == 'stop_live_call':
                session_id = input_data.get('session_id')
                if session_id:
                    asyncio.create_task(self.stop_streaming(session_id))
        
        return {
            "agent_type": self.agent_type,
            "status": "ready",
            "active_sessions": len(self.active_sessions)
        }
    
    async def start_realtime_processing(self, session_id: str, audio_filename: str, websocket_callback: Callable) -> Dict[str, Any]:
        try:
            logger.info(f"🎵 Starting clean processing: {audio_filename}")
            
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            if not audio_path.exists():
                return {"error": f"Audio file not found: {audio_filename}"}
            
            # Check audio properties
            with wave.open(str(audio_path), 'rb') as wav_file:
                channels = wav_file.getnchannels()
                duration = wav_file.getnframes() / wav_file.getframerate()
                
            if channels != 2:
                return {"error": "Only stereo audio supported"}
            
            # Setup session
            self.active_sessions[session_id] = {
                "start_time": datetime.now(),
                "status": "ready",
                "segments": [],
                "stopped": False
            }
            
            self.audio_buffers[session_id] = AudioBuffer()
            self.aggregators[session_id] = SegmentAggregator()
            
            # Send UI start signal
            await websocket_callback({
                "type": "ui_player_start",
                "data": {
                    "session_id": session_id,
                    "audio_filename": audio_filename,
                    "duration": duration
                }
            })
            
            # Start processing
            await asyncio.sleep(1.0)  # UI sync delay
            
            transcription_task = asyncio.create_task(self._start_transcription(session_id, websocket_callback))
            audio_task = asyncio.create_task(self._process_audio(session_id, audio_path))
            
            self.active_sessions[session_id]["tasks"] = [transcription_task, audio_task]
            
            return {
                "session_id": session_id,
                "status": "started",
                "duration": duration,
                "channel_mapping": "LEFT=Customer, RIGHT=Agent"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting: {e}")
            return {"error": str(e)}
    
    async def _process_audio(self, session_id: str, audio_path: Path):
        """Simple audio processing"""
        try:
            with wave.open(str(audio_path), 'rb') as wav_file:
                buffer = self.audio_buffers[session_id]
                buffer.start()
                
                frames_per_chunk = int(self.sample_rate * self.chunk_duration_ms / 1000)
                start_time = time.time()
                chunk_count = 0
                
                while True:
                    if self.active_sessions[session_id].get("stopped"):
                        break
                        
                    stereo_chunk = wav_file.readframes(frames_per_chunk)
                    if not stereo_chunk:
                        break
                    
                    buffer.add_chunk(stereo_chunk)
                    chunk_count += 1
                    
                    # Maintain timing
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    wait_time = expected_time - time.time()
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        
            # Mark as complete
            self.active_sessions[session_id]["audio_complete"] = True
            await asyncio.sleep(3.0)
            buffer.stop()
            
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
    
    async def _start_transcription(self, session_id: str, websocket_callback: Callable):
        """Simple transcription"""
        try:
            if not self.google_client:
                raise Exception("Google STT not available")
            
            buffer = self.audio_buffers[session_id]
            aggregator = self.aggregators[session_id]
            
            # Simple Google STT config
            config = self.speech_types.RecognitionConfig(
                encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="en-GB",
                audio_channel_count=2,
                enable_separate_recognition_per_channel=True,
                model="phone_call",
                use_enhanced=True,
                enable_automatic_punctuation=True
            )
            
            streaming_config = self.speech_types.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False
            )
            
            def audio_generator():
                for chunk in buffer.get_chunks():
                    if self.active_sessions[session_id].get("stopped"):
                        break
                    request = self.speech_types.StreamingRecognizeRequest()
                    request.audio_content = chunk
                    yield request
            
            # Process responses
            responses = self.google_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )
            
            for response in responses:
                if self.active_sessions[session_id].get("stopped"):
                    break
                    
                await self._process_response(session_id, response, aggregator, websocket_callback)
                
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
    
    async def _process_response(self, session_id: str, response, aggregator: SegmentAggregator, websocket_callback: Callable):
        """Simple response processing"""
        try:
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                if not transcript:
                    continue
                
                # Get channel and map to speaker
                channel = getattr(result, 'channel_tag', None)
                speaker = self.channel_mapping.get(channel, "agent")
                
                # Aggregate segments
                aggregated_text = aggregator.add_segment(speaker, transcript, result.is_final)
                
                if aggregated_text:
                    # Send complete segment
                    segment_data = {
                        "session_id": session_id,
                        "speaker": aggregator.current_speaker,
                        "text": aggregated_text,
                        "is_final": True,
                        "channel": channel,
                        "timestamp": get_current_timestamp()
                    }
                    
                    self.active_sessions[session_id]["segments"].append(segment_data)
                    
                    await websocket_callback({
                        "type": "transcription_segment",
                        "data": segment_data
                    })
                    
                    logger.info(f"📝 [{aggregator.current_speaker.upper()}]: {aggregated_text[:60]}...")
                    
                    if aggregator.current_speaker == "customer":
                        await websocket_callback({
                            "type": "progressive_fraud_analysis_trigger",
                            "data": {
                                "session_id": session_id,
                                "customer_text": aggregated_text,
                                "timestamp": get_current_timestamp()
                            }
                        })
                
        except Exception as e:
            logger.error(f"❌ Response processing error: {e}")
    
    async def stop_streaming(self, session_id: str) -> Dict[str, Any]:
        """Simple stop"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            session["stopped"] = True
            
            # Send UI stop
            if "websocket_callback" in session:
                await session["websocket_callback"]({
                    "type": "ui_player_stop",
                    "data": {"session_id": session_id}
                })
            
            # Stop buffer
            if session_id in self.audio_buffers:
                self.audio_buffers[session_id].stop()
            
            # Cancel tasks
            for task in session.get("tasks", []):
                if not task.done():
                    task.cancel()
            
            # Cleanup
            segments_count = len(session.get("segments", []))
            
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            if session_id in self.aggregators:
                del self.aggregators[session_id]
            
            return {
                "session_id": session_id,
                "status": "stopped",
                "segments_processed": segments_count
            }
            
        except Exception as e:
            logger.error(f"❌ Stop error: {e}")
            return {"error": str(e)}
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        return {
            "active_sessions": len(self.active_sessions),
            "google_stt_available": self.google_client is not None,
            "processing_mode": "clean_native_stereo",
            "timestamp": get_current_timestamp()
        }
    
    def get_session_transcription(self, session_id: str) -> Optional[List[Dict]]:
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].get("segments", [])
        return None
    
    def get_customer_speech_history(self, session_id: str) -> Optional[List[str]]:
        if session_id in self.active_sessions:
            segments = self.active_sessions[session_id].get("segments", [])
            return [seg["text"] for seg in segments if seg.get("speaker") == "customer"]
        return None

# Create global instance
audio_processor_agent = AudioProcessorAgent()
