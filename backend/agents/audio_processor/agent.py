# backend/agents/audio_processor/agent.py - REAL-TIME CHUNK PROCESSING
"""
Audio Processing Agent for Large Files (16MB+) with Real-time Streaming
Processes audio in chunks as it "plays" for real-time fraud detection
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
import json
import math

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings
from ...utils import get_current_timestamp

logger = logging.getLogger(__name__)

class AudioProcessorAgent(BaseAgent):
    """
    Real-time Audio Chunk Processor for Large Files
    Simulates real-time audio playback while streaming chunks to Google STT
    """
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Real-time Audio Chunk Processor"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Real-time processing settings
        self.chunk_size_bytes = 8 * 1024 * 1024  # 8MB chunks (under 10MB limit)
        self.real_time_chunk_duration = 2.0  # Send chunks every 2 seconds
        self.sample_rate = 16000
        self.bytes_per_second = self.sample_rate * 2  # 16-bit audio = 2 bytes per sample
        
        # Initialize Google STT
        self._initialize_google_stt()
              
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"🎵 {self.agent_name} ready for large file processing")
        logger.info(f"📊 Chunk size: {self.chunk_size_bytes / (1024*1024):.1f}MB")
        logger.info(f"⏱️  Real-time chunk duration: {self.real_time_chunk_duration}s")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        settings = get_settings()
        
        base_config = {
            **settings.get_agent_config("audio_processor"),
            "audio_base_path": "data/sample_audio",
            "processing_delay_ms": 100,  # Faster for real-time
            "enable_interim_results": True,  # Get partial results
        }
        
        return base_config
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get capabilities"""
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_google_stt(self):
        """Initialize Google Speech-to-Text streaming"""
        
        try:
            logger.info("🔄 Initializing Google STT for streaming...")
            from google.cloud import speech_v1p1beta1 as speech
            
            self.google_client = speech.SpeechClient()
            self.speech_module = speech
            self.transcription_source = "google_stt_streaming_chunks"
            
            logger.info("✅ Google STT streaming client ready!")
                        
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
            file_path = input_data.get('file_path', input_data.get('audio_file_path', ''))
        else:
            file_path = str(input_data)
        
        try:
            if not self.google_client:
                return {
                    "error": "Google Speech-to-Text not available",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "agent_type": self.agent_type,
                "status": "ready_for_realtime_chunking",
                "file_path": file_path,
                "chunk_processing": True,
                "real_time_simulation": True,
                "transcription_source": self.transcription_source,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.handle_error(e, "process")
            return {
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def start_realtime_processing(
        self, 
        session_id: str, 
        audio_filename: str,
        websocket_callback: Callable[[Dict], None]
    ) -> Dict[str, Any]:
        """Start real-time chunk processing for large audio files"""
        
        try:
            logger.info(f"🎵 STARTING REAL-TIME CHUNK PROCESSING")
            logger.info(f"🎯 Session: {session_id}, File: {audio_filename}")
            
            # Validate audio file
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if session_id in self.active_sessions:
                raise ValueError(f"Session {session_id} already active")
            
            if not self.google_client:
                raise Exception("Google Speech-to-Text not available")
            
            # Get audio file info
            audio_info = await self._get_audio_info(audio_path)
            file_size_mb = audio_info["size_bytes"] / (1024 * 1024)
            
            logger.info(f"📁 File size: {file_size_mb:.1f}MB")
            logger.info(f"⏱️  Estimated duration: {audio_info['duration']:.1f}s")
            
            # Calculate chunking strategy
            chunk_strategy = self._calculate_chunk_strategy(audio_info)
            logger.info(f"📊 Chunk strategy: {chunk_strategy}")
            
            # Initialize session
            self.active_sessions[session_id] = {
                "filename": audio_filename,
                "audio_path": str(audio_path),
                "audio_info": audio_info,
                "chunk_strategy": chunk_strategy,
                "websocket_callback": websocket_callback,
                "start_time": datetime.now(),
                "status": "active",
                "current_position": 0.0,
                "transcribed_segments": [],
                "transcription_source": self.transcription_source,
                "chunks_processed": 0
            }
            
            # Start real-time chunk processing task
            processing_task = asyncio.create_task(
                self._process_realtime_chunks(session_id)
            )
            self.processing_tasks[session_id] = processing_task
            
            # Send initial confirmation
            await websocket_callback({
                "type": "realtime_chunk_processing_started",
                "data": {
                    "session_id": session_id,
                    "filename": audio_filename,
                    "audio_info": audio_info,
                    "chunk_strategy": chunk_strategy,
                    "processing_mode": "realtime_chunks",
                    "transcription_engine": self.transcription_source,
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {
                "session_id": session_id,
                "status": "started",
                "processing_mode": "realtime_chunks",
                "file_size_mb": file_size_mb,
                "expected_chunks": chunk_strategy["total_chunks"],
                "chunk_duration": chunk_strategy["chunk_duration_seconds"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting real-time processing: {e}")
            return {"error": str(e)}
    
    def _calculate_chunk_strategy(self, audio_info: Dict) -> Dict[str, Any]:
        """Calculate optimal chunking strategy for real-time processing"""
        
        file_size_bytes = audio_info["size_bytes"]
        duration_seconds = audio_info["duration"]
        
        # Calculate how many chunks we need to stay under 8MB per chunk
        min_chunks_for_size = math.ceil(file_size_bytes / self.chunk_size_bytes)
        
        # Calculate how many chunks for real-time processing (every 2 seconds)
        chunks_for_realtime = math.ceil(duration_seconds / self.real_time_chunk_duration)
        
        # Use the larger number to ensure both constraints are met
        total_chunks = max(min_chunks_for_size, chunks_for_realtime, 1)
        
        # Calculate actual chunk size and timing
        chunk_size_bytes = math.ceil(file_size_bytes / total_chunks)
        chunk_duration_seconds = duration_seconds / total_chunks
        
        return {
            "total_chunks": total_chunks,
            "chunk_size_bytes": chunk_size_bytes,
            "chunk_size_mb": chunk_size_bytes / (1024 * 1024),
            "chunk_duration_seconds": chunk_duration_seconds,
            "file_size_mb": file_size_bytes / (1024 * 1024),
            "total_duration": duration_seconds,
            "real_time_simulation": True
        }
    
    async def _process_realtime_chunks(self, session_id: str) -> None:
        """Process audio file in real-time chunks using Google STT streaming"""
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_path = Path(session["audio_path"])
        chunk_strategy = session["chunk_strategy"]
        
        try:
            logger.info(f"🔄 Starting real-time chunk processing for {session_id}")
            
            # Read the entire audio file
            with open(audio_path, 'rb') as audio_file:
                audio_content = audio_file.read()
            
            logger.info(f"📖 Read {len(audio_content)} bytes from {audio_path.name}")
            
            # Create audio chunks
            chunks = self._create_audio_chunks(audio_content, chunk_strategy)
            logger.info(f"📦 Created {len(chunks)} chunks")
            
            # Process chunks in real-time simulation
            start_time = time.time()
            current_audio_time = 0.0
            
            for chunk_index, (chunk_data, chunk_start_time, chunk_duration) in enumerate(chunks):
                if session_id not in self.active_sessions:
                    break
                
                # Wait for real-time timing (simulate audio playback)
                target_time = start_time + chunk_start_time
                current_time = time.time()
                
                if current_time < target_time:
                    wait_time = target_time - current_time
                    logger.debug(f"⏳ Waiting {wait_time:.2f}s for chunk {chunk_index}")
                    await asyncio.sleep(wait_time)
                
                # Process chunk with Google STT streaming
                logger.info(f"🎯 Processing chunk {chunk_index + 1}/{len(chunks)} at {chunk_start_time:.1f}s")
                
                # Send chunk processing notification
                await websocket_callback({
                    "type": "chunk_processing_started",
                    "data": {
                        "session_id": session_id,
                        "chunk_index": chunk_index,
                        "chunk_start_time": chunk_start_time,
                        "chunk_duration": chunk_duration,
                        "chunk_size_mb": len(chunk_data) / (1024 * 1024),
                        "timestamp": get_current_timestamp()
                    }
                })
                
                # Process chunk through Google STT
                try:
                    chunk_results = await self._process_chunk_with_stt(
                        chunk_data, chunk_index, chunk_start_time, chunk_duration
                    )
                    
                    # Send each transcription segment
                    for segment in chunk_results:
                        segment_data = {
                            "session_id": session_id,
                            "speaker": segment["speaker"],
                            "start": chunk_start_time + segment["relative_start"],
                            "end": chunk_start_time + segment["relative_end"],
                            "duration": segment["duration"],
                            "text": segment["text"],
                            "confidence": segment["confidence"],
                            "chunk_index": chunk_index,
                            "processing_mode": "realtime_chunks",
                            "transcription_source": self.transcription_source,
                            "timestamp": get_current_timestamp()
                        }
                        
                        session["transcribed_segments"].append(segment_data)
                        
                        await websocket_callback({
                            "type": "transcription_segment",
                            "data": segment_data
                        })
                        
                        logger.info(f"📤 Sent segment: {segment['speaker']} - '{segment['text'][:50]}...'")
                
                except Exception as chunk_error:
                    logger.error(f"❌ Error processing chunk {chunk_index}: {chunk_error}")
                    await websocket_callback({
                        "type": "chunk_error",
                        "data": {
                            "session_id": session_id,
                            "chunk_index": chunk_index,
                            "error": str(chunk_error),
                            "timestamp": get_current_timestamp()
                        }
                    })
                
                session["chunks_processed"] += 1
                current_audio_time = chunk_start_time + chunk_duration
                
                # Small delay between chunks for processing
                await asyncio.sleep(0.1)
            
            # Send completion
            await websocket_callback({
                "type": "realtime_processing_complete",
                "data": {
                    "session_id": session_id,
                    "total_chunks_processed": session["chunks_processed"],
                    "total_segments": len(session["transcribed_segments"]),
                    "total_duration": current_audio_time,
                    "processing_mode": "realtime_chunks",
                    "timestamp": get_current_timestamp()
                }
            })
            
            logger.info(f"✅ Real-time chunk processing completed for {session_id}")
            
        except asyncio.CancelledError:
            logger.info(f"🛑 Real-time processing cancelled for {session_id}")
        except Exception as e:
            logger.error(f"❌ Error in real-time chunk processing: {e}")
            await websocket_callback({
                "type": "error",
                "data": {
                    "session_id": session_id,
                    "error": str(e),
                    "processing_mode": "realtime_chunks",
                    "timestamp": get_current_timestamp()
                }
            })
    
    def _create_audio_chunks(self, audio_content: bytes, chunk_strategy: Dict) -> List[tuple]:
        """Create audio chunks with timing information"""
        
        total_chunks = chunk_strategy["total_chunks"]
        chunk_size_bytes = chunk_strategy["chunk_size_bytes"]
        chunk_duration = chunk_strategy["chunk_duration_seconds"]
        
        chunks = []
        
        for i in range(total_chunks):
            start_byte = i * chunk_size_bytes
            end_byte = min((i + 1) * chunk_size_bytes, len(audio_content))
            
            chunk_data = audio_content[start_byte:end_byte]
            chunk_start_time = i * chunk_duration
            
            chunks.append((chunk_data, chunk_start_time, chunk_duration))
            
            logger.debug(f"📦 Chunk {i}: {len(chunk_data)} bytes, {chunk_start_time:.1f}s")
        
        return chunks
    
    async def _process_chunk_with_stt(
        self, 
        chunk_data: bytes, 
        chunk_index: int, 
        chunk_start_time: float, 
        chunk_duration: float
    ) -> List[Dict]:
        """Process a single chunk with Google STT streaming"""
        
        try:
            logger.debug(f"🎯 STT processing chunk {chunk_index}: {len(chunk_data)} bytes")
            
            # Create streaming configuration (without speaker diarization to avoid errors)
            config = self.speech_module.RecognitionConfig(
                encoding=self.speech_module.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-GB",
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                max_alternatives=1
            )
            
            streaming_config = self.speech_module.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False
            )
            
            # Create request generator for streaming
            def request_generator():
                # First request with config
                yield self.speech_module.StreamingRecognizeRequest(
                    streaming_config=streaming_config
                )
                
                # Split chunk into smaller pieces for streaming (4KB each)
                stream_chunk_size = 4096
                for i in range(0, len(chunk_data), stream_chunk_size):
                    stream_chunk = chunk_data[i:i + stream_chunk_size]
                    yield self.speech_module.StreamingRecognizeRequest(
                        audio_content=stream_chunk
                    )
            
            # Process streaming response
            responses = self.google_client.streaming_recognize(request_generator())
            
            # Collect final results
            segments = []
            
            for response in responses:
                if response.error:
                    logger.error(f"❌ STT error in chunk {chunk_index}: {response.error}")
                    continue
                
                for result in response.results:
                    if result.is_final and result.alternatives:
                        alternative = result.alternatives[0]
                        transcript = alternative.transcript.strip()
                        
                        if transcript:
                            # Estimate speaker (alternate for demo purposes)
                            speaker = "customer" if chunk_index % 2 == 0 else "agent"
                            
                            segments.append({
                                "speaker": speaker,
                                "relative_start": 0.0,  # Relative to chunk start
                                "relative_end": chunk_duration,
                                "duration": chunk_duration,
                                "text": transcript,
                                "confidence": alternative.confidence,
                                "chunk_index": chunk_index
                            })
                            
                            logger.debug(f"📝 Chunk {chunk_index} result: '{transcript[:50]}...'")
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ STT processing failed for chunk {chunk_index}: {e}")
            # Return empty segments rather than failing completely
            return []
    
    async def _get_audio_info(self, audio_path: Path) -> Dict[str, Any]:
        """Get audio file information"""
        try:
            file_size = audio_path.stat().st_size
            
            # Estimate duration based on file size and sample rate
            # For 16-bit mono audio at 16kHz: bytes_per_second = 16000 * 2 = 32000
            estimated_duration = file_size / self.bytes_per_second
            
            return {
                "filename": audio_path.name,
                "duration": estimated_duration,
                "sample_rate": self.sample_rate,
                "channels": 1,
                "format": "wav",
                "size_bytes": file_size,
                "transcription_engine": self.transcription_source
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting audio info: {e}")
            return {
                "filename": audio_path.name,
                "duration": 30.0,  # Default
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "size_bytes": 0,
                "transcription_engine": self.transcription_source,
                "error": str(e)
            }
    
    async def stop_realtime_processing(self, session_id: str) -> Dict[str, Any]:
        """Stop real-time processing"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            # Cancel processing task
            if session_id in self.processing_tasks:
                self.processing_tasks[session_id].cancel()
                del self.processing_tasks[session_id]
            
            # Get session info
            session = self.active_sessions[session_id]
            duration = (datetime.now() - session["start_time"]).total_seconds()
            
            # Cleanup
            del self.active_sessions[session_id]
            
            logger.info(f"🛑 Stopped real-time processing for {session_id}")
            
            return {
                "session_id": session_id,
                "status": "stopped",
                "duration": duration,
                "chunks_processed": session.get("chunks_processed", 0),
                "segments_transcribed": len(session.get("transcribed_segments", [])),
                "processing_mode": "realtime_chunks"
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping session: {e}")
            return {"error": str(e)}
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all sessions"""
        return {
            "active_sessions": len(self.active_sessions),
            "processing_tasks": len(self.processing_tasks),
            "transcription_source": self.transcription_source,
            "chunk_size_mb": self.chunk_size_bytes / (1024 * 1024),
            "realtime_chunk_duration": self.real_time_chunk_duration,
            "google_client_available": self.google_client is not None
        }

# Create global instance
audio_processor_agent = AudioProcessorAgent()
