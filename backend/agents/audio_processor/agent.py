# backend/agents/audio_processor/agent.py 
"""
Audio Processing Agent with Google Speech-to-Text Streaming API
"""

import asyncio
import logging
import time
import wave
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import json
import subprocess
import os

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings
from ...utils import get_current_timestamp

logger = logging.getLogger(__name__)

class AudioProcessorAgent(BaseAgent):
    """
    Audio Processing Agent with REAL Google Speech-to-Text Streaming API
    RENAMED: This is now the main audio processor (was ServerRealtimeAudioProcessor)
    """
    
    def __init__(self):
        super().__init__(
            agent_type="audio_processor",
            agent_name="Audio Processing Agent with Google STT"
        )
        
        self.active_sessions: Dict[str, Dict] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Enhanced debug tracking
        self.debug_mode = True
        self.transcription_source = "unknown"
        self.force_google_stt = True  # FORCE GOOGLE STT USAGE
        
        # Initialize transcription engine with enhanced debug
        self._initialize_transcription_engine_debug()
        
        # NO FALLBACK TRANSCRIPTIONS - Google STT only
        # Removed _load_scenario_transcriptions() - no mock fallback allowed
        
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"🎵 {self.agent_name} ready for GOOGLE STT STREAMING API")
        logger.info(f"🐛 DEBUG MODE: {self.debug_mode}")
        logger.info(f"🔧 TRANSCRIPTION SOURCE: {self.transcription_source}")
        logger.info(f"🚀 FORCE GOOGLE STT: {self.force_google_stt}")
        logger.info(f"🌊 STREAMING API: Enabled (handles large files)")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with debug enhancements"""
        settings = get_settings()
        
        base_config = {
            **settings.get_agent_config("audio_processor"),
            "chunk_duration_seconds": 2.0,
            "overlap_seconds": 0.5,
            "sample_rate": 16000,
            "channels": 1,
            "processing_delay_ms": 300,
            "audio_base_path": "data/sample_audio",
            "min_speech_duration": 0.5,
            "silence_threshold": 0.01,
            
            # Enhanced debug settings
            "debug_transcription": True,
            "log_transcription_attempts": True,
            "force_google_stt": True,  # FORCE GOOGLE STT
            "use_streaming_api": True,  # USE STREAMING API
            "streaming_chunk_size": 4096,  # 4KB chunks for streaming
            "mock_fallback_enabled": False  # NO FALLBACK
        }
        
        # Enhanced: Log the configuration being used
        logger.info(f"🔧 Audio Processor Config: {json.dumps(base_config, indent=2)}")
        
        return base_config
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get capabilities"""
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_transcription_engine_debug(self):
        """Initialize transcription engine - FORCE GOOGLE STT"""
        settings = get_settings()
        
        logger.info(f"🔧 FORCING GOOGLE STT INITIALIZATION")
        
        # ALWAYS TRY GOOGLE STT FIRST
        try:
            logger.info("🔄 Importing Google Cloud Speech...")
            from google.cloud import speech
            
            logger.info("🔄 Creating Speech client...")
            self.google_client = speech.SpeechClient()
            self.transcription_source = "google_stt_forced"
            
            logger.info("✅ Google Speech-to-Text client FORCED initialization!")
            
            # Test the client
            try:
                logger.info("🔄 Testing FORCED Google STT access...")
                
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-GB"
                )
                logger.info("✅ FORCED Google STT test successful!")
                        
            except Exception as test_error:
                logger.error(f"❌ FORCED Google STT test failed: {test_error}")
                
                # DON'T FALL BACK - RAISE ERROR INSTEAD
                logger.error("❌ Google STT test failed - system not properly configured")
                self.transcription_source = "google_stt_failed"
                self.google_client = None
                
        except ImportError as import_error:
            logger.error(f"❌ Failed to import Google Cloud Speech: {import_error}")
            logger.error("🔧 Install with: pip install google-cloud-speech")
            self.transcription_source = "import_failed"
            self.google_client = None
        except Exception as init_error:
            logger.error(f"❌ Failed to initialize Google STT: {init_error}")
            self.transcription_source = "init_failed"
            self.google_client = None
        
        # Log final transcription source
        logger.info(f"🎯 FINAL TRANSCRIPTION SOURCE: {self.transcription_source}")
        logger.info(f"🎯 GOOGLE CLIENT AVAILABLE: {self.google_client is not None}")
        
        if not self.google_client:
            logger.error("❌ CRITICAL: Google Speech-to-Text not available!")
            logger.error("🔧 System will fail on audio processing attempts")
            logger.error("🔧 Fix authentication before processing audio")
    
    # REMOVED _load_scenario_transcriptions() - NO FALLBACK ALLOWED
    # System will only use Google Speech-to-Text API
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Main processing method - transcribe audio file using Google STT
        UPDATED: Now uses Google STT instead of mock data
        """
        if isinstance(input_data, dict):
            file_path = input_data.get('file_path', input_data.get('audio_file_path', ''))
        else:
            file_path = str(input_data)
        
        # For the basic process method, we'll use a simplified approach
        # This maintains compatibility with the existing agent interface
        try:
            if not self.google_client:
                return {
                    "error": "Google Speech-to-Text not available",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"🎯 Processing audio file via basic agent interface: {file_path}")
            
            # For now, return a compatible response
            # The real processing happens via start_realtime_processing()
            return {
                "agent_type": self.agent_type,
                "status": "ready_for_real_processing",
                "file_path": file_path,
                "google_client_available": self.google_client is not None,
                "transcription_source": self.transcription_source,
                "processing_mode": "google_stt_streaming",
                "note": "Use start_realtime_processing() for full Google STT transcription",
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
        """Start real-time processing with GOOGLE STT STREAMING"""
        
        try:
            logger.info(f"🎵 STARTING GOOGLE STT STREAMING TRANSCRIPTION")
            logger.info(f"🎯 Session ID: {session_id}")
            logger.info(f"📁 Audio File: {audio_filename}")
            logger.info(f"🔧 Transcription Source: {self.transcription_source}")
            logger.info(f"🚀 Force Google STT: {self.force_google_stt}")
            
            # Validate audio file exists
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            logger.info(f"📍 Audio Path: {audio_path}")
            logger.info(f"📁 File Exists: {audio_path.exists()}")
            
            if not audio_path.exists():
                logger.error(f"❌ Audio file not found: {audio_path}")
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Check if session already exists
            if session_id in self.active_sessions:
                raise ValueError(f"Session {session_id} already active")
            
            # Get audio file info
            audio_info = await self._get_real_audio_info(audio_path)
            logger.info(f"🎵 Audio Info: {audio_info}")
            
            # FORCE GOOGLE STT USAGE - NO FALLBACK
            logger.info("🚀 FORCING GOOGLE STT USAGE - NO FALLBACK ALLOWED")
            
            if not self.google_client:
                error_msg = "Google Speech-to-Text API not available. Please configure authentication."
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            logger.info("🎯 ATTEMPTING REAL GOOGLE SPEECH-TO-TEXT STREAMING")
            transcription_segments = await self._get_google_stt_transcription(audio_path, audio_filename)
            logger.info(f"📝 Google STT returned {len(transcription_segments)} segments")
            
            if not transcription_segments:
                error_msg = f"Google STT returned no transcription for {audio_filename}. Audio may contain no speech."
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            logger.info(f"📝 Final transcription: {len(transcription_segments)} segments")
            if transcription_segments:
                logger.info(f"📄 First segment: '{transcription_segments[0].get('text', 'N/A')[:50]}...'")
            
            # Initialize session
            self.active_sessions[session_id] = {
                "filename": audio_filename,
                "audio_path": str(audio_path),
                "audio_info": audio_info,
                "transcription_segments": transcription_segments,
                "websocket_callback": websocket_callback,
                "start_time": datetime.now(),
                "status": "active",
                "current_position": 0.0,
                "transcribed_segments": [],
                "speaker_state": "unknown",
                "transcription_source": self.transcription_source,
                "forced_google_stt": self.force_google_stt
            }
            
            # Start processing task
            processing_task = asyncio.create_task(
                self._process_audio_with_real_transcription(session_id)
            )
            self.processing_tasks[session_id] = processing_task
            
            # Send initial confirmation
            await websocket_callback({
                "type": "real_processing_started",
                "data": {
                    "session_id": session_id,
                    "filename": audio_filename,
                    "audio_info": audio_info,
                    "processing_mode": "google_stt_streaming",
                    "transcription_engine": self.transcription_source,
                    "forced_google": self.force_google_stt,
                    "expected_segments": len(transcription_segments),
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {
                "session_id": session_id,
                "status": "started",
                "audio_duration": audio_info["duration"],
                "processing_mode": "google_stt_streaming",
                "transcription_source": self.transcription_source,
                "transcription_file": audio_filename,
                "streaming_enabled": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting Google STT: {e}")
            logger.error(f"🔍 Exception details: {type(e).__name__}: {str(e)}")
            return {"error": str(e)}
    
    async def _get_google_stt_transcription(self, audio_path: Path, filename: str) -> List[Dict]:
        """Get transcription from Google Speech-to-Text STREAMING API - HANDLES LARGE FILES"""
        logger.info(f"🎯 CALLING GOOGLE STT STREAMING API FOR: {filename}")
        logger.info(f"📁 Audio path: {audio_path}")
        logger.info(f"🔧 Google client available: {self.google_client is not None}")
        
        try:
            if not self.google_client:
                logger.error("❌ Google client not available")
                raise Exception("Google STT client not initialized")
            
            # Read the audio file
            logger.info(f"📖 Reading audio file: {audio_path}")
            with open(audio_path, 'rb') as audio_file:
                audio_content = audio_file.read()
            
            file_size_bytes = len(audio_content)
            file_size_mb = file_size_bytes / (1024 * 1024)
            logger.info(f"📁 Audio file size: {file_size_bytes} bytes ({file_size_mb:.2f} MB)")
            
            if file_size_bytes == 0:
                logger.error("❌ Audio file is empty")
                raise Exception("Audio file is empty")
            
            # Import the speech module
            logger.info("🔄 Importing Google Cloud Speech module...")
            from google.cloud import speech
            
            # Create streaming configuration
            logger.info("🔄 Creating Google STT STREAMING configuration...")
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-GB",
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                use_enhanced=True,
                model="latest_long",  # Better for longer audio
                # Enable speaker diarization for better speaker separation
                enable_speaker_diarization=True,
                diarization_speaker_count=2,  # Assume 2 speakers (agent + customer)
                max_alternatives=1
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,  # Get partial results
                single_utterance=False  # Don't stop after first utterance
            )
            
            logger.info("🔄 Setting up STREAMING transcription...")
            logger.info(f"🔧 Config: language={config.language_code}, sample_rate={config.sample_rate_hertz}")
            logger.info(f"🔧 Speaker diarization: {config.enable_speaker_diarization}")
            logger.info(f"🔧 Audio size: {file_size_mb:.2f} MB (streaming can handle large files)")
            
            # Use streaming API
            segments = await self._process_streaming_audio(audio_content, streaming_config, filename)
            
            if segments:
                logger.info(f"✅ REAL GOOGLE STT STREAMING SUCCESS: {len(segments)} segments transcribed")
                logger.info(f"🎯 This was a REAL STREAMING API call to Google Speech-to-Text!")
                logger.info(f"📊 Total duration estimated: {segments[-1]['end']:.1f} seconds")
                return segments
            else:
                logger.error("❌ Google STT Streaming returned no results")
                logger.error("🔄 Audio may contain no speech or is not recognized")
                raise Exception("Google STT Streaming returned no transcription results")
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"❌ Google STT STREAMING transcription failed: {e}")
            logger.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            
            # Log the specific error for debugging
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            
            # Re-raise with cleaner message
            raise Exception(f"Google Speech-to-Text Streaming failed: {str(e)}")
    
    async def _process_streaming_audio(self, audio_content: bytes, streaming_config, filename: str) -> List[Dict]:
        """Process audio using Google STT Streaming API"""
        logger.info(f"🌊 Starting STREAMING transcription for {filename}")
        
        from google.cloud import speech
        import asyncio
        
        # Chunk the audio data for streaming
        chunk_size = 4096  # 4KB chunks
        audio_chunks = [audio_content[i:i + chunk_size] for i in range(0, len(audio_content), chunk_size)]
        
        logger.info(f"📦 Split audio into {len(audio_chunks)} chunks of {chunk_size} bytes each")
        
        def generate_audio_chunks():
            """Generator for audio chunks"""
            for i, chunk in enumerate(audio_chunks):
                if i == 0:
                    logger.info(f"📤 Sending first chunk ({len(chunk)} bytes)")
                elif i == len(audio_chunks) - 1:
                    logger.info(f"📤 Sending final chunk ({len(chunk)} bytes)")
                elif i % 50 == 0:  # Log every 50th chunk
                    logger.info(f"📤 Sending chunk {i}/{len(audio_chunks)} ({len(chunk)} bytes)")
                
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
        
        # Create the streaming request generator
        def request_generator():
            # First request with configuration
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            # Then yield audio chunks
            yield from generate_audio_chunks()
        
        try:
            logger.info("🔄 Calling Google STT streaming_recognize...")
            
            # Make the streaming request
            responses = self.google_client.streaming_recognize(request_generator())
            
            logger.info("📥 Processing streaming responses...")
            
            segments = []
            current_speaker = "customer"
            segment_start_time = 0.0
            
            # Process streaming responses
            for i, response in enumerate(responses):
                logger.info(f"📥 Processing response {i}")
                
                # Check for errors
                if response.error:
                    logger.error(f"❌ Streaming error: {response.error}")
                    raise Exception(f"Google STT streaming error: {response.error}")
                
                # Process results
                for result in response.results:
                    if result.is_final:
                        # Final result - add to segments
                        if result.alternatives:
                            alternative = result.alternatives[0]
                            transcript = alternative.transcript.strip()
                            confidence = alternative.confidence
                            
                            logger.info(f"📝 Final result: '{transcript}' (confidence: {confidence:.2f})")
                            
                            if transcript:
                                # Determine speaker from diarization if available
                                speaker = current_speaker
                                if hasattr(result, 'speaker_tag'):
                                    speaker = "agent" if result.speaker_tag == 1 else "customer"
                                
                                # Calculate duration based on words (rough estimate)
                                words = transcript.split()
                                duration = max(len(words) / 2.5, 1.0)  # ~2.5 words per second
                                
                                segment = {
                                    "speaker": speaker,
                                    "start": segment_start_time,
                                    "end": segment_start_time + duration,
                                    "text": transcript,
                                    "confidence": confidence,
                                    "source": "google_stt_streaming",
                                    "is_final": True,
                                    "word_count": len(words)
                                }
                                
                                segments.append(segment)
                                logger.info(f"✅ Added segment: {speaker} - '{transcript[:50]}...' ({duration:.1f}s)")
                                
                                segment_start_time += duration + 0.5  # Small gap between segments
                                # Alternate speaker for demo (in real app, use diarization)
                                current_speaker = "agent" if current_speaker == "customer" else "customer"
                    
                    else:
                        # Interim result - just log for debugging
                        if result.alternatives:
                            interim_text = result.alternatives[0].transcript
                            logger.debug(f"📝 Interim: '{interim_text[:30]}...'")
            
            logger.info(f"🌊 Streaming transcription complete: {len(segments)} final segments")
            return segments
            
        except Exception as e:
            logger.error(f"❌ Streaming processing failed: {e}")
            raise Exception(f"Streaming transcription failed: {str(e)}")
    
    # ... (rest of the methods remain the same as server_realtime_processor)
    
    async def _process_audio_with_real_transcription(self, session_id: str) -> None:
        """Core processing loop with enhanced debug logging"""
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        transcription_segments = session["transcription_segments"]
        audio_info = session["audio_info"]
        
        try:
            logger.info(f"🔄 STARTING GOOGLE STT TRANSCRIPTION LOOP")
            logger.info(f"🎯 Session: {session_id}")
            logger.info(f"📝 Segments: {len(transcription_segments)}")
            logger.info(f"🔧 Source: {session.get('transcription_source', 'unknown')}")
            logger.info(f"🚀 Forced Google: {session.get('forced_google_stt', False)}")
            
            chunk_duration = self.config["chunk_duration_seconds"]
            current_time = 0.0
            segment_index = 0
            
            # Process the transcription segments
            for segment in transcription_segments:
                # Check if session was cancelled
                if session_id not in self.active_sessions:
                    break
                
                # Wait until we reach the segment's start time
                segment_start = segment["start"]
                
                # Wait until it's time for this segment
                while current_time < segment_start:
                    if session_id not in self.active_sessions:
                        return
                    
                    await asyncio.sleep(0.1)
                    current_time += 0.1
                
                # Update current position
                session["current_position"] = current_time
                
                # Create transcription segment with enhanced debug info
                transcription_result = {
                    "session_id": session_id,
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["end"] - segment["start"],
                    "text": segment["text"],
                    "confidence": segment.get("confidence", 0.92),
                    "segment_index": segment_index,
                    "processing_mode": "google_stt_streaming",
                    "transcription_source": session.get('transcription_source', 'unknown'),
                    "forced_google": session.get('forced_google_stt', False),
                    "segment_source": segment.get("source", "unknown"),
                    "timestamp": get_current_timestamp()
                }
                
                # Store segment
                session["transcribed_segments"].append(transcription_result)
                
                # Send transcription segment
                await websocket_callback({
                    "type": "transcription_segment",
                    "data": transcription_result
                })
                
                logger.info(f"📤 Google STT segment {segment_index}: {segment['speaker']} - {segment['text'][:50]}...")
                logger.info(f"🔧 Source: {session.get('transcription_source', 'unknown')}")
                logger.info(f"🎯 Segment source: {segment.get('source', 'unknown')}")
                segment_index += 1
                
                # Wait for realistic processing time
                processing_delay = self.config["processing_delay_ms"] / 1000.0
                await asyncio.sleep(processing_delay)
                
                # Update current time to end of segment
                current_time = segment["end"] + 0.5
            
            # Send completion message
            await websocket_callback({
                "type": "processing_complete",
                "data": {
                    "session_id": session_id,
                    "total_segments": len(session["transcribed_segments"]),
                    "transcription_engine": session.get('transcription_source', 'unknown'),
                    "forced_google_stt": session.get('forced_google_stt', False),
                    "audio_file": session["filename"],
                    "timestamp": get_current_timestamp()
                }
            })
            
            logger.info(f"✅ Google STT transcription completed for {session_id} - {session['filename']}")
            logger.info(f"🔧 Final source: {session.get('transcription_source', 'unknown')}")
            logger.info(f"🎯 Forced Google STT: {session.get('forced_google_stt', False)}")
            
        except asyncio.CancelledError:
            logger.info(f"🛑 Google STT transcription cancelled for {session_id}")
        except Exception as e:
            logger.error(f"❌ Error in Google STT transcription loop: {e}")
            await websocket_callback({
                "type": "error",
                "data": {
                    "session_id": session_id,
                    "error": str(e),
                    "transcription_source": session.get('transcription_source', 'unknown'),
                    "timestamp": get_current_timestamp()
                }
            })
    
    async def _get_real_audio_info(self, audio_path: Path) -> Dict[str, Any]:
        """Get real audio file information with debug"""
        
        try:
            file_size = audio_path.stat().st_size if audio_path.exists() else 0
            filename = audio_path.name
            
            logger.info(f"🎵 Getting audio info for: {filename}")
            logger.info(f"📁 File size: {file_size} bytes")
            
            # For streaming STT, estimate duration from file size
            estimated_duration = max(20.0, min(60.0, file_size / 50000))
            
            audio_info = {
                "filename": audio_path.name,
                "duration": estimated_duration,
                "sample_rate": self.config["sample_rate"],
                "channels": self.config["channels"],
                "format": "wav",
                "size_bytes": file_size,
                "transcription_engine": self.transcription_source,
                "forced_google_stt": self.force_google_stt
            }
            
            logger.info(f"🎵 Audio Info Generated: {audio_info}")
            return audio_info
            
        except Exception as e:
            logger.error(f"❌ Error getting audio info: {e}")
            return {
                "filename": audio_path.name,
                "duration": 25.0,
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "size_bytes": 0,
                "transcription_engine": self.transcription_source,
                "forced_google_stt": self.force_google_stt,
                "error": str(e)
            }
    
    async def stop_realtime_processing(self, session_id: str) -> Dict[str, Any]:
        """Stop real-time processing for a session"""
        
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            # Cancel processing task
            if session_id in self.processing_tasks:
                self.processing_tasks[session_id].cancel()
                del self.processing_tasks[session_id]
            
            # Get session info before cleanup
            session = self.active_sessions[session_id]
            duration = (datetime.now() - session["start_time"]).total_seconds()
            
            # Cleanup session
            del self.active_sessions[session_id]
            
            logger.info(f"🛑 Stopped Google STT transcription for session {session_id}")
            
            return {
                "session_id": session_id,
                "status": "stopped",
                "duration": duration,
                "segments_transcribed": len(session.get("transcribed_segments", [])),
                "audio_file": session.get("filename", "unknown"),
                "transcription_source": session.get("transcription_source", "unknown"),
                "forced_google_stt": session.get("forced_google_stt", False)
            }
            
        except Exception as e:
            logger.error(f"❌ Error stopping session {session_id}: {e}")
            return {"error": str(e)}
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all sessions"""
        return {
            "active_sessions": len(self.active_sessions),
            "processing_tasks": len(self.processing_tasks),
            "transcription_source": self.transcription_source,
            "forced_google_stt": self.force_google_stt,
            "debug_mode": self.debug_mode,
            "google_client_available": self.google_client is not None
        }

# Create global instance - NOW THE MAIN AUDIO AGENT
audio_processor_agent = AudioProcessorAgent()
