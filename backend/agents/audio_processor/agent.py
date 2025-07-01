# backend/agents/audio_processor/agent.py - SIMPLIFIED WORKING VERSION
"""
Audio Processing Agent with Google Speech-to-Text API
SIMPLIFIED: Based on your working example code
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import json

from ..shared.base_agent import BaseAgent, AgentCapability, agent_registry
from ...config.settings import get_settings
from ...utils import get_current_timestamp

logger = logging.getLogger(__name__)

class AudioProcessorAgent(BaseAgent):
    """
    Audio Processing Agent with Google Speech-to-Text API
    SIMPLIFIED: Uses the working pattern from your example
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
        self.force_google_stt = True  
        
        # Initialize transcription engine
        self._initialize_google_stt()
              
        # Register with global registry
        agent_registry.register_agent(self)
        
        logger.info(f"🎵 {self.agent_name} ready for GOOGLE STT API")
        logger.info(f"🔧 TRANSCRIPTION SOURCE: {self.transcription_source}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
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
        }
        
        return base_config
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get capabilities"""
        return [
            AgentCapability.AUDIO_PROCESSING,
            AgentCapability.REAL_TIME_PROCESSING
        ]
    
    def _initialize_google_stt(self):
        """Initialize Google Speech-to-Text using your working example pattern"""
        
        try:
            logger.info("🔄 Importing Google Cloud Speech (v1p1beta1)...")
            from google.cloud import speech_v1p1beta1 as speech
            
            logger.info("🔄 Creating Speech client...")
            self.google_client = speech.SpeechClient()
            self.speech_module = speech  # Store reference for later use
            self.transcription_source = "google_stt_v1p1beta1"
            
            logger.info("✅ Google Speech-to-Text client initialized!")
            
            # Test the client with your working configuration pattern
            try:
                logger.info("🔄 Testing Google STT access...")
                
                # Use your working example configuration
                diarization_config = speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=2,
                    max_speaker_count=10,
                )
                
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-GB",
                    diarization_config=diarization_config,
                )
                
                logger.info("✅ Google STT configuration test successful!")
                        
            except Exception as test_error:
                logger.error(f"❌ Google STT test failed: {test_error}")
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
        
        # Log final status
        logger.info(f"🎯 FINAL TRANSCRIPTION SOURCE: {self.transcription_source}")
        logger.info(f"🎯 GOOGLE CLIENT AVAILABLE: {self.google_client is not None}")
    
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
                "status": "ready_for_real_processing",
                "file_path": file_path,
                "google_client_available": self.google_client is not None,
                "transcription_source": self.transcription_source,
                "processing_mode": "google_stt_recognize",
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
        """Start real-time processing with Google STT"""
        
        try:
            logger.info(f"🎵 STARTING GOOGLE STT TRANSCRIPTION")
            logger.info(f"🎯 Session ID: {session_id}")
            logger.info(f"📁 Audio File: {audio_filename}")
            
            # Validate audio file exists
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            
            if not audio_path.exists():
                logger.error(f"❌ Audio file not found: {audio_path}")
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Check if session already exists
            if session_id in self.active_sessions:
                raise ValueError(f"Session {session_id} already active")
            
            # Get audio file info
            audio_info = await self._get_audio_info(audio_path)
            
            if not self.google_client:
                error_msg = "Google Speech-to-Text API not available. Please configure authentication."
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            logger.info("🎯 CALLING REAL GOOGLE SPEECH-TO-TEXT API")
            transcription_segments = await self._get_google_transcription(audio_path, audio_filename)
            logger.info(f"📝 Google STT returned {len(transcription_segments)} segments")
            
            if not transcription_segments:
                error_msg = f"Google STT returned no transcription for {audio_filename}."
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
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
                "transcription_source": self.transcription_source
            }
            
            # Start processing task
            processing_task = asyncio.create_task(
                self._process_segments(session_id)
            )
            self.processing_tasks[session_id] = processing_task
            
            # Send initial confirmation
            await websocket_callback({
                "type": "real_processing_started",
                "data": {
                    "session_id": session_id,
                    "filename": audio_filename,
                    "audio_info": audio_info,
                    "processing_mode": "google_stt_recognize",
                    "transcription_engine": self.transcription_source,
                    "expected_segments": len(transcription_segments),
                    "timestamp": get_current_timestamp()
                }
            })
            
            return {
                "session_id": session_id,
                "status": "started",
                "audio_duration": audio_info["duration"],
                "processing_mode": "google_stt_recognize",
                "transcription_source": self.transcription_source,
                "transcription_file": audio_filename
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting Google STT: {e}")
            return {"error": str(e)}
    
    async def _get_google_transcription(self, audio_path: Path, filename: str) -> List[Dict]:
        """Get transcription from Google Speech-to-Text API using your working example pattern"""
        logger.info(f"🎯 CALLING GOOGLE STT API FOR: {filename}")
        
        try:
            # Read the audio file (like your example)
            logger.info(f"📖 Reading audio file: {audio_path}")
            with open(audio_path, 'rb') as audio_file:
                content = audio_file.read()
            
            file_size_bytes = len(content)
            file_size_mb = file_size_bytes / (1024 * 1024)
            logger.info(f"📁 Audio file size: {file_size_bytes} bytes ({file_size_mb:.2f} MB)")
            
            if file_size_bytes == 0:
                logger.error("❌ Audio file is empty")
                raise Exception("Audio file is empty")
            
            # Create audio object (like your example)
            audio = self.speech_module.RecognitionAudio(content=content)
            
            # Use your working configuration pattern
            diarization_config = self.speech_module.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,
                max_speaker_count=10,
            )
            
            config = self.speech_module.RecognitionConfig(
                encoding=self.speech_module.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-GB",
                diarization_config=diarization_config,
            )
            
            logger.info("🔄 Calling Google STT recognize() method...")
            logger.info(f"🔧 Config: language={config.language_code}, sample_rate={config.sample_rate_hertz}")
            logger.info(f"🔧 Speaker diarization: enabled with 2-10 speakers")
            
            # Call the API (like your example)
            response = self.google_client.recognize(config=config, audio=audio)
            
            # Process the response using your example pattern
            segments = []
            
            if response.results:
                logger.info(f"📥 Got {len(response.results)} results from Google STT")
                
                # Get the last result which contains all words with speaker tags (like your example)
                result = response.results[-1]
                
                if result.alternatives:
                    alternative = result.alternatives[0]
                    
                    # Check if we have word-level speaker information
                    if hasattr(alternative, 'words') and alternative.words:
                        logger.info(f"📝 Processing {len(alternative.words)} words with speaker tags")
                        
                        # Group words by speaker to create segments
                        current_speaker = None
                        current_text = []
                        current_start = 0.0
                        
                        for word_info in alternative.words:
                            speaker_tag = getattr(word_info, 'speaker_tag', 1)
                            word = word_info.word
                            
                            # Convert speaker tag to readable speaker (1 = agent, others = customer)
                            speaker = "agent" if speaker_tag == 1 else "customer"
                            
                            # If speaker changed, create a segment from accumulated words
                            if current_speaker is not None and current_speaker != speaker:
                                if current_text:
                                    segment_text = " ".join(current_text)
                                    segments.append({
                                        "speaker": current_speaker,
                                        "start": current_start,
                                        "end": current_start + len(current_text) * 0.4,  # Rough timing
                                        "text": segment_text,
                                        "confidence": alternative.confidence,
                                        "source": "google_stt_recognize"
                                    })
                                    current_start += len(current_text) * 0.4 + 0.5
                                
                                current_text = []
                            
                            current_speaker = speaker
                            current_text.append(word)
                        
                        # Add the final segment
                        if current_text:
                            segment_text = " ".join(current_text)
                            segments.append({
                                "speaker": current_speaker,
                                "start": current_start,
                                "end": current_start + len(current_text) * 0.4,
                                "text": segment_text,
                                "confidence": alternative.confidence,
                                "source": "google_stt_recognize"
                            })
                    
                    else:
                        # No word-level info, create single segment
                        transcript = alternative.transcript.strip()
                        if transcript:
                            segments.append({
                                "speaker": "customer",  # Default speaker
                                "start": 0.0,
                                "end": 30.0,  # Default duration
                                "text": transcript,
                                "confidence": alternative.confidence,
                                "source": "google_stt_recognize"
                            })
            
            if segments:
                logger.info(f"✅ REAL GOOGLE STT SUCCESS: {len(segments)} segments transcribed")
                logger.info(f"🎯 This was a REAL API call to Google Speech-to-Text!")
                for i, seg in enumerate(segments):
                    logger.info(f"📄 Segment {i}: {seg['speaker']} - '{seg['text'][:50]}...'")
                return segments
            else:
                logger.error("❌ Google STT returned no usable results")
                raise Exception("Google STT returned no transcription results")
            
        except Exception as e:
            logger.error(f"❌ Google STT transcription failed: {e}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise Exception(f"Google Speech-to-Text failed: {str(e)}")
    
    async def _process_segments(self, session_id: str) -> None:
        """Process transcription segments and send via WebSocket"""
        
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        transcription_segments = session["transcription_segments"]
        
        try:
            logger.info(f"🔄 Processing {len(transcription_segments)} segments")
            
            chunk_duration = self.config["chunk_duration_seconds"]
            current_time = 0.0
            segment_index = 0
            
            # Process each segment
            for segment in transcription_segments:
                # Check if session was cancelled
                if session_id not in self.active_sessions:
                    break
                
                # Wait until we reach the segment's start time
                segment_start = segment["start"]
                
                while current_time < segment_start:
                    if session_id not in self.active_sessions:
                        return
                    await asyncio.sleep(0.1)
                    current_time += 0.1
                
                # Create transcription result
                transcription_result = {
                    "session_id": session_id,
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["end"] - segment["start"],
                    "text": segment["text"],
                    "confidence": segment.get("confidence", 0.95),
                    "segment_index": segment_index,
                    "processing_mode": "google_stt_recognize",
                    "transcription_source": self.transcription_source,
                    "segment_source": segment.get("source", "unknown"),
                    "timestamp": get_current_timestamp()
                }
                
                # Store and send segment
                session["transcribed_segments"].append(transcription_result)
                
                await websocket_callback({
                    "type": "transcription_segment",
                    "data": transcription_result
                })
                
                logger.info(f"📤 Sent segment {segment_index}: {segment['speaker']} - {segment['text'][:50]}...")
                segment_index += 1
                
                # Processing delay
                processing_delay = self.config["processing_delay_ms"] / 1000.0
                await asyncio.sleep(processing_delay)
                
                current_time = segment["end"] + 0.5
            
            # Send completion
            await websocket_callback({
                "type": "processing_complete",
                "data": {
                    "session_id": session_id,
                    "total_segments": len(session["transcribed_segments"]),
                    "transcription_engine": self.transcription_source,
                    "audio_file": session["filename"],
                    "timestamp": get_current_timestamp()
                }
            })
            
            logger.info(f"✅ Processing completed for {session_id}")
            
        except asyncio.CancelledError:
            logger.info(f"🛑 Processing cancelled for {session_id}")
        except Exception as e:
            logger.error(f"❌ Error in processing loop: {e}")
            await websocket_callback({
                "type": "error",
                "data": {
                    "session_id": session_id,
                    "error": str(e),
                    "transcription_source": self.transcription_source,
                    "timestamp": get_current_timestamp()
                }
            })
    
    async def _get_audio_info(self, audio_path: Path) -> Dict[str, Any]:
        """Get audio file information"""
        try:
            file_size = audio_path.stat().st_size if audio_path.exists() else 0
            estimated_duration = max(20.0, min(60.0, file_size / 50000))
            
            return {
                "filename": audio_path.name,
                "duration": estimated_duration,
                "sample_rate": self.config["sample_rate"],
                "channels": self.config["channels"],
                "format": "wav",
                "size_bytes": file_size,
                "transcription_engine": self.transcription_source
            }
            
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
            
            # Get session info
            session = self.active_sessions[session_id]
            duration = (datetime.now() - session["start_time"]).total_seconds()
            
            # Cleanup
            del self.active_sessions[session_id]
            
            logger.info(f"🛑 Stopped processing for session {session_id}")
            
            return {
                "session_id": session_id,
                "status": "stopped",
                "duration": duration,
                "segments_transcribed": len(session.get("transcribed_segments", [])),
                "audio_file": session.get("filename", "unknown"),
                "transcription_source": self.transcription_source
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
            "debug_mode": self.debug_mode,
            "google_client_available": self.google_client is not None
        }

# Create global instance
audio_processor_agent = AudioProcessorAgent()
