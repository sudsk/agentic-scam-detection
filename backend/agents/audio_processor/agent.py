async def _process_stereo_response(
        self, 
        session_id: str, 
        response, 
        speaker_tracker, 
        websocket_callback: Callable,
        last_final_text: str
    ) -> Optional[Dict]:
        """Process stereo streaming response using channel information"""
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
                    logger.debug(f"🚫 Skipping exact duplicate: '{transcript[:30]}...'")
                    continue
                
                # Extract channel and speaker information
                channel_tag = None
                speaker_tag = None
                
                if hasattr(alternative, 'words') and alternative.words:
                    # Get channel tags (for stereo)
                    channel_tags = [w.channel_tag for w in alternative.words 
                                   if hasattr(w, 'channel_tag') and w.channel_tag is not None]
                    if channel_tags:
                        channel_tag = max(set(channel_tags), key=channel_tags.count)
                    
                    # Get speaker tags (backup)
                    speaker_tags = [w.speaker_tag for w in alternative.words 
                                   if hasattr(w, 'speaker_tag') and w.speaker_tag is not None]
                    if speaker_tags:
                        speaker_tag = max(set(speaker_tags), key=speaker_tags.count)
                
                # Detect speaker using channel information
                detected_speaker = speaker_tracker.detect_speaker_stereo(channel_tag, speaker_tag, transcript)
                
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
                    "channel_tag": channel_tag,
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
                    
                    logger.info(f"📝 STEREO FINAL [{detected_speaker}] (ch:{channel_tag}, spk:{speaker_tag}): '{transcript[:50]}...'")
                    
                    # Trigger fraud analysis for customer speech
                    if detected_speaker == "customer":
                        await self._trigger_fraud_analysis(
                            session_id, transcript, websocket_callback
                        )
                    
                    return segment_data
                else:
                    # Send interim results
                    if len(transcript) > 5:
                        await websocket_callback({
                            "type": "transcription_interim",
                            "data": segment_data
                        })
        
        except Exception as e:
            logger.error(f"❌ Error processing stereo response: {e}")
        
        return None


class StereoChannelTracker:
    """Speaker tracker for stereo audio using Google's channel information"""
    
    def __init__(self):
        self.speaker_history = []
        self.turn_count = {"agent": 0, "customer": 0}
        self.segment_count = 0
        
    def detect_speaker_stereo(self, channel_tag: Optional[int], speaker_tag: Optional[int], transcript: str) -> str:
        """Detect speaker using channel information (100% accurate for proper stereo)"""
        
        self.segment_count += 1
        
        # PRIMARY: Use channel information if available
        if channel_tag is not None:
            if channel_tag == 0:
                detected_speaker = "agent"     # Left channel = Agent
            elif channel_tag == 1:
                detected_speaker = "customer"  # Right channel = Customer
            else:
                # Unexpected channel, fallback to content
                detected_speaker = self._detect_by_content(transcript)
            
            logger.info(f"🎯 STEREO channel detection: Channel {channel_tag} → {detected_speaker}")
            
        # FALLBACK: Use speaker tag if no channel info
        elif speaker_tag is not None:
            # First segment is always agent
            if self.segment_count == 1:
                detected_speaker = "agent"
            else:
                # Use content detection with speaker tag as hint
                detected_speaker = self._detect_by_content(transcript)
            
            logger.info(f"🎯 STEREO fallback: Speaker tag {speaker_tag} + content → {detected_speaker}")
            
        # LAST RESORT: Content-based detection
        else:
            detected_speaker = self._detect_by_content(transcript)
            logger.info(f"🎯 STEREO content-only: '{transcript[:20]}...' → {detected_speaker}")
        
        # Update tracking
        self.turn_count[detected_speaker] += 1
        self.speaker_history.append(detected_speaker)
        
        return detected_speaker
    
    def _detect_by_content(self, transcript: str) -> str:
        """Content-based detection for stereo fallback"""
        transcript_lower = transcript.lower()
        
        # Strong customer indicators
        customer_phrases = [
            "hi", "hello", "yes", "i need", "i want", "please", "urgent", "emergency",
            "alex", "patricia", "williams", "turkey", "istanbul", "stuck", "hospital",
            "boyfriend", "girlfriend", "partner", "dating", "money", "help"
        ]
        
        # Strong agent indicators
        agent_phrases = [
            "good morning", "good afternoon", "hsbc", "customer services",
            "how can i assist", "mrs williams", "i understand", "i see",
            "can you tell me", "fraud", "scam", "protection", "verify"
        ]
        
        customer_score = sum(1 for phrase in customer_phrases if phrase in transcript_lower)
        agent_score = sum(1 for phrase in agent_phrases if phrase in transcript_lower)
        
        if customer_score > agent_score:
            return "customer"
        elif agent_score > customer_score:
            return "agent"
        else:
            # Default based on segment position
            return "agent" if self.segment_count <= 2 else "customer"
    
    def get_stats(self) -> Dict:
        return {
            "type": "stereo_channel_tracker",
            "turn_count": self.turn_count.copy(),
            "total_segments": len(self.speaker_history),
            "detection_method": "channel_based"
        }


class EnhancedSpeakerTracker(SpeakerTracker):
    """Enhanced speaker tracker for mono audio with better content detection"""
    
    def __init__(self):
        super().__init__()
        self.confidence_scores = []
        
    def detect_speaker(self, google_speaker_tag: Optional[int], transcript: str) -> str:
        """Enhanced speaker detection for mono audio"""
        
        self.segment_count += 1
        
        # First segment is always agent
        if self.segment_count == 1:
            self.current_speaker = "agent"
            self.turn_count["agent"] += 1
            self.speaker_history.append("agent")
            self.last_speaker_tag = google_speaker_tag
            self.confidence_scores.append(1.0)  # 100% confident about first segment
            logger.info(f"🎯 MONO first segment: agent (tag:{google_speaker_tag})")
            return "agent"
        
        # Use Google tags if they change
        if google_speaker_tag is not None and self.last_speaker_tag is not None:
            if google_speaker_tag != self.last_speaker_tag:
                detected_speaker = "customer" if self.current_speaker == "agent" else "agent"
                confidence = 0.9  # High confidence in Google tag changes
                
                logger.info(f"🔄 MONO tag change: {self.last_speaker_tag} → {google_speaker_tag}, switching to {detected_speaker}")
                
                self.current_speaker = detected_speaker
                self.turn_count[detected_speaker] += 1
                self.last_speaker_tag = google_speaker_tag
                self.confidence_scores.append(confidence)
                self.speaker_history.append(detected_speaker)
                return detected_speaker
        
        # Enhanced content-based detection
        content_speaker, confidence = self._enhanced_content_detection(transcript)
        
        # Use content detection if confident or if Google tags are unreliable
        if confidence > 0.7 or google_speaker_tag is None:
            if content_speaker != self.current_speaker:
                logger.info(f"🔍 MONO content switch: {self.current_speaker} → {content_speaker} (confidence: {confidence:.2f})")
                self.current_speaker = content_speaker
                self.turn_count[content_speaker] += 1
            
            self.confidence_scores.append(confidence)
            self.speaker_history.append(content_speaker)
            return content_speaker
        
        # Default: keep current speaker
        self.confidence_scores.append(0.5)
        self.speaker_history.append(self.current_speaker)
        return self.current_speaker
    
    def _enhanced_content_detection(self, transcript: str) -> tuple:
        """Enhanced content detection with confidence scoring"""
        transcript_lower = transcript.lower()
        
        # High-confidence customer phrases
        strong_customer = ["hi", "hello", "yes", "i need", "urgent", "emergency", "alex", "patricia", "stuck"]
        # High-confidence agent phrases  
        strong_agent = ["hsbc", "customer services", "mrs williams", "fraud", "scam", "i understand"]
        
        # Medium-confidence phrases
        medium_customer = ["please", "help", "money", "transfer", "turkey", "hospital", "boyfriend"]
        medium_agent = ["can you tell me", "verify", "security", "protection", "policy"]
        
        # Calculate weighted scores
        strong_customer_score = sum(2 for phrase in strong_customer if phrase in transcript_lower)
        strong_agent_score = sum(2 for phrase in strong_agent if phrase in transcript_lower)
        medium_customer_score = sum(1 for phrase in medium_customer if phrase in transcript_lower)
        medium_agent_score = sum(1 for phrase in medium_agent if phrase in transcript_lower)
        
        total_customer = strong_customer_score + medium_customer_score
        total_agent = strong_agent_score + medium_agent_score
        
        # Determine speaker and confidence
        if total_customer > total_agent:
            confidence = min(0.9, 0.6 + (total_customer - total_agent) * 0.1)
            return "customer", confidence
        elif total_agent > total_customer:
            confidence = min(0.9, 0.6 + (total_agent - total_customer) * 0.1)
            return "agent", confidence
        else:
            # No clear winner - use alternation logic with low confidence
            if len(self.speaker_history) >= 2:
                last_speaker = self.speaker_history[-1]
                return "customer" if last_speaker == "agent" else "agent", 0.4
            return self.current_speaker, 0.3
    
    def get_stats(self) -> Dict:
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
        return {
            "type": "enhanced_mono_tracker",
            "turn_count": self.turn_count.copy(),
            "total_segments": len(self.speaker_history),
            "detection_method": "google_tags_plus_enhanced_content",
            "average_confidence": round(avg_confidence, 2),
            "transitions": len(self.speaker_transitions)
        }


# Update the original StereoSpeakerTracker to be compatible
class StereoSpeakerTracker(SpeakerTracker):
    """Legacy stereo speaker tracker - now redirects to StereoChannelTracker"""
    
    def __init__(self):
        super().__init__()
        self.channel_tracker = StereoChannelTracker()
        
    def detect_speaker(self, google_speaker_tag: Optional[int], transcript: str) -> str:
        """Redirect to stereo channel detection"""
        return self.channel_tracker.detect_speaker_stereo(None, google_speaker_tag, transcript)
    
    def detect_speaker_stereo(self, channel_tag: Optional[int], speaker_tag: Optional[int], transcript: str) -> str:
        """Use the new stereo channel tracker"""
        return self.channel_tracker.detect_speaker_stereo(channel_tag, speaker_tag, transcript)# backend/agents/audio_processor/agent.py - IMPROVED VERSION
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
    """Improved speaker tracking for agent-first conversations"""
    
    def __init__(self):
        self.speaker_history = []
        self.current_speaker = "agent"  # Agent always starts
        self.turn_count = {"agent": 0, "customer": 0}
        self.last_speaker_tag = None
        self.segment_count = 0
        self.speaker_transitions = []
        
    def detect_speaker(self, google_speaker_tag: Optional[int], transcript: str) -> str:
        """ENHANCED speaker detection with content-based fallback"""
        
        self.segment_count += 1
        
        # RULE 1: First segment is always agent
        if self.segment_count == 1:
            self.current_speaker = "agent"
            self.turn_count["agent"] += 1
            self.speaker_history.append("agent")
            self.last_speaker_tag = google_speaker_tag
            logger.info(f"🎯 First segment: agent (tag:{google_speaker_tag})")
            return "agent"
        
        # RULE 2: If Google provides speaker tags, use them
        if google_speaker_tag is not None:
            if self.last_speaker_tag is not None and google_speaker_tag != self.last_speaker_tag:
                # Speaker tag changed - switch speakers
                detected_speaker = "customer" if self.current_speaker == "agent" else "agent"
                
                logger.info(f"🔄 Google tag change: {self.last_speaker_tag} → {google_speaker_tag}, switching to {detected_speaker}")
                
                self.current_speaker = detected_speaker
                self.turn_count[detected_speaker] += 1
                self.last_speaker_tag = google_speaker_tag
                
                self.speaker_history.append(detected_speaker)
                return detected_speaker
            else:
                # Same tag or first tag - keep current or use content
                if self.last_speaker_tag is None:
                    self.last_speaker_tag = google_speaker_tag
                
                # Use content detection as backup even with same tag
                content_speaker = self._detect_by_content(transcript)
                if content_speaker != self.current_speaker:
                    logger.info(f"🔍 Content override: Google tag:{google_speaker_tag} but content suggests {content_speaker}")
                    self.current_speaker = content_speaker
                    self.turn_count[content_speaker] += 1
                
                self.speaker_history.append(self.current_speaker)
                return self.current_speaker
        else:
            # RULE 3: No Google tag - use content-based detection
            logger.info(f"🏷️ No Google tag for: '{transcript[:20]}...', using content detection")
            detected_speaker = self._detect_by_content(transcript)
            
            if detected_speaker != self.current_speaker:
                logger.info(f"🔍 Content detection switch: {self.current_speaker} → {detected_speaker}")
                self.current_speaker = detected_speaker
                self.turn_count[detected_speaker] += 1
            
            self.speaker_history.append(detected_speaker)
            return detected_speaker
    
    def _detect_by_content(self, transcript: str) -> str:
        """Detect speaker based on conversation content - ENHANCED"""
        
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
            # Short responses like "Hi", "Yes" are likely customer
            customer_score += 2
            
        if "williams" in transcript_lower or "patricia" in transcript_lower or "alex" in transcript_lower:
            # Names mentioned are often customer context
            customer_score += 1
            
        if any(phrase in transcript_lower for phrase in ["hsbc", "customer services", "can i take", "mrs williams"]):
            # Professional banking language is agent
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
                    # Same speaker for 2 segments, likely time to switch
                    return "customer" if self.current_speaker == "agent" else "agent"
            
            # Default: keep current speaker
            return self.current_speaker
    
    def get_stats(self) -> Dict:
        return {
            "current_speaker": self.current_speaker,
            "turn_count": self.turn_count.copy(),
            "total_segments": len(self.speaker_history),
            "transitions": len(self.speaker_transitions),
            "last_transition": self.speaker_transitions[-1] if self.speaker_transitions else None
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
            logger.info(f"🎙️ Starting streaming recognition for session {session_id}")
            
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not prepared")
            
            if not self.google_client:
                raise Exception("Google Speech-to-Text not available")
            
            # Start audio buffer
            audio_buffer = self.audio_buffers[session_id]
            if not audio_buffer.is_active:
                audio_buffer.start()
                logger.info(f"✅ Audio buffer started for {session_id}")
            
            # Update session
            session = self.active_sessions[session_id]
            session["status"] = "streaming"
            session["websocket_callback"] = websocket_callback
            
            # Start streaming task
            streaming_task = asyncio.create_task(
                self._streaming_with_restart(session_id)
            )
            self.streaming_tasks[session_id] = streaming_task
            
            logger.info(f"✅ Streaming task created for {session_id}")
            
            # Send confirmation
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
        """Streaming with automatic restart - BACK TO STABLE VERSION"""
        session = self.active_sessions[session_id]
        restart_interval = 240  # BACK TO 4 minutes - more stable
        
        while session_id in self.active_sessions and session.get("status") == "streaming":
            try:
                await asyncio.wait_for(
                    self._do_streaming_recognition(session_id),
                    timeout=restart_interval
                )
            except asyncio.TimeoutError:
                session["restart_count"] += 1
                logger.info(f"🔄 Restarting recognition for {session_id} after {restart_interval}s")
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"❌ Recognition error: {e}")
                session["restart_count"] += 1
                if session["restart_count"] > 3:
                    break
                await asyncio.sleep(2.0)
    
    async def _do_streaming_recognition(self, session_id: str):
        """Core streaming recognition logic with automatic mono/stereo detection"""
        session = self.active_sessions[session_id]
        websocket_callback = session["websocket_callback"]
        audio_buffer = self.audio_buffers[session_id]
        speaker_tracker = self.speaker_trackers[session_id]
        
        # Detect if we're processing stereo or mono
        is_stereo = isinstance(speaker_tracker, StereoSpeakerTracker)
        
        logger.info(f"🎙️ Starting Google STT streaming for {session_id} ({'STEREO' if is_stereo else 'MONO'} mode)")
        
        # Create recognition config based on audio type
        if is_stereo:
            recognition_config = self.speech_types.RecognitionConfig(
                encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="en-GB",
                audio_channel_count=2,  # STEREO: 2 channels
                enable_separate_recognition_per_channel=True,  # STEREO: Process channels separately
                model="phone_call",
                use_enhanced=True,
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                enable_word_time_offsets=True,
                # Optional diarization as backup
                diarization_config=self.speech_types.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=2,
                    max_speaker_count=2
                ),
                speech_contexts=[
                    self.speech_types.SpeechContext(
                        phrases=[
                            "HSBC", "customer services", "Patricia Williams", "Alex",
                            "urgent", "emergency", "Turkey", "Istanbul", "hospital",
                            "transfer", "pounds", "investment", "fraud", "scam"
                        ],
                        boost=15
                    )
                ]
            )
            logger.info("🔧 Using STEREO config: 2 channels, separate recognition")
        else:
            recognition_config = self.speech_types.RecognitionConfig(
                encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="en-GB",
                audio_channel_count=1,  # MONO: 1 channel
                model="phone_call",
                use_enhanced=True,
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                enable_word_time_offsets=True,
                # Strong diarization for mono
                diarization_config=self.speech_types.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=2,
                    max_speaker_count=2
                ),
                speech_contexts=[
                    self.speech_types.SpeechContext(
                        phrases=[
                            "HSBC", "customer services", "Patricia Williams", "Alex",
                            "urgent", "emergency", "Turkey", "Istanbul", "hospital",
                            "transfer", "pounds", "investment", "fraud", "scam"
                        ],
                        boost=15
                    )
                ]
            )
            logger.info("🔧 Using MONO config: 1 channel, diarization enabled")
        
        streaming_config = self.speech_types.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
            single_utterance=False
        )
        
        # Audio request generator with better chunk management
        def audio_generator():
            logger.info(f"🎵 Audio generator starting for {session_id} ({'STEREO' if is_stereo else 'MONO'})")
            chunk_count = 0
            empty_count = 0
            max_empty = 30
            
            for chunk in audio_buffer.get_chunks():
                if session_id not in self.active_sessions:
                    logger.info(f"🛑 Session {session_id} ended, stopping audio generator")
                    break
                
                if not chunk or len(chunk) == 0:
                    empty_count += 1
                    if empty_count >= max_empty:
                        logger.info(f"🎵 No more audio chunks, ending generator")
                        break
                    continue
                
                chunk_count += 1
                empty_count = 0
                
                if chunk_count <= 3 or chunk_count % 25 == 0:
                    logger.info(f"🎵 Sending {'stereo' if is_stereo else 'mono'} chunk {chunk_count} ({len(chunk)} bytes)")
                
                request = self.speech_types.StreamingRecognizeRequest()
                request.audio_content = chunk
                yield request
            
            logger.info(f"🎵 Audio generator completed: {chunk_count} chunks sent")
        
        # Start streaming with error handling
        try:
            logger.info(f"🚀 Starting Google STT v1p1beta1 streaming for {session_id}")
            
            responses = self.google_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator()
            )
            
            # Process responses with stereo/mono logic
            response_count = 0
            last_final_text = ""
            
            for response in responses:
                if session_id not in self.active_sessions:
                    logger.info(f"🛑 Session {session_id} ended, stopping response processing")
                    break
                
                response_count += 1
                if response_count <= 3 or response_count % 10 == 0:
                    logger.info(f"📨 Processing {'stereo' if is_stereo else 'mono'} response {response_count}")
                
                # Process with appropriate logic
                if is_stereo:
                    processed = await self._process_stereo_response(
                        session_id, response, speaker_tracker, websocket_callback, last_final_text
                    )
                else:
                    processed = await self._process_response_with_dedup(
                        session_id, response, speaker_tracker, websocket_callback, last_final_text
                    )
                
                if processed and processed.get("is_final"):
                    last_final_text = processed.get("text", "")
            
            logger.info(f"✅ Google STT streaming completed for {session_id}: {response_count} responses processed")
                
        except Exception as e:
            logger.error(f"❌ Streaming error for {session_id}: {e}")
            
            # Send error to client
            try:
                await websocket_callback({
                    "type": "streaming_error",
                    "data": {
                        "session_id": session_id,
                        "error": str(e),
                        "audio_type": "stereo" if is_stereo else "mono",
                        "timestamp": get_current_timestamp()
                    }
                })
            except:
                pass
            
            raise
    
    async def _process_response_with_dedup(
        self, 
        session_id: str, 
        response, 
        speaker_tracker: SpeakerTracker, 
        websocket_callback: Callable,
        last_final_text: str
    ) -> Optional[Dict]:
        """Process streaming response with MINIMAL filtering to preserve good segments"""
        try:
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                
                if not transcript:
                    continue
                
                # MINIMAL filtering - only skip very obvious duplicates
                if result.is_final and last_final_text:
                    if transcript == last_final_text:  # Exact match only
                        logger.debug(f"🚫 Skipping exact duplicate: '{transcript[:30]}...'")
                        continue
                
                # Extract speaker tag from words
                speaker_tag = None
                if hasattr(alternative, 'words') and alternative.words:
                    tags = [w.speaker_tag for w in alternative.words 
                           if hasattr(w, 'speaker_tag') and w.speaker_tag is not None]
                    if tags:
                        speaker_tag = max(set(tags), key=tags.count)
                
                # Detect speaker with improved logic
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
                    
                    logger.info(f"📝 FINAL [{detected_speaker}] (tag:{speaker_tag}): '{transcript[:50]}...'")
                    
                    # Trigger fraud analysis for customer speech
                    if detected_speaker == "customer":
                        await self._trigger_fraud_analysis(
                            session_id, transcript, websocket_callback
                        )
                    
                    return segment_data
                else:
                    # Send substantial interim results only
                    if len(transcript) > 5:  # Reduced threshold
                        await websocket_callback({
                            "type": "transcription_interim",
                            "data": segment_data
                        })
        
        except Exception as e:
            logger.error(f"❌ Error processing response: {e}")
        
        return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
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
        """Start real-time processing with FAST audio start and parallel transcription"""
        try:
            logger.info(f"🎵 Starting realtime processing: {audio_filename}")
            
            # Prepare session
            prepare_result = self._prepare_live_call({"session_id": session_id})
            if "error" in prepare_result:
                return prepare_result
            
            # Load audio file
            audio_path = Path(self.config["audio_base_path"]) / audio_filename
            if not audio_path.exists():
                return {"error": f"Audio file not found: {audio_filename}"}
            
            # Check audio properties
            with wave.open(str(audio_path), 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                duration = frames / sample_rate
                
                logger.info(f"📁 Audio: {duration:.1f}s, {sample_rate}Hz, {channels}ch")
            
            # FAST START: Start audio processing immediately (like before)
            logger.info("🚀 Starting audio immediately with parallel transcription")
            
            if channels == 2:
                audio_task = asyncio.create_task(
                    self._process_stereo_audio(session_id, audio_path, websocket_callback)
                )
            else:
                audio_task = asyncio.create_task(
                    self._process_mono_audio_fast_fixed(session_id, audio_path, websocket_callback)
                )
            
            # Store the task
            self.active_sessions[session_id]["audio_task"] = audio_task
            
            # Start transcription in parallel with small delay
            await asyncio.sleep(0.3)  # Reduced from 0.8s to 0.3s
            
            transcription_task = asyncio.create_task(
                self._start_transcription_parallel(session_id, websocket_callback)
            )
            self.active_sessions[session_id]["transcription_task"] = transcription_task
            
            return {
                "session_id": session_id,
                "audio_filename": audio_filename,
                "audio_duration": duration,
                "channels": channels,
                "processing_mode": f"{'stereo' if channels == 2 else 'mono'}_fast_parallel",
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting processing: {e}")
            return {"error": str(e)}
    
    async def _start_transcription_parallel(self, session_id: str, websocket_callback: Callable):
        """Start transcription in parallel with audio playback"""
        try:
            logger.info(f"🎙️ Starting parallel transcription for {session_id}")
            
            # Start streaming recognition
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
            logger.error(f"❌ Error in parallel transcription: {e}")
    
    async def _process_mono_audio_fast_fixed(self, session_id: str, audio_path: Path, websocket_callback: Callable):
        """Process mono audio with fast start and proper completion detection"""
        try:
            logger.info(f"🎵 Starting FAST mono audio processing for {session_id}")
            
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)
                audio_buffer = self.audio_buffers[session_id]
                
                if not audio_buffer.is_active:
                    audio_buffer.start()
                
                chunk_count = 0
                total_frames = wav_file.getnframes()
                total_chunks = total_frames // frames_per_chunk
                
                logger.info(f"🎵 Fast processing: {total_chunks} chunks total")
                
                while True:
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    if not audio_chunk:
                        logger.info(f"📁 Audio file completed: {chunk_count} chunks processed")
                        break
                    
                    audio_buffer.add_chunk(audio_chunk)
                    chunk_count += 1
                    
                    # Fast processing with minimal delay
                    await asyncio.sleep(0.02)  # 20ms delay only
                    
                    if chunk_count % 50 == 0:
                        progress = (chunk_count / total_chunks) * 100 if total_chunks > 0 else 0
                        logger.info(f"🎵 Fast audio: {progress:.1f}% complete")
                        
            logger.info(f"✅ Fast audio complete: {chunk_count}/{total_chunks} chunks")
            
            # Wait for processing to complete, then stop buffer
            await asyncio.sleep(6.0)
            audio_buffer.stop()
            logger.info(f"🛑 Audio buffer stopped after completion")
            
        except Exception as e:
            logger.error(f"❌ Error in fast mono audio: {e}")
            raise
    
    async def _process_mono_audio_realtime(self, session_id: str, audio_path: Path, websocket_callback: Callable):
        """Process mono audio with CONSERVATIVE real-time timing and proper completion"""
        try:
            logger.info(f"🎵 Starting CONSERVATIVE mono audio processing for {session_id}")
            
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)
                audio_buffer = self.audio_buffers[session_id]
                
                if not audio_buffer.is_active:
                    audio_buffer.start()
                
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                total_frames = wav_file.getnframes()
                total_chunks = total_frames // frames_per_chunk
                
                logger.info(f"🎵 Processing with REAL-TIME timing: {frames_per_chunk} frames per chunk, total: {total_chunks}")
                
                while True:
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    if not audio_chunk:
                        logger.info(f"📁 Reached end of audio file after {chunk_count} chunks")
                        break
                    
                    audio_buffer.add_chunk(audio_chunk)
                    chunk_count += 1
                    
                    # REAL-TIME timing - maintain proper chunk intervals
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = asyncio.get_event_loop().time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    
                    # Log every 25 chunks (5 seconds)
                    if chunk_count % 25 == 0:
                        elapsed = chunk_count * self.chunk_duration_ms / 1000
                        progress = (chunk_count / total_chunks) * 100 if total_chunks > 0 else 0
                        logger.info(f"🎵 Real-time audio: {elapsed:.1f}s ({progress:.1f}%), buffer: {audio_buffer.buffer.qsize()}")
                        
            logger.info(f"✅ Real-time mono audio complete: {chunk_count} chunks processed")
            
            # IMPORTANT: Signal completion by stopping the buffer after a delay
            await asyncio.sleep(8.0)  # Wait for final processing
            
            # Stop the audio buffer to signal completion
            audio_buffer.stop()
            logger.info(f"🛑 Audio buffer stopped - processing should complete soon")
            
        except Exception as e:
            logger.error(f"❌ Error in real-time mono audio: {e}")
            raise
    
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
            logger.info(f"🎵 Starting mono audio processing for {session_id}")
            
            with wave.open(str(audio_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)
                audio_buffer = self.audio_buffers[session_id]
                
                if not audio_buffer.is_active:
                    audio_buffer.start()
                    logger.info("✅ Audio buffer started for mono processing")
                
                chunk_count = 0
                start_time = asyncio.get_event_loop().time()
                
                logger.info(f"🎵 Will process audio in {frames_per_chunk} frame chunks")
                
                while True:
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    if not audio_chunk:
                        logger.info("📁 Reached end of audio file")
                        break
                    
                    audio_buffer.add_chunk(audio_chunk)
                    chunk_count += 1
                    
                    # Log progress every 25 chunks (5 seconds)
                    if chunk_count % 25 == 0:
                        elapsed = chunk_count * self.chunk_duration_ms / 1000
                        logger.info(f"🎵 Audio progress: {elapsed:.1f}s processed, buffer size: {audio_buffer.buffer.qsize()}")
                    
                    # Maintain real-time timing
                    expected_time = start_time + (chunk_count * self.chunk_duration_ms / 1000)
                    current_time = asyncio.get_event_loop().time()
                    wait_time = expected_time - current_time
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        
            logger.info(f"✅ Mono audio processing complete: {chunk_count} chunks processed")
            
            # Keep buffer active for a few more seconds to let processing catch up
            await asyncio.sleep(3.0)
            
        except Exception as e:
            logger.error(f"❌ Error processing mono audio: {e}")
            raise
    
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
