# Fixed sections for backend/agents/audio_processor/agent.py

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
        
        logger.info(f"🎯 Native STT mapping: Channel 0/2 (LEFT)=Customer, Channel 1/3 (RIGHT)=Agent")
        
    def detect_speaker_from_channel(self, channel: Optional[int]) -> str:
        """FIXED: Detect speaker from Google STT channel information"""
        self.segment_count += 1
        
        # RULE 1: First segment is ALWAYS agent (banking convention)
        if self.segment_count == 1:
            detected_speaker = "agent"
            logger.info(f"🎯 First segment: agent (channel:{channel})")
        
        # RULE 2: Use Google's channel information if available
        elif channel is not None:
            if channel in self.channel_mapping:
                detected_speaker = self.channel_mapping[channel]
                logger.info(f"🎯 Channel {channel} mapped to: {detected_speaker}")
            else:
                # Unknown channel - use alternation logic
                detected_speaker = "customer" if self.last_detected_speaker == "agent" else "agent"
                logger.warning(f"🎯 Unknown channel {channel}, alternating to: {detected_speaker}")
        
        # RULE 3: No channel info - use simple alternation
        else:
            # Simple alternation: agent -> customer -> agent -> customer
            detected_speaker = "customer" if self.last_detected_speaker == "agent" else "agent"
            logger.info(f"🎯 No channel info, alternating: {self.last_detected_speaker} -> {detected_speaker}")
        
        # Update tracking
        self.turn_count[detected_speaker] += 1
        self.last_detected_speaker = detected_speaker
        
        return detected_speaker

# FIXED: Enhanced response processing with better channel extraction
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
                logger.info(f"🔍 Found result.channel_tag: {channel_tag}")
            
            # Method 2: Extract from words (alternative approach)
            elif hasattr(alternative, 'words') and alternative.words:
                word_channels = []
                for word in alternative.words:
                    if hasattr(word, 'channel_tag') and word.channel_tag is not None:
                        word_channels.append(word.channel_tag)
                
                if word_channels:
                    # Use most common channel from words
                    channel_tag = max(set(word_channels), key=word_channels.count)
                    logger.info(f"🔍 Extracted from words, channel: {channel_tag}")
            
            # Method 3: Check if result has channel attribute
            elif hasattr(result, 'channel') and result.channel is not None:
                channel_tag = result.channel
                logger.info(f"🔍 Found result.channel: {channel_tag}")
            
            # Log what we found
            if channel_tag is not None:
                logger.info(f"🎯 Channel detected: {channel_tag} for text: '{transcript[:30]}...'")
            else:
                logger.warning(f"🎯 No channel info for text: '{transcript[:30]}...'")
            
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
                logger.info(f"📝 NATIVE [{detected_speaker.upper()}] {channel_display}: '{transcript[:50]}...'")
                
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

# ADDITIONAL: Debug method to inspect Google STT response structure
def _debug_google_response(self, response, transcript: str):
    """Debug helper to understand Google STT response structure"""
    try:
        for result in response.results:
            logger.info(f"🔍 DEBUG Response for '{transcript[:20]}...':")
            logger.info(f"  - result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
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

# ENHANCED: Recognition config with explicit channel settings
async def _do_native_stereo_recognition(self, session_id: str):
    """ENHANCED: Core recognition with NATIVE GOOGLE STT STEREO support"""
    session = self.active_sessions[session_id]
    websocket_callback = session["websocket_callback"]
    audio_buffer = self.audio_buffers[session_id]
    speaker_tracker = self.speaker_trackers[session_id]
    
    logger.info(f"🎙️ Starting NATIVE STEREO Google STT for {session_id}")
    
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
            
        logger.info(f"🎵 Native stereo generator sent {chunk_count} stereo chunks")
    
    try:
        logger.info(f"🚀 Starting Google STT with ENHANCED NATIVE STEREO for {session_id}")
        
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
                        if transcript:
                            self._debug_google_response(response, transcript)
            
            processed = await self._process_native_stereo_response(
                session_id, response, speaker_tracker, websocket_callback, last_final_text
            )
            
            if processed and processed.get("is_final"):
                last_final_text = processed.get("text", "")
        
        logger.info(f"✅ Native stereo streaming completed: {response_count} responses")
            
    except Exception as e:
        if "timeout" in str(e).lower():
            logger.warning(f"⚠️ Native stereo timeout: {e}")
        else:
            logger.error(f"❌ Native stereo streaming error: {e}")
        raise
