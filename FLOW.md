# Complete End-to-End ADK Fraud Detection Flow

## 🎵 **AUDIO PROCESSING PHASE** (Steps 1-6: UNCHANGED)

### **Step 1: Frontend Audio File Selection & Play**
**Code Location**: `frontend/src/App.tsx`
```tsx
// User clicks demo call button
const DemoCallButton = ({ audioFile }) => (
  <button onClick={() => !isPlaying ? startServerProcessing(audioFile) : null}>
    {audioFile.title}
  </button>
)

// Triggers audio processing
const startServerProcessing = async (audioFile: RealAudioFile) => {
  // Creates WebSocket message
  const message = {
    type: 'process_audio',
    data: {
      filename: audioFile.filename,
      session_id: newSessionId
    }
  };
  ws.send(JSON.stringify(message));  // ← WebSocket message sent
  
  // Start HTML5 audio playback
  await startAudioPlayback(audioFile);
}
```

### **Step 2: WebSocket Message Routing**
**Code Location**: `backend/app.py` → `backend/orchestrator/fraud_detection_orchestrator.py`
```python
# app.py WebSocket endpoint
@app.websocket("/ws/fraud-detection/{client_id}")
async def fraud_detection_websocket(websocket: WebSocket, client_id: str):
    # Receives message from frontend
    data = await websocket.receive_text()
    message_data = json.loads(data)
    
    # Delegates to WebSocket handler
    await fraud_ws_handler.handle_message(websocket, client_id, message_data)

# FraudDetectionWebSocketHandler (in orchestrator file)
async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict):
    if message_type == 'process_audio':
        await self.handle_audio_processing(websocket, client_id, data)

# Delegates to orchestrator
async def handle_audio_processing(self, websocket: WebSocket, client_id: str, data: Dict):
    result = await self.orchestrator.orchestrate_audio_processing(
        filename=filename,
        session_id=session_id,
        callback=websocket_callback  # ← Callback for WebSocket messages
    )
```

### **Step 3: Orchestrator Initializes Audio Processing**
**Code Location**: `backend/orchestrator/fraud_detection_orchestrator.py`
```python
class FraudDetectionOrchestrator:
    async def orchestrate_audio_processing(self, filename: str, session_id: str, callback: Callable):
        # Initialize session tracking
        await self._initialize_session(session_id, filename, callback)
        
        # Create orchestrator callback that will handle transcription
        audio_callback = self._create_audio_callback(session_id, callback)
        
        # Start audio processing with UNCHANGED audio processor
        result = await audio_processor_agent.start_realtime_processing(
            session_id=session_id,
            audio_filename=filename,
            websocket_callback=audio_callback  # ← Orchestrator's callback
        )

    def _create_audio_callback(self, session_id: str, external_callback: Callable):
        async def orchestrator_audio_callback(message_data: Dict) -> None:
            # Forward to WebSocket (frontend)
            if external_callback:
                await external_callback(message_data)  # ← To frontend
            
            # Handle orchestration logic
            await self._handle_audio_message(session_id, message_data)  # ← Trigger agents
        
        return orchestrator_audio_callback
```

### **Step 4: Audio Processor - Google STT Processing**
**Code Location**: `backend/agents/audio_processor/agent.py` (UNCHANGED)
```python
class AudioProcessorAgent:
    async def start_realtime_processing(self, session_id: str, audio_filename: str, websocket_callback: Callable):
        # Load audio file
        audio_path = Path(self.config["audio_base_path"]) / audio_filename
        
        # Start Google STT transcription
        transcription_task = asyncio.create_task(
            self._start_native_stereo_transcription(session_id, websocket_callback)
        )
        
        # Start audio file processing
        audio_task = asyncio.create_task(
            self._process_native_stereo_audio(session_id, audio_path, websocket_callback)
        )

    async def _do_native_stereo_recognition(self, session_id: str):
        # UNCHANGED Google STT v1p1beta1 native stereo processing
        recognition_config = self.speech_types.RecognitionConfig(
            encoding=self.speech_types.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code="en-GB",
            audio_channel_count=2,  # Stereo
            enable_separate_recognition_per_channel=True,  # Separate channels
            model="phone_call"
        )
        
        # Process stereo chunks through Google STT
        responses = self.google_client.streaming_recognize(
            config=streaming_config,
            requests=stereo_audio_generator()  # ← Stereo chunks
        )
        
        for response in responses:
            await self._process_native_stereo_response(
                session_id, response, speaker_tracker, websocket_callback
            )

    async def _process_native_stereo_response(self, session_id: str, response, speaker_tracker, websocket_callback):
        # Extract channel information and detect speaker
        channel_tag = result.channel_tag  # Google STT channel info
        detected_speaker = speaker_tracker.detect_speaker_from_channel(channel_tag)
        
        # Create transcription segment
        segment_data = {
            "session_id": session_id,
            "speaker": detected_speaker,  # "agent" or "customer"
            "text": transcript,
            "confidence": confidence,
            "is_final": result.is_final,
            "native_stereo": True,
            "start": start_time,
            "end": end_time
        }
        
        # Send to orchestrator callback
        await websocket_callback({
            "type": "transcription_segment",
            "data": segment_data
        })
```

### **Step 5: WebSocket Sends Transcription to Frontend**
**Code Location**: Orchestrator callback → WebSocket → Frontend
```python
# Orchestrator callback (step 3)
async def orchestrator_audio_callback(message_data: Dict):
    # Forward to frontend via WebSocket
    if external_callback:
        await external_callback(message_data)  # ← This sends to frontend
    
    # Also trigger orchestration
    await self._handle_audio_message(session_id, message_data)
```

### **Step 6: Frontend Displays Live Transcription**
**Code Location**: `frontend/src/App.tsx`
```tsx
// WebSocket message handler
const handleWebSocketMessage = (message: WebSocketMessage): void => {
  switch (message.type) {
    case 'transcription_segment':
      handleTranscriptionSegment(message.data);  // ← Display transcription
      break;
    case 'transcription_interim':
      handleInterimTranscription(message.data);
      break;
  }
};

const handleTranscriptionSegment = (data: any) => {
  // Create final transcription segment
  const finalSegment: AudioSegment = {
    speaker: data.speaker,  // "agent" or "customer"
    start: data.start,
    text: data.text,
    confidence: data.confidence,
    is_final: true
  };
  
  // Add to display
  setShowingSegments(prev => [...prev, finalSegment]);
  
  // Update customer text for analysis
  if (data.speaker === 'customer') {
    setTranscription(prev => prev + ' ' + data.text);
  }
};
```

---

## 🤖 **FRAUD ANALYSIS PHASE** (Step 7+: NEW ADK Orchestration)

### **Step 7: Orchestrator Receives Customer Speech**
**Code Location**: `backend/orchestrator/fraud_detection_orchestrator.py`
```python
async def _handle_audio_message(self, session_id: str, message_data: Dict):
    message_type = message_data.get('type')
    data = message_data.get('data', {})
    
    if message_type == 'transcription_segment':
        speaker = data.get('speaker')
        text = data.get('text', '')
        
        # Only process customer speech for fraud analysis
        if speaker == 'customer' and text.strip():
            await self._process_customer_speech(session_id, text, data)  # ← Trigger pipeline

async def _process_customer_speech(self, session_id: str, customer_text: str, segment_data: Dict):
    # Add to speech buffer
    self.customer_speech_buffer[session_id].append({
        'text': customer_text,
        'timestamp': segment_data.get('start', 0),
        'confidence': segment_data.get('confidence', 0.0)
    })
    
    # Accumulate customer speech
    accumulated_speech = " ".join([
        segment['text'] for segment in self.customer_speech_buffer[session_id]
    ])
    
    # Trigger agent pipeline when enough text
    if len(accumulated_speech.strip()) >= 15:
        await self._run_agent_pipeline(session_id, accumulated_speech)  # ← Start ADK agents
```

### **Step 8: ADK Agent Pipeline Execution**
**Code Location**: `backend/orchestrator/fraud_detection_orchestrator.py`
```python
async def _run_agent_pipeline(self, session_id: str, customer_text: str, callback: Optional[Callable] = None):
    # STEP 8A: Scam Detection Agent
    logger.info("🤖 Orchestrator: Running Scam Detection Agent")
    await self._notify_agent_start('scam_detection', session_id, callback)  # ← Notify frontend
    
    scam_analysis = await self._run_scam_detection(session_id, customer_text)
    parsed_analysis = await self._parse_and_accumulate_patterns(session_id, scam_analysis)
    
    await self._notify_agent_complete('scam_detection', session_id, parsed_analysis, callback)
    
    risk_score = parsed_analysis.get('risk_score', 0)
    
    # STEP 8B: Policy Guidance Agent (if medium+ risk)
    if risk_score >= 40:
        logger.info("📚 Orchestrator: Running Policy Guidance Agent")
        await self._notify_agent_start('policy_guidance', session_id, callback)
        
        policy_result = await self._run_policy_guidance(session_id, parsed_analysis)
        session_data['policy_guidance'] = policy_result
        
        await self._notify_agent_complete('policy_guidance', session_id, policy_result, callback)
    
    # STEP 8C: Decision Agent (if high risk)
    if risk_score >= 60:
        logger.info("🎯 Orchestrator: Running Decision Agent")
        await self._notify_agent_start('decision', session_id, callback)
        
        decision_result = await self._run_decision_agent(session_id, parsed_analysis)
        session_data['decision_result'] = decision_result
        
        await self._notify_agent_complete('decision', session_id, decision_result, callback)
    
    # STEP 8D: Case Management (if required)
    if risk_score >= 50:
        logger.info("📋 Orchestrator: Running Case Management Agent")
        await self._run_case_management(session_id, parsed_analysis, callback)

async def _run_scam_detection(self, session_id: str, customer_text: str) -> str:
    # Call ADK Scam Detection Agent
    context = {
        'session_id': session_id,
        'customer_profile': session_data.get('customer_profile', {}),
        'call_duration': call_duration
    }
    
    prompt = f"""
    FRAUD ANALYSIS REQUEST:
    CUSTOMER SPEECH: "{customer_text}"
    CONTEXT: {json.dumps(context, indent=2)}
    """
    
    return await self.scam_detection_agent.run(prompt, context)  # ← ADK Agent call
```

### **Step 9: Individual ADK Agents Processing**

#### **9A: Scam Detection Agent**
**Code Location**: `backend/agents/scam_detection/adk_agent.py`
```python
# ADK Agent created with Google Gemini
def create_scam_detection_agent() -> Agent:
    return Agent(
        name="HSBC Scam Detection Agent",
        model="gemini-1.5-flash",
        instruction=SCAM_DETECTION_INSTRUCTION,  # ← Detailed fraud analysis prompt
        tools=[fraud_pattern_analyzer, risk_calculator]
    )

# Agent processes customer text and returns structured analysis
# Output: risk_score, scam_type, detected_patterns, confidence, etc.
```

#### **9B: Policy Guidance Agent**
**Code Location**: `backend/agents/policy_guidance/adk_agent.py`
```python
def create_policy_guidance_agent() -> Agent:
    return Agent(
        name="HSBC Policy Guidance Agent",
        model="gemini-1.5-pro",
        instruction=POLICY_GUIDANCE_INSTRUCTION,  # ← Policy lookup and guidance
        tools=[policy_database_search, escalation_procedures]
    )

# Agent returns: immediate_alerts, recommended_actions, key_questions, escalation_threshold
```

#### **9C: Decision Agent**
**Code Location**: `backend/agents/decision/adk_agent.py`
```python
def create_decision_agent() -> Agent:
    return Agent(
        name="HSBC Decision Agent",
        model="gemini-1.5-pro", 
        instruction=DECISION_AGENT_INSTRUCTION,  # ← Final decision making
        tools=[agent_coordinator, decision_engine]
    )

# Agent returns: decision, reasoning, priority, actions, case_creation, escalation_path
```

### **Step 10: Case Management Agent Integration**
**Code Location**: `backend/orchestrator/fraud_detection_orchestrator.py` → `backend/agents/case_management/adk_agent.py`
```python
async def _run_case_management(self, session_id: str, analysis: Dict, callback: Callable):
    await self._notify_agent_start('case_management', session_id, callback)
    
    # Prepare incident data for ADK Case Management Agent
    incident_data = {
        'customer_name': session_data.get('customer_profile', {}).get('name', 'Unknown'),
        'customer_account': session_data.get('customer_profile', {}).get('account', 'Unknown'),
        'risk_score': analysis.get('risk_score', 0),
        'scam_type': analysis.get('scam_type', 'unknown'),
        'session_id': session_id,
        'detected_patterns': self.accumulated_patterns.get(session_id, {}),
        'auto_created': True,
        'creation_method': 'orchestrator_auto_creation'
    }
    
    # Call ADK Case Management Agent
    case_result = await self.case_management_agent.create_fraud_incident(
        incident_data=incident_data,
        context={'session_id': session_id}
    )
    
    session_data['case_info'] = case_result
    await self._notify_agent_complete('case_management', session_id, case_result, callback)

# In Case Management Agent:
class PureADKCaseManagementAgent:
    async def create_fraud_incident(self, incident_data: Dict, context: Dict) -> Dict:
        # Prepare comprehensive prompt for ADK agent
        incident_prompt = self._prepare_incident_prompt(incident_data, context)
        
        # Run ADK agent to structure the incident
        adk_result = await self._execute_adk_agent(incident_prompt)
        
        # Parse ADK result into ServiceNow structure
        parsed_incident = self._parse_incident_result(adk_result)
        
        # Create actual ServiceNow incident
        if self.servicenow_service:
            servicenow_result = await self._create_servicenow_incident(parsed_incident, incident_data)
        else:
            servicenow_result = await self._create_mock_incident(parsed_incident, incident_data)
        
        return servicenow_result
```

### **Step 11: ServiceNow Integration**
**Code Location**: `backend/services/servicenow_service.py`
```python
class ServiceNowService:
    async def create_incident(self, incident_data: Dict) -> Dict:
        # Use Basic Authentication to ServiceNow
        credentials = f"{self.username}:{self.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {encoded_credentials}'
        }
        
        session = aiohttp.ClientSession(headers=headers)
        url = f"{self.instance_url}/api/now/table/incident"
        
        # Create incident in ServiceNow
        async with session.post(url, json=incident_data) as response:
            if response.status == 201:
                result = await response.json()
                return {
                    "success": True,
                    "incident_number": result["result"]["number"],
                    "incident_sys_id": result["result"]["sys_id"],
                    "incident_url": f"{self.instance_url}/nav_to.do?uri=incident.do?sys_id={result['result']['sys_id']}"
                }
```

### **Step 12: Frontend Receives Analysis Updates**
**Code Location**: `frontend/src/App.tsx`
```tsx
// Frontend receives WebSocket messages during agent processing
const handleWebSocketMessage = (message: WebSocketMessage): void => {
  switch (message.type) {
    case 'fraud_analysis_update':
      handleFraudAnalysisUpdate(message.data);  // ← Update risk score, patterns
      break;
    case 'policy_guidance_ready':
      handlePolicyGuidanceReady(message.data);   // ← Display policy guidance
      break;
    case 'adk_orchestrator_decision':
      handleOrchestratorDecision(message.data);  // ← Show final decision
      break;
    case 'auto_incident_created':
      handleAutoIncidentCreated(message.data);   // ← Show case creation
      break;
  }
};

const handleFraudAnalysisUpdate = (data: any) => {
  setRiskScore(data.risk_score || 0);
  setRiskLevel(data.risk_level || 'MINIMAL');
  setScamType(data.scam_type || 'unknown');
  setDetectedPatterns(data.detected_patterns || {});
  // Update UI with risk information
};

const handleAutoIncidentCreated = (data: any) => {
  setAutoIncidentCreated(data.incident_number);
  // Show confirmation dialog with ServiceNow link
  if (window.confirm(`ADK Auto-Incident Created!\nIncident: ${data.incident_number}\nOpen in ServiceNow?`)) {
    window.open(data.incident_url, '_blank');
  }
};
```

---

## 🔄 **COMPLETE FLOW SUMMARY**

### **Audio Processing (Steps 1-6): UNCHANGED**
1. **Frontend** → Audio file selection & WebSocket message
2. **WebSocket Handler** → Route message to orchestrator  
3. **Orchestrator** → Initialize session, delegate to audio processor
4. **Audio Processor** → Google STT native stereo processing (UNCHANGED)
5. **Google STT** → Real-time transcription via WebSocket
6. **Frontend** → Live transcription display (UNCHANGED)

### **Fraud Analysis (Steps 7-12): NEW ADK Architecture**
7. **Orchestrator** → Receive customer speech, accumulate text
8. **ADK Pipeline** → Sequential agent execution:
   - 8A: **Scam Detection Agent** → Risk scoring & pattern detection
   - 8B: **Policy Guidance Agent** → Procedural guidance (if risk ≥ 40%)
   - 8C: **Decision Agent** → Final decision making (if risk ≥ 60%)
   - 8D: **Case Management Agent** → ServiceNow preparation (if risk ≥ 50%)
9. **ADK Agents** → Individual Google Gemini-powered analysis
10. **Case Management** → ADK-structured ServiceNow incident creation
11. **ServiceNow API** → Real incident creation with Basic Auth
12. **Frontend Updates** → Real-time display of analysis results

### **Key Architecture Points**
- **Audio processing**: Completely unchanged, still uses Google STT v1p1beta1
- **Orchestration**: New centralized coordination via `FraudDetectionOrchestrator`
- **Agent pipeline**: Sequential ADK agent execution based on risk thresholds
- **Case management**: Professional ServiceNow integration with ADK structuring
- **Frontend**: Same UI, enhanced with ADK analysis results display

The **audio core remains identical** while the **fraud analysis is now sophisticated multi-agent ADK coordination**! 🎵🤖
