// frontend/src/App.tsx - CHUNK 1: Imports and Types

import React, { useState, useEffect, useRef } from 'react';
import { 
  AlertCircle, 
  CheckCircle, 
  Upload, 
  Mic, 
  Shield, 
  FileText, 
  Users, 
  Play, 
  Pause, 
  Volume2,
  PhoneCall,
  TrendingUp,
  AlertTriangle,
  Eye,
  Brain,
  Wifi,
  WifiOff,
  Phone,
  User,
  Clock,
  Hash,
  Settings,
  Server,
  UserCheck,
  Headphones,
  Bot,
  Zap,
  BookOpen
} from 'lucide-react';

// Environment configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

// ===== TYPE DEFINITIONS =====

interface AudioSegment {
  speaker: 'agent' | 'customer';
  start: number;
  duration: number;
  text: string;
  confidence?: number;
  is_final?: boolean;
  is_interim?: boolean;
  speaker_tag?: number;
  segment_id?: string;
}

interface RealAudioFile {
  id: string;
  filename: string;
  title: string;
  description: string;
  icon: string;
  file_path: string;
  size_bytes: number;
  duration: number;
  scam_type?: string;
}

interface DetectedPattern {
  pattern_name: string;
  matches: string[];
  count: number;
  weight: number;
  severity: 'critical' | 'high' | 'medium';
}

interface PolicyGuidance {
  policy_id: string;
  policy_title: string;
  policy_version: string;
  immediate_alerts: string[];
  recommended_actions: string[];
  key_questions: string[];
  customer_education: string[];
  escalation_threshold: number;
  rag_powered?: boolean;
}

interface WebSocketMessage {
  type: string;
  data: any;
}

interface CustomerProfile {
  name: string;
  account: string;
  status: string;
  segment: string;
  riskProfile: string;
  recentActivity: {
    lastCall: string;
    description: string;
    additionalInfo?: string;
  };
  demographics: {
    age: number;
    location: string;
    relationship: string;
  };
  alerts: {
    type: string;
    date: string;
    description: string;
  }[];
}

interface ADKProcessingState {
  scamAnalysisActive: boolean;
  policyAnalysisActive: boolean;
  summarizationActive: boolean;
  orchestratorActive: boolean;
  currentAgent: string;
  processingSteps: string[];
}

// frontend/src/App.tsx - CHUNK 2: Configuration Data

// ===== CONFIGURATION DATA =====

const customerProfiles: Record<string, CustomerProfile> = {
  'romance_scam_1.wav': {
    name: "Mrs Patricia Williams",
    account: "****3847",
    status: "Active",
    segment: "Premier Banking",
    riskProfile: "Medium",
    recentActivity: {
      lastCall: "3 weeks ago",
      description: "Balance inquiry and international transfer questions",
      additionalInfo: "Asked about sending money overseas multiple times"
    },
    demographics: {
      age: 67,
      location: "Bournemouth, UK",
      relationship: "Widow"
    },
    alerts: [
      {
        type: "Unusual Activity",
        date: "2 weeks ago",
        description: "Attempted £3,000 transfer to Turkey - blocked by fraud filters"
      }
    ]
  },
  'investment_scam_live_call.wav': {
    name: "Mr David Chen",
    account: "****5691",
    status: "Active", 
    segment: "Business Banking",
    riskProfile: "High",
    recentActivity: {
      lastCall: "5 days ago",
      description: "Investment account setup inquiry",
      additionalInfo: "Mentioned guaranteed returns opportunity"
    },
    demographics: {
      age: 42,
      location: "Manchester, UK",
      relationship: "Married"
    },
    alerts: [
      {
        type: "Investment Warning",
        date: "1 week ago", 
        description: "Customer inquired about transferring £25,000 for 'guaranteed 15% monthly returns'"
      }
    ]
  },
  'impersonation_scam_live_call.wav': {
    name: "Ms Sarah Thompson",
    account: "****7234",
    status: "Active",
    segment: "Personal Banking", 
    riskProfile: "Low",
    recentActivity: {
      lastCall: "1 day ago",
      description: "Reported suspicious call claiming to be from HSBC security",
      additionalInfo: "Customer was asked for PIN - correctly refused"
    },
    demographics: {
      age: 34,
      location: "Birmingham, UK",
      relationship: "Single"
    },
    alerts: [
      {
        type: "Security Alert",
        date: "Yesterday",
        description: "Customer reported impersonation attempt - no account compromise detected"
      }
    ]
  }
};

const defaultCustomerProfile: CustomerProfile = {
  name: "Customer",
  account: "****0000",
  status: "Active",
  segment: "Personal Banking", 
  riskProfile: "Unknown",
  recentActivity: {
    lastCall: "N/A",
    description: "No recent activity"
  },
  demographics: {
    age: 0,
    location: "Unknown",
    relationship: "Unknown"
  },
  alerts: []
};

// frontend/src/App.tsx - CHUNK 3: Utility Functions

// ===== UTILITY FUNCTIONS =====

const getSpeakerIcon = (speaker: string) => {
  if (speaker === 'customer') {
    return <User className="w-3 h-3 text-blue-600" />;
  } else {
    return <UserCheck className="w-3 h-3 text-green-600" />;
  }
};

const getSpeakerLabel = (speaker: string) => {
  return speaker === 'customer' ? 'Customer' : 'Agent';
};

const getSpeakerColor = (speaker: string, isInterim: boolean = false) => {
  if (speaker === 'customer') {
    return isInterim ? 'bg-blue-50 text-blue-700 border-blue-200' : 'bg-blue-100 text-blue-800 border-blue-300';
  } else {
    return isInterim ? 'bg-green-50 text-green-700 border-green-200' : 'bg-green-100 text-green-800 border-green-300';
  }
};

const getRiskColor = (riskScore: number): string => {
  if (riskScore >= 80) return 'bg-red-100 text-red-800 border-red-300';
  if (riskScore >= 60) return 'bg-orange-100 text-orange-800 border-orange-300';
  if (riskScore >= 40) return 'bg-yellow-100 text-yellow-800 border-yellow-300';
  return 'bg-green-100 text-green-800 border-green-300';
};

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

// frontend/src/App.tsx - CHUNK 4: Component State

// ===== MAIN COMPONENT =====

function App() {
  // ===== CORE STATE =====
  const [selectedAudioFile, setSelectedAudioFile] = useState<RealAudioFile | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  
  // ===== ANALYSIS STATE =====
  const [transcription, setTranscription] = useState<string>('');
  const [riskScore, setRiskScore] = useState<number>(0);
  const [riskLevel, setRiskLevel] = useState<string>('MINIMAL');
  const [detectedPatterns, setDetectedPatterns] = useState<Record<string, DetectedPattern>>({});
  const [policyGuidance, setPolicyGuidance] = useState<PolicyGuidance | null>(null);
  const [processingStage, setProcessingStage] = useState<string>('');
  const [showingSegments, setShowingSegments] = useState<AudioSegment[]>([]);
  const [scamType, setScamType] = useState<string>('unknown');
  const [currentCustomer, setCurrentCustomer] = useState<CustomerProfile>(defaultCustomerProfile);
  
  // ===== BACKEND INTEGRATION STATE =====
  const [audioFiles, setAudioFiles] = useState<RealAudioFile[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState<boolean>(true);
  const [sessionId, setSessionId] = useState<string>('');
  
  // ===== WEBSOCKET STATE =====
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [serverProcessing, setServerProcessing] = useState<boolean>(false);
  
  // ===== ADK STATE =====
  const [adkProcessing, setAdkProcessing] = useState<ADKProcessingState>({
    scamAnalysisActive: false,
    policyAnalysisActive: false,
    summarizationActive: false,
    orchestratorActive: false,
    currentAgent: '',
    processingSteps: []
  });
  const [autoIncidentCreated, setAutoIncidentCreated] = useState<string | null>(null);
  const [accumulatedPatterns, setAccumulatedPatterns] = useState<Record<string, DetectedPattern>>({});
  const [analysisHistory, setAnalysisHistory] = useState<any[]>([]);
  
  // ===== REFS =====
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

// frontend/src/App.tsx - CHUNK 5: WebSocket Functions

  // ===== WEBSOCKET FUNCTIONS =====

  const connectWebSocket = (): void => {
    try {
      const websocket = new WebSocket(`${WS_BASE_URL}/ws/fraud-detection/agent-${Date.now()}`);
      
      websocket.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setWs(websocket);
        
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
      };
      
      websocket.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error);
        }
      };
      
      websocket.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setWs(null);
        
        if (event.code !== 1000) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('🔄 Attempting to reconnect WebSocket...');
            connectWebSocket();
          }, 3000);
        }
      };
      
      websocket.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };
      
    } catch (error) {
      console.error('❌ Error creating WebSocket connection:', error);
    }
  };

// frontend/src/App.tsx - CHUNK 6: Message Handling Part 1

  const handleWebSocketMessage = (message: WebSocketMessage): void => {
    console.log('📨 WebSocket message received:', message);
    
    switch (message.type) {
      case 'processing_started':
        setProcessingStage('🖥️ ADK Server processing started');
        setServerProcessing(true);
        resetADKState();
        break;
        
      case 'streaming_started':
        setProcessingStage('🎙️ Audio streaming started');
        break;
        
      // ===== TRANSCRIPTION HANDLING =====
      case 'transcription_segment':
        handleTranscriptionSegment(message.data);
        break;
        
      case 'transcription_interim':
        handleInterimTranscription(message.data);
        break;
        
      // ===== ADK AGENT MESSAGES =====
      case 'adk_scam_analysis_started':
        setAdkProcessing(prev => ({
          ...prev,
          scamAnalysisActive: true,
          currentAgent: 'ADK Scam Detection Agent',
          processingSteps: [...prev.processingSteps, '🤖 ADK Scam Detection started']
        }));
        setProcessingStage('🤖 ADK Scam Detection analyzing...');
        break;
        
      case 'fraud_analysis_update':
        handleFraudAnalysisUpdate(message.data);
        break;
        
      case 'rag_policy_analysis_started':
        setAdkProcessing(prev => ({
          ...prev,
          policyAnalysisActive: true,
          currentAgent: 'RAG Policy System',
          processingSteps: [...prev.processingSteps, '📚 RAG Policy retrieval started']
        }));
        setProcessingStage('📚 RAG Policy System analyzing...');
        break;
        
      case 'policy_guidance_ready':
        handlePolicyGuidanceReady(message.data);
        break;
		    
// frontend/src/App.tsx - CHUNK 7: Message Handling Part 2

      case 'adk_orchestrator_started':
        setAdkProcessing(prev => ({
          ...prev,
          orchestratorActive: true,
          currentAgent: 'ADK Orchestrator Agent',
          processingSteps: [...prev.processingSteps, '🎭 ADK Orchestrator deciding']
        }));
        setProcessingStage('🎭 ADK Orchestrator making decision...');
        break;
        
      case 'adk_orchestrator_decision':
        setAdkProcessing(prev => ({
          ...prev,
          orchestratorActive: false,
          processingSteps: [...prev.processingSteps, '✅ ADK Orchestrator decision complete']
        }));
        setProcessingStage('✅ ADK Orchestrator decision ready');
        break;
        
      case 'adk_summarization_started':
        setAdkProcessing(prev => ({
          ...prev,
          summarizationActive: true,
          currentAgent: 'ADK Summarization Agent',
          processingSteps: [...prev.processingSteps, '📝 ADK Summarization started']
        }));
        setProcessingStage('📝 ADK creating incident summary...');
        break;
        
      case 'adk_summarization_complete':
        setAdkProcessing(prev => ({
          ...prev,
          summarizationActive: false,
          processingSteps: [...prev.processingSteps, '✅ ADK Summary complete']
        }));
        setProcessingStage('✅ ADK incident summary ready');
        break;
        
      case 'auto_incident_creation_started':
        setProcessingStage('📋 ADK auto-creating incident...');
        break;
        
      case 'auto_incident_created':
        handleAutoIncidentCreated(message.data);
        break;
        
      case 'processing_complete':
        setProcessingStage('✅ ADK processing complete');
        setServerProcessing(false);
        setAdkProcessing(prev => ({
          ...prev,
          scamAnalysisActive: false,
          policyAnalysisActive: false,
          orchestratorActive: false,
          processingSteps: [...prev.processingSteps, '🏁 Processing complete']
        }));
        break;
        
      case 'processing_stopped':
        setProcessingStage('🛑 Processing stopped');
        setServerProcessing(false);
        resetADKState();
        break;
        
      case 'error':
        setProcessingStage(`❌ Error: ${message.data.error || message.data.message}`);
        setServerProcessing(false);
        resetADKState();
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  };

// frontend/src/App.tsx - CHUNK 8: Message Handlers

  // ===== MESSAGE HANDLERS =====

  const handleTranscriptionSegment = (data: any) => {
    const finalSegment: AudioSegment = {
      speaker: data.speaker,
      start: data.start || 0,
      duration: data.duration || 0,
      text: data.text,
      confidence: data.confidence,
      is_final: true,
      speaker_tag: data.speaker_tag,
      segment_id: `final-${Date.now()}-${Math.random()}`
    };
    
    setShowingSegments(prev => [...prev, finalSegment]);
    
    if (data.speaker === 'customer') {
      setTranscription(prev => {
        const customerText = prev + ' ' + data.text;
        return customerText.trim();
      });
    }
    
    setProcessingStage(`🎙️ ${data.speaker}: ${data.text.slice(0, 30)}...`);
  };

  const handleInterimTranscription = (data: any) => {
    const interimSegment: AudioSegment = {
      speaker: data.speaker,
      start: data.start || 0,
      duration: data.duration || 0,
      text: data.text,
      confidence: data.confidence,
      is_final: false,
      is_interim: true,
      speaker_tag: data.speaker_tag,
      segment_id: `interim-${Date.now()}-${Math.random()}`
    };
    
    setShowingSegments(prev => {
      const withoutLastInterim = prev.filter(s => 
        s.is_final || s.speaker !== data.speaker || !s.is_interim
      );
      return [...withoutLastInterim, interimSegment];
    });
    
    setProcessingStage(`🎙️ ${data.speaker} speaking...`);
  };

  const handleFraudAnalysisUpdate = (data: any) => {
    setRiskScore(data.risk_score || 0);
    setRiskLevel(data.risk_level || 'MINIMAL');
    setScamType(data.scam_type || 'unknown');
    
    const patterns = data.detected_patterns || {};
    setDetectedPatterns(patterns);
    setAccumulatedPatterns(patterns);
    
    setAnalysisHistory(prev => [...prev, {
      timestamp: data.timestamp,
      risk_score: data.risk_score,
      patterns: patterns,
      analysis_number: data.analysis_number,
      adk_powered: data.adk_powered
    }]);
    
    setAdkProcessing(prev => ({
      ...prev,
      scamAnalysisActive: false,
      processingSteps: [...prev.processingSteps, `✅ ADK Analysis: ${data.risk_score}% risk`]
    }));
    
    setProcessingStage(`🤖 ADK Risk: ${data.risk_score}% (${data.pattern_count || 0} patterns)`);
  };

  const handlePolicyGuidanceReady = (data: any) => {
    setPolicyGuidance({
      policy_id: data.policy_id,
      policy_title: data.policy_title,
      policy_version: data.policy_version,
      immediate_alerts: data.immediate_alerts || [],
      recommended_actions: data.recommended_actions || [],
      key_questions: data.key_questions || [],
      customer_education: data.customer_education || [],
      escalation_threshold: data.escalation_threshold,
      rag_powered: data.rag_powered
    });
    
    setAdkProcessing(prev => ({
      ...prev,
      policyAnalysisActive: false,
      processingSteps: [...prev.processingSteps, '✅ RAG Policy guidance ready']
    }));
    
    setProcessingStage('📚 RAG Policy guidance ready');
  };

  const handleAutoIncidentCreated = (data: any) => {
    setAutoIncidentCreated(data.incident_number);
    setProcessingStage(`✅ ADK Auto-incident: ${data.incident_number}`);
    
    setAdkProcessing(prev => ({
      ...prev,
      processingSteps: [...prev.processingSteps, `📋 Auto-incident: ${data.incident_number}`]
    }));
    
    if (window.confirm(
      `ADK Auto-Incident Created!\n\n` +
      `Incident: ${data.incident_number}\n` +
      `Risk Score: ${data.risk_score}%\n` +
      `Method: ADK Agents\n\n` +
      `Open in ServiceNow?`
    )) {
      window.open(data.incident_url, '_blank');
    }
  };

// frontend/src/App.tsx - CHUNK 9: Audio Processing Functions

  // ===== AUDIO PROCESSING FUNCTIONS =====

  const startServerProcessing = async (audioFile: RealAudioFile): Promise<void> => {
    try {
      if (!ws || !isConnected) {
        setProcessingStage('🔌 WebSocket not connected - attempting to connect...');
        connectWebSocket();
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        if (!ws || !isConnected) {
          throw new Error('Could not establish WebSocket connection');
        }
      }
      
      if (selectedAudioFile?.id !== audioFile.id) {
        setSelectedAudioFile(audioFile);
        const customerProfile = customerProfiles[audioFile.filename] || defaultCustomerProfile;
        setCurrentCustomer(customerProfile);
        resetState();
      }
      
      const newSessionId = `adk_session_${Date.now()}`;
      setSessionId(newSessionId);
      setIsPlaying(true);
      setServerProcessing(true);
      setProcessingStage('🖥️ Starting ADK server processing...');
      
      // Send WebSocket message
      const message = {
        type: 'process_audio',
        data: {
          filename: audioFile.filename,
          session_id: newSessionId
        }
      };
      ws.send(JSON.stringify(message));
      
      // Start audio playback with enhanced loading
      await startAudioPlayback(audioFile);
      
    } catch (error) {
      console.error('❌ Error starting ADK processing:', error);
      setProcessingStage(`❌ Failed to start ADK processing: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setIsPlaying(false);
      setServerProcessing(false);
      resetADKState();
    }
  };

  const startAudioPlayback = async (audioFile: RealAudioFile): Promise<void> => {
    const audio = new Audio(`${API_BASE_URL}/api/v1/audio/sample-files/${audioFile.filename}`);
    setAudioElement(audio);
    
    // Enhanced audio event listeners
    audio.addEventListener('timeupdate', () => {
      setCurrentTime(audio.currentTime);
    });
    
    audio.addEventListener('play', () => {
      setIsPlaying(true);
    });
    
    audio.addEventListener('pause', () => {
      setIsPlaying(false);
    });
    
    audio.addEventListener('ended', () => {
      setIsPlaying(false);
      setProcessingStage('✅ Audio complete - ADK processing finishing');
    });
    
    audio.addEventListener('error', (e) => {
      console.error('❌ Audio playback error:', e);
      setProcessingStage('❌ Error playing audio file');
      setIsPlaying(false);
      stopServerProcessing();
    });
    
    // Enhanced audio loading with pre-loading
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Audio load timeout')), 5000);
      
      audio.addEventListener('canplaythrough', () => {
        clearTimeout(timeout);
        resolve(true);
      }, { once: true });
      
      audio.addEventListener('error', () => {
        clearTimeout(timeout);
        reject(new Error('Audio load failed'));
      }, { once: true });
      
      audio.load();
    });
    
    // Start playback with sync delay
    setTimeout(async () => {
      try {
        await audio.play();
        setIsPlaying(true);
        setProcessingStage('🎵 Audio + ADK processing active');
      } catch (error) {
        console.error('❌ Audio play failed:', error);
        setProcessingStage('❌ Failed to start audio playback');
        stopServerProcessing();
      }
    }, 100);
  };

  const stopServerProcessing = (): void => {
    if (ws && isConnected && sessionId) {
      const message = {
        type: 'stop_processing',
        data: {
          session_id: sessionId
        }
      };
      ws.send(JSON.stringify(message));
    }
    
    setIsPlaying(false);
    setServerProcessing(false);
    setProcessingStage('🛑 ADK processing stopped');
    resetADKState();
    
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }
  };

// frontend/src/App.tsx - CHUNK 10: Utility and Service Functions

  // ===== UTILITY FUNCTIONS =====

  const resetState = (): void => {
    setCurrentTime(0);
    setTranscription('');
    setRiskScore(0);
    setRiskLevel('MINIMAL');
    setDetectedPatterns({});
    setAccumulatedPatterns({});
    setAnalysisHistory([]);
    setPolicyGuidance(null);
    setProcessingStage('');
    setShowingSegments([]);
    setSessionId('');
    setServerProcessing(false);
    setAutoIncidentCreated(null);
    resetADKState();
  };

  const resetADKState = (): void => {
    setAdkProcessing({
      scamAnalysisActive: false,
      policyAnalysisActive: false,
      summarizationActive: false,
      orchestratorActive: false,
      currentAgent: '',
      processingSteps: []
    });
  };

  const loadAudioFiles = async (): Promise<void> => {
    try {
      setIsLoadingFiles(true);
      const response = await fetch(`${API_BASE_URL}/api/v1/audio/sample-files`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setAudioFiles(data.files);
      } else {
        console.error('Failed to load audio files:', data);
      }
    } catch (error) {
      console.error('Error loading audio files:', error);
    } finally {
      setIsLoadingFiles(false);
    }
  };

  const createServiceNowCase = async () => {
    try {
      setProcessingStage('📋 Creating ServiceNow case...');
      
      const caseData = {
        customer_name: currentCustomer.name,
        customer_account: currentCustomer.account,
        customer_phone: currentCustomer.demographics?.location || 'Unknown',
        risk_score: riskScore,
        scam_type: scamType,
        session_id: sessionId,
        confidence_score: riskScore / 100,
        transcript_segments: showingSegments.map(segment => ({
          speaker: segment.speaker,
          start: segment.start,
          text: segment.text,
          confidence: segment.confidence
        })),
        detected_patterns: accumulatedPatterns,
        recommended_actions: policyGuidance?.recommended_actions || [
          "Review customer interaction",
          "Verify transaction details", 
          "Follow standard fraud procedures"
        ],
        adk_powered: true
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/cases/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(caseData),
      });

      const result = await response.json();

      if (response.ok && result.status === 'success') {
        const caseInfo = result.data;
        setProcessingStage(`✅ ServiceNow case created: ${caseInfo.case_number}`);
        
        const successMessage = `
ServiceNow Case Created Successfully!

Case Number: ${caseInfo.case_number}
Priority: ${caseInfo.priority}
Risk Score: ${riskScore}%
Scam Type: ${scamType.replace('_', ' ').toUpperCase()}
Method: ADK Enhanced

Click OK to open the case in ServiceNow.
        `.trim();
        
        if (window.confirm(successMessage)) {
          window.open(caseInfo.case_url, '_blank');
        }
        
      } else {
        const errorMessage = result.error || 'Failed to create case';
        setProcessingStage(`❌ Error: ${errorMessage}`);
        alert(`Failed to create ServiceNow case: ${errorMessage}`);
      }
      
    } catch (error) {
      console.error('Error creating ServiceNow case:', error);
      setProcessingStage(`❌ Error: ${error instanceof Error ? error.me// frontend/src/App.tsx - CHUNK 10: Utility and Service Functions

  // ===== UTILITY FUNCTIONS =====

// frontend/src/App.tsx - CHUNK 11: Component Functions

  // ===== COMPONENT FUNCTIONS =====

  const DemoCallButton = ({ audioFile }: { audioFile: RealAudioFile }) => (
    <button
      onClick={() => !isPlaying ? startServerProcessing(audioFile) : null}
      disabled={isPlaying}
      className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm transition-colors ${
        selectedAudioFile?.id === audioFile.id 
          ? 'bg-blue-100 text-blue-800 border-2 border-blue-300' 
          : 'bg-gray-50 text-gray-700 hover:bg-gray-100 border-2 border-transparent'
      } ${isPlaying ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
    >
      <span className="text-lg">{audioFile.icon}</span>
      {!isPlaying ? (
        <Play className="w-4 h-4" />
      ) : selectedAudioFile?.id === audioFile.id ? (
        <Pause className="w-4 h-4" />
      ) : (
        <Play className="w-4 h-4" />
      )}
      <span className="font-medium">{audioFile.title}</span>
      {selectedAudioFile?.id === audioFile.id && isPlaying && (
        <div className="flex items-center space-x-1">
          <span className="text-xs bg-red-500 text-white px-2 py-1 rounded-full">
            ADK LIVE
          </span>
        </div>
      )}
    </button>
  );

  const ADKProcessingIndicator = () => (
    <div className="flex items-center space-x-2 text-xs">
      {serverProcessing ? (
        <div className="flex items-center space-x-1 text-purple-600">
          <Bot className="w-4 h-4 animate-pulse" />
          <span>ADK Processing</span>
          {adkProcessing.currentAgent && (
            <span className="text-purple-500">({adkProcessing.currentAgent})</span>
          )}
        </div>
      ) : (
        <div className="flex items-center space-x-1 text-gray-500">
          <Bot className="w-4 h-4" />
          <span>ADK Ready</span>
        </div>
      )}
    </div>
  );

  // ===== EFFECTS =====

  useEffect(() => {
    loadAudioFiles();
  }, []);

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (ws) {
        ws.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

// frontend/src/App.tsx - CHUNK 12: JSX Header and Demo Section

  // ===== RENDER =====

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Enhanced Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="flex justify-between items-center px-6 py-3">
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <img src="/hsbc-uk.svg" alt="HSBC" className="h-8 w-auto" />
            </div>
            <span className="text-lg font-medium text-gray-900">Agent Desktop</span>
            <div className="flex items-center space-x-1 text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
              <Bot className="w-3 h-3" />
              <span>ADK Enhanced</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-6 text-sm text-gray-600">
            <span>Agent: James Bond</span>
            <span>ID: JB2024</span>
            <span>Shift: 09:00-17:00</span>
            <div className="flex items-center space-x-2 text-sm">
              <span className="text-gray-600">ServiceNow:</span>
              <span className="text-green-600">Ready</span>
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            </div>
            <ADKProcessingIndicator />
            <Settings className="w-4 h-4" />
          </div>
        </div>
      </header>

      {/* Enhanced Demo Call Selection */}
      <div className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-gray-700">ADK Demo Calls:</span>
            <div className="flex space-x-2">
              {audioFiles.map((audioFile) => (
                <DemoCallButton key={audioFile.id} audioFile={audioFile} />
              ))}
            </div>
            
            {isPlaying && selectedAudioFile && (
              <button
                onClick={stopServerProcessing}
                className="ml-4 px-3 py-2 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700 flex items-center space-x-1"
              >
                <AlertTriangle className="w-4 h-4" />
                <span>Stop ADK Processing</span>
              </button>
            )}
          </div>
          
          <div className="flex items-center space-x-4">
            {/* ADK Processing Steps */}
            {adkProcessing.processingSteps.length > 0 && (
              <div className="flex items-center space-x-2 text-xs">
                <span className="text-purple-600">ADK Steps:</span>
                <span className="text-purple-500">{adkProcessing.processingSteps.length}</span>
              </div>
            )}
            
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <div className="flex items-center space-x-1 text-green-600">
                  <Wifi className="w-4 h-4" />
                  <span className="text-xs">Connected</span>
                </div>
              ) : (
                <div className="flex items-center space-x-1 text-red-600">
                  <WifiOff className="w-4 h-4" />
                  <span className="text-xs">Disconnected</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
// frontend/src/App.tsx - CHUNK 13: Left Sidebar Customer Info

      {/* Main Content Grid */}
      <div className="flex h-[calc(100vh-120px)]">
        
        {/* Left Sidebar - Enhanced Customer Information */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          {/* Customer Information */}
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
              <User className="w-4 h-4 mr-2" />
              Customer Information
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Name:</span>
                <span className="font-medium">{currentCustomer.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Account:</span>
                <span className="font-medium">{currentCustomer.account}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Status:</span>
                <span className="text-green-600 font-medium">{currentCustomer.status}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Segment:</span>
                <span className="font-medium">{currentCustomer.segment}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Risk Profile:</span>
                <span className={`font-medium ${
                  currentCustomer.riskProfile === 'High' ? 'text-red-600' :
                  currentCustomer.riskProfile === 'Medium' ? 'text-yellow-600' :
                  'text-green-600'
                }`}>
                  {currentCustomer.riskProfile}
                </span>
              </div>
            </div>
          </div>

          {/* Enhanced Call Controls */}
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
              <PhoneCall className="w-4 h-4 mr-2" />
              Call Controls
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Call Duration:</span>
                <span className="font-medium">{formatTime(currentTime)}</span>
              </div>
              {selectedAudioFile && (
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Audio File:</span>
                  <span className="font-medium text-xs">{selectedAudioFile.filename}</span>
                </div>
              )}
              <div className="flex space-x-2">
                <button className="flex-1 flex items-center justify-center space-x-1 px-3 py-2 bg-yellow-100 text-yellow-700 rounded text-sm hover:bg-yellow-200">
                  <Volume2 className="w-4 h-4" />
                  <span>Hold</span>
                </button>
                <button className="flex-1 flex items-center justify-center space-x-1 px-3 py-2 bg-blue-100 text-blue-700 rounded text-sm hover:bg-blue-200">
                  <Users className="w-4 h-4" />
                  <span>Transfer</span>
                </button>
              </div>
              <div className="flex space-x-2 mt-2">
                <button className="flex-1 flex items-center justify-center space-x-1 px-3 py-2 bg-green-100 text-green-700 rounded text-sm hover:bg-green-200">
                  <PhoneCall className="w-4 h-4" />
                  <span>Mute</span>
                </button>
                <button className="flex-1 flex items-center justify-center space-x-1 px-3 py-2 bg-red-100 text-red-700 rounded text-sm hover:bg-red-200">
                  <Phone className="w-4 h-4" />
                  <span>Hang Up</span>
                </button>
              </div>
            </div>
          </div>
// frontend/src/App.tsx - CHUNK 14: Left Sidebar Quick Actions and ADK Status

          {/* Enhanced Quick Actions */}
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
              <Zap className="w-4 h-4 mr-2" />
              Quick Actions
            </h3>
            <div className="space-y-2">
              <button className="w-full flex items-center space-x-2 px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 rounded">
                <User className="w-4 h-4" />
                <span>Verify Identity</span>
              </button>
              
              {/* Enhanced ServiceNow Case Creation */}
              <button 
                onClick={createServiceNowCase}
                disabled={riskScore < 40}
                className={`w-full flex items-center space-x-2 px-3 py-2 text-left text-sm rounded ${
                  riskScore < 40 
                    ? 'text-gray-400 cursor-not-allowed' 
                    : 'text-gray-700 hover:bg-gray-50'
                }`}
              >
                <FileText className="w-4 h-4" />
                <span>Create ServiceNow Case</span>
                {riskScore >= 50 && (
                  <span className="ml-auto text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                    Manual
                  </span>
                )}
              </button>
              
              {/* Auto-Created Incident Display */}
              {autoIncidentCreated && (
                <div className="w-full flex items-center space-x-2 px-3 py-2 text-left text-sm bg-green-50 text-green-700 rounded">
                  <CheckCircle className="w-4 h-4" />
                  <span>ADK Auto-Incident: {autoIncidentCreated}</span>
                  <button 
                    onClick={() => {
                      const url = `https://dev303197.service-now.com/incident/${autoIncidentCreated}`;
                      window.open(url, '_blank');
                    }}
                    className="ml-auto text-xs text-green-600 hover:text-green-800"
                  >
                    View
                  </button>
                </div>
              )}
              
              {/* Critical Risk Auto-Escalation */}
              {riskScore >= 80 && (
                <button 
                  onClick={createServiceNowCase}
                  className="w-full flex items-center space-x-2 px-3 py-2 text-left text-sm bg-red-100 text-red-700 rounded hover:bg-red-200"
                >
                  <AlertTriangle className="w-4 h-4" />
                  <span>ESCALATE to Fraud Team</span>
                  <span className="ml-auto text-xs bg-red-200 text-red-800 px-2 py-1 rounded">
                    URGENT
                  </span>
                </button>
              )}
            </div>
          </div>

          {/* ADK Processing Status */}
          {adkProcessing.processingSteps.length > 0 && (
            <div className="p-4 border-b border-gray-200">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <Bot className="w-4 h-4 mr-2 text-purple-600" />
                ADK Processing Steps
              </h3>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {adkProcessing.processingSteps.slice(-5).map((step, index) => (
                  <div key={index} className="text-xs p-2 bg-purple-50 rounded text-purple-700">
                    {step}
                  </div>
                ))}
              </div>
              {adkProcessing.currentAgent && (
                <div className="mt-2 text-xs text-purple-600 font-medium">
                  Active: {adkProcessing.currentAgent}
                </div>
              )}
            </div>
          )}
// frontend/src/App.tsx - CHUNK 15: Left Sidebar Recent Activity and Center Transcription

          {/* Recent Activity */}
          <div className="p-4 flex-1">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Recent Activity</h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-600">Last Call:</span>
                <span>{currentCustomer.recentActivity.lastCall}</span>
              </div>
              <div className="text-gray-600">{currentCustomer.recentActivity.description}</div>
              
              {currentCustomer.recentActivity.additionalInfo && (
                <div className="text-blue-600 bg-blue-50 p-2 rounded text-xs mt-2">
                  <strong>Note:</strong> {currentCustomer.recentActivity.additionalInfo}
                </div>
              )}
              
              {/* Customer-specific alerts */}
              {currentCustomer.alerts.length > 0 && currentCustomer.alerts.map((alert, index) => (
                <div key={index} className="mt-4 p-2 bg-orange-50 border border-orange-200 rounded">
                  <div className="flex justify-between">
                    <span className="text-orange-800 font-medium">{alert.type}:</span>
                    <span className="text-orange-600">{alert.date}</span>
                  </div>
                  <div className="text-orange-700 text-xs mt-1">{alert.description}</div>
                </div>
              ))}
              
              {/* ADK Live Fraud Alert */}
              {riskScore > 60 && (
                <div className="mt-4 p-2 bg-red-50 border border-red-200 rounded">
                  <div className="flex justify-between">
                    <span className="text-red-800 font-medium flex items-center">
                      <Bot className="w-3 h-3 mr-1" />
                      ADK Fraud Alert:
                    </span>
                    <span className="text-red-600">Live</span>
                  </div>
                  <div className="text-red-700 text-xs mt-1">
                    {riskScore >= 80 ? 'CRITICAL RISK - ADK immediate intervention required' :
                     'HIGH RISK - ADK enhanced monitoring active'}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Center - Enhanced Live Transcription */}
        <div className="flex-1 bg-white border-r border-gray-200">
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <Headphones className="w-5 h-5 mr-2" />
              Live Transcription
              {serverProcessing && (
                <span className="ml-2 text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                  ADK Processing
                </span>
              )}
            </h3>
            <div className="flex items-center space-x-4">
              {/* Analysis Progress */}
              {analysisHistory.length > 0 && (
                <div className="text-xs text-gray-600">
                  ADK Analyses: {analysisHistory.length}
                </div>
              )}
              
              {/* Processing Status */}
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-red-600 font-medium">
                  {serverProcessing ? 'ADK Processing' : 'Ready'}
                </span>
              </div>
            </div>
          </div>


// frontend/src/App.tsx - CHUNK 16: Transcription Content

          <div className="p-4 h-full overflow-y-auto">
            {showingSegments.length === 0 ? (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <div className="text-center">
                  <Bot className="w-12 h-12 text-purple-400 mx-auto mb-4" />
                  <p className="text-lg font-medium text-gray-700">ADK Enhanced Processing</p>
                  <p className="text-sm text-gray-500 mt-1">Select a demo call to see live transcription with Google ADK agents</p>
                  {processingStage && (
                    <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <p className="text-blue-700 text-sm font-medium">{processingStage}</p>
                      {processingStage.includes('ServiceNow case created') && (
                        <button 
                          onClick={() => setProcessingStage('')}
                          className="mt-2 text-xs text-blue-600 hover:text-blue-800 underline"
                        >
                          Clear
                        </button>
                      )}
                    </div>  
                  )}
                  
                  {/* ADK System Status */}
                  <div className="mt-4 text-xs text-gray-400">
                    <div className="flex items-center justify-center space-x-2 mb-2">
                      <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                      <span>4 ADK Agents Ready</span>
                    </div>
                    <div>Scam Detection • RAG Policy • Summarization • Orchestrator</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {showingSegments.map((segment, index) => {
                  const isInterim = !segment.is_final;
                  
                  return (
                    <div key={segment.segment_id || `${segment.speaker}-${index}-${isInterim ? 'interim' : 'final'}`} 
                         className={`animate-in slide-in-from-bottom duration-500 ${isInterim ? 'opacity-70' : ''}`}>
                      
                      {/* Enhanced Speaker Label */}
                      <div className="flex items-start space-x-3">
                        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium border ${
                          getSpeakerColor(segment.speaker, isInterim)
                        }`}>
                          {getSpeakerIcon(segment.speaker)}
                          <span>{getSpeakerLabel(segment.speaker)}</span>
                          {segment.confidence && (
                            <span className="text-xs opacity-75">
                              {Math.round(segment.confidence * 100)}%
                            </span>
                          )}
                        </div>
                        
                        {isInterim && (
                          <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-1 rounded animate-pulse">
                            typing...
                          </span>
                        )}
                        
                        {/* ADK Processing Indicator */}
                        {segment.speaker === 'customer' && !isInterim && adkProcessing.scamAnalysisActive && (
                          <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded flex items-center">
                            <Bot className="w-3 h-3 mr-1" />
                            ADK analyzing...
                          </span>
                        )}
                      </div>
                      
                      {/* Enhanced Text Display */}
                      <div className={`mt-2 ml-3 p-3 rounded-lg border-l-4 ${
                        isInterim 
                          ? 'bg-yellow-50 border-yellow-400' 
                          : segment.speaker === 'customer'
                          ?		
