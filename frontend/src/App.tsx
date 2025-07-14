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
  Headphones
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
  immediate_alerts: string[];
  recommended_actions: string[];
  key_questions: string[];
  customer_education: string[];
  policy_id?: string;
  policy_title?: string;
  policy_version?: string;
  escalation_threshold?: number;
  regulatory_requirements?: string[];
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

// frontend/src/App.tsx - CHUNK 2: Main Component State and Functions (Simplified)

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
  
  // ===== REFS =====
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

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

// Replace the entire handleWebSocketMessage function with this enhanced version:

const handleWebSocketMessage = (message: WebSocketMessage): void => {
  console.log('📨 WebSocket message received:', message);
  
  switch (message.type) {
    case 'processing_started':
    case 'server_processing_started':
      setProcessingStage('🖥️ Server processing started');
      setServerProcessing(true);
      break;
      
    case 'streaming_started':
      setProcessingStage('🎙️ Audio streaming started');
      break;
      
    case 'transcription_segment':
      const transcriptData = message.data;
      
      const finalSegment: AudioSegment = {
        speaker: transcriptData.speaker,
        start: transcriptData.start || 0,
        duration: transcriptData.duration || 0,
        text: transcriptData.text,
        confidence: transcriptData.confidence,
        is_final: true,
        speaker_tag: transcriptData.speaker_tag,
        segment_id: `final-${Date.now()}-${Math.random()}`
      };
      
      setShowingSegments(prev => [...prev, finalSegment]);
      
      if (transcriptData.speaker === 'customer') {
        setTranscription(prev => {
          const customerText = prev + ' ' + transcriptData.text;
          return customerText.trim();
        });
      }
      
      setProcessingStage(`🎙️ ${transcriptData.speaker}: ${transcriptData.text.slice(0, 30)}...`);
      break;
      
    case 'transcription_interim':
      const interimData = message.data;
      
      const interimSegment: AudioSegment = {
        speaker: interimData.speaker,
        start: interimData.start || 0,
        duration: interimData.duration || 0,
        text: interimData.text,
        confidence: interimData.confidence,
        is_final: false,
        is_interim: true,
        speaker_tag: interimData.speaker_tag,
        segment_id: `interim-${Date.now()}-${Math.random()}`
      };
      
      setShowingSegments(prev => {
        const withoutLastInterim = prev.filter(s => 
          s.is_final || s.speaker !== interimData.speaker || !s.is_interim
        );
        return [...withoutLastInterim, interimSegment];
      });
      
      setProcessingStage(`🎙️ ${interimData.speaker} speaking...`);
      break;
      
    case 'progressive_fraud_analysis_trigger':
      console.log('🔍 Progressive fraud analysis triggered for customer speech');
      setProcessingStage('🔍 Analyzing customer speech for fraud patterns...');
      break;
      
    case 'fraud_analysis_started':
      setProcessingStage('🔍 Analyzing for fraud patterns...');
      break;
      
    case 'fraud_analysis_update':
      const analysis = message.data;
      
      // DEBUG: Log analysis data structure
      console.log('🔍 Fraud Analysis Update:');
      console.log('  - risk_score:', analysis.risk_score);
      console.log('  - detected_patterns:', analysis.detected_patterns);
      console.log('  - detected_patterns type:', typeof analysis.detected_patterns);
      console.log('  - detected_patterns keys:', Object.keys(analysis.detected_patterns || {}));        
     
      setRiskScore(analysis.risk_score || 0);
      setRiskLevel(analysis.risk_level || 'MINIMAL');
      setScamType(analysis.scam_type || 'unknown');
      setDetectedPatterns(analysis.detected_patterns || {});
      setProcessingStage(`🔍 Risk updated: ${analysis.risk_score}%`);
      break;
      
    case 'processing_progress':
      const progressData = message.data;
      console.log('📊 Processing progress:', progressData);
      setProcessingStage(`📊 Progress: ${progressData.progress || 0}% - ${progressData.chunks_processed || 0}/${progressData.total_chunks || 0} chunks`);
      break;
      
    case 'policy_analysis_started':
      setProcessingStage('📚 Retrieving policy guidance...');
      break;
      
    case 'policy_guidance_ready':
      const policyMessage = message.data;
      console.log('📚 Policy Guidance Ready - Enhanced Debug:');
      console.log('  - Full message data:', policyMessage);
      console.log('  - policy_guidance property:', policyMessage.policy_guidance);
      
      // Check if data is nested
      const actualPolicyData = policyMessage.policy_guidance || policyMessage;
      console.log('  - actual policy data:', actualPolicyData);
      console.log('  - immediate_alerts:', actualPolicyData?.immediate_alerts);
      console.log('  - recommended_actions:', actualPolicyData?.recommended_actions);
      
      setPolicyGuidance(actualPolicyData);
      setProcessingStage('📚 Policy guidance ready');
      break;
      
    case 'decision_made':
      const decisionData = message.data;
      console.log('🎯 Decision made:', decisionData);
      setProcessingStage(`🎯 Decision: ${decisionData.decision} (${decisionData.priority} priority)`);
      
      // Show decision in UI
      if (decisionData.decision === 'BLOCK_AND_ESCALATE') {
        setProcessingStage('🚨 CRITICAL: BLOCK TRANSACTION - Escalating to fraud team');
      } else if (decisionData.decision === 'VERIFY_AND_MONITOR') {
        setProcessingStage('⚠️ HIGH RISK: Enhanced verification required');
      }
      break;
      
    case 'analysis_complete':
      console.log('✅ Analysis pipeline complete');
      setProcessingStage('✅ Multi-agent analysis complete');
      
      // Show summary of what was completed
      const completionData = message.data;
      if (completionData.final_risk_score !== undefined) {
        console.log(`🎯 Final analysis: ${completionData.final_risk_score}% risk, ${completionData.agents_involved?.length || 0} agents involved`);
      }
      break;
      
    case 'ui_player_start':
      console.log('🎵 UI player start signal received');
      setProcessingStage('🎵 Audio player synchronized with server');
      break;
      
    case 'ui_player_stop':
      console.log('🎵 UI player stop signal received');
      setProcessingStage('🎵 Audio player stopped');
      break;
      
    case 'case_creation_started':
      setProcessingStage('📋 Creating fraud case...');
      break;
      
    case 'case_created':
      setProcessingStage(`📋 Case created: ${message.data.case_number || message.data.case_id}`);
      break;
      
    case 'case_creation_error':
      setProcessingStage(`❌ Case creation failed: ${message.data.error}`);
      break;
      
    case 'processing_complete':
      setProcessingStage('✅ Server processing complete');
      setServerProcessing(false);
      break;
      
    case 'processing_stopped':
      setProcessingStage('🛑 Processing stopped');
      setServerProcessing(false);
      break;
      
    case 'error':
      setProcessingStage(`❌ Error: ${message.data.error || message.data.message}`);
      setServerProcessing(false);
      console.error('Server error:', message.data);
      break;
      
    default:
      console.log('❓ Unknown message type:', message.type);
      // Don't update processing stage for unknown messages to avoid confusion
  }
};
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
      
      const newSessionId = `server_session_${Date.now()}`;
      setSessionId(newSessionId);
      setIsPlaying(true);
      setServerProcessing(true);
      setProcessingStage('🖥️ Starting server-side processing...');
      
      const message = {
        type: 'process_audio',
        data: {
          filename: audioFile.filename,
          session_id: newSessionId
        }
      };
      ws.send(JSON.stringify(message));
      
      const audio = new Audio(`${API_BASE_URL}/api/v1/audio/sample-files/${audioFile.filename}`);
      setAudioElement(audio);
      
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
        setProcessingStage('✅ Audio playback complete - server processing finishing');
      });
      
      audio.addEventListener('error', (e) => {
        console.error('❌ Audio playback error:', e);
        setProcessingStage('❌ Error playing audio file');
        setIsPlaying(false);
        stopServerProcessing();
      });
      
      await audio.play();
      
    } catch (error) {
      console.error('❌ Error starting server processing:', error);
      setProcessingStage(`❌ Failed to start processing: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setIsPlaying(false);
      setServerProcessing(false);
    }
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
    setProcessingStage('🛑 Processing stopped');
    
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }
  };

  // ===== UTILITY FUNCTIONS =====

  const resetState = (): void => {
    setCurrentTime(0);
    setTranscription('');
    setRiskScore(0);
    setRiskLevel('MINIMAL');
    setDetectedPatterns({});
    setPolicyGuidance(null);
    setProcessingStage('');
    setShowingSegments([]);
    setSessionId('');
    setServerProcessing(false);
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
        detected_patterns: detectedPatterns,
        recommended_actions: policyGuidance?.recommended_actions || [
          "Review customer interaction",
          "Verify transaction details",
          "Follow standard fraud procedures"
        ]
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
      setProcessingStage(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      alert(`Error creating ServiceNow case: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

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
            LIVE
          </span>
        </div>
      )}
    </button>
  );

  const ProcessingStatus = () => (
    <div className="flex items-center space-x-2 text-xs">
      {serverProcessing ? (
        <div className="flex items-center space-x-1 text-blue-600">
          <Server className="w-4 h-4 animate-pulse" />
          <span>Server Processing</span>
        </div>
      ) : (
        <div className="flex items-center space-x-1 text-gray-500">
          <Server className="w-4 h-4" />
          <span>Server Ready</span>
        </div>
      )}
    </div>
  );

  // Safe helper function to check arrays
  const safeArray = (arr: any): any[] => {
    if (Array.isArray(arr)) return arr;
    if (typeof arr === 'string') return [arr];
    return [];
  };
  
  // Safe helper to get array length
  const safeLength = (arr: any): number => {
    return safeArray(arr).length;
  };

 
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

  // ===== RENDER =====

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Simplified Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="flex justify-between items-center px-6 py-3">
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <img src="/hsbc-uk.svg" alt="HSBC" className="h-8 w-auto" />
            </div>
            <span className="text-lg font-medium text-gray-900">Agent Desktop</span>
          </div>
          
          <div className="flex items-center space-x-6 text-sm text-gray-600">
            <span>Agent: James Bond</span>
            <span>ID: JB2024</span>
            <span>Shift: 09:00-17:00</span>
          </div>
        </div>
      </header>

      {/* Simplified Demo Call Selection */}
      <div className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-gray-700">Demo Calls:</span>
          <div className="flex space-x-2">
            {audioFiles.map((audioFile) => (
              <DemoCallButton key={audioFile.id} audioFile={audioFile} />
            ))}
          </div>
          
          {isPlaying && selectedAudioFile && (
            <button
              onClick={stopServerProcessing}
              className="ml-4 px-3 py-2 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700"
            >
              Stop Processing
            </button>
          )}
          
          <div className="flex items-center space-x-2 ml-4">
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

      {/* Main Content Grid */}
      <div className="flex h-[calc(100vh-120px)]">
        
        {/* Left Sidebar - Simplified Customer Information */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          {/* Customer Information */}
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Customer Information</h3>
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

          {/* Call Controls */}
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Call Controls</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Call Duration:</span>
                <span className="font-medium">{formatTime(currentTime)}</span>
              </div>
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

          {/* Quick Actions */}
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Quick Actions</h3>
            <div className="space-y-2">
              <button className="w-full flex items-center space-x-2 px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 rounded">
                <User className="w-4 h-4" />
                <span>Verify Identity</span>
              </button>
              
              {/* ServiceNow Case Creation Button */}
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
                {riskScore >= 80 && (
                  <span className="ml-auto text-xs bg-red-100 text-red-700 px-2 py-1 rounded">
                    Auto
                  </span>
                )}
              </button>
              
              {/* High Risk Auto-Escalation */}
              {riskScore >= 80 && (
                <button 
                  onClick={createServiceNowCase}
                  className="w-full flex items-center space-x-2 px-3 py-2 text-left text-sm bg-red-100 text-red-700 rounded hover:bg-red-200"
                >
                  <AlertTriangle className="w-4 h-4" />
                  <span>ESCALATE to Fraud Team</span>
                </button>
              )}
            </div>
          </div>

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
              
              {/* Risk-based dynamic alert */}
              {riskScore > 60 && (
                <div className="mt-4 p-2 bg-red-50 border border-red-200 rounded">
                  <div className="flex justify-between">
                    <span className="text-red-800 font-medium">Live Fraud Alert:</span>
                    <span className="text-red-600">Now</span>
                  </div>
                  <div className="text-red-700 text-xs mt-1">
                    {riskScore >= 80 ? 'HIGH RISK - Immediate intervention required' :
                     'MEDIUM RISK - Enhanced monitoring active'}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        {/* Center - Live Transcription */}
        <div className="flex-1 bg-white border-r border-gray-200">
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Live Transcription - for demo only</h3>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-red-600 font-medium">Processing</span>
            </div>
          </div>
          
          <div className="p-4 h-full overflow-y-auto">
            {showingSegments.length === 0 ? (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <div className="text-center">
                  <Server className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p>Select a demo call to see live transcription</p>
                  {processingStage && (
                    <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded">
                      <p className="text-blue-700 text-sm">{processingStage}</p>
                      {processingStage.includes('ServiceNow case created') && (
                        <button 
                          onClick={() => setProcessingStage('')}
                          className="mt-1 text-xs text-blue-600 hover:text-blue-800"
                        >
                          Clear
                        </button>
                      )}
                    </div>  
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {showingSegments.map((segment, index) => {
                  const isInterim = !segment.is_final;
                  
                  return (
                    <div key={segment.segment_id || `${segment.speaker}-${index}-${isInterim ? 'interim' : 'final'}`} 
                         className={`animate-in slide-in-from-bottom duration-500 ${isInterim ? 'opacity-70' : ''}`}>
                      
                      {/* Speaker Label */}
                      <div className="flex items-start space-x-3">
                        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium border ${
                          getSpeakerColor(segment.speaker, isInterim)
                        }`}>
                          {getSpeakerIcon(segment.speaker)}
                          <span>{getSpeakerLabel(segment.speaker)}</span>
                        </div>
                        
                        {isInterim && (
                          <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-1 rounded animate-pulse">
                            typing...
                          </span>
                        )}
                      </div>
                      
                      {/* Text Display */}
                      <div className={`mt-2 ml-3 p-3 rounded-lg border-l-4 ${
                        isInterim 
                          ? 'bg-yellow-50 border-yellow-400' 
                          : 'bg-gray-50 border-blue-400'
                      }`}>
                        <p className={`text-sm text-gray-800 ${isInterim ? 'italic' : ''}`}>
                          {segment.text}
                          {isInterim && <span className="animate-pulse ml-1">|</span>}
                        </p>
                      </div>
                    </div>
                  );
                })}
                
                {serverProcessing && (
                  <div className="flex items-center space-x-2 text-gray-500 ml-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span className="text-sm">Server processing audio...</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Sidebar - Restored AI Agent Assist */}
        <div className="w-96 bg-white flex flex-col">
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">AI Agent Assist</h3>
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              ) : (
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              )}
              <span className="text-sm text-gray-600">Analysis</span>
            </div>
          </div>

          {/* Risk Score Display */}
          <div className="p-4 border-b border-gray-200">
            <div className="text-center">
              <div className="relative mx-auto w-24 h-24 mb-3">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 24 24">
                  <circle
                    cx="12"
                    cy="12"
                    r="10"
                    fill="none"
                    stroke="#e5e7eb"
                    strokeWidth="2"
                  />
                  <circle
                    cx="12"
                    cy="12"
                    r="10"
                    fill="none"
                    stroke={riskScore >= 80 ? '#dc2626' : riskScore >= 60 ? '#ea580c' : riskScore >= 40 ? '#ca8a04' : '#16a34a'}
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeDasharray={`${(riskScore / 100) * 62.83} 62.83`}
                    className="transition-all duration-500"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-xl font-bold text-gray-900">{riskScore}%</div>
                    <div className="text-xs text-gray-600">Risk</div>
                  </div>
                </div>
              </div>
              
              <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                riskScore >= 80 ? 'bg-red-100 text-red-800' :
                riskScore >= 60 ? 'bg-orange-100 text-orange-800' :
                riskScore >= 40 ? 'bg-yellow-100 text-yellow-800' :
                'bg-green-100 text-green-800'
              }`}>
                {riskScore >= 80 ? 'CRITICAL RISK' :
                 riskScore >= 60 ? 'HIGH RISK' :
                 riskScore >= 40 ? 'MEDIUM RISK' : 'LOW RISK'}
              </div>
            </div>
          </div>

          {/* Live Alerts */}
          {(riskScore >= 40 || Object.keys(detectedPatterns).length > 0) && (
            <div className="p-4 border-b border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <AlertCircle className="w-4 h-4 text-red-500 mr-2" />
                Live Alerts
              </h4>
              
              {riskScore >= 80 && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-3">
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-red-600" />
                    <span className="text-sm font-medium text-red-800">
                      {scamType === 'romance_scam' ? 'ROMANCE SCAM DETECTED' :
                       scamType === 'investment_scam' ? 'INVESTMENT SCAM DETECTED' :
                       scamType === 'impersonation_scam' ? 'IMPERSONATION SCAM DETECTED' :
                       'FRAUD DETECTED'}
                    </span>
                  </div>
                  <p className="text-xs text-red-700 mt-1">
                    Analysis: {Object.keys(detectedPatterns).join(', ') || 'Multiple fraud indicators detected'}
                  </p>
                </div>
              )}
              
              {riskScore >= 60 && riskScore < 80 && (
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 mb-3">
                  <div className="flex items-center space-x-2">
                    <AlertCircle className="w-4 h-4 text-orange-600" />
                    <span className="text-sm font-medium text-orange-800">
                      {scamType === 'romance_scam' ? 'EMOTIONAL MANIPULATION' :
                       scamType === 'investment_scam' ? 'URGENCY PRESSURE' :
                       'SUSPICIOUS ACTIVITY'}
                    </span>
                  </div>
                  <p className="text-xs text-orange-700 mt-1">
                    Server detected: {Object.keys(detectedPatterns).slice(0, 2).join(', ')} patterns  
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Recommended Actions */}
          {policyGuidance && safeLength(policyGuidance.recommended_actions) > 0 && (
            <div className="p-4 border-b border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                Recommendations
              </h4>
              
              <div className="space-y-2">
                {safeArray(policyGuidance.recommended_actions).slice(0, 4).map((action: string, index: number) => (
                  <div key={index} className="flex items-start space-x-3 p-2 bg-blue-50 rounded-lg">
                    <span className="flex-shrink-0 w-5 h-5 bg-blue-100 text-blue-800 rounded-full flex items-center justify-center text-xs font-medium">
                      {index + 1}
                    </span>
                    <span className="text-xs text-blue-900">{action}</span>
                  </div>
                ))}
                
                {riskScore >= 80 && (
                  <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-medium text-red-800">1. ALERT: STOP TRANSACTION</span>
                    </div>
                    <p className="text-xs text-red-700 mt-1">Detection alert - Do not process any transfers</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Policy Guidance */}
          {policyGuidance && policyGuidance.policy_id && (
            <div className="p-4 border-b border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <FileText className="w-4 h-4 text-blue-500 mr-2" />
                Policy Guidance
              </h4>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="text-xs font-medium text-blue-800 mb-2">
                  {policyGuidance.policy_title} ({policyGuidance.policy_id})
                </div>
                {policyGuidance.policy_version && (
                  <div className="text-xs text-blue-600 mb-2">Version: {policyGuidance.policy_version}</div>
                )}
                {safeLength(policyGuidance.customer_education) > 0 ? (
                  <ul className="text-xs text-blue-700 space-y-1">
                    {safeArray(policyGuidance.customer_education).slice(0, 4).map((point: string, index: number) => (
                      <li key={index}>• {point}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-blue-700">Policy guidance loading...</p>
                )}
              </div>
            </div>
          )}

          {/* Key Questions */}
          {policyGuidance && safeLength(policyGuidance.key_questions) > 0 && (
            <div className="p-4 border-b border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <Eye className="w-4 h-4 text-purple-500 mr-2" />
                Key Questions
              </h4>
              
              <div className="space-y-2">
                {safeArray(policyGuidance.key_questions).slice(0, 3).map((question: string, index: number) => (
                  <div key={index} className="p-2 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100 transition-colors cursor-pointer">
                    <p className="text-xs text-green-900 font-medium">"{question}"</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Customer Education */}
          {policyGuidance && safeLength(policyGuidance.customer_education) > 0 && (
            <div className="p-4 flex-1">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <Users className="w-4 h-4 text-green-500 mr-2" />
                Customer Education
              </h4>
              
              <div className="space-y-2">
                {safeArray(policyGuidance.customer_education).slice(0, 3).map((point: string, index: number) => (
                  <div key={index} className="p-2 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-xs text-yellow-900">"{point}"</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Default State */}
          {riskScore === 0 && Object.keys(detectedPatterns).length === 0 && !policyGuidance && (
            <div className="flex-1 flex items-center justify-center p-4">
              <div className="text-center text-gray-500">
                <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h4 className="text-sm font-medium text-gray-900 mb-2">AI Processing</h4>
                <p className="text-xs text-gray-600">
                  Start a demo call to see fraud detection
                </p>
                <div className="mt-4 space-y-2 text-xs">
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>6 Agents Ready</span>
                  </div>
                  <div className="text-gray-400">Audio • Scam • Policy • Decision • Summarization • Case • Orchestrator</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;        
