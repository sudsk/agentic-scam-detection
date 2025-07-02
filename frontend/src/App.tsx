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
  Server
} from 'lucide-react';

import HSBCLogo from '/hsbc-uk.svg';

// Environment configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

// Type definitions
interface AudioSegment {
  speaker: 'agent' | 'customer';
  start: number;
  duration: number;
  text: string;
  confidence?: number;
  is_final?: boolean;      // NEW: Track if this is a final result
  is_interim?: boolean;    // NEW: Track if this is an interim result
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
}

interface WebSocketMessage {
  type: string;
  data: any;
}

// Utility functions
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

function App() {
  // Core state
  const [selectedAudioFile, setSelectedAudioFile] = useState<RealAudioFile | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  
  // Analysis state
  const [transcription, setTranscription] = useState<string>('');
  const [riskScore, setRiskScore] = useState<number>(0);
  const [riskLevel, setRiskLevel] = useState<string>('MINIMAL');
  const [detectedPatterns, setDetectedPatterns] = useState<Record<string, DetectedPattern>>({});
  const [policyGuidance, setPolicyGuidance] = useState<PolicyGuidance | null>(null);
  const [processingStage, setProcessingStage] = useState<string>('');
  const [showingSegments, setShowingSegments] = useState<AudioSegment[]>([]);
  
  // Backend integration state
  const [audioFiles, setAudioFiles] = useState<RealAudioFile[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState<boolean>(true);
  const [sessionId, setSessionId] = useState<string>('');
  
  // WebSocket state
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [serverProcessing, setServerProcessing] = useState<boolean>(false);
  
  // Refs
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Mock customer data
  const customerData = {
    name: "Michael Thompson",
    account: "****5678",
    status: "Active",
    segment: "Personal Banking",
    riskProfile: "Medium"
  };

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

  const handleWebSocketMessage = (message: WebSocketMessage): void => {
    console.log('📨 WebSocket message received:', message);
    
    switch (message.type) {
      case 'server_processing_started':
        setProcessingStage('🖥️ Server processing started');
        setServerProcessing(true);
        break;
        
      // FINAL transcription segments - add as new segments
      case 'transcription_segment':
        const transcriptData = message.data;
        
        // Add final transcription segment
        setShowingSegments(prev => [...prev, {
          speaker: transcriptData.speaker,
          start: transcriptData.start,
          duration: transcriptData.duration,
          text: transcriptData.text,
          confidence: transcriptData.confidence,
          is_final: true  // Mark as final
        }]);
        
        // Update full transcription text with final results only
        if (transcriptData.is_final) {
          setTranscription(prev => {
            // Get all final segments
            const finalSegments = [...showingSegments.filter(s => s.is_final), {
              speaker: transcriptData.speaker,
              text: transcriptData.text,
              is_final: true
            }];
            
            // Join final customer segments only
            const customerText = finalSegments
              .filter(s => s.speaker === 'customer')
              .map(s => s.text)
              .join(' ');
              
            return customerText;
          });
        }
        
        setProcessingStage(`🎙️ FINAL: ${transcriptData.speaker} speaking...`);
        break;
        
      // NEW interim result - add as interim segment
      case 'transcription_interim_new':
        const interimData = message.data;
        
        // Remove any existing interim segments for this speaker
        setShowingSegments(prev => [
          ...prev.filter(s => s.is_final), // Keep all final segments
          {
            speaker: interimData.speaker,
            start: interimData.start,
            duration: interimData.duration,
            text: interimData.text,
            confidence: interimData.confidence,
            is_final: false,  // Mark as interim
            is_interim: true
          }
        ]);
        
        setProcessingStage(`🎙️ interim: ${interimData.speaker} - "${interimData.text.slice(0, 30)}..."`);
        break;
        
      // UPDATE existing interim result - replace the interim segment
      case 'transcription_interim_update':
        const updateData = message.data;
        
        // Replace the interim segment for this speaker
        setShowingSegments(prev => [
          ...prev.filter(s => s.is_final), // Keep all final segments
          {
            speaker: updateData.speaker,
            start: updateData.start,
            duration: updateData.duration,
            text: updateData.text,
            confidence: updateData.confidence,
            is_final: false,  // Mark as interim
            is_interim: true
          }
        ]);
        
        setProcessingStage(`🎙️ updating: ${updateData.speaker} - "${updateData.text.slice(0, 30)}..."`);
        break;
        
      case 'fraud_analysis_started':
        setProcessingStage('🔍 Analyzing for fraud patterns...');
        break;
        
      case 'fraud_analysis_update':
        const analysis = message.data;
        setRiskScore(analysis.risk_score || 0);
        setRiskLevel(analysis.risk_level || 'MINIMAL');
        setDetectedPatterns(analysis.detected_patterns || {});
        setProcessingStage(`🔍 Risk updated: ${analysis.risk_score}%`);
        break;
        
      case 'policy_analysis_started':
        setProcessingStage('📚 Retrieving policy guidance...');
        break;
        
      case 'policy_guidance_ready':
        setPolicyGuidance(message.data);
        setProcessingStage('📚 Policy guidance ready');
        break;
        
      case 'case_creation_started':
        setProcessingStage('📋 Creating fraud case...');
        break;
        
      case 'case_created':
        setProcessingStage(`📋 Case created: ${message.data.case_id}`);
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
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  // ===== AUDIO ANALYSIS FUNCTIONS =====

  const startServerProcessing = async (audioFile: RealAudioFile): Promise<void> => {
    try {
      // Check WebSocket connection
      if (!ws || !isConnected) {
        setProcessingStage('🔌 WebSocket not connected - attempting to connect...');
        connectWebSocket();
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        if (!ws || !isConnected) {
          throw new Error('Could not establish WebSocket connection');
        }
      }
      
      // Reset state for new analysis
      if (selectedAudioFile?.id !== audioFile.id) {
        setSelectedAudioFile(audioFile);
        resetState();
      }
      
      const newSessionId = `server_session_${Date.now()}`;
      setSessionId(newSessionId);
      setIsPlaying(true);
      setServerProcessing(true);
      setProcessingStage('🖥️ Starting server-side processing...');
      
      // Send WebSocket message to start server processing
      const message = {
        type: 'process_audio',
        data: {
          filename: audioFile.filename,
          session_id: newSessionId
        }
      };
      ws.send(JSON.stringify(message));
      
      // Start audio playback simultaneously
      const audio = new Audio(`${API_BASE_URL}/api/v1/audio/sample-files/${audioFile.filename}`);
      setAudioElement(audio);
      
      // Set up audio event listeners
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
      
      // Start playback
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
          {serverProcessing && (
            <span className="text-xs bg-blue-500 text-white px-2 py-1 rounded-full">
              SERVER
            </span>
          )}
        </div>
      )}
    </button>
  );

  const ProcessingStatus = () => (
    <div className="flex items-center space-x-2 text-xs">
      {serverProcessing ? (
        <div className="flex items-center space-x-1 text-blue-600">
          <Server className="w-4 h-4 animate-pulse" />
          <span>Server Processing Active</span>
        </div>
      ) : (
        <div className="flex items-center space-x-1 text-gray-500">
          <Server className="w-4 h-4" />
          <span>Server Ready</span>
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

  // ===== RENDER =====

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="flex justify-between items-center px-6 py-3">
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <img src="/hsbc-uk.svg" alt="HSBC" className="h-8 w-auto" />
            </div>
            <span className="text-lg font-medium text-gray-900">Agent Desktop</span>
          </div>
          
          <div className="flex items-center space-x-6 text-sm text-gray-600">
            <span>Agent: Sarah Mitchell</span>
            <span>ID: SM2024</span>
            <span>Shift: 09:00-17:00</span>
            <ProcessingStatus />
            <Settings className="w-4 h-4" />
          </div>
        </div>
      </header>

      {/* Demo Call Selection - Compact Widget Bar */}
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
        
        {/* Left Sidebar - Customer Information */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          {/* Customer Information */}
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Customer Information</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Name:</span>
                <span className="font-medium">{customerData.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Account:</span>
                <span className="font-medium">{customerData.account}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Status:</span>
                <span className="text-green-600 font-medium">{customerData.status}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Segment:</span>
                <span className="font-medium">{customerData.segment}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Risk Profile:</span>
                <span className="text-yellow-600 font-medium">{customerData.riskProfile}</span>
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
              <button className="w-full flex items-center space-x-2 px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 rounded">
                <FileText className="w-4 h-4" />
                <span>Create Case</span>
              </button>
              {riskScore >= 80 && (
                <button className="w-full flex items-center space-x-2 px-3 py-2 text-left text-sm bg-red-100 text-red-700 rounded hover:bg-red-200">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Escalate to Fraud Team</span>
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
                <span>2 days ago</span>
              </div>
              <div className="text-gray-600">Account balance inquiry</div>
              
              {riskScore > 0 && (
                <div className="mt-4 p-2 bg-orange-50 border border-orange-200 rounded">
                  <div className="flex justify-between">
                    <span className="text-orange-800 font-medium">Fraud Alert:</span>
                    <span className="text-orange-600">4 weeks ago</span>
                  </div>
                  <div className="text-orange-700 text-xs mt-1">Suspicious payment blocked - £1,200 to unknown recipient</div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Center - Live Transcription */}
        <div className="flex-1 bg-white border-r border-gray-200">
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Transcription</h3>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-red-600 font-medium">Processing</span>
              {serverProcessing && (
                <span className="text-xs bg-blue-500 text-white px-2 py-1 rounded-full">
                  SERVER
                </span>
              )}
            </div>
          </div>
          
          <div className="p-4 h-full overflow-y-auto">
            {showingSegments.length === 0 ? (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <div className="text-center">
                  <Server className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p>Select a demo call to see transcription</p>
                  <p className="text-sm text-blue-600 mt-2">Audio plays locally, server processes simultaneously</p>
                  {processingStage && (
                    <p className="text-sm text-green-600 mt-2">{processingStage}</p>
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {showingSegments.map((segment, index) => {
                  const isInterim = !segment.is_final;
                  
                  return (
                    <div key={`${segment.speaker}-${index}-${isInterim ? 'interim' : 'final'}`} 
                         className={`animate-in slide-in-from-bottom duration-500 ${isInterim ? 'opacity-70' : ''}`}>
                      <div className="flex items-start space-x-3">
                        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium ${
                          segment.speaker === 'customer' 
                            ? 'bg-blue-100 text-blue-800' 
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {segment.speaker === 'customer' ? (
                            <>
                              <User className="w-3 h-3" />
                              <span>Customer</span>
                            </>
                          ) : (
                            <>
                              <Shield className="w-3 h-3" />
                              <span>Agent</span>
                            </>
                          )}
                          {isInterim && <span className="text-xs opacity-60 ml-1">(live)</span>}
                        </div>
                        <span className="text-xs text-gray-500 mt-1">
                          {formatTime(segment.start)}
                        </span>
                        {segment.confidence && (
                          <span className="text-xs text-gray-400 mt-1">
                            {Math.round(segment.confidence * 100)}%
                          </span>
                        )}
                        <span className="text-xs bg-blue-100 text-blue-700 px-1 rounded">
                          SERVER
                        </span>
                        {isInterim && (
                          <span className="text-xs bg-yellow-100 text-yellow-700 px-1 rounded animate-pulse">
                            LIVE
                          </span>
                        )}
                      </div>
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

        {/* Right Sidebar - AI Agent Assist */}
        <div className="w-96 bg-white flex flex-col">
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">AI Agent Assist</h3>
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              ) : (
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              )}
              <span className="text-sm text-gray-600">Server Analysis</span>
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
                    <div className="text-xs text-gray-600">Fraud Risk</div>
                  </div>
                </div>
              </div>
              
              <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                riskScore >= 80 ? 'bg-red-100 text-red-800' :
                riskScore >= 60 ? 'bg-orange-100 text-orange-800' :
                riskScore >= 40 ? 'bg-yellow-100 text-yellow-800' :
                'bg-green-100 text-green-800'
              }`}>
                {riskScore >= 80 ? 'HIGH RISK' :
                 riskScore >= 60 ? 'MEDIUM RISK' :
                 riskScore >= 40 ? 'LOW RISK' : 'MINIMAL RISK'}
              </div>
            </div>
          </div>

          {/* Live Alerts */}
          {(riskScore >= 40 || Object.keys(detectedPatterns).length > 0) && (
            <div className="p-4 border-b border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <AlertCircle className="w-4 h-4 text-red-500 mr-2" />
                Server Alerts
              </h4>
              
              {riskScore >= 80 && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-3">
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-red-600" />
                    <span className="text-sm font-medium text-red-800">INVESTMENT SCAM DETECTED</span>
                  </div>
                  <p className="text-xs text-red-700 mt-1">Server analysis: guaranteed returns, urgency, third-party instructions</p>
                </div>
              )}
              
              {riskScore >= 60 && riskScore < 80 && (
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 mb-3">
                  <div className="flex items-center space-x-2">
                    <AlertCircle className="w-4 h-4 text-orange-600" />
                    <span className="text-sm font-medium text-orange-800">URGENCY PRESSURE</span>
                  </div>
                  <p className="text-xs text-orange-700 mt-1">Server detected: "immediately" and time pressure patterns</p>
                </div>
              )}
            </div>
          )}

          {/* Recommended Actions */}
          {policyGuidance && policyGuidance.recommended_actions.length > 0 && (
            <div className="p-4 border-b border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                Recommended Actions
              </h4>
              
              <div className="space-y-2">
                {policyGuidance.recommended_actions.slice(0, 4).map((action: string, index: number) => (
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
                      <span className="text-xs font-medium text-red-800">1. STOP TRANSACTION</span>
                    </div>
                    <p className="text-xs text-red-700 mt-1">Do not process any transfers</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* HSBC Policy */}
          {policyGuidance && policyGuidance.policy_id && (
            <div className="p-4 border-b border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <FileText className="w-4 h-4 text-blue-500 mr-2" />
                HSBC Policy
              </h4>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="text-xs font-medium text-blue-800 mb-2">
                  {policyGuidance.policy_title || 'Policy Guidance'} ({policyGuidance.policy_id})
                </div>
                {policyGuidance.customer_education && policyGuidance.customer_education.length > 0 && (
                  <ul className="text-xs text-blue-700 space-y-1">
                    {policyGuidance.customer_education.slice(0, 4).map((point: string, index: number) => (
                      <li key={index}>• {point}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}

          {/* Key Questions to Ask */}
          {policyGuidance && policyGuidance.key_questions.length > 0 && (
            <div className="p-4 border-b border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <Eye className="w-4 h-4 text-purple-500 mr-2" />
                Key Questions to Ask
              </h4>
              
              <div className="space-y-2">
                {policyGuidance.key_questions.slice(0, 3).map((question: string, index: number) => (
                  <div key={index} className="p-2 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100 transition-colors cursor-pointer">
                    <p className="text-xs text-green-900 font-medium">"{question}"</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Explain to Customer */}
          {policyGuidance && policyGuidance.customer_education.length > 0 && (
            <div className="p-4 flex-1">
              <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                <Users className="w-4 h-4 text-green-500 mr-2" />
                Explain to Customer
              </h4>
              
              <div className="space-y-2">
                {policyGuidance.customer_education.slice(0, 3).map((point: string, index: number) => (
                  <div key={index} className="p-2 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-xs text-yellow-900">"{point}"</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Default State - No Analysis */}
          {riskScore === 0 && Object.keys(detectedPatterns).length === 0 && !policyGuidance && (
            <div className="flex-1 flex items-center justify-center p-4">
              <div className="text-center text-gray-500">
                <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h4 className="text-sm font-medium text-gray-900 mb-2">Server-side AI Processing</h4>
                <p className="text-xs text-gray-600">
                  Start a demo call to see server-side fraud detection and policy guidance
                </p>
                <div className="mt-4 space-y-2 text-xs">
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>6 Agents Ready</span>
                  </div>
                  <div className="text-gray-400">Server Audio • Fraud • Policy • Case • Compliance • Orchestrator</div>
                  {processingStage && (
                    <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded">
                      <p className="text-blue-700">{processingStage}</p>
                    </div>
                  )}
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
