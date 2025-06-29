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
  WifiOff
} from 'lucide-react';

// Environment configuration - FIXED: No hardcoded IPs, pure environment variables
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

// Log configuration for debugging (will be removed in production)
console.log('🔧 Frontend Configuration:');
console.log('  API_BASE_URL:', API_BASE_URL);
console.log('  WS_BASE_URL:', WS_BASE_URL);
console.log('  Environment:', process.env.REACT_APP_ENVIRONMENT || 'development');

// Type definitions
interface AudioSegment {
  speaker: 'agent' | 'customer';
  start: number;
  duration: number;
  text: string;
  confidence?: number;
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

interface BackendAnalysisResult {
  session_id: string;
  fraud_analysis?: {
    risk_score: number;
    risk_level: string;
    scam_type: string;
    detected_patterns: Record<string, DetectedPattern>;
    confidence: number;
  };
  policy_guidance?: PolicyGuidance;
  transcription?: {
    speaker_segments: any[];
  };
  orchestrator_decision?: {
    decision: string;
    reasoning: string;
    actions: string[];
  };
}

// Risk level colors
const getRiskColor = (riskScore: number): string => {
  if (riskScore >= 80) return 'bg-red-100 text-red-800 border-red-300';
  if (riskScore >= 60) return 'bg-orange-100 text-orange-800 border-orange-300';
  if (riskScore >= 40) return 'bg-yellow-100 text-yellow-800 border-yellow-300';
  return 'bg-green-100 text-green-800 border-green-300';
};

const getRiskColorRaw = (riskScore: number): string => {
  if (riskScore >= 80) return 'rgb(239, 68, 68)';
  if (riskScore >= 60) return 'rgb(245, 101, 39)';
  if (riskScore >= 40) return 'rgb(245, 158, 11)';
  return 'rgb(34, 197, 94)';
};

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
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
  
  // Refs
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Reset state when new analysis starts
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
  };

  // Load audio files from backend on component mount
  useEffect(() => {
    loadAudioFiles();
  }, []);

  // WebSocket connection management
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

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Load real audio files from backend
  const loadAudioFiles = async (): Promise<void> => {
    try {
      setIsLoadingFiles(true);
      const response = await fetch(`${API_BASE_URL}/api/v1/audio/sample-files`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setAudioFiles(data.files);
        console.log('✅ Loaded audio files:', data.files);
      } else {
        console.error('❌ Failed to load audio files:', data);
        setProcessingStage('❌ Failed to load audio files from backend');
      }
    } catch (error) {
      console.error('❌ Error loading audio files:', error);
      setProcessingStage('❌ Backend connection error - check if server is running');
    } finally {
      setIsLoadingFiles(false);
    }
  };

  // WebSocket connection with auto-reconnect
  const connectWebSocket = (): void => {
    try {
      const websocket = new WebSocket(`${WS_BASE_URL}/ws/fraud-detection-${Date.now()}`);
      
      websocket.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setWs(websocket);
        
        // Clear any pending reconnection attempts
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
        
        // Attempt to reconnect after 3 seconds if not manually closed
        if (event.code !== 1000) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('🔄 Attempting to reconnect WebSocket...');
            connectWebSocket();
          }, 3000);
        }
      };
      
      websocket.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setProcessingStage('❌ WebSocket connection error');
      };
      
    } catch (error) {
      console.error('❌ Error creating WebSocket connection:', error);
    }
  };

  // Handle real-time WebSocket messages from backend
  const handleWebSocketMessage = (message: WebSocketMessage): void => {
    console.log('📨 WebSocket message received:', message);
    
    switch (message.type) {
      case 'transcription_segment':
        const segment = message.data;
        setShowingSegments(prev => [...prev, {
          speaker: segment.speaker,
          start: segment.start,
          duration: segment.duration,
          text: segment.text,
          confidence: segment.confidence
        }]);
        
        setTranscription(prev => prev + (prev ? ' ' : '') + segment.text);
        setProcessingStage('🎵 Live transcription streaming...');
        break;
        
      case 'fraud_analysis_update':
        const analysis = message.data;
        setRiskScore(analysis.risk_score || 0);
        setRiskLevel(analysis.risk_level || 'MINIMAL');
        setDetectedPatterns(analysis.detected_patterns || {});
        setProcessingStage('🔍 Fraud patterns detected...');
        break;
        
      case 'policy_guidance_ready':
        setPolicyGuidance(message.data);
        setProcessingStage('📚 Policy guidance retrieved...');
        break;
        
      case 'case_created':
        setProcessingStage(`📋 Case created: ${message.data.case_id}`);
        break;
        
      case 'processing_complete':
        setProcessingStage('✅ All agents completed analysis');
        break;
        
      case 'processing_started':
        setProcessingStage(`🤖 ${message.data.agent}: ${message.data.stage}`);
        break;
        
      case 'error':
        setProcessingStage(`❌ Error: ${message.data.error}`);
        break;
        
      case 'pong':
        // Handle ping/pong for connection health
        console.log('📡 WebSocket pong received');
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  // Start audio analysis with backend integration
  const startAnalysis = async (audioFile: RealAudioFile): Promise<void> => {
    try {
      // Reset state for new analysis
      if (selectedAudioFile?.id !== audioFile.id) {
        setSelectedAudioFile(audioFile);
        resetState();
      }
      
      const newSessionId = `session_${Date.now()}`;
      setSessionId(newSessionId);
      setIsPlaying(true);
      setProcessingStage('🎵 Loading audio and starting analysis...');
      
      // Start backend processing via REST API
      await processAudioWithBackend(audioFile, newSessionId);
      
      // Start audio playback
      const audio = new Audio(`${API_BASE_URL}/api/v1/audio/sample-files/${audioFile.filename}`);
      setAudioElement(audio);
      
      // Set up audio event listeners
      audio.addEventListener('timeupdate', () => {
        setCurrentTime(audio.currentTime);
      });
      
      audio.addEventListener('ended', () => {
        setIsPlaying(false);
        setProcessingStage('✅ Audio playback complete');
      });
      
      audio.addEventListener('error', (e) => {
        console.error('❌ Audio playback error:', e);
        setProcessingStage('❌ Error playing audio file');
        setIsPlaying(false);
      });
      
      // Start playback
      await audio.play();
      
    } catch (error) {
      console.error('❌ Error starting analysis:', error);
      setProcessingStage('❌ Failed to start analysis');
      setIsPlaying(false);
    }
  };

  // Process audio with backend agents via REST API
  const processAudioWithBackend = async (audioFile: RealAudioFile, sessionId: string): Promise<void> => {
    try {
      setProcessingStage('🔍 Sending to backend multi-agent system...');
      
      const response = await fetch(`${API_BASE_URL}/api/v1/audio/process-audio-realtime`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: audioFile.filename,
          session_id: sessionId
        })
      });
      
      if (!response.ok) {
        throw new Error(`Backend responded with status: ${response.status}`);
      }
      
      const result: { status: string; analysis?: BackendAnalysisResult; error?: string } = await response.json();
      
      if (result.status === 'success' && result.analysis) {
        setProcessingStage('✅ Backend processing complete');
        
        // Update UI with real results from backend
        const analysis = result.analysis;
        
        if (analysis.fraud_analysis) {
          setRiskScore(analysis.fraud_analysis.risk_score);
          setRiskLevel(analysis.fraud_analysis.risk_level);
          setDetectedPatterns(analysis.fraud_analysis.detected_patterns);
        }
        
        if (analysis.policy_guidance) {
          setPolicyGuidance(analysis.policy_guidance);
        }
        
        if (analysis.transcription?.speaker_segments) {
          // Simulate streaming transcription from real results
          simulateStreamingFromReal(analysis.transcription.speaker_segments);
        }
        
      } else {
        throw new Error(result.error || 'Backend processing failed');
      }
      
    } catch (error) {
      console.error('❌ Error processing with backend:', error);
      setProcessingStage(`❌ Backend error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  // Simulate streaming effect from real transcription results
  const simulateStreamingFromReal = (segments: any[]): void => {
    let segmentIndex = 0;
    
    const showNextSegment = () => {
      if (segmentIndex < segments.length && isPlaying) {
        const segment = segments[segmentIndex];
        
        setShowingSegments(prev => [...prev, {
          speaker: segment.speaker,
          start: segment.start,
          duration: segment.end - segment.start,
          text: segment.text,
          confidence: segment.confidence || 0.9
        }]);
        
        setTranscription(prev => 
          prev + (prev ? ' ' : '') + segment.text
        );
        
        segmentIndex++;
        
        // Schedule next segment with realistic timing
        if (segmentIndex < segments.length) {
          setTimeout(showNextSegment, 2000 + Math.random() * 1000);
        }
      }
    };
    
    // Start showing segments after a brief delay
    setTimeout(showNextSegment, 1000);
  };

  // Stop analysis and audio playback
  const stopAnalysis = (): void => {
    setIsPlaying(false);
    setProcessingStage('⏸️ Analysis stopped');
    
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  };

  // Send ping to keep WebSocket alive
  const sendPing = (): void => {
    if (ws && isConnected) {
      ws.send(JSON.stringify({ type: 'ping', data: { timestamp: new Date().toISOString() } }));
    }
  };

  // Ping WebSocket every 30 seconds to keep connection alive
  useEffect(() => {
    const pingInterval = setInterval(sendPing, 30000);
    return () => clearInterval(pingInterval);
  }, [ws, isConnected]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <Shield className="w-8 h-8 text-red-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  HSBC Fraud Detection System
                </h1>
                <p className="text-sm text-gray-500">
                  Real-time Multi-Agent Call Analysis Demo
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                {isConnected ? (
                  <Wifi className="w-4 h-4 text-green-600" />
                ) : (
                  <WifiOff className="w-4 h-4 text-red-600" />
                )}
                <span className={`px-3 py-1 rounded-full text-sm ${
                  isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div className="text-sm text-gray-600">6 Agents Active</div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        {/* Real Audio Files Section */}
        <div className="mb-8">
          <div className="flex items-center space-x-2 mb-6">
            <PhoneCall className="w-6 h-6 text-blue-600" />
            <h2 className="text-2xl font-bold text-gray-900">Real Call Recordings</h2>
            <span className="text-sm text-gray-500">Click to analyze with backend agents</span>
          </div>
          
          {isLoadingFiles ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="ml-3 text-gray-600">Loading audio files...</span>
            </div>
          ) : audioFiles.length === 0 ? (
            <div className="flex items-center justify-center py-12 bg-yellow-50 border border-yellow-200 rounded-lg">
              <AlertTriangle className="w-8 h-8 text-yellow-600 mr-3" />
              <div>
                <p className="text-yellow-800 font-medium">No audio files available</p>
                <p className="text-yellow-600 text-sm">Make sure the backend server is running and has sample files</p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {audioFiles.map((audioFile) => (
                <div
                  key={audioFile.id}
                  className={`bg-white rounded-lg shadow-md p-6 cursor-pointer border-2 transition-all hover:shadow-lg ${
                    selectedAudioFile?.id === audioFile.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                  }`}
                  onClick={() => !isPlaying ? startAnalysis(audioFile) : null}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-2xl">{audioFile.icon}</span>
                    <span className="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      Ready
                    </span>
                  </div>
                  
                  <h3 className="font-semibold text-gray-900 mb-2">{audioFile.title}</h3>
                  <p className="text-sm text-gray-600 mb-3">{audioFile.description}</p>
                  
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <span>{formatTime(audioFile.duration)}</span>
                    <span>{(audioFile.size_bytes / 1024 / 1024).toFixed(1)} MB</span>
                  </div>
                  
                  {selectedAudioFile?.id === audioFile.id && (
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <div className="flex items-center justify-between">
                        {!isPlaying ? (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              startAnalysis(audioFile);
                            }}
                            className="flex items-center space-x-2 px-3 py-1 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition-colors"
                          >
                            <Play className="w-4 h-4" />
                            <span>Analyze</span>
                          </button>
                        ) : (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              stopAnalysis();
                            }}
                            className="flex items-center space-x-2 px-3 py-1 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700 transition-colors"
                          >
                            <Pause className="w-4 h-4" />
                            <span>Stop</span>
                          </button>
                        )}
                        <span className="text-xs text-gray-600">
                          {audioElement ? formatTime(currentTime) : '0:00'} / {formatTime(audioFile.duration)}
                        </span>
                      </div>
                      
                      {/* Progress bar */}
                      {audioElement && (
                        <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-100"
                            style={{ width: `${(currentTime / audioFile.duration) * 100}%` }}
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Processing Status */}
        {processingStage && (
          <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <Brain className="w-5 h-5 text-blue-600 animate-pulse" />
              <span className="text-blue-800 font-medium">{processingStage}</span>
              {sessionId && (
                <span className="text-blue-600 text-sm">Session: {sessionId}</span>
              )}
            </div>
          </div>
        )}

        {/* Main Analysis Grid */}
        {selectedAudioFile && (riskScore > 0 || showingSegments.length > 0) && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Left Column - Live Transcription */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <Mic className="w-5 h-5 text-green-600" />
                  <h2 className="text-lg font-semibold text-gray-900">Live Transcription</h2>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-red-500">LIVE</span>
                  </div>
                </div>
                
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {showingSegments.map((segment, index) => (
                    <div key={index} className="animate-in slide-in-from-bottom duration-500">
                      <div className="flex items-start space-x-3">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          segment.speaker === 'customer' 
                            ? 'bg-blue-100 text-blue-800' 
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {segment.speaker === 'customer' ? 'Customer' : 'Agent'}
                        </span>
                        <span className="text-xs text-gray-500 mt-1">
                          {formatTime(segment.start)}
                        </span>
                        {segment.confidence && (
                          <span className="text-xs text-gray-400 mt-1">
                            {Math.round(segment.confidence * 100)}%
                          </span>
                        )}
                      </div>
                      <div className="mt-1 p-3 bg-gray-50 rounded-lg border-l-4 border-blue-400">
                        <p className="text-sm text-gray-800">{segment.text}</p>
                      </div>
                    </div>
                  ))}
                  
                  {isPlaying && (
                    <div className="flex items-center space-x-2 text-gray-500">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                      <span className="text-sm">Listening...</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Middle Column - Risk Assessment */}
            <div className="lg:col-span-1">
              <div className="space-y-6">
                
                {/* Risk Score Display */}
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="flex items-center space-x-2 mb-4">
                    <TrendingUp className="w-5 h-5 text-red-600" />
                    <h2 className="text-lg font-semibold text-gray-900">Risk Assessment</h2>
                    {riskScore > 0 && (
                      <Eye className="w-4 h-4 text-blue-500 animate-pulse" />
                    )}
                  </div>
                  
                  <div className="text-center">
                    <div className="relative">
                      <svg className="w-32 h-32 mx-auto" viewBox="0 0 120 120">
                        <circle
                          cx="60"
                          cy="60"
                          r="50"
                          fill="none"
                          stroke="#e5e7eb"
                          strokeWidth="8"
                        />
                        <circle
                          cx="60"
                          cy="60"
                          r="50"
                          fill="none"
                          stroke={getRiskColorRaw(riskScore)}
                          strokeWidth="8"
                          strokeLinecap="round"
                          strokeDasharray={`${(riskScore / 100) * 314.16} 314.16`}
                          transform="rotate(-90 60 60)"
                          className="transition-all duration-500"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-gray-900">{riskScore}%</div>
                          <div className="text-sm text-gray-600">{riskLevel}</div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-4">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {riskLevel} RISK
                      </h3>
                    </div>
                  </div>
                </div>

                {/* Detected Patterns */}
                {Object.keys(detectedPatterns).length > 0 && (
                  <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-center space-x-2 mb-4">
                      <AlertTriangle className="w-5 h-5 text-orange-600" />
                      <h3 className="text-lg font-semibold text-gray-900">Detected Patterns</h3>
                    </div>
                    
                    <div className="space-y-3">
                      {Object.entries(detectedPatterns).map(([pattern, data]) => (
                        <div key={pattern} className="border-l-4 border-red-400 pl-4 animate-in slide-in-from-left duration-500">
                          <div className="flex items-center justify-between">
                            <h4 className="font-medium text-gray-900 capitalize">
                              {pattern.replace('_', ' ')}
                            </h4>
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              data.severity === 'critical' ? 'bg-red-100 text-red-800' :
                              data.severity === 'high' ? 'bg-orange-100 text-orange-800' :
                              'bg-yellow-100 text-yellow-800'
                            }`}>
                              {data.severity}
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 mt-1">
                            {data.count} matches detected (Weight: {data.weight})
                          </p>
                          <div className="mt-2">
                            {data.matches.map((match: string, index: number) => (
                              <span key={index} className="inline-block bg-red-50 text-red-700 px-2 py-1 rounded text-xs mr-1 mb-1">
                                "{match}"
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Policy Guidance */}
            <div className="lg:col-span-1">
              {policyGuidance ? (
                <div className="space-y-6">
                  
                  {/* Immediate Alerts */}
                  {policyGuidance.immediate_alerts.length > 0 && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-6 animate-in slide-in-from-right duration-500">
                      <h3 className="text-lg font-semibold text-red-900 mb-4 flex items-center space-x-2">
                        <AlertCircle className="w-5 h-5" />
                        <span>Immediate Alerts</span>
                      </h3>
                      <ul className="space-y-2">
                        {policyGuidance.immediate_alerts.map((alert: string, index: number) => (
                          <li key={index} className="text-sm text-red-800 font-medium flex items-start space-x-2">
                            <span className="text-red-600 mt-0.5">•</span>
                            <span>{alert}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Recommended Actions */}
                  {policyGuidance.recommended_actions.length > 0 && (
                    <div className="bg-white rounded-lg shadow p-6 animate-in slide-in-from-right duration-700">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                        <CheckCircle className="w-5 h-5 text-blue-600" />
                        <span>Recommended Actions</span>
                      </h3>
                      <ol className="space-y-3">
                        {policyGuidance.recommended_actions.map((action: string, index: number) => (
                          <li key={index} className="text-sm text-gray-700 flex items-start space-x-3">
                            <span className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-800 rounded-full flex items-center justify-center text-xs font-medium">
                              {index + 1}
                            </span>
                            <span>{action}</span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {/* Key Questions */}
                  {policyGuidance.key_questions.length > 0 && (
                    <div className="bg-white rounded-lg shadow p-6 animate-in slide-in-from-right duration-1000">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                        <FileText className="w-5 h-5 text-purple-600" />
                        <span>Key Questions to Ask</span>
                      </h3>
                      <div className="space-y-2">
                        {policyGuidance.key_questions.map((question: string, index: number) => (
                          <div key={index} className="p-3 bg-blue-50 border border-blue-200 rounded-lg hover:bg-blue-100 transition-colors cursor-pointer">
                            <p className="text-sm text-blue-900 font-medium">"{question}"</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Customer Education */}
                  {policyGuidance.customer_education.length > 0 && (
                    <div className="bg-white rounded-lg shadow p-6 animate-in slide-in-from-right duration-1200">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                        <Users className="w-5 h-5 text-green-600" />
                        <span>Customer Education</span>
                      </h3>
                      <div className="space-y-2">
                        {policyGuidance.customer_education.map((point: string, index: number) => (
                          <div key={index} className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                            <p className="text-sm text-yellow-900">"{point}"</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-white rounded-lg shadow p-6 text-center">
                  <Shield className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900">Policy Guidance</h3>
                  <p className="text-sm text-gray-600 mt-2">
                    Agent guidance will appear here during call analysis
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Agent Status Dashboard */}
        <div className="mt-12 bg-white rounded-lg shadow p-6">
          <div className="flex items-center space-x-2 mb-6">
            <Brain className="w-6 h-6 text-purple-600" />
            <h2 className="text-xl font-bold text-gray-900">Multi-Agent System Status</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { name: 'Audio Processing Agent', status: 'Ready', icon: '🎵', description: 'Real-time transcription' },
              { name: 'Fraud Detection Agent', status: 'Active', icon: '🔍', description: 'Pattern recognition' },
              { name: 'Policy Guidance Agent', status: 'Ready', icon: '📚', description: 'HSBC procedures' },
              { name: 'Case Management Agent', status: 'Standby', icon: '📋', description: 'Investigation cases' },
              { name: 'Compliance Agent', status: 'Active', icon: '🔒', description: 'GDPR/FCA compliance' },
              { name: 'Master Orchestrator', status: 'Coordinating', icon: '🎭', description: 'Workflow management' }
            ].map((agent, index) => (
              <div key={index} className="p-4 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-lg">{agent.icon}</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    agent.status === 'Active' || agent.status === 'Coordinating' ? 'bg-green-100 text-green-800' :
                    agent.status === 'Ready' ? 'bg-blue-100 text-blue-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {agent.status}
                  </span>
                </div>
                <h4 className="font-medium text-gray-900 mb-1">{agent.name}</h4>
                <p className="text-sm text-gray-600">{agent.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Demo Instructions */}
        <div className="mt-8 bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-4 flex items-center space-x-2">
            <Volume2 className="w-5 h-5" />
            <span>Real-time Demo Instructions</span>
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-blue-800 mb-3">🎯 How It Works</h4>
              <ul className="text-sm text-blue-700 space-y-2">
                <li className="flex items-start space-x-2">
                  <span className="text-blue-600 mt-0.5">1.</span>
                  <span><strong>Select a call</strong> from the demo recordings above</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-blue-600 mt-0.5">2.</span>
                  <span><strong>Watch live transcription</strong> stream as audio plays</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-blue-600 mt-0.5">3.</span>
                  <span><strong>See risk score</strong> update in real-time</span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="text-blue-600 mt-0.5">4.</span>
                  <span><strong>Get instant policy guidance</strong> from agents</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-blue-800 mb-3">🚀 Multi-Agent Processing</h4>
              <ul className="text-sm text-blue-700 space-y-2">
                <li>• <strong>Audio Agent:</strong> Converts speech to text with speaker identification</li>
                <li>• <strong>Fraud Agent:</strong> Detects scam patterns in real-time</li>
                <li>• <strong>Policy Agent:</strong> Retrieves HSBC response procedures</li>
                <li>• <strong>Orchestrator:</strong> Coordinates all agent responses</li>
                <li>• <strong>Compliance:</strong> Ensures regulatory compliance</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-blue-100 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>💡 Pro Tip:</strong> Try different call types to see how the system adapts its response. 
              The system now uses <strong>real backend agents</strong> for authentic fraud detection results.
            </p>
          </div>
        </div>

        {/* System Performance Metrics */}
        {selectedAudioFile && riskScore > 0 && (
          <div className="mt-8 bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
              <TrendingUp className="w-5 h-5" />
              <span>Real-time Performance Metrics</span>
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{formatTime(currentTime)}</div>
                <div className="text-sm text-blue-800">Processing Time</div>
              </div>
              
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{showingSegments.length}</div>
                <div className="text-sm text-green-800">Segments Processed</div>
              </div>
              
              <div className="text-center p-4 bg-orange-50 rounded-lg">
                <div className="text-2xl font-bold text-orange-600">{Object.keys(detectedPatterns).length}</div>
                <div className="text-sm text-orange-800">Patterns Detected</div>
              </div>
              
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {riskScore >= 80 ? 'Critical' : riskScore >= 60 ? 'High' : riskScore >= 40 ? 'Medium' : 'Low'}
                </div>
                <div className="text-sm text-purple-800">Risk Level</div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
