import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, Upload, Mic, Shield, FileText, Users } from 'lucide-react';

interface FraudAnalysis {
  session_id: string;
  fraud_detection: {
    risk_score: number;
    risk_level: string;
    scam_type: string;
    confidence: number;
    detected_patterns: any;
    explanation: string;
  };
  agent_guidance: {
    immediate_alerts: string[];
    recommended_actions: string[];
    key_questions: string[];
    customer_education: string[];
  };
  orchestrator_decision: {
    decision: string;
    reasoning: string;
    actions: string[];
    priority: string;
  };
  requires_escalation: boolean;
}

interface TranscriptionData {
  text: string;
  speaker: string;
  confidence: number;
  timestamp: string;
}

interface SystemStatus {
  system_status: string;
  agents_count: number;
  agents: Record<string, string>;
  framework: string;
  last_updated: string;
}

function App() {
  const [fraudAnalysis, setFraudAnalysis] = useState<FraudAnalysis | null>(null);
  const [transcription, setTranscription] = useState<TranscriptionData | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processedFiles, setProcessedFiles] = useState<string[]>([]);

  // Load system status on component mount
  useEffect(() => {
    fetchSystemStatus();
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/v1/system/status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Error fetching system status:', error);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setIsProcessing(true);
    setFraudAnalysis(null);
    setTranscription(null);

    try {
      // Upload file
      const formData = new FormData();
      formData.append('file', file);

      const uploadResponse = await fetch('/api/v1/audio/upload', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('File upload failed');
      }

      // Process through ADK pipeline
      const processResponse = await fetch('/api/v1/fraud/process-audio', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: file.name,
          session_id: `session_${Date.now()}`
        }),
      });

      if (!processResponse.ok) {
        throw new Error('Processing failed');
      }

      const result = await processResponse.json();
      
      // Set transcription data
      if (result.transcription) {
        setTranscription(result.transcription);
      }

      // Set fraud analysis data
      if (result.fraud_analysis) {
        setFraudAnalysis(result.fraud_analysis);
      }

      // Add to processed files
      setProcessedFiles(prev => [...prev, file.name]);

    } catch (error) {
      console.error('Error processing file:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const getRiskColor = (riskScore: number) => {
    if (riskScore >= 80) return 'bg-red-100 text-red-800 border-red-300';
    if (riskScore >= 60) return 'bg-orange-100 text-orange-800 border-orange-300';
    if (riskScore >= 40) return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    return 'bg-green-100 text-green-800 border-green-300';
  };

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'BLOCK_AND_ESCALATE': return 'bg-red-500';
      case 'VERIFY_AND_MONITOR': return 'bg-orange-500';
      case 'MONITOR_AND_EDUCATE': return 'bg-yellow-500';
      default: return 'bg-green-500';
    }
  };

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
                  Powered by Google ADK 1.4.2 Multi-Agent Architecture
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {systemStatus && (
                <div className={`px-3 py-1 rounded-full text-sm ${
                  systemStatus.system_status === 'operational' 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  System: {systemStatus.system_status}
                </div>
              )}
              <div className="text-sm text-gray-600">
                {systemStatus?.agents_count} Agents Active
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Column - Audio Upload & Processing */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Audio Upload Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Upload className="w-5 h-5 text-blue-600" />
                <h2 className="text-lg font-semibold text-gray-900">Audio Upload</h2>
              </div>
              
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                <input
                  type="file"
                  accept=".wav,.mp3,.m4a"
                  onChange={handleFileUpload}
                  disabled={isProcessing}
                  className="hidden"
                  id="audio-upload"
                />
                <label
                  htmlFor="audio-upload"
                  className={`cursor-pointer ${isProcessing ? 'opacity-50' : ''}`}
                >
                  <Mic className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-sm text-gray-600">
                    {isProcessing ? 'Processing...' : 'Click to upload audio file'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Supports WAV, MP3, M4A formats
                  </p>
                </label>
              </div>

              {selectedFile && (
                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-blue-900">
                    Selected: {selectedFile.name}
                  </p>
                  <p className="text-xs text-blue-700">
                    Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              )}
            </div>

            {/* Transcription Results */}
            {transcription && (
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <FileText className="w-5 h-5 text-green-600" />
                  <h2 className="text-lg font-semibold text-gray-900">Transcription</h2>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <span className="text-sm font-medium text-gray-700">Speaker:</span>
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      transcription.speaker === 'customer' 
                        ? 'bg-blue-100 text-blue-800' 
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {transcription.speaker}
                    </span>
                    <span className="text-sm text-gray-600">
                      ({(transcription.confidence * 100).toFixed(1)}% confidence)
                    </span>
                  </div>
                  
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-800">{transcription.text}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Agent Status */}
            {systemStatus && (
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <Users className="w-5 h-5 text-purple-600" />
                  <h2 className="text-lg font-semibold text-gray-900">ADK Agents</h2>
                </div>
                
                <div className="space-y-2">
                  {Object.entries(systemStatus.agents).map(([agentName, status]) => (
                    <div key={agentName} className="flex items-center justify-between">
                      <span className="text-sm text-gray-700 capitalize">
                        {agentName.replace('_', ' ')}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        status === 'ready' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {status}
                      </span>
                    </div>
                  ))}
                </div>
                
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <p className="text-xs text-gray-500">
                    Framework: {systemStatus.framework}
                  </p>
                  <p className="text-xs text-gray-500">
                    Last Updated: {new Date(systemStatus.last_updated).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Middle Column - Fraud Detection Results */}
          <div className="lg:col-span-1">
            {fraudAnalysis ? (
              <div className="space-y-6">
                
                {/* Risk Score Display */}
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="flex items-center space-x-2 mb-4">
                    <AlertCircle className="w-5 h-5 text-red-600" />
                    <h2 className="text-lg font-semibold text-gray-900">Risk Assessment</h2>
                  </div>
                  
                  <div className="text-center">
                    <div className={`inline-flex items-center justify-center w-24 h-24 rounded-full text-2xl font-bold ${getRiskColor(fraudAnalysis.fraud_detection.risk_score)}`}>
                      {fraudAnalysis.fraud_detection.risk_score}%
                    </div>
                    
                    <div className="mt-4">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {fraudAnalysis.fraud_detection.risk_level} RISK
                      </h3>
                      <p className="text-sm text-gray-600 mt-1">
                        {fraudAnalysis.fraud_detection.scam_type.replace('_', ' ').toUpperCase()}
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        Confidence: {(fraudAnalysis.fraud_detection.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                {/* Detected Patterns */}
                {Object.keys(fraudAnalysis.fraud_detection.detected_patterns).length > 0 && (
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Detected Patterns</h3>
                    
                    <div className="space-y-3">
                      {Object.entries(fraudAnalysis.fraud_detection.detected_patterns).map(([pattern, data]: [string, any]) => (
                        <div key={pattern} className="border-l-4 border-red-400 pl-4">
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
                            {data.count} matches (Weight: {data.weight})
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Orchestrator Decision */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">System Decision</h3>
                  
                  <div className={`p-4 rounded-lg ${getDecisionColor(fraudAnalysis.orchestrator_decision.decision)} text-white`}>
                    <h4 className="font-bold text-lg mb-2">
                      {fraudAnalysis.orchestrator_decision.decision.replace('_', ' ')}
                    </h4>
                    <p className="text-sm opacity-90">
                      {fraudAnalysis.orchestrator_decision.reasoning}
                    </p>
                  </div>
                  
                  <div className="mt-4">
                    <h5 className="font-medium text-gray-900 mb-2">Required Actions:</h5>
                    <ul className="text-sm text-gray-700 space-y-1">
                      {fraudAnalysis.orchestrator_decision.actions.map((action, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <span className="text-gray-400">•</span>
                          <span>{action}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">Priority:</span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        fraudAnalysis.orchestrator_decision.priority === 'CRITICAL' ? 'bg-red-100 text-red-800' :
                        fraudAnalysis.orchestrator_decision.priority === 'HIGH' ? 'bg-orange-100 text-orange-800' :
                        fraudAnalysis.orchestrator_decision.priority === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {fraudAnalysis.orchestrator_decision.priority}
                      </span>
                    </div>
                    
                    {fraudAnalysis.requires_escalation && (
                      <div className="mt-2 p-2 bg-red-50 rounded-lg">
                        <p className="text-sm text-red-800 font-medium">
                          🚨 ESCALATION REQUIRED
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow p-6 text-center">
                <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900">No Analysis Yet</h3>
                <p className="text-sm text-gray-600 mt-2">
                  Upload an audio file to see fraud detection results
                </p>
              </div>
            )}
          </div>

          {/* Right Column - Agent Guidance */}
          <div className="lg:col-span-1">
            {fraudAnalysis?.agent_guidance ? (
              <div className="space-y-6">
                
                {/* Immediate Alerts */}
                {fraudAnalysis.agent_guidance.immediate_alerts.length > 0 && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-red-900 mb-4">🚨 Immediate Alerts</h3>
                    <ul className="space-y-2">
                      {fraudAnalysis.agent_guidance.immediate_alerts.map((alert, index) => (
                        <li key={index} className="text-sm text-red-800 font-medium">
                          {alert}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Recommended Actions */}
                {fraudAnalysis.agent_guidance.recommended_actions.length > 0 && (
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">📋 Recommended Actions</h3>
                    <ol className="space-y-2">
                      {fraudAnalysis.agent_guidance.recommended_actions.map((action, index) => (
                        <li key={index} className="text-sm text-gray-700 flex items-start space-x-2">
                          <span className="font-medium text-blue-600">{index + 1}.</span>
                          <span>{action}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}

                {/* Key Questions */}
                {fraudAnalysis.agent_guidance.key_questions.length > 0 && (
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">❓ Key Questions to Ask</h3>
                    <div className="space-y-2">
                      {fraudAnalysis.agent_guidance.key_questions.map((question, index) => (
                        <div key={index} className="p-3 bg-blue-50 border border-blue-200 rounded-lg cursor-pointer hover:bg-blue-100 transition-colors">
                          <p className="text-sm text-blue-900">"{question}"</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Customer Education */}
                {fraudAnalysis.agent_guidance.customer_education.length > 0 && (
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">📖 Customer Education</h3>
                    <div className="space-y-2">
                      {fraudAnalysis.agent_guidance.customer_education.map((point, index) => (
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
                <CheckCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900">No Guidance Available</h3>
                <p className="text-sm text-gray-600 mt-2">
                  Agent guidance will appear here after fraud analysis
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Demo Instructions */}
        <div className="mt-12 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-4">
            🧪 Demo Instructions - Google ADK 1.4.2 Multi-Agent System
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-blue-800 mb-2">Sample Audio Files</h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• <code>investment_scam_live_call.wav</code> - Investment fraud (High Risk)</li>
                <li>• <code>romance_scam_live_call.wav</code> - Romance scam (Medium-High Risk)</li>
                <li>• <code>impersonation_scam_live_call.wav</code> - Bank impersonation (Critical Risk)</li>
                <li>• <code>legitimate_call.wav</code> - Normal banking inquiry (Low Risk)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-blue-800 mb-2">ADK 1.4.2 Agents Active</h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• <strong>Audio Processing Agent</strong> - Stream handling & transcription</li>
                <li>• <strong>Live Scam Detection Agent</strong> - Real-time fraud pattern recognition</li>
                <li>• <strong>RAG Policy Agent</strong> - HSBC policy retrieval & guidance</li>
                <li>• <strong>Master Orchestrator Agent</strong> - Workflow coordination</li>
                <li>• <strong>Case Management Agent</strong> - Investigation case creation</li>
                <li>• <strong>Compliance Agent</strong> - GDPR/FCA compliance checking</li>
              </ul>
            </div>
          </div>
          
          {processedFiles.length > 0 && (
            <div className="mt-4 p-3 bg-blue-100 rounded">
              <h5 className="font-medium text-blue-800 mb-2">Processed Files This Session:</h5>
              <div className="flex flex-wrap gap-2">
                {processedFiles.map((filename, index) => (
                  <span key={index} className="px-2 py-1 bg-blue-200 text-blue-800 rounded text-xs">
                    {filename}
                  </span>
                ))}
              </div>
            </div>
          )}
          
          <div className="mt-4 p-3 bg-blue-100 rounded">
            <p className="text-sm text-blue-800">
              <strong>System Architecture:</strong> This demo showcases Google ADK 1.4.2 multi-agent architecture 
              with specialized agents working together to detect fraud in real-time HSBC call center operations.
              Each agent has specific responsibilities and they communicate through ADK's built-in orchestration.
            </p>
          </div>
        </div>

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-sm mx-4">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span className="text-gray-900">Processing through ADK agents...</span>
              </div>
              <div className="mt-3 space-y-1 text-xs text-gray-600">
                <div>🎵 Audio Processing Agent</div>
                <div>🔍 Live Scam Detection Agent</div>
                <div>📚 RAG Policy Agent</div>
                <div>🎭 Master Orchestrator</div>
                <div>🔒 Compliance Agent</div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
