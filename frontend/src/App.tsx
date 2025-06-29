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
  Brain
} from 'lucide-react';

// Type definitions
interface AudioSegment {
  speaker: 'agent' | 'customer';
  start: number;
  duration: number;
  text: string;
}

interface SampleCall {
  id: string;
  title: string;
  description: string;
  duration: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  expectedRisk: number;
  scamType: 'investment_scam' | 'romance_scam' | 'impersonation_scam' | 'none';
  icon: string;
  segments: AudioSegment[];
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

// Sample audio calls data
const SAMPLE_CALLS: SampleCall[] = [
  {
    id: 'investment_scam_1',
    title: 'Investment Scam Call',
    description: 'Guaranteed returns promise with margin call urgency',
    duration: 45,
    riskLevel: 'HIGH',
    expectedRisk: 87,
    scamType: 'investment_scam',
    icon: '💰',
    segments: [
      { speaker: 'agent', start: 0, duration: 5, text: "Good afternoon, HSBC customer service. This is Sarah, how can I help you today?" },
      { speaker: 'customer', start: 5, duration: 8, text: "Hi Sarah, I need to make an urgent transfer. My investment advisor called this morning saying I need to add more funds to my trading account to cover a margin call." },
      { speaker: 'agent', start: 13, duration: 4, text: "I understand. Can you tell me more about this trading account and how much you need to transfer?" },
      { speaker: 'customer', start: 17, duration: 12, text: "They said I need to transfer fifteen thousand pounds immediately or I'll lose all my previous investments. The advisor, James, has been guaranteeing thirty-five percent monthly returns and said this is just a temporary margin requirement." },
      { speaker: 'agent', start: 29, duration: 6, text: "I see. Have you been able to withdraw any profits from this investment previously?" },
      { speaker: 'customer', start: 35, duration: 10, text: "Well, no, not yet. But James said all successful traders go through this and I just need to be brave and trust the process. He said the returns are guaranteed." }
    ]
  },
  {
    id: 'romance_scam_1',
    title: 'Romance Scam Call',
    description: 'Online relationship emergency payment request',
    duration: 38,
    riskLevel: 'MEDIUM',
    expectedRisk: 72,
    scamType: 'romance_scam',
    icon: '💕',
    segments: [
      { speaker: 'agent', start: 0, duration: 3, text: "Good morning, HSBC. How can I assist you today?" },
      { speaker: 'customer', start: 3, duration: 7, text: "I need to send four thousand pounds to Turkey urgently. It's for my partner Alex who is stuck in Istanbul." },
      { speaker: 'agent', start: 10, duration: 3, text: "May I ask who this payment is for and how you know them?" },
      { speaker: 'customer', start: 13, duration: 10, text: "We've been together for seven months. We met online and have been talking every day. Alex is a doctor working overseas and needs emergency funds for a visa issue." },
      { speaker: 'agent', start: 23, duration: 4, text: "Have you met Alex in person, and why can't they get help from their employer?" },
      { speaker: 'customer', start: 27, duration: 11, text: "We haven't met in person yet, but we video call all the time. Alex said the hospital won't help with personal visa problems and family can't send money from their country. This is really urgent." }
    ]
  },
  {
    id: 'impersonation_scam_1',
    title: 'Bank Impersonation Scam',
    description: 'Fake HSBC security team requesting credentials',
    duration: 35,
    riskLevel: 'CRITICAL',
    expectedRisk: 94,
    scamType: 'impersonation_scam',
    icon: '🎭',
    segments: [
      { speaker: 'agent', start: 0, duration: 4, text: "Good afternoon, HSBC customer service. This is Mike, how can I help?" },
      { speaker: 'customer', start: 4, duration: 9, text: "Someone from bank security called saying my account has been compromised. They asked me to confirm my card details and PIN to verify my identity immediately." },
      { speaker: 'agent', start: 13, duration: 5, text: "I see. Did they ask for your PIN or online banking password over the phone?" },
      { speaker: 'customer', start: 18, duration: 8, text: "Yes, they said they needed my full card number, expiry date, and PIN to secure my account before it gets frozen." },
      { speaker: 'agent', start: 26, duration: 9, text: "I need to stop you there. HSBC will never ask for your PIN or full card details over the phone. This sounds like a scam. Have you given them any information?" }
    ]
  },
  {
    id: 'legitimate_call_1',
    title: 'Legitimate Banking Inquiry',
    description: 'Normal account balance and transaction query',
    duration: 25,
    riskLevel: 'LOW',
    expectedRisk: 8,
    scamType: 'none',
    icon: '✅',
    segments: [
      { speaker: 'agent', start: 0, duration: 4, text: "Good morning, HSBC customer service. This is Emma, how can I help you today?" },
      { speaker: 'customer', start: 4, duration: 4, text: "Hi Emma, I'd like to check my current account balance please." },
      { speaker: 'agent', start: 8, duration: 5, text: "Certainly, I can help you with that. Can I take you through security verification first?" },
      { speaker: 'customer', start: 13, duration: 6, text: "Of course. I also wanted to ask about setting up a standing order for my rent payment." },
      { speaker: 'agent', start: 19, duration: 6, text: "No problem at all. Let me help you with both the balance inquiry and setting up that standing order." }
    ]
  }
];

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

// Pattern detection simulation
const detectPatterns = (text: string, scamType: string): Record<string, DetectedPattern> => {
  const patterns: Record<string, Record<string, string[]>> = {
    investment_scam: {
      'guaranteed_returns': ['guaranteed', 'guarantee', 'thirty-five percent', 'monthly returns'],
      'urgency_pressure': ['urgent', 'immediately', 'margin call', 'lose all'],
      'third_party_instructions': ['advisor', 'James', 'called', 'said']
    },
    romance_scam: {
      'romance_exploitation': ['partner', 'together', 'online', 'seven months'],
      'urgency_pressure': ['urgent', 'emergency', 'stuck'],
      'overseas_story': ['Turkey', 'Istanbul', 'doctor', 'visa issue']
    },
    impersonation_scam: {
      'authority_impersonation': ['bank security', 'account compromised', 'verify'],
      'credential_requests': ['card details', 'PIN', 'card number', 'expiry date'],
      'urgency_pressure': ['immediately', 'frozen', 'secure']
    }
  };

  const detected: Record<string, DetectedPattern> = {};
  const scamPatterns = patterns[scamType] || {};
  
  Object.entries(scamPatterns).forEach(([patternName, keywords]) => {
    const matches = keywords.filter(keyword => 
      text.toLowerCase().includes(keyword.toLowerCase())
    );
    
    if (matches.length > 0) {
      detected[patternName] = {
        pattern_name: patternName,
        matches: matches,
        count: matches.length,
        weight: patternName === 'credential_requests' ? 40 : 
                patternName === 'authority_impersonation' ? 35 : 
                patternName === 'guaranteed_returns' ? 30 : 25,
        severity: patternName === 'credential_requests' ? 'critical' : 'high'
      };
    }
  });
  
  return detected;
};

// Policy guidance based on detected patterns and scam type
const getPolicyGuidance = (scamType: string, riskScore: number, detectedPatterns: Record<string, DetectedPattern>): PolicyGuidance => {
  const guidanceMap: Record<string, PolicyGuidance> = {
    investment_scam: {
      immediate_alerts: [
        "🚨 INVESTMENT SCAM DETECTED - Guaranteed returns claim",
        "🛑 DO NOT PROCESS TRANSFER - High fraud risk",
        "📞 ESCALATE TO FRAUD TEAM IMMEDIATELY"
      ],
      recommended_actions: [
        "Immediately halt any investment-related transfers",
        "Ask: 'Have you been able to withdraw any profits from this investment?'",
        "Verify investment company against regulatory register",
        "Explain: All legitimate investments carry risk - guarantees indicate fraud",
        "Document company name and promised returns for investigation"
      ],
      key_questions: [
        "Have you been able to withdraw any money from this investment?",
        "Did they guarantee specific percentage returns?",
        "How did this investment company first contact you?",
        "Can you check if they are regulated by financial authorities?"
      ],
      customer_education: [
        "All legitimate investments carry risk - guaranteed returns are impossible",
        "Real investment firms are registered and regulated by financial authorities",
        "High-pressure sales tactics are warning signs of fraud",
        "Take time to research - legitimate opportunities don't require immediate action"
      ]
    },
    romance_scam: {
      immediate_alerts: [
        "⚠️ ROMANCE SCAM SUSPECTED - Online relationship money request",
        "🔍 ENHANCED VERIFICATION REQUIRED",
        "📋 DOCUMENT ALL DETAILS THOROUGHLY"
      ],
      recommended_actions: [
        "Stop transfers to individuals customer has never met in person",
        "Ask: 'Have you met this person face-to-face?'",
        "Explain romance scam patterns: relationship building followed by emergency requests",
        "Check customer's transfer history for previous payments",
        "Advise customer to discuss with family/friends before proceeding"
      ],
      key_questions: [
        "Have you met this person in person?",
        "Have they asked for money before?",
        "Why can't they get help from family or their employer?",
        "How long have you known them?",
        "What platform did you meet them on?"
      ],
      customer_education: [
        "Romance scammers build relationships over months to gain trust",
        "They often claim to be overseas, in military, or traveling for work",
        "Real partners don't repeatedly ask for money, especially for emergencies",
        "Video calls can be faked - only meeting in person confirms identity"
      ]
    },
    impersonation_scam: {
      immediate_alerts: [
        "🚨 CRITICAL: Bank impersonation scam detected",
        "🛑 DO NOT PROCESS ANY PAYMENTS - STOP TRANSACTION",
        "📞 ESCALATE TO FRAUD TEAM IMMEDIATELY",
        "🔒 CONSIDER ACCOUNT SECURITY MEASURES"
      ],
      recommended_actions: [
        "Immediately inform customer: Banks never ask for PINs over phone",
        "Instruct customer to hang up and call official number independently",
        "Explain: Legitimate authorities send official letters before taking action",
        "Document all impersonation details including claimed authority",
        "Report incident to relevant authority fraud departments"
      ],
      key_questions: [
        "What number did they call you from?",
        "Did they ask for your PIN or online banking password?",
        "Have you received any official letters about this matter?",
        "Why do they claim payment is needed immediately?"
      ],
      customer_education: [
        "Banks will never ask for your PIN, password, or full card details over the phone",
        "Legitimate authorities send official letters before taking any action",
        "Government agencies don't accept payments via bank transfer or gift cards",
        "When in doubt, hang up and call the organization's official number"
      ]
    }
  };

  const guidance = guidanceMap[scamType] || {
    immediate_alerts: [],
    recommended_actions: ["Continue standard customer service procedures"],
    key_questions: [],
    customer_education: []
  };

  // Add escalation if high risk
  if (riskScore >= 80 && !guidance.immediate_alerts.some((alert: string) => alert.includes('ESCALATE'))) {
    guidance.immediate_alerts.push("📞 ESCALATE TO FRAUD TEAM IMMEDIATELY");
  }

  return guidance;
};

function App() {
  const [selectedCall, setSelectedCall] = useState<SampleCall | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [currentSegment, setCurrentSegment] = useState<number>(0);
  const [transcription, setTranscription] = useState<string>('');
  const [riskScore, setRiskScore] = useState<number>(0);
  const [riskLevel, setRiskLevel] = useState<string>('MINIMAL');
  const [detectedPatterns, setDetectedPatterns] = useState<Record<string, DetectedPattern>>({});
  const [policyGuidance, setPolicyGuidance] = useState<PolicyGuidance | null>(null);
  const [processingStage, setProcessingStage] = useState<string>('');
  const [showingSegments, setShowingSegments] = useState<AudioSegment[]>([]);
  
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const hasProcessedSegments = useRef<Set<number>>(new Set());

  // Reset state when new call is selected
  const resetState = (): void => {
    setCurrentTime(0);
    setCurrentSegment(0);
    setTranscription('');
    setRiskScore(0);
    setRiskLevel('MINIMAL');
    setDetectedPatterns({});
    setPolicyGuidance(null);
    setProcessingStage('');
    setShowingSegments([]);
    hasProcessedSegments.current.clear();
  };

  // Stop playing
  const stopPlaying = (): void => {
    setIsPlaying(false);
    setProcessingStage('⏸️ Paused');
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

// Start playing audio simulation
  const startPlaying = (call: SampleCall): void => {
    if (selectedCall?.id !== call.id) {
      setSelectedCall(call);
      resetState();
    }
    
    setIsPlaying(true);
    setProcessingStage('🎵 Audio Processing Agent - Streaming audio...');
    
    intervalRef.current = setInterval(() => {
      setCurrentTime(prevTime => {
        const newTime = prevTime + 0.1;
        
        // Check if we should show a new segment
        const activeSegment = call.segments.find(
          (seg: AudioSegment) => newTime >= seg.start && newTime < (seg.start + seg.duration)
        );
        
        if (activeSegment && !hasProcessedSegments.current.has(activeSegment.start)) {
          hasProcessedSegments.current.add(activeSegment.start);
          
          // Add to showing segments
          setTimeout(() => {
            setShowingSegments(prev => [...prev, activeSegment]);
            
            // Update transcription
            setTranscription(prev => {
              const newText = prev + (prev ? ' ' : '') + activeSegment.text;
              
              // Only process customer speech for fraud detection
              if (activeSegment.speaker === 'customer') {
                setProcessingStage('🔍 Fraud Detection Agent - Analyzing patterns...');
                
                // Simulate fraud detection processing
                setTimeout(() => {
                  const patterns = detectPatterns(newText, call.scamType);
                  setDetectedPatterns(patterns);
                  
                  // Calculate risk score based on patterns and progress
                  const progressFactor = Math.min(newTime / call.duration, 1);
                  const targetRisk = call.expectedRisk;
                  const newRisk = Math.floor(targetRisk * progressFactor);
                  
                  setRiskScore(newRisk);
                  
                  if (newRisk >= 80) setRiskLevel('CRITICAL');
                  else if (newRisk >= 60) setRiskLevel('HIGH');
                  else if (newRisk >= 40) setRiskLevel('MEDIUM');
                  else if (newRisk >= 20) setRiskLevel('LOW');
                  else setRiskLevel('MINIMAL');
                  
                  setProcessingStage('📚 Policy Guidance Agent - Retrieving procedures...');
                  
                  // Get policy guidance
                  setTimeout(() => {
                    const guidance = getPolicyGuidance(call.scamType, newRisk, patterns);
                    setPolicyGuidance(guidance);
                    setProcessingStage('🎭 Master Orchestrator - Coordinating response...');
                  }, 500);
                }, 300);
              }
              
              return newText;
            });
          }, 100);
        }
        
        // Stop at end of call
        if (newTime >= call.duration) {
          setIsPlaying(false);
          setProcessingStage('✅ Analysis Complete - All agents processed');
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
          }
          return call.duration;
        }
        
        return newTime;
      });
    }, 100);
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
                  Real-time Multi-Agent Call Analysis Demo
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="px-3 py-1 rounded-full text-sm bg-green-100 text-green-800">
                System: Operational
              </div>
              <div className="text-sm text-gray-600">6 Agents Active</div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Sample Calls Section */}
        <div className="mb-8">
          <div className="flex items-center space-x-2 mb-6">
            <PhoneCall className="w-6 h-6 text-blue-600" />
            <h2 className="text-2xl font-bold text-gray-900">Demo Call Recordings</h2>
            <span className="text-sm text-gray-500">Click to analyze in real-time</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {SAMPLE_CALLS.map((call) => (
              <div
                key={call.id}
                className={`bg-white rounded-lg shadow-md p-6 cursor-pointer border-2 transition-all hover:shadow-lg ${
                  selectedCall?.id === call.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                }`}
                onClick={() => !isPlaying ? startPlaying(call) : null}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-2xl">{call.icon}</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    call.riskLevel === 'CRITICAL' ? 'bg-red-100 text-red-800' :
                    call.riskLevel === 'HIGH' ? 'bg-orange-100 text-orange-800' :
                    call.riskLevel === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {call.riskLevel}
                  </span>
                </div>
                
                <h3 className="font-semibold text-gray-900 mb-2">{call.title}</h3>
                <p className="text-sm text-gray-600 mb-3">{call.description}</p>
                
                <div className="flex items-center justify-between text-sm text-gray-500">
                  <span>{formatTime(call.duration)}</span>
                  <span>Risk: {call.expectedRisk}%</span>
                </div>
                
                {selectedCall?.id === call.id && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <div className="flex items-center justify-between">
                      {!isPlaying ? (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            startPlaying(call);
                          }}
                          className="flex items-center space-x-2 px-3 py-1 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
                        >
                          <Play className="w-4 h-4" />
                          <span>Play</span>
                        </button>
                      ) : (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            stopPlaying();
                          }}
                          className="flex items-center space-x-2 px-3 py-1 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700"
                        >
                          <Pause className="w-4 h-4" />
                          <span>Pause</span>
                        </button>
                      )}
                      <span className="text-xs text-gray-600">
                        {formatTime(currentTime)} / {formatTime(call.duration)}
                      </span>
                    </div>
                    
                    {/* Progress bar */}
                    <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-100"
                        style={{ width: `${(currentTime / call.duration) * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Processing Status */}
        {processingStage && (
          <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <Brain className="w-5 h-5 text-blue-600 animate-pulse" />
              <span className="text-blue-800 font-medium">{processingStage}</span>
            </div>
          </div>
        )}

        {/* Main Analysis Grid */}
        {selectedCall && (
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
                      <p className="text-sm text-gray-600 mt-1 capitalize">
                        {selectedCall.scamType.replace('_', ' ')}
                      </p>
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
              Notice how <strong>Investment Scam</strong> calls trigger immediate escalation, while 
              <strong>Romance Scam</strong> calls focus on education and verification questions.
            </p>
          </div>
        </div>

        {/* System Performance Metrics */}
        {selectedCall && riskScore > 0 && (
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
