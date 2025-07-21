## File: backend/config/demo_scripts.ts

```typescript
// backend/config/demo_scripts.ts - Complete Demo Scripts Configuration
"""
Complete scripted demo system with pre-defined timelines, risk progression,
and automatic question triggering for professional banking demonstrations
"""

import { DetectedPattern } from '../api/models';

// ===== DEMO SCRIPT INTERFACES =====

interface DemoScriptEvent {
  triggerAtSeconds: number;
  customerText: string;
  detectedPatterns: Record<string, DetectedPattern>;
  riskScore: number;
  riskFactors: string[];
  suggestedQuestion: {
    question: string;
    context: string;
    urgency: 'low' | 'medium' | 'high' | 'critical';
    explanation: string;
    pattern?: string;
  };
  simulateAskedAfterMs: number;
  agentAction: string;
  complianceNote: string;
  bankingContext?: {
    transactionType?: string;
    amount?: string;
    beneficiary?: string;
    riskFlags?: string[];
  };
}

interface DemoScript {
  title: string;
  description: string;
  customerProfile: string;
  initialRisk: number;
  timeline: DemoScriptEvent[];
  duration: number;
  scamType: string;
  expectedOutcome: string;
  complianceLevel: 'standard' | 'enhanced' | 'critical';
}

// ===== COMPLETE DEMO SCRIPTS =====

export const DEMO_SCRIPTS: Record<string, DemoScript> = {
  'romance_scam_1.wav': {
    title: 'Romance Scam Investigation Demo',
    description: 'Progressive detection of romance scam with emotional manipulation tactics',
    customerProfile: 'Mrs Patricia Williams - Premier Banking Customer',
    initialRisk: 5,
    duration: 65,
    scamType: 'romance_scam',
    expectedOutcome: 'Transaction blocked, case escalated to Financial Crime Team',
    complianceLevel: 'critical',
    timeline: [
      {
        triggerAtSeconds: 12,
        customerText: "I need to make an urgent international transfer to Turkey",
        detectedPatterns: {
          'Urgency Pressure': {
            pattern_name: 'Urgency Pressure',
            count: 1,
            weight: 15,
            severity: 'medium' as const,
            matches: ['urgent', 'international transfer']
          }
        },
        riskScore: 20,
        riskFactors: ['urgency_pressure', 'overseas_beneficiary'],
        suggestedQuestion: {
          question: "What specifically makes this transaction urgent?",
          context: "Urgency pressure detected in customer request",
          urgency: 'medium',
          explanation: "AI detected urgency keywords 'urgent' and 'international' - common fraud indicators",
          pattern: "Urgency Pressure"
        },
        simulateAskedAfterMs: 3000,
        agentAction: "Agent investigates the claimed urgency",
        complianceNote: "Following FCA guidelines for APP fraud prevention",
        bankingContext: {
          transactionType: "International Transfer",
          amount: "£3,000",
          beneficiary: "Unknown - Turkey",
          riskFlags: ["Urgency", "International", "High-risk country"]
        }
      },
      
      {
        triggerAtSeconds: 28,
        customerText: "My boyfriend is stuck overseas and needs money for hospital bills",
        detectedPatterns: {
          'Urgency Pressure': {
            pattern_name: 'Urgency Pressure',
            count: 2,
            weight: 15,
            severity: 'medium' as const,
            matches: ['urgent', 'stuck', 'needs money']
          },
          'Romance Exploitation': {
            pattern_name: 'Romance Exploitation',
            count: 1,
            weight: 25,
            severity: 'high' as const,
            matches: ['boyfriend', 'overseas', 'money']
          },
          'Emergency Money Request': {
            pattern_name: 'Emergency Money Request',
            count: 1,
            weight: 18,
            severity: 'high' as const,
            matches: ['hospital bills', 'needs money']
          }
        },
        riskScore: 45,
        riskFactors: ['urgency_pressure', 'romance_exploitation', 'emergency_money_request', 'overseas_beneficiary'],
        suggestedQuestion: {
          question: "How did you first meet this person?",
          context: "Romance scam indicators detected - investigating relationship origin",
          urgency: 'high',
          explanation: "AI detected romance scam pattern: boyfriend + overseas + emergency money request",
          pattern: "Romance Exploitation"
        },
        simulateAskedAfterMs: 3500,
        agentAction: "Agent probes relationship background and authenticity",
        complianceNote: "Enhanced due diligence required for relationship-based transfers",
        bankingContext: {
          transactionType: "Emergency Transfer",
          amount: "£3,000",
          beneficiary: "Romantic partner - Turkey",
          riskFlags: ["Romance scam", "Never met in person", "Emergency request", "Medical emergency"]
        }
      },
      
      {
        triggerAtSeconds: 45,
        customerText: "We met online through a dating app about 3 weeks ago. He's very sweet and caring",
        detectedPatterns: {
          'Romance Exploitation': {
            pattern_name: 'Romance Exploitation',
            count: 2,
            weight: 25,
            severity: 'high' as const,
            matches: ['met online', 'dating app', 'sweet and caring']
          },
          'Online Relationship': {
            pattern_name: 'Online Relationship',
            count: 1,
            weight: 12,
            severity: 'medium' as const,
            matches: ['met online', 'dating app']
          },
          'Short Relationship Duration': {
            pattern_name: 'Short Relationship Duration',
            count: 1,
            weight: 8,
            severity: 'medium' as const,
            matches: ['3 weeks ago']
          }
        },
        riskScore: 68,
        riskFactors: ['urgency_pressure', 'romance_exploitation', 'emergency_money_request', 'overseas_beneficiary', 'online_relationship', 'short_relationship'],
        suggestedQuestion: {
          question: "Have you met this person in person?",
          context: "Short online relationship with emergency money request - high fraud risk",
          urgency: 'high',
          explanation: "Critical red flag: Online relationship + 3 weeks duration + emergency money = classic romance scam",
          pattern: "Online Relationship"
        },
        simulateAskedAfterMs: 3000,
        agentAction: "Agent investigates physical meeting to confirm authenticity",
        complianceNote: "APP fraud prevention - verifying genuine relationship",
        bankingContext: {
          transactionType: "High-Risk Transfer",
          amount: "£3,000",
          beneficiary: "Online dating contact - Never met",
          riskFlags: ["Online only", "Short duration", "Dating app", "No verification"]
        }
      },
      
      {
        triggerAtSeconds: 62,
        customerText: "No, we haven't met in person yet, but he says he loves me and needs my help. He asked me not to tell anyone about this",
        detectedPatterns: {
          'Romance Exploitation': {
            pattern_name: 'Romance Exploitation',
            count: 3,
            weight: 25,
            severity: 'critical' as const,
            matches: ['loves me', 'needs my help', 'never met']
          },
          'Never Met Person': {
            pattern_name: 'Never Met Beneficiary',
            count: 1,
            weight: 20,
            severity: 'high' as const,
            matches: ['haven\'t met in person']
          },
          'Secrecy Request': {
            pattern_name: 'Secrecy Request',
            count: 1,
            weight: 22,
            severity: 'critical' as const,
            matches: ['not to tell anyone']
          }
        },
        riskScore: 95,
        riskFactors: ['urgency_pressure', 'romance_exploitation', 'emergency_money_request', 'overseas_beneficiary', 'online_relationship', 'short_relationship', 'never_met_person', 'secrecy_request'],
        suggestedQuestion: {
          question: "Why would someone who loves you ask you to keep financial help secret from family and friends?",
          context: "CRITICAL: All major romance scam indicators present",
          urgency: 'critical',
          explanation: "FRAUD ALERT: Never met + love claims + secrecy + emergency money = 95% fraud probability",
          pattern: "Secrecy Request"
        },
        simulateAskedAfterMs: 4000,
        agentAction: "Agent challenges the secrecy and emotional manipulation tactics",
        complianceNote: "MANDATORY INTERVENTION: All APP fraud indicators present - transaction must be blocked",
        bankingContext: {
          transactionType: "BLOCKED TRANSACTION",
          amount: "£3,000",
          beneficiary: "SUSPECTED ROMANCE SCAMMER",
          riskFlags: ["All fraud indicators", "Mandatory block", "FCA compliance", "Customer protection"]
        }
      }
    ]
  },

  'investment_scam_1.wav': {
    title: 'Investment Scam Detection Demo',
    description: 'Progressive detection of high-yield investment fraud with pressure tactics',
    customerProfile: 'Mr Robert Chen - Business Banking Customer',
    initialRisk: 8,
    duration: 58,
    scamType: 'investment_scam',
    expectedOutcome: 'Investment blocked, regulatory warnings provided',
    complianceLevel: 'enhanced',
    timeline: [
      {
        triggerAtSeconds: 15,
        customerText: "I want to invest £25,000 in a cryptocurrency opportunity with guaranteed 15% monthly returns",
        detectedPatterns: {
          'Investment Fraud': {
            pattern_name: 'Investment Fraud',
            count: 1,
            weight: 20,
            severity: 'high' as const,
            matches: ['cryptocurrency', 'guaranteed', '15% monthly']
          },
          'Unrealistic Returns': {
            pattern_name: 'Unrealistic Returns',
            count: 1,
            weight: 25,
            severity: 'high' as const,
            matches: ['guaranteed 15% monthly']
          }
        },
        riskScore: 45,
        riskFactors: ['investment_fraud', 'unrealistic_returns', 'large_amount'],
        suggestedQuestion: {
          question: "What regulatory body oversees this investment opportunity?",
          context: "Unrealistic returns claimed - potential investment scam",
          urgency: 'high',
          explanation: "AI detected guaranteed high returns claim - major red flag for investment fraud",
          pattern: "Investment Fraud"
        },
        simulateAskedAfterMs: 3500,
        agentAction: "Agent investigates regulatory compliance and legitimacy",
        complianceNote: "FCA requires verification of investment firms and realistic return expectations",
        bankingContext: {
          transactionType: "Investment Transfer",
          amount: "£25,000",
          beneficiary: "Cryptocurrency platform",
          riskFlags: ["Guaranteed returns", "Unregulated", "High yield", "Crypto investment"]
        }
      },

      {
        triggerAtSeconds: 32,
        customerText: "They said this is a limited time offer and I need to invest today or I'll miss out. No regulatory oversight needed because it's blockchain-based",
        detectedPatterns: {
          'Investment Fraud': {
            pattern_name: 'Investment Fraud',
            count: 2,
            weight: 20,
            severity: 'high' as const,
            matches: ['limited time', 'no regulatory oversight', 'blockchain-based']
          },
          'Urgency Pressure': {
            pattern_name: 'Urgency Pressure',
            count: 1,
            weight: 15,
            severity: 'medium' as const,
            matches: ['today', 'miss out', 'limited time']
          },
          'Regulatory Avoidance': {
            pattern_name: 'Regulatory Avoidance',
            count: 1,
            weight: 18,
            severity: 'high' as const,
            matches: ['no regulatory oversight']
          }
        },
        riskScore: 75,
        riskFactors: ['investment_fraud', 'unrealistic_returns', 'urgency_pressure', 'regulatory_avoidance', 'large_amount'],
        suggestedQuestion: {
          question: "Why would a legitimate investment opportunity claim to avoid regulatory oversight?",
          context: "Regulatory avoidance and time pressure - classic investment scam tactics",
          urgency: 'high',
          explanation: "Multiple red flags: time pressure + regulatory avoidance + unrealistic returns",
          pattern: "Regulatory Avoidance"
        },
        simulateAskedAfterMs: 4000,
        agentAction: "Agent explains regulatory requirements and warns about scam tactics",
        complianceNote: "All legitimate investments require FCA authorization - explain regulatory protection",
        bankingContext: {
          transactionType: "SUSPICIOUS INVESTMENT",
          amount: "£25,000",
          beneficiary: "Unregulated crypto scheme",
          riskFlags: ["No regulation", "Time pressure", "Blockchain excuse", "Classic scam"]
        }
      },

      {
        triggerAtSeconds: 50,
        customerText: "But they showed me other investors making huge profits. I don't want to miss this chance to make money quickly",
        detectedPatterns: {
          'Investment Fraud': {
            pattern_name: 'Investment Fraud',
            count: 3,
            weight: 20,
            severity: 'critical' as const,
            matches: ['huge profits', 'make money quickly']
          },
          'Social Proof Manipulation': {
            pattern_name: 'Social Proof Manipulation',
            count: 1,
            weight: 15,
            severity: 'medium' as const,
            matches: ['other investors making huge profits']
          },
          'Get Rich Quick': {
            pattern_name: 'Get Rich Quick',
            count: 1,
            weight: 12,
            severity: 'medium' as const,
            matches: ['make money quickly']
          }
        },
        riskScore: 88,
        riskFactors: ['investment_fraud', 'unrealistic_returns', 'urgency_pressure', 'regulatory_avoidance', 'social_proof_manipulation', 'get_rich_quick', 'large_amount'],
        suggestedQuestion: {
          question: "How can you verify these claimed profits are real and not fabricated testimonials?",
          context: "CRITICAL: Classic investment scam with fabricated social proof",
          urgency: 'critical',
          explanation: "SCAM ALERT: Fake testimonials + get-rich-quick + no regulation = investment fraud",
          pattern: "Social Proof Manipulation"
        },
        simulateAskedAfterMs: 3000,
        agentAction: "Agent educates about fake testimonials and investment scam protection",
        complianceNote: "BLOCK TRANSACTION: Classic investment scam - customer protection required",
        bankingContext: {
          transactionType: "BLOCKED - INVESTMENT SCAM",
          amount: "£25,000",
          beneficiary: "CONFIRMED SCAM OPERATION",
          riskFlags: ["All scam indicators", "Customer protection", "Transaction blocked", "FCA compliance"]
        }
      }
    ]
  },

  'impersonation_scam_1.wav': {
    title: 'Bank Impersonation Scam Demo',
    description: 'Detection of criminals impersonating HSBC security team',
    customerProfile: 'Mrs Sarah Foster - Personal Banking Customer',
    initialRisk: 10,
    duration: 42,
    scamType: 'impersonation_scam',
    expectedOutcome: 'Impersonation detected, security measures activated',
    complianceLevel: 'enhanced',
    timeline: [
      {
        triggerAtSeconds: 18,
        customerText: "I received a call from HSBC security saying my account was compromised and I need to verify my details immediately",
        detectedPatterns: {
          'Impersonation Scam': {
            pattern_name: 'Impersonation Scam',
            count: 1,
            weight: 22,
            severity: 'high' as const,
            matches: ['HSBC security', 'account compromised', 'verify details']
          },
          'Urgency Pressure': {
            pattern_name: 'Urgency Pressure',
            count: 1,
            weight: 15,
            severity: 'medium' as const,
            matches: ['immediately']
          }
        },
        riskScore: 42,
        riskFactors: ['impersonation_scam', 'urgency_pressure', 'credential_request'],
        suggestedQuestion: {
          question: "What specific department did they claim to be calling from, and did they provide a reference number?",
          context: "Potential bank impersonation - verifying authenticity",
          urgency: 'high',
          explanation: "AI detected bank impersonation pattern - HSBC never requests verification over phone",
          pattern: "Impersonation Scam"
        },
        simulateAskedAfterMs: 3200,
        agentAction: "Agent verifies call legitimacy and explains security protocols",
        complianceNote: "Bank impersonation scams require immediate customer education",
        bankingContext: {
          transactionType: "Security Verification",
          amount: "N/A",
          beneficiary: "Fraudulent caller",
          riskFlags: ["Bank impersonation", "Credential harvesting", "Phone fraud"]
        }
      },

      {
        triggerAtSeconds: 35,
        customerText: "They asked for my PIN and said they needed to move my money to a safe account. They knew some of my details already",
        detectedPatterns: {
          'Impersonation Scam': {
            pattern_name: 'Impersonation Scam',
            count: 2,
            weight: 22,
            severity: 'critical' as const,
            matches: ['asked for PIN', 'safe account', 'knew details']
          },
          'Credential Request': {
            pattern_name: 'Credential Request',
            count: 1,
            weight: 25,
            severity: 'critical' as const,
            matches: ['asked for my PIN']
          },
          'Safe Account Scam': {
            pattern_name: 'Safe Account Scam',
            count: 1,
            weight: 20,
            severity: 'high' as const,
            matches: ['move money to safe account']
          }
        },
        riskScore: 85,
        riskFactors: ['impersonation_scam', 'credential_request', 'safe_account_scam', 'social_engineering'],
        suggestedQuestion: {
          question: "Did you provide your PIN or any banking details to this caller?",
          context: "CRITICAL: PIN request confirms impersonation scam - immediate security action needed",
          urgency: 'critical',
          explanation: "SCAM CONFIRMED: HSBC NEVER requests PINs by phone - classic impersonation fraud",
          pattern: "Credential Request"
        },
        simulateAskedAfterMs: 2500,
        agentAction: "Agent initiates immediate security protocol and account protection",
        complianceNote: "CRITICAL: If PIN disclosed, immediate account freeze and card replacement required",
        bankingContext: {
          transactionType: "SECURITY BREACH",
          amount: "Full account access",
          beneficiary: "CRIMINAL IMPERSONATOR",
          riskFlags: ["PIN request", "Impersonation confirmed", "Account compromise", "Emergency action"]
        }
      }
    ]
  },

  'app_scam_1.wav': {
    title: 'Authorized Push Payment Scam Demo',
    description: 'Detection of APP fraud with invoice manipulation',
    customerProfile: 'Mr Adam Henderson - Business Banking Customer',
    initialRisk: 12,
    duration: 48,
    scamType: 'app_scam',
    expectedOutcome: 'APP fraud detected, payment verification required',
    complianceLevel: 'enhanced',
    timeline: [
      {
        triggerAtSeconds: 20,
        customerText: "I need to pay an urgent invoice that just came in. The supplier changed their bank details and needs payment today",
        detectedPatterns: {
          'APP Fraud': {
            pattern_name: 'APP Fraud',
            count: 1,
            weight: 18,
            severity: 'medium' as const,
            matches: ['urgent invoice', 'changed bank details']
          },
          'Urgency Pressure': {
            pattern_name: 'Urgency Pressure',
            count: 1,
            weight: 15,
            severity: 'medium' as const,
            matches: ['urgent', 'today']
          },
          'Bank Detail Change': {
            pattern_name: 'Bank Detail Change',
            count: 1,
            weight: 20,
            severity: 'high' as const,
            matches: ['changed their bank details']
          }
        },
        riskScore: 38,
        riskFactors: ['app_fraud', 'urgency_pressure', 'bank_detail_change'],
        suggestedQuestion: {
          question: "How did you receive notification of the bank detail change?",
          context: "Bank detail changes are common in APP fraud - verification needed",
          urgency: 'medium',
          explanation: "AI detected APP fraud pattern: urgent payment + changed bank details",
          pattern: "Bank Detail Change"
        },
        simulateAskedAfterMs: 3400,
        agentAction: "Agent investigates communication method and authenticity",
        complianceNote: "APP fraud prevention requires verification of bank detail changes",
        bankingContext: {
          transactionType: "Business Payment",
          amount: "£8,500",
          beneficiary: "Supplier with changed details",
          riskFlags: ["Changed bank details", "Urgent payment", "APP fraud risk"]
        }
      },

      {
        triggerAtSeconds: 38,
        customerText: "I got an email that looked exactly like their usual invoices. The managing director called me personally to confirm it was urgent",
        detectedPatterns: {
          'APP Fraud': {
            pattern_name: 'APP Fraud',
            count: 2,
            weight: 18,
            severity: 'high' as const,
            matches: ['email looked exactly like', 'managing director called']
          },
          'Email Spoofing': {
            pattern_name: 'Email Spoofing',
            count: 1,
            weight: 16,
            severity: 'medium' as const,
            matches: ['email looked exactly like usual invoices']
          },
          'Authority Impersonation': {
            pattern_name: 'Authority Impersonation',
            count: 1,
            weight: 14,
            severity: 'medium' as const,
            matches: ['managing director called personally']
          }
        },
        riskScore: 72,
        riskFactors: ['app_fraud', 'urgency_pressure', 'bank_detail_change', 'email_spoofing', 'authority_impersonation'],
        suggestedQuestion: {
          question: "Did you call the supplier back on their usual number to verify this change?",
          context: "HIGH RISK: Email spoofing and authority impersonation detected",
          urgency: 'high',
          explanation: "ALERT: Spoofed emails + impersonation calls = classic APP fraud setup",
          pattern: "Email Spoofing"
        },
        simulateAskedAfterMs: 3800,
        agentAction: "Agent recommends verification through established contact methods",
        complianceNote: "APP fraud: Always verify bank changes through known contact details",
        bankingContext: {
          transactionType: "SUSPICIOUS PAYMENT",
          amount: "£8,500",
          beneficiary: "POTENTIALLY COMPROMISED SUPPLIER",
          riskFlags: ["Email spoofing", "Impersonation", "No verification", "APP fraud"]
        }
      }
    ]
  },

  'legitimate_call_1.wav': {
    title: 'Legitimate Banking Transaction Demo',
    description: 'Example of normal banking conversation with low fraud risk',
    customerProfile: 'Mr Michael Thompson - Personal Banking Customer',
    initialRisk: 2,
    duration: 35,
    scamType: 'none',
    expectedOutcome: 'Standard processing approved',
    complianceLevel: 'standard',
    timeline: [
      {
        triggerAtSeconds: 25,
        customerText: "I'd like to transfer £500 to my daughter's account for her university expenses",
        detectedPatterns: {},
        riskScore: 8,
        riskFactors: ['family_transfer'],
        suggestedQuestion: {
          question: "Can you confirm your daughter's account details for verification?",
          context: "Standard family transfer - routine verification",
          urgency: 'low',
          explanation: "Low risk family transfer - standard verification procedures apply",
          pattern: "Family Transfer"
        },
        simulateAskedAfterMs: 4000,
        agentAction: "Agent performs standard identity and account verification",
        complianceNote: "Standard KYC procedures for family transfers",
        bankingContext: {
          transactionType: "Family Transfer",
          amount: "£500",
          beneficiary: "Family member",
          riskFlags: ["Standard verification", "Low risk", "Legitimate transaction"]
        }
      }
    ]
  }
};

// ===== DEMO CONFIGURATION SETTINGS =====

export const DEMO_CONFIG = {
  // Timing settings
  DEFAULT_SIMULATION_DELAY: 3000,
  MIN_SIMULATION_DELAY: 1500,
  MAX_SIMULATION_DELAY: 5000,
  
  // UI settings
  SHOW_EXPLANATIONS_DEFAULT: true,
  AUTO_SIMULATE_DEFAULT: true,
  SHOW_RISK_BREAKDOWN_DEFAULT: true,
  
  // Demo control settings
  ALLOW_MANUAL_OVERRIDE: true,
  SHOW_DEMO_CONTROLS: true,
  SHOW_PROGRESS_TRACKING: true,
  
  // Explanation display settings
  EXPLANATION_DURATION: 4000,
  RISK_UPDATE_ANIMATION_DURATION: 800,
  QUESTION_HIGHLIGHT_DURATION: 6000,
  
  // Risk thresholds for demo annotations
  RISK_THRESHOLDS: {
    LOW: 25,
    MEDIUM: 45,
    HIGH: 70,
    CRITICAL: 85
  }
} as const;

// ===== DEMO UTILITIES =====

export function getDemoScript(filename: string): DemoScript | null {
  return DEMO_SCRIPTS[filename] || null;
}

export function getAllDemoScripts(): Record<string, DemoScript> {
  return DEMO_SCRIPTS;
}

export function getDemoScriptList(): Array<{id: string, title: string, scamType: string, duration: number}> {
  return Object.entries(DEMO_SCRIPTS).map(([id, script]) => ({
    id,
    title: script.title,
    scamType: script.scamType,
    duration: script.duration
  }));
}

export function getRiskLevelFromScore(score: number): string {
  const { RISK_THRESHOLDS } = DEMO_CONFIG;
  if (score >= RISK_THRESHOLDS.CRITICAL) return 'CRITICAL';
  if (score >= RISK_THRESHOLDS.HIGH) return 'HIGH';
  if (score >= RISK_THRESHOLDS.MEDIUM) return 'MEDIUM';
  if (score >= RISK_THRESHOLDS.LOW) return 'LOW';
  return 'MINIMAL';
}

export function getDemoExplanation(riskScore: number, patterns: string[]): string {
  const riskLevel = getRiskLevelFromScore(riskScore);
  const patternText = patterns.length > 0 ? patterns.join(', ') : 'No patterns';
  
  return `Risk Level: ${riskLevel} (${riskScore}%) - Detected: ${patternText}`;
}

// ===== DEMO EVENT HANDLERS =====

export interface DemoEventHandler {
  onDemoEvent: (event: DemoScriptEvent) => Promise<void>;
  onDemoComplete: () => Promise<void>;
  onDemoError: (error: string) => Promise<void>;
}

export class DemoScriptRunner {
  private script: DemoScript;
  private handlers: DemoEventHandler;
  private currentEventIndex: number = 0;
  private startTime: number = 0;
  private isRunning: boolean = false;
  private timeouts: NodeJS.Timeout[] = [];
  
  constructor(script: DemoScript, handlers: DemoEventHandler) {
    this.script = script;
    this.handlers = handlers;
  }
  
  start(): void {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.startTime = Date.now();
    this.currentEventIndex = 0;
    this.scheduleEvents();
  }
  
  stop(): void {
    this.isRunning = false;
    this.timeouts.forEach(timeout => clearTimeout(timeout));
    this.timeouts = [];
  }
  
  private scheduleEvents(): void {
    this.script.timeline.forEach((event, index) => {
      const timeout = setTimeout(async () => {
        if (this.isRunning) {
          this.currentEventIndex = index + 1;
          await this.handlers.onDemoEvent(event);
          
          // Check if this was the last event
          if (index === this.script.timeline.length - 1) {
            await this.handlers.onDemoComplete();
          }
        }
      }, event.triggerAtSeconds * 1000);
      
      this.timeouts.push(timeout);
    });
  }
  
  getCurrentProgress(): number {
    return this.currentEventIndex / this.script.timeline.length;
  }
  
  getNextEvent(): DemoScriptEvent | null {
    return this.script.timeline[this.currentEventIndex] || null;
  }
}

export default DEMO_SCRIPTS;
```
