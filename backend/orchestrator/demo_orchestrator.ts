## File: backend/orchestrator/demo_orchestrator.ts

```typescript
// backend/orchestrator/demo_orchestrator.ts - Demo Mode Integration
"""
Demo orchestrator that integrates scripted demo events with the main fraud detection orchestrator
Provides seamless demo experience while maintaining production code paths
"""

import { DemoScript, DemoScriptEvent, DemoScriptRunner, DemoEventHandler, DEMO_CONFIG } from '../config/demo_scripts';
import { FraudDetectionOrchestrator } from './fraud_detection_orchestrator';
import { get_current_timestamp } from '../utils';

interface DemoState {
  enabled: boolean;
  scriptedFlow: boolean;
  autoSimulateAgent: boolean;
  showExplanations: boolean;
  showRiskBreakdown: boolean;
  currentScript: DemoScript | null;
  currentRunner: DemoScriptRunner | null;
  lastTriggerTime: number;
  demoStartTime: number;
}

interface DemoExplanation {
  title: string;
  message: string;
  type: 'pattern' | 'question' | 'action' | 'risk' | 'compliance';
  riskLevel?: number;
  duration?: number;
}

export class DemoOrchestrator {
  private fraudOrchestrator: FraudDetectionOrchestrator;
  private demoState: DemoState;
  private explanationQueue: DemoExplanation[] = [];
  
  constructor(fraudOrchestrator: FraudDetectionOrchestrator) {
    this.fraudOrchestrator = fraudOrchestrator;
    this.demoState = {
      enabled: false,
      scriptedFlow: true,
      autoSimulateAgent: true,
      showExplanations: true,
      showRiskBreakdown: true,
      currentScript: null,
      currentRunner: null,
      lastTriggerTime: 0,
      demoStartTime: 0
    };
  }
  
  // ===== DEMO LIFECYCLE METHODS =====
  
  async startDemo(script: DemoScript, callback?: Function): Promise<boolean> {
    try {
      console.log(`🎭 Starting demo: ${script.title}`);
      
      this.demoState.enabled = true;
      this.demoState.currentScript = script;
      this.demoState.demoStartTime = Date.now();
      
      // Create demo event handlers
      const handlers: DemoEventHandler = {
        onDemoEvent: async (event: DemoScriptEvent) => {
          await this.handleDemoEvent(event, callback);
        },
        onDemoComplete: async () => {
          await this.handleDemoComplete(callback);
        },
        onDemoError: async (error: string) => {
          await this.handleDemoError(error, callback);
        }
      };
      
      // Create and start demo runner
      this.demoState.currentRunner = new DemoScriptRunner(script, handlers);
      this.demoState.currentRunner.start();
      
      // Send initial demo started message
      if (callback) {
        await callback({
          type: 'demo_started',
          data: {
            script_title: script.title,
            duration: script.duration,
            scam_type: script.scamType,
            customer_profile: script.customerProfile,
            timeline_events: script.timeline.length,
            demo_config: this.getDemoConfig(),
            timestamp: get_current_timestamp()
          }
        });
      }
      
      return true;
      
    } catch (error) {
      console.error('❌ Demo start error:', error);
      return false;
    }
  }
  
  async stopDemo(callback?: Function): Promise<void> {
    try {
      console.log('🛑 Stopping demo');
      
      if (this.demoState.currentRunner) {
        this.demoState.currentRunner.stop();
      }
      
      this.demoState.enabled = false;
      this.demoState.currentScript = null;
      this.demoState.currentRunner = null;
      this.explanationQueue = [];
      
      if (callback) {
        await callback({
          type: 'demo_stopped',
          data: {
            timestamp: get_current_timestamp()
          }
        });
      }
      
    } catch (error) {
      console.error('❌ Demo stop error:', error);
    }
  }
  
  // ===== DEMO EVENT HANDLING =====
  
  private async handleDemoEvent(event: DemoScriptEvent, callback?: Function): Promise<void> {
    try {
      console.log(`🎭 Demo event triggered at ${event.triggerAtSeconds}s: ${event.suggestedQuestion.question}`);
      
      this.demoState.lastTriggerTime = Date.now();
      
      // Send risk and pattern updates first
      await this.sendRiskUpdate(event, callback);
      
      // Show explanation if enabled
      if (this.demoState.showExplanations) {
        await this.showExplanation({
          title: 'Pattern Detection',
          message: event.suggestedQuestion.explanation,
          type: 'pattern',
          riskLevel: event.riskScore,
          duration: DEMO_CONFIG.EXPLANATION_DURATION
        }, callback);
      }
      
      // Send question prompt
      await this.sendQuestionPrompt(event, callback);
      
      // Auto-simulate agent response if enabled
      if (this.demoState.autoSimulateAgent) {
        setTimeout(async () => {
          await this.simulateAgentResponse(event, callback);
        }, event.simulateAskedAfterMs);
      }
      
    } catch (error) {
      console.error('❌ Demo event handling error:', error);
    }
  }
  
  private async sendRiskUpdate(event: DemoScriptEvent, callback?: Function): Promise<void> {
    if (!callback) return;
    
    await callback({
      type: 'fraud_analysis_update',
      data: {
        session_id: 'demo_session',
        risk_score: event.riskScore,
        risk_level: this.getRiskLevel(event.riskScore),
        scam_type: this.demoState.currentScript?.scamType || 'unknown',
        detected_patterns: event.detectedPatterns,
        confidence: Math.min(event.riskScore / 100, 0.95),
        explanation: event.suggestedQuestion.explanation,
        demo_mode: true,
        banking_context: event.bankingContext,
        compliance_note: event.complianceNote,
        timestamp: get_current_timestamp()
      }
    });
  }
  
  private async sendQuestionPrompt(event: DemoScriptEvent, callback?: Function): Promise<void> {
    if (!callback) return;
    
    await callback({
      type: 'question_prompt_ready',
      data: {
        session_id: 'demo_session',
        question: event.suggestedQuestion.question,
        context: event.suggestedQuestion.context,
        urgency: event.suggestedQuestion.urgency,
        pattern: event.suggestedQuestion.pattern || 'demo_pattern',
        risk_level: event.riskScore,
        question_id: `demo_q_${Date.now()}`,
        agent_action: event.agentAction,
        banking_context: event.bankingContext,
        auto_simulate_in_ms: this.demoState.autoSimulateAgent ? event.simulateAskedAfterMs : null,
        demo_mode: true,
        timestamp: get_current_timestamp()
      }
    });
  }
  
  private async simulateAgentResponse(event: DemoScriptEvent, callback?: Function): Promise<void> {
    if (!callback) return;
    
    // Show agent action explanation
    if (this.demoState.showExplanations) {
      await this.showExplanation({
        title: 'Agent Action',
        message: event.agentAction,
        type: 'action',
        duration: 3000
      }, callback);
    }
    
    // Send agent response simulation
    await callback({
      type: 'agent_question_simulated',
      data: {
        session_id: 'demo_session',
        question_asked: event.suggestedQuestion.question,
        agent_action: event.agentAction,
        compliance_note: event.complianceNote,
        banking_context: event.bankingContext,
        demo_mode: true,
        timestamp: get_current_timestamp()
      }
    });
  }
  
  private async showExplanation(explanation: DemoExplanation, callback?: Function): Promise<void> {
    if (!callback || !this.demoState.showExplanations) return;
    
    await callback({
      type: 'demo_explanation',
      data: {
        session_id: 'demo_session',
        explanation: explanation,
        demo_config: this.getDemoConfig(),
        timestamp: get_current_timestamp()
      }
    });
  }
  
  private async handleDemoComplete(callback?: Function): Promise<void> {
    try {
      console.log('✅ Demo completed');
      
      const script = this.demoState.currentScript;
      if (!script) return;
      
      // Send completion summary
      if (callback) {
        await callback({
          type: 'demo_complete',
          data: {
            script_title: script.title,
            final_risk_score: this.getFinalRiskScore(),
            scam_type: script.scamType,
            expected_outcome: script.expectedOutcome,
            compliance_level: script.complianceLevel,
            total_events: script.timeline.length,
            demo_duration: (Date.now() - this.demoState.demoStartTime) / 1000,
            timestamp: get_current_timestamp()
          }
        });
      }
      
      // Auto-cleanup after a delay
      setTimeout(() => {
        this.stopDemo(callback);
      }, 5000);
      
    } catch (error) {
      console.error('❌ Demo completion error:', error);
    }
  }
  
  private async handleDemoError(error: string, callback?: Function): Promise<void> {
    console.error('❌ Demo error:', error);
    
    if (callback) {
      await callback({
        type: 'demo_error',
        data: {
          error: error,
          timestamp: get_current_timestamp()
        }
      });
    }
    
    await this.stopDemo(callback);
  }
  
  // ===== DEMO UTILITIES =====
  
  private getRiskLevel(score: number): string {
    if (score >= 80) return 'CRITICAL';
    if (score >= 60) return 'HIGH';
    if (score >= 40) return 'MEDIUM';
    if (score >= 20) return 'LOW';
    return 'MINIMAL';
  }
  
  private getFinalRiskScore(): number {
    if (!this.demoState.currentScript) return 0;
    const timeline = this.demoState.currentScript.timeline;
    return timeline.length > 0 ? timeline[timeline.length - 1].riskScore : 0;
  }
  
  private getDemoConfig() {
    return {
      enabled: this.demoState.enabled,
      scriptedFlow: this.demoState.scriptedFlow,
      autoSimulateAgent: this.demoState.autoSimulateAgent,
      showExplanations: this.demoState.showExplanations,
      showRiskBreakdown: this.demoState.showRiskBreakdown,
      current_script: this.demoState.currentScript?.title || null,
      progress: this.demoState.currentRunner?.getCurrentProgress() || 0
    };
  }
  
  // ===== PUBLIC METHODS =====
  
  isDemoEnabled(): boolean {
    return this.demoState.enabled;
  }
  
  getDemoProgress(): number {
    return this.demoState.currentRunner?.getCurrentProgress() || 0;
  }
  
  getNextDemoEvent(): DemoScriptEvent | null {
    return this.demoState.currentRunner?.getNextEvent() || null;
  }
  
  updateDemoSettings(settings: Partial<DemoState>): void {
    Object.assign(this.demoState, settings);
  }
  
  getDemoStatus() {
    return {
      ...this.demoState,
      progress: this.getDemoProgress(),
      next_event: this.getNextDemoEvent()?.suggestedQuestion.question || null,
      time_since_last_trigger: this.demoState.lastTriggerTime ? 
        Date.now() - this.demoState.lastTriggerTime : 0
    };
  }
}

export default DemoOrchestrator;
```
