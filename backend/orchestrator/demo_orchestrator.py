# backend/orchestrator/demo_orchestrator.py - MINIMAL ESSENTIAL VERSION
"""
Minimal Demo Orchestrator - Essential functionality only
Just triggers timed questions - no overlays, no complex coordination
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class DemoOrchestrator:
    """Minimal demo orchestrator - just timed question triggers"""
    
    def __init__(self, fraud_orchestrator=None):
        self.active_sessions: Dict[str, asyncio.Task] = {}
    
    async def start_demo_overlay(self, script: Dict, callback: Callable, session_id: str) -> bool:
        """Start minimal demo - just question timing"""
        try:
            timeline = script.get('timeline', [])
            if not timeline:
                return False
            
            # Start background task for timed questions
            task = asyncio.create_task(self._run_question_timeline(session_id, timeline, callback))
            self.active_sessions[session_id] = task
            return True
            
        except Exception:
            return False
    
    async def _run_question_timeline(self, session_id: str, timeline: list, callback: Callable):
        """Run timed question prompts"""
        try:
            last_trigger_time = 0
            
            for event in timeline:
                trigger_time = event.get('triggerAtSeconds', 10)
                
                # Calculate interval since last event
                interval = trigger_time - last_trigger_time
                
                if interval > 0:
                    print(f"🎭 DEMO: Waiting {interval} seconds (total: {trigger_time}s)")
                    await asyncio.sleep(interval)
                
                last_trigger_time = trigger_time
                
                # Check if session still active
                if session_id not in self.active_sessions:
                    break
                
                # Send question prompt
                question_data = event.get('suggestedQuestion', {})
                if question_data.get('question'):
                    print(f"🎭 DEMO: Sending question at {trigger_time}s: {question_data['question']}")
                    await callback({
                        'type': 'question_prompt_ready',
                        'data': {
                            'session_id': session_id,
                            'question': question_data['question'],
                            'context': question_data.get('context', ''),
                            'urgency': question_data.get('urgency', 'medium'),
                            'pattern': 'demo_timing',
                            'risk_level': event.get('riskScore', 0),
                            'question_id': f"demo_{session_id}_{trigger_time}"
                        }
                    })
                    
        except asyncio.CancelledError:
            print(f"🎭 DEMO: Timeline cancelled for {session_id}")
            pass
    
    async def stop_session_demo(self, session_id: str):
        """Stop demo session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].cancel()
            del self.active_sessions[session_id]
    
    # Minimal required methods for compatibility
    async def handle_real_customer_speech(self, session_id: str, text: str, speech_data: Dict):
        pass
    
    async def coordinate_completion(self, session_id: str):
        await self.stop_session_demo(session_id)

# Minimal demo script data
DEMO_SCRIPTS = {
    'romance_scam_1.wav': {
        'title': 'Romance Scam Demo',
        'timeline': [
            {
                'triggerAtSeconds': 22.8,
                'riskScore': 25,
                'suggestedQuestion': {
                    'question': 'Can you tell me more about the purpose for this transfer?',
                    'context': 'Initial transfer inquiry',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 40,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'How do you know the person?',
                    'context': 'Verify beneficiary',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 52.7,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'Have they tried reaching out to their own family or embassy for assistance?',
                    'context': 'Emergency assistance verification',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 69,
                'riskScore': 80,
                'suggestedQuestion': {
                    'question': 'How long have you known this person, and have you ever met face-to-face?',
                    'context': 'Relationship verification needed',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 83,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Can you walk me through how you first got in touch with this person?',
                    'context': 'Contact method verification',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 100.5,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Inform customer we are concerned that this situation shows signs of a romance scam',
                    'context': 'Fraud warning to customer',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 123.5,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Have you had any video calls or seen recent photos of this person?',
                    'context': 'Visual verification check',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 132.5,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Is this the first time they have requested financial help from you?',
                    'context': 'Previous requests investigation',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 152.5,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Inform customer we are placing a temporary hold on this transfer and escalating to our fraud team',
                    'context': 'Final escalation action',
                    'urgency': 'critical'
                }
            }
        ]
    },
    'impersonation_scam_1.wav': {
        'title': 'Impersonation Scam Demo',
        'timeline': [
            {
                'triggerAtSeconds': 12,
                'riskScore': 25,
                'suggestedQuestion': {
                    'question': 'Can you tell me which department they said they were from and the number they called from?',
                    'context': 'Caller identity verification',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 24,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'What specific instructions or requests did they make during that call?',
                    'context': 'Suspicious request investigation',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 35,
                'riskScore': 80,
                'suggestedQuestion': {
                    'question': 'Did they request any banking information, passwords, or account details?',
                    'context': 'Information harvesting check',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 60,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Did they specifically ask you to provide your PIN or any security codes?',
                    'context': 'Credential theft attempt',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 70,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'What other personal or financial information did they try to obtain?',
                    'context': 'Additional information gathering',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 152,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Have you received similar suspicious calls like this in the past?',
                    'context': 'Previous incident investigation',
                    'urgency': 'critical'
                }
            }
        ]
    },
    'investment_scam_1.wav': {
        'title': 'Investment Scam Demo',
        'timeline': [
            {
                'triggerAtSeconds': 30,
                'riskScore': 25,
                'suggestedQuestion': {
                    'question': 'Could you provide more details about this investment opportunity you mentioned?',
                    'context': 'Investment verification',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 42,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'How did you originally come across this investment advisor or platform?',
                    'context': 'Advisor source verification',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 60,
                'riskScore': 80,
                'suggestedQuestion': {
                    'question': 'Have you been able to withdraw any returns or profits from previous investments?',
                    'context': 'Investment legitimacy check',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 75,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Can you confirm if this advisor is authorized and regulated by the FCA?',
                    'context': 'Regulatory compliance check',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 90,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'What is the exact name of the trading platform or investment company?',
                    'context': 'Platform verification',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 125,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Have you felt any pressure from the advisor to invest additional funds?',
                    'context': 'Pressure tactics investigation',
                    'urgency': 'critical'
                }
            }
        ]
    },
    'app_scam_1.wav': {
        'title': 'APP Scam Demo',
        'timeline': [
            {
                'triggerAtSeconds': 30,
                'riskScore': 25,
                'suggestedQuestion': {
                    'question': 'Can you tell me how you received these updated banking details?',
                    'context': 'Bank detail change verification',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 46,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'Have you confirmed these new bank details directly with your solicitor by phone?',
                    'context': 'Direct contact verification',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 63,
                'riskScore': 80,
                'suggestedQuestion': {
                    'question': 'Could you provide the name of the solicitor firm and the account holder name for these new details?',
                    'context': 'Solicitor and account verification',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 85,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Does the sender email address match previous communications from your solicitor?',
                    'context': 'Email authenticity check',
                    'urgency': 'critical'
                }
            }
        ]
    },
    'legitimate_call_1.wav': {
        'title': 'Legitimate Call Demo',
        'timeline': [
            {
                'triggerAtSeconds': 94,
                'riskScore': 25,
                'suggestedQuestion': {
                    'question': 'Could you share your travel dates and which countries you plan to visit?',
                    'context': 'Travel information verification',
                    'urgency': 'low'
                }
            }
        ]
    }
}

def getDemoScript(filename: str) -> Optional[Dict]:
    """Get demo script"""
    return DEMO_SCRIPTS.get(filename)
