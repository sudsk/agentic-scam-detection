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
                'triggerAtSeconds': 23,
                'riskScore': 25,
                'suggestedQuestion': {
                    'question': 'What is the purpose of the transfer?',
                    'context': 'Initial transfer inquiry',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 49,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'Has the person contacted his family for help?',
                    'context': 'Emergency assistance verification',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 64,
                'riskScore': 80,
                'suggestedQuestion': {
                    'question': 'Have you met the person in person?',
                    'context': 'Relationship verification needed',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 76,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'How did the person contact you?',
                    'context': 'Contact method verification',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 93,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Raise concern about potential Romance scam',
                    'context': 'Fraud warning to customer',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 110,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Have you been able to video call?',
                    'context': 'Visual verification check',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 125,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Has the person asked for money before?',
                    'context': 'Previous requests investigation',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 145,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Recommend temporary hold and escalate to fraud team.',
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
                    'question': 'Who did they speak to and what number did they call from?',
                    'context': 'Caller identity verification',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 24,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'What did they ask the customer to do?',
                    'context': 'Suspicious request investigation',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 35,
                'riskScore': 80,
                'suggestedQuestion': {
                    'question': 'Did they ask for any account details?',
                    'context': 'Information harvesting check',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 60,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Did they ask for your PIN?',
                    'context': 'Credential theft attempt',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 70,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Did they ask for any other personal details?',
                    'context': 'Additional information gathering',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 152,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Has this happened before?',
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
                    'question': 'Ask details about the investment opportunity',
                    'context': 'Investment verification',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 42,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'How did they find the advisor?',
                    'context': 'Advisor source verification',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 60,
                'riskScore': 80,
                'suggestedQuestion': {
                    'question': 'Have they withdrawn any profits?',
                    'context': 'Investment legitimacy check',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 75,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Is the advisor regulated by FCA?',
                    'context': 'Regulatory compliance check',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 90,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'What is the name of the trading platform?',
                    'context': 'Platform verification',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 125,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Has the advisor pressured you to invest more?',
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
                    'question': 'How did they receive the updated bank details?',
                    'context': 'Bank detail change verification',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 46,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'Did they speak directly to their solicitor about these new bank details?',
                    'context': 'Direct contact verification',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 63,
                'riskScore': 80,
                'suggestedQuestion': {
                    'question': "What's the name of their solicitor firm and the account name for the new bank details?",
                    'context': 'Solicitor and account verification',
                    'urgency': 'high'
                }
            },
            {
                'triggerAtSeconds': 85,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Does the email address look the same as previous correspondence?',
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
                    'question': 'Ask about travel dates and destination',
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
