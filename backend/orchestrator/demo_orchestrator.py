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
                    'context': 'Urgent transfer detected',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 49,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'Has the person contacted his family for help?',
                    'context': 'Romance scam indicators',
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
                    'question': 'How the person contacted you?',
                    'context': 'Secrecy request detected',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 93,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Raise concern about potential Romance scam',
                    'context': 'Secrecy request detected',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 110,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Have you been able to video call?',
                    'context': 'Secrecy request detected',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 125,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Has the person asked money before?',
                    'context': 'Secrecy request detected',
                    'urgency': 'critical'
                }
            },
            {
                'triggerAtSeconds': 145,
                'riskScore': 90,
                'suggestedQuestion': {
                    'question': 'Recommend temporary hold and escalate to fraud team.',
                    'context': 'Secrecy request detected',
                    'urgency': 'critical'
                }
            }
        ]
    }
}

def getDemoScript(filename: str) -> Optional[Dict]:
    """Get demo script"""
    return DEMO_SCRIPTS.get(filename)
