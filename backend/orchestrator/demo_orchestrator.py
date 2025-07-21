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
            for event in timeline:
                await asyncio.sleep(event.get('triggerAtSeconds', 10))
                
                # Check if session still active
                if session_id not in self.active_sessions:
                    break
                
                # Send question prompt
                question_data = event.get('suggestedQuestion', {})
                if question_data.get('question'):
                    await callback({
                        'type': 'question_prompt_ready',
                        'data': {
                            'session_id': session_id,
                            'question': question_data['question'],
                            'context': question_data.get('context', ''),
                            'urgency': question_data.get('urgency', 'medium'),
                            'pattern': 'demo_timing',
                            'risk_level': event.get('riskScore', 0),
                            'question_id': f"demo_{session_id}_{event.get('triggerAtSeconds', 0)}"
                        }
                    })
                    
        except asyncio.CancelledError:
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
                'triggerAtSeconds': 15,
                'riskScore': 25,
                'suggestedQuestion': {
                    'question': 'What makes this transaction urgent?',
                    'context': 'Urgency detected',
                    'urgency': 'medium'
                }
            },
            {
                'triggerAtSeconds': 35,
                'riskScore': 65,
                'suggestedQuestion': {
                    'question': 'Have you met this person in person?',
                    'context': 'Romance scam indicators',
                    'urgency': 'high'
                }
            }
        ]
    }
}

def getDemoScript(filename: str) -> Optional[Dict]:
    """Get demo script"""
    return DEMO_SCRIPTS.get(filename)
