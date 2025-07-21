# backend/orchestrator/demo_orchestrator.py
"""
Demo Orchestrator for Hybrid Mode - Two-Track Approach
Track 1: Real audio processing with actual fraud detection
Track 2: Demo enhancements with perfect timing and educational content
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
import uuid

logger = logging.getLogger(__name__)

class DemoOrchestrator:
    """
    Orchestrates demo enhancements alongside real fraud detection processing
    Implements Two-Track Approach: Real Processing + Demo Enhancements
    """
    
    def __init__(self, fraud_orchestrator=None):
        self.fraud_orchestrator = fraud_orchestrator
        
        # Demo state management
        self.demo_state = {
            'enabled': False,
            'mode': 'hybrid',  # hybrid, pure_script, or production
            'active_sessions': {},  # session_id -> demo_data
            'coordination_enabled': True
        }
        
        # Demo timing and coordination
        self.demo_tasks = {}  # session_id -> asyncio.Task
        self.coordination_data = {}  # session_id -> coordination_info
        
        logger.info("🎭 Demo Orchestrator initialized for Two-Track Approach")
    
    # ===== HYBRID DEMO MAIN METHODS =====
    
    async def start_demo_overlay(
        self, 
        script: Dict, 
        callback: Optional[Callable],
        session_id: str
    ) -> bool:
        """
        Start demo overlay that coordinates with real processing
        This is the main entry point for Hybrid Demo mode
        """
        try:
            logger.info(f"🎭 Starting Two-Track Demo: {script.get('title', 'Unknown')}")
            logger.info(f"   Track 1: Real audio processing will handle actual fraud detection")
            logger.info(f"   Track 2: Demo enhancements will provide perfect timing and education")
            
            # Store demo session data
            self.demo_state['active_sessions'][session_id] = {
                'script': script,
                'callback': callback,
                'start_time': time.time(),
                'events_triggered': [],
                'coordination_mode': 'hybrid',
                'real_speech_detected': False,
                'timeline_position': 0
            }
            
            # Initialize coordination data
            self.coordination_data[session_id] = {
                'real_processing_active': True,
                'demo_enhancements_active': True,
                'last_real_event': None,
                'last_demo_event': None,
                'sync_status': 'coordinated'
            }
            
            # Send demo overlay started message
            if callback:
                await callback({
                    'type': 'hybrid_demo_started',
                    'data': {
                        'session_id': session_id,
                        'script_title': script.get('title', 'Demo Script'),
                        'mode': 'hybrid',
                        'track_1': 'Real audio processing + fraud detection',
                        'track_2': 'Demo enhancements + educational content',
                        'coordination_enabled': True,
                        'timeline_events': len(script.get('timeline', [])),
                        'duration': script.get('duration', 60),
                        'timestamp': self._get_timestamp()
                    }
                })
            
            # Start demo timeline in background (Track 2)
            demo_task = asyncio.create_task(
                self._run_demo_timeline(session_id, script, callback)
            )
            self.demo_tasks[session_id] = demo_task
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo overlay start error: {e}")
            return False
    
    async def _run_demo_timeline(
        self, 
        session_id: str, 
        script: Dict, 
        callback: Optional[Callable]
    ):
        """
        Run demo timeline (Track 2) coordinated with real audio processing (Track 1)
        """
        try:
            logger.info(f"🎭 Track 2: Starting demo timeline for {session_id}")
            
            demo_session = self.demo_state['active_sessions'][session_id]
            timeline = script.get('timeline', [])
            
            for event_index, event in enumerate(timeline):
                # Wait for the right time (coordinated with Track 1 audio)
                wait_time = event.get('triggerAtSeconds', 0)
                logger.info(f"🎭 Track 2: Waiting {wait_time}s for event {event_index + 1}")
                await asyncio.sleep(wait_time)
                
                # Check if session still active
                if session_id not in self.demo_state['active_sessions']:
                    logger.info(f"🎭 Track 2: Session {session_id} no longer active, stopping timeline")
                    break
                
                # Update timeline position
                demo_session['timeline_position'] = event_index + 1
                
                # Send demo explanation (Track 2 enhancement)
                await self._send_demo_explanation(session_id, event, callback)
                
                # Wait a moment, then send question enhancement
                await asyncio.sleep(1.0)
                await self._send_demo_question_enhancement(session_id, event, callback)
                
                # Track triggered events
                demo_session['events_triggered'].append({
                    'event_index': event_index,
                    'trigger_time': time.time(),
                    'event_type': 'demo_enhancement',
                    'question': event.get('suggestedQuestion', {}).get('question', ''),
                    'risk_score': event.get('riskScore', 0)
                })
                
                # Update coordination status
                if session_id in self.coordination_data:
                    self.coordination_data[session_id]['last_demo_event'] = {
                        'time': time.time(),
                        'event_index': event_index,
                        'type': 'timeline_event'
                    }
                
                logger.info(f"🎭 Track 2: Event {event_index + 1}/{len(timeline)} triggered for {session_id}")
            
            # Demo timeline complete
            await self._handle_demo_timeline_complete(session_id, callback)
            
        except asyncio.CancelledError:
            logger.info(f"🎭 Track 2: Demo timeline cancelled for {session_id}")
        except Exception as e:
            logger.error(f"❌ Track 2: Demo timeline error: {e}")
    
    # ===== COORDINATION WITH REAL PROCESSING (Track 1) =====
    
    async def handle_real_customer_speech(
        self, 
        session_id: str, 
        text: str, 
        speech_data: Dict
    ):
        """
        Handle real customer speech from Track 1 and coordinate with Track 2
        """
        if session_id not in self.demo_state['active_sessions']:
            return
        
        demo_session = self.demo_state['active_sessions'][session_id]
        demo_session['real_speech_detected'] = True
        
        # Update coordination data
        if session_id in self.coordination_data:
            self.coordination_data[session_id]['last_real_event'] = {
                'time': time.time(),
                'type': 'customer_speech',
                'text': text[:50] + '...' if len(text) > 50 else text
            }
        
        # Log coordination between tracks
        logger.info(f"🎭 COORDINATION: Real speech (Track 1) detected for {session_id}")
        logger.info(f"   Text: '{text[:100]}...'")
        logger.info(f"   Demo timeline position: {demo_session.get('timeline_position', 0)}")
        
        # Could add logic here to adjust demo timing based on real speech
        # For now, just log that coordination is working
    
    async def coordinate_completion(self, session_id: str):
        """
        Coordinate demo completion with real processing completion
        Called when Track 1 (real processing) completes
        """
        if session_id not in self.demo_state['active_sessions']:
            return
            
        logger.info(f"🎭 COORDINATION: Real processing (Track 1) complete for {session_id}")
        
        demo_session = self.demo_state['active_sessions'][session_id]
        callback = demo_session['callback']
        
        # Stop demo timeline if still running
        if session_id in self.demo_tasks:
            demo_task = self.demo_tasks[session_id]
            if not demo_task.done():
                demo_task.cancel()
                logger.info(f"🎭 Track 2: Demo timeline stopped due to Track 1 completion")
        
        # Send coordination completion message
        if callback:
            await callback({
                'type': 'hybrid_coordination_complete',
                'data': {
                    'session_id': session_id,
                    'track_1_status': 'Real processing complete',
                    'track_2_status': 'Demo enhancements coordinated',
                    'events_triggered': len(demo_session['events_triggered']),
                    'coordination_successful': True,
                    'timestamp': self._get_timestamp()
                }
            })
    
    # ===== DEMO ENHANCEMENT METHODS =====
    
    async def _send_demo_explanation(
        self, 
        session_id: str, 
        event: Dict, 
        callback: Optional[Callable]
    ):
        """Send demo explanation overlay (Track 2 enhancement)"""
        if not callback:
            return
        
        suggested_question = event.get('suggestedQuestion', {})
        explanation = suggested_question.get('explanation', 'Demo explanation')
        risk_score = event.get('riskScore', 0)
        
        await callback({
            'type': 'demo_explanation',
            'data': {
                'session_id': session_id,
                'explanation': {
                    'title': 'Track 2: Demo Enhancement',
                    'message': f"DEMO TIMING: {explanation}",
                    'type': 'pattern',
                    'riskLevel': risk_score,
                    'duration': 4000
                },
                'mode': 'hybrid',
                'track': '2',
                'banking_context': event.get('bankingContext', {}),
                'compliance_note': event.get('complianceNote', ''),
                'note': 'This is Track 2 demo enhancement - Track 1 handles real analysis',
                'timestamp': self._get_timestamp()
            }
        })
        
        logger.info(f"🎭 Track 2: Sent explanation for {session_id}: {explanation[:50]}...")
    
    async def _send_demo_question_enhancement(
        self, 
        session_id: str, 
        event: Dict, 
        callback: Optional[Callable]
    ):
        """Send demo question enhancement (Track 2, doesn't replace real questions)"""
        if not callback:
            return
        
        suggested_question = event.get('suggestedQuestion', {})
        question = suggested_question.get('question', 'Demo question')
        context = suggested_question.get('context', 'Demo context')
        
        await callback({
            'type': 'demo_question_enhancement',
            'data': {
                'session_id': session_id,
                'suggested_question': question,
                'context': context,
                'urgency': suggested_question.get('urgency', 'medium'),
                'agent_action': event.get('agentAction', 'Demo agent action'),
                'banking_context': event.get('bankingContext', {}),
                'mode': 'hybrid_enhancement',
                'track': '2',
                'note': 'Track 2 demo enhancement - Real questions come from Track 1 AI analysis',
                'timestamp': self._get_timestamp()
            }
        })
        
        logger.info(f"🎭 Track 2: Sent question enhancement for {session_id}: {question}")
    
    async def _handle_demo_timeline_complete(self, session_id: str, callback: Optional[Callable]):
        """Handle demo timeline completion"""
        logger.info(f"🎭 Track 2: Demo timeline complete for {session_id}")
        
        if callback:
            await callback({
                'type': 'demo_timeline_complete',
                'data': {
                    'session_id': session_id,
                    'track': '2',
                    'status': 'Demo timeline complete',
                    'note': 'Track 1 (real processing) may still be active',
                    'timestamp': self._get_timestamp()
                }
            })
    
    # ===== SESSION MANAGEMENT =====
    
    async def stop_session_demo(self, session_id: str):
        """Stop demo for specific session"""
        logger.info(f"🎭 Stopping demo for session {session_id}")
        
        # Cancel demo task
        if session_id in self.demo_tasks:
            demo_task = self.demo_tasks[session_id]
            if not demo_task.done():
                demo_task.cancel()
            del self.demo_tasks[session_id]
        
        # Clean up session data
        if session_id in self.demo_state['active_sessions']:
            del self.demo_state['active_sessions'][session_id]
        
        if session_id in self.coordination_data:
            del self.coordination_data[session_id]
        
        logger.info(f"🎭 Demo stopped and cleaned up for session {session_id}")
    
    # ===== STATUS AND MONITORING =====
    
    def get_demo_status(self) -> Dict[str, Any]:
        """Get current demo status"""
        return {
            'enabled': self.demo_state['enabled'],
            'mode': self.demo_state['mode'],
            'active_sessions': len(self.demo_state['active_sessions']),
            'coordination_enabled': self.demo_state['coordination_enabled'],
            'approach': 'Two-Track (Real Processing + Demo Enhancements)',
            'sessions': {
                session_id: {
                    'script_title': session_data['script'].get('title', 'Unknown'),
                    'timeline_position': session_data.get('timeline_position', 0),
                    'events_triggered': len(session_data.get('events_triggered', [])),
                    'real_speech_detected': session_data.get('real_speech_detected', False),
                    'coordination_mode': session_data.get('coordination_mode', 'hybrid')
                }
                for session_id, session_data in self.demo_state['active_sessions'].items()
            },
            'coordination_status': self.coordination_data,
            'timestamp': self._get_timestamp()
        }
    
    def is_demo_active(self, session_id: str) -> bool:
        """Check if demo is active for session"""
        return session_id in self.demo_state['active_sessions']
    
    def get_coordination_status(self, session_id: str) -> Dict[str, Any]:
        """Get coordination status between Track 1 and Track 2"""
        if session_id not in self.coordination_data:
            return {'status': 'not_coordinated'}
        
        coord_data = self.coordination_data[session_id]
        return {
            'track_1_status': 'Real processing active' if coord_data['real_processing_active'] else 'Inactive',
            'track_2_status': 'Demo enhancements active' if coord_data['demo_enhancements_active'] else 'Inactive',
            'last_real_event': coord_data.get('last_real_event'),
            'last_demo_event': coord_data.get('last_demo_event'),
            'sync_status': coord_data.get('sync_status', 'unknown'),
            'session_active': session_id in self.demo_state['active_sessions']
        }
    
    # ===== UTILITY METHODS =====
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def __str__(self) -> str:
        return f"DemoOrchestrator(mode={self.demo_state['mode']}, active_sessions={len(self.demo_state['active_sessions'])})"

# ===== DEMO SCRIPT UTILITIES =====

def getDemoScript(filename: str) -> Optional[Dict]:
    """
    Get demo script for filename - simplified version
    In a full implementation, this would import from demo_scripts.py
    """
    # For now, return a simple demo script structure
    # Later this will import from the full demo_scripts.py file
    
    demo_scripts = {
        'romance_scam_1.wav': {
            'title': 'Romance Scam Investigation Demo',
            'description': 'Progressive detection of romance scam with emotional manipulation',
            'duration': 65,
            'scamType': 'romance_scam',
            'timeline': [
                {
                    'triggerAtSeconds': 12,
                    'riskScore': 20,
                    'suggestedQuestion': {
                        'question': 'What specifically makes this transaction urgent?',
                        'context': 'Urgency pressure detected',
                        'urgency': 'medium',
                        'explanation': 'AI detected urgency keywords - common fraud indicator'
                    },
                    'agentAction': 'Agent investigates claimed urgency',
                    'complianceNote': 'Following FCA guidelines for APP fraud prevention',
                    'bankingContext': {
                        'transactionType': 'International Transfer',
                        'amount': '£3,000',
                        'riskFlags': ['Urgency', 'International']
                    }
                },
                {
                    'triggerAtSeconds': 28,
                    'riskScore': 45,
                    'suggestedQuestion': {
                        'question': 'How did you first meet this person?',
                        'context': 'Romance scam indicators detected',
                        'urgency': 'high',
                        'explanation': 'AI detected romance scam pattern: boyfriend + overseas + emergency'
                    },
                    'agentAction': 'Agent probes relationship background',
                    'complianceNote': 'Enhanced due diligence required',
                    'bankingContext': {
                        'transactionType': 'Emergency Transfer',
                        'amount': '£3,000',
                        'riskFlags': ['Romance scam', 'Never met', 'Emergency']
                    }
                }
            ]
        }
    }
    
    return demo_scripts.get(filename)

# Export the main class
__all__ = ['DemoOrchestrator', 'getDemoScript']
