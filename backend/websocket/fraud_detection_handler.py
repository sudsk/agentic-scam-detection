# backend/websocket/fraud_detection_handler.py - COMPLETE ENHANCED VERSION

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from ..agents.fraud_detection_system import fraud_detection_system
from ..agents.audio_processor.agent import audio_processor_agent 
from ..websocket.connection_manager import ConnectionManager
from ..utils import get_current_timestamp, create_error_response

logger = logging.getLogger(__name__)

class EnhancedFraudDetectionWebSocketHandler:
    """Complete enhanced WebSocket handler with all improvements"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_sessions: Dict[str, Dict] = {}
        self.customer_speech_buffer: Dict[str, List[str]] = {}
        
        # Enhanced tracking for better risk calculation
        self.session_analysis_history: Dict[str, List[Dict]] = {}
        self.accumulated_patterns: Dict[str, Dict] = {}
        
        logger.info("🔌 Enhanced Fraud Detection WebSocket handler initialized")
    
    async def handle_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages"""
        
        message_type = message.get('type')
        data = message.get('data', {})
        
        try:
            if message_type == 'process_audio':
                await self.handle_audio_processing(websocket, client_id, data)
            elif message_type == 'start_realtime_session':
                await self.handle_audio_processing(websocket, client_id, data)
            elif message_type == 'stop_processing':
                await self.handle_stop_processing(websocket, client_id, data)
            elif message_type == 'get_status':
                await self.handle_status_request(websocket, client_id)
            elif message_type == 'ping':
                await self.send_message(websocket, client_id, {
                    'type': 'pong',
                    'data': {'timestamp': get_current_timestamp()}
                })
            else:
                await self.send_error(websocket, client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling message from {client_id}: {e}")
            await self.send_error(websocket, client_id, f"Message processing error: {str(e)}")
    
    async def handle_audio_processing(self, websocket: WebSocket, client_id: str, data: Dict) -> None:
        """Start audio processing with enhanced fraud detection"""
        
        filename = data.get('filename')
        session_id = data.get('session_id') or f"session_{uuid.uuid4().hex[:8]}"
        
        if not filename:
            await self.send_error(websocket, client_id, "Missing filename parameter")
            return
        
        if session_id in self.active_sessions:
            await self.send_error(websocket, client_id, f"Session {session_id} already active")
            return
        
        try:
            logger.info(f"🎵 Starting enhanced audio processing: {filename} for {client_id}")
            
            # Initialize enhanced session tracking
            self.active_sessions[session_id] = {
                'client_id': client_id,
                'filename': filename,
                'websocket': websocket,
                'start_time': datetime.now(),
                'status': 'processing'
            }
            self.customer_speech_buffer[session_id] = []
            self.session_analysis_history[session_id] = []
            self.accumulated_patterns[session_id] = {}
            
            # AUDIO SYNC FIX: Add small delay to sync with frontend audio loading
            await asyncio.sleep(0.5)
            
            # Create callback for audio processor
            async def enhanced_audio_callback(message_data: Dict) -> None:
                await self.handle_enhanced_audio_message(websocket, client_id, session_id, message_data)
            
            # Start audio processing
            result = await audio_processor_agent.start_realtime_processing(
                session_id=session_id,
                audio_filename=filename,
                websocket_callback=enhanced_audio_callback
            )
            
            if 'error' in result:
                await self.send_error(websocket, client_id, result['error'])
                self.cleanup_session(session_id)
                return
            
            # Send confirmation
            await self.send_message(websocket, client_id, {
                'type': 'processing_started',
                'data': {
                    'session_id': session_id,
                    'filename': filename,
                    'audio_duration': result.get('audio_duration', 0),
                    'channels': result.get('channels', 1),
                    'processing_mode': result.get('processing_mode', 'enhanced_realtime'),
                    'timestamp': get_current_timestamp()
                }
            })
            
            logger.info(f"✅ Enhanced processing started for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error starting enhanced processing: {e}")
            await self.send_error(websocket, client_id, f"Failed to start processing: {str(e)}")
            self.cleanup_session(session_id)
    
    async def handle_enhanced_audio_message(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        message_data: Dict
    ) -> None:
        """Handle messages from audio processor with enhanced tracking"""
        
        message_type = message_data.get('type')
        data = message_data.get('data', {})
        
        try:
            if message_type == 'transcription_segment':
                # Forward transcription to client
                await self.send_message(websocket, client_id, message_data)
                
                # Enhanced customer speech processing
                speaker = data.get('speaker')
                text = data.get('text', '')
                
                if speaker == 'customer' and text.strip():
                    await self.enhanced_customer_speech_processing(websocket, client_id, session_id, text)
            
            elif message_type == 'transcription_interim':
                # Forward interim results
                await self.send_message(websocket, client_id, message_data)
            
            elif message_type == 'processing_complete':
                # Auto-create incident for medium+ risk
                await self.handle_processing_complete(websocket, client_id, session_id)
                # Forward the completion message
                await self.send_message(websocket, client_id, message_data)
            
            else:
                # Forward other messages
                await self.send_message(websocket, client_id, message_data)
                
        except Exception as e:
            logger.error(f"❌ Error handling enhanced audio message: {e}")
    
    async def enhanced_customer_speech_processing(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Enhanced customer speech processing with better risk accumulation"""
        
        try:
            # Add to speech buffer
            if session_id not in self.customer_speech_buffer:
                self.customer_speech_buffer[session_id] = []
            
            self.customer_speech_buffer[session_id].append(customer_text)
            
            # SCAM RISK FIX: More frequent analysis for better risk accumulation
            buffer_length = len(self.customer_speech_buffer[session_id])
            
            # Trigger analysis every customer segment for better risk escalation
            if buffer_length >= 1:
                # Use accumulated speech for analysis
                accumulated_speech = " ".join(self.customer_speech_buffer[session_id])
                
                if len(accumulated_speech.strip()) >= 10:  # Lower threshold for faster detection
                    await self.run_enhanced_fraud_analysis(websocket, client_id, session_id, accumulated_speech)
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced customer speech processing: {e}")
    
    async def run_enhanced_fraud_analysis(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        customer_text: str
    ) -> None:
        """Enhanced fraud analysis with pattern accumulation for better risk scoring"""
        
        try:
            logger.info(f"🔍 Running enhanced fraud analysis for session {session_id}")
            
            # Send analysis started notification
            await self.send_message(websocket, client_id, {
                'type': 'fraud_analysis_started',
                'data': {
                    'session_id': session_id,
                    'text_length': len(customer_text),
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Small delay for realistic processing
            await asyncio.sleep(0.3)
            
            # Run fraud detection
            fraud_result = fraud_detection_system.fraud_detector.process(customer_text)
            
            if 'error' not in fraud_result:
                # SCAM RISK FIX: Enhanced pattern accumulation
                current_patterns = fraud_result.get('detected_patterns', {})
                session_patterns = self.accumulated_patterns.get(session_id, {})
                
                # Merge patterns, keeping highest counts and weights
                for pattern_name, pattern_data in current_patterns.items():
                    if pattern_name in session_patterns:
                        # Keep higher count and add severity multipliers
                        existing = session_patterns[pattern_name]
                        pattern_data['count'] = max(pattern_data.get('count', 1), existing.get('count', 1)) + 1
                        pattern_data['matches'] = list(set(
                            pattern_data.get('matches', []) + existing.get('matches', [])
                        ))
                    session_patterns[pattern_name] = pattern_data
                
                self.accumulated_patterns[session_id] = session_patterns
                
                # SCAM RISK FIX: Calculate enhanced risk score based on accumulated patterns
                enhanced_risk_score = self.calculate_enhanced_risk_score(session_patterns, customer_text)
                
                # Store analysis in history
                analysis_record = {
                    'timestamp': get_current_timestamp(),
                    'risk_score': enhanced_risk_score,
                    'patterns': session_patterns,
                    'scam_type': fraud_result.get('scam_type', 'unknown'),
                    'confidence': fraud_result.get('confidence', 0.0)
                }
                self.session_analysis_history[session_id].append(analysis_record)
                
                # SCAM RISK FIX: Send ENHANCED fraud analysis results
                await self.send_message(websocket, client_id, {
                    'type': 'fraud_analysis_update',
                    'data': {
                        'session_id': session_id,
                        'risk_score': enhanced_risk_score,  # Use enhanced score
                        'risk_level': self.get_risk_level(enhanced_risk_score),
                        'scam_type': fraud_result.get('scam_type', 'unknown'),
                        'detected_patterns': session_patterns,  # Use accumulated patterns
                        'confidence': fraud_result.get('confidence', 0.0),
                        'explanation': fraud_result.get('explanation', ''),
                        'pattern_count': len(session_patterns),
                        'analysis_number': len(self.session_analysis_history[session_id]),
                        'timestamp': get_current_timestamp()
                    }
                })
                
                # Get policy guidance for medium/high risk
                if enhanced_risk_score >= 40:
                    await self.get_enhanced_policy_guidance(websocket, client_id, session_id, {
                        'risk_score': enhanced_risk_score,
                        'scam_type': fraud_result.get('scam_type', 'unknown'),
                        'detected_patterns': session_patterns
                    })
                
                logger.info(f"✅ Enhanced fraud analysis complete: {enhanced_risk_score}% risk ({len(session_patterns)} patterns)")
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced fraud analysis: {e}")
    
    def calculate_enhanced_risk_score(self, accumulated_patterns: Dict, customer_text: str) -> float:
        """SCAM RISK FIX: Calculate enhanced risk score for better romance scam detection"""
        total_score = 0.0
        
        # Enhanced scoring for romance scam patterns
        for pattern_name, pattern_data in accumulated_patterns.items():
            weight = pattern_data.get('weight', 20)
            count = pattern_data.get('count', 1)
            severity_multiplier = {
                'critical': 2.5,  # Increased multiplier
                'high': 2.0,      # Increased multiplier  
                'medium': 1.5     # Increased multiplier
            }.get(pattern_data.get('severity', 'medium'), 1.0)
            
            pattern_score = weight * count * severity_multiplier
            
            # ROMANCE SCAM FIX: Special boost for romance scam indicators
            if pattern_name == 'romance_exploitation':
                pattern_score *= 1.5  # 50% boost for romance patterns
            
            total_score += pattern_score
        
        # Additional contextual scoring for romance scams
        text_lower = customer_text.lower()
        if 'never met' in text_lower and 'emergency' in text_lower:
            total_score += 25  # Big boost for this combination
        if 'stuck' in text_lower and ('turkey' in text_lower or 'istanbul' in text_lower):
            total_score += 20  # Location-specific boost
        if 'camera broken' in text_lower or "can't video" in text_lower:
            total_score += 15  # Video avoidance boost
        
        # Cap at 100 but be more aggressive in scoring
        enhanced_score = min(total_score * 0.9, 100)  # Slightly more aggressive
        return round(enhanced_score, 1)
    
    def get_risk_level(self, risk_score: float) -> str:
        """Get risk level from score"""
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"
    
    async def get_enhanced_policy_guidance(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        fraud_analysis: Dict
    ) -> None:
        """Get enhanced policy guidance"""
        
        try:
            await self.send_message(websocket, client_id, {
                'type': 'policy_analysis_started',
                'data': {
                    'session_id': session_id,
                    'timestamp': get_current_timestamp()
                }
            })
            
            await asyncio.sleep(0.3)
            
            # Get policy guidance
            policy_result = fraud_detection_system.policy_guide.process({
                'scam_type': fraud_analysis.get('scam_type', 'unknown'),
                'risk_score': fraud_analysis.get('risk_score', 0)
            })
            
            if 'error' not in policy_result:
                primary_policy = policy_result.get('primary_policy', {})
                agent_guidance = policy_result.get('agent_guidance', {})
                
                await self.send_message(websocket, client_id, {
                    'type': 'policy_guidance_ready',
                    'data': {
                        'session_id': session_id,
                        'policy_id': primary_policy.get('policy_id', ''),
                        'policy_title': primary_policy.get('title', ''),
                        'policy_version': primary_policy.get('version', ''),
                        'immediate_alerts': agent_guidance.get('immediate_alerts', []),
                        'recommended_actions': agent_guidance.get('recommended_actions', []),
                        'key_questions': agent_guidance.get('key_questions', []),
                        'customer_education': agent_guidance.get('customer_education', []),
                        'escalation_threshold': primary_policy.get('escalation_threshold', 80),
                        'timestamp': get_current_timestamp()
                    }
                })
                
        except Exception as e:
            logger.error(f"❌ Error getting enhanced policy guidance: {e}")
    
    async def handle_processing_complete(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str
    ) -> None:
        """AUTO-INCIDENT CREATION: Handle processing completion with auto-incident creation"""
        
        try:
            # Get final analysis
            analysis_history = self.session_analysis_history.get(session_id, [])
            if not analysis_history:
                logger.warning(f"No analysis history for session {session_id}")
                return
            
            final_analysis = analysis_history[-1]  # Get latest analysis
            final_risk_score = final_analysis.get('risk_score', 0)
            
            logger.info(f"🏁 Processing complete for {session_id}, final risk: {final_risk_score}%")
            
            # AUTO-INCIDENT CREATION: Create incident for medium+ risk (>= 50%)
            if final_risk_score >= 50:
                await self.auto_create_incident_with_summary(websocket, client_id, session_id, final_analysis)
            
        except Exception as e:
            logger.error(f"❌ Error handling processing completion: {e}")
    
    async def auto_create_incident_with_summary(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        session_id: str, 
        final_analysis: Dict
    ) -> None:
        """AUTO-INCIDENT CREATION: Auto-create ServiceNow incident with enhanced summary"""
        
        try:
            logger.info(f"📋 Auto-creating incident for session {session_id}")
            
            # Send notification
            await self.send_message(websocket, client_id, {
                'type': 'auto_incident_creation_started',
                'data': {
                    'session_id': session_id,
                    'risk_score': final_analysis.get('risk_score', 0),
                    'reason': 'Medium/High risk detected - auto-creating incident',
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Send summarization progress
            await self.send_message(websocket, client_id, {
                'type': 'summarization_started',
                'data': {
                    'session_id': session_id,
                    'agent': 'Enhanced Summarization System',
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Generate enhanced summary
            summary_start_time = time.time()
            transcript_summary = await self.generate_enhanced_summary(session_id, final_analysis)
            summary_time = time.time() - summary_start_time
            
            # Send summarization complete
            await self.send_message(websocket, client_id, {
                'type': 'summarization_complete',
                'data': {
                    'session_id': session_id,
                    'summary_length': len(transcript_summary),
                    'processing_time': summary_time,
                    'agent': 'enhanced_summarization_system',
                    'timestamp': get_current_timestamp()
                }
            })
            
            # Prepare enhanced incident data
            risk_score = final_analysis.get('risk_score', 0)
            scam_type = final_analysis.get('scam_type', 'unknown')
            detected_patterns = final_analysis.get('patterns', {})
            
            # Get customer info
            customer_info = self.get_customer_info_from_session(session_id)
            
            incident_data = {
                'customer_name': customer_info.get('name', 'Unknown Customer'),
                'customer_account': customer_info.get('account', 'Unknown'),
                'customer_phone': customer_info.get('phone', 'Unknown'),
                'risk_score': risk_score,
                'scam_type': scam_type,
                'session_id': session_id,
                'confidence_score': final_analysis.get('confidence', 0.0),
                'transcript_summary': transcript_summary,
                'detected_patterns': detected_patterns,
                'recommended_actions': [
                    f"Review {scam_type.replace('_', ' ')} indicators",
                    "Verify customer safety and education provided",
                    "Document fraud prevention actions taken",
                    "Follow up with customer within 24 hours"
                ],
                'auto_created': True,
                'creation_reason': f'Auto-created for {risk_score}% risk score',
                'summary_generation_time': summary_time
            }
            
            # Create incident via existing API (mock for demo)
            incident_result = await self.create_servicenow_incident(incident_data)
            
            if incident_result.get('success'):
                await self.send_message(websocket, client_id, {
                    'type': 'auto_incident_created',
                    'data': {
                        'session_id': session_id,
                        'incident_number': incident_result.get('incident_number'),
                        'incident_url': incident_result.get('incident_url'),
                        'risk_score': risk_score,
                        'summary_length': len(transcript_summary),
                        'summary_generation_time': summary_time,
                        'auto_created': True,
                        'timestamp': get_current_timestamp()
                    }
                })
                
                logger.info(f"✅ Auto-created incident {incident_result.get('incident_number')} for session {session_id}")
            else:
                logger.error(f"❌ Failed to auto-create incident: {incident_result.get('error')}")
            
        except Exception as e:
            logger.error(f"❌ Error auto-creating incident: {e}")
    
    async def generate_enhanced_summary(self, session_id: str, final_analysis: Dict) -> str:
        """Generate enhanced summary of the call transcription"""
        
        try:
            # Get full transcript and analysis data
            customer_speech = self.customer_speech_buffer.get(session_id, [])
            detected_patterns = final_analysis.get('patterns', {})
            risk_score = final_analysis.get('risk_score', 0)
            scam_type = final_analysis.get('scam_type', 'unknown')
            confidence = final_analysis.get('confidence', 0.0)
            
            if not customer_speech:
                return "No customer speech recorded for analysis."
            
            # Get customer info
            customer_info = self.get_customer_info_from_session(session_id)
            customer_text = " ".join(customer_speech)
            pattern_names = list(detected_patterns.keys())
            
            # Create enhanced structured summary
            summary = f"""FRAUD DETECTION INCIDENT SUMMARY

EXECUTIVE SUMMARY:
{risk_score}% risk {scam_type.replace('_', ' ')} detected involving {customer_info.get('name', 'customer')} (Account: {customer_info.get('account', 'unknown')}). {len(pattern_names)} fraud indicators identified requiring {'immediate intervention' if risk_score >= 80 else 'enhanced monitoring' if risk_score >= 60 else 'standard procedures with increased vigilance'}.

CUSTOMER REQUEST ANALYSIS:
Customer communication: "{customer_text[:300]}{'...' if len(customer_text) > 300 else ''}"

Key request patterns suggest {self.analyze_request_type(customer_text, scam_type)}.

FRAUD INDICATORS DETECTED:
{self.format_patterns_summary(detected_patterns)}

CUSTOMER BEHAVIOR ASSESSMENT:
{self.assess_behavior_summary(customer_text, scam_type, risk_score)}

RECOMMENDED ACTIONS:
{self.get_summary_recommendations(risk_score, scam_type)}

INCIDENT CLASSIFICATION:
- Severity: {self.get_risk_level_description(risk_score)}
- Financial Risk: {self.assess_financial_risk(risk_score)}
- Escalation: {'Required' if risk_score >= 80 else 'Recommended' if risk_score >= 60 else 'Standard Review'}
- Confidence: {confidence:.1%}

ANALYSIS METADATA:
- Session ID: {session_id}
- Total Customer Segments: {len(customer_speech)}
- Patterns Detected: {len(pattern_names)}
- Analysis Iterations: {len(self.session_analysis_history.get(session_id, []))}

Summary generated by Enhanced Fraud Detection System
Generated: {get_current_timestamp()}
"""
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def analyze_request_type(self, customer_text: str, scam_type: str) -> str:
        """Analyze what the customer was requesting"""
        text_lower = customer_text.lower()
        
        if "transfer" in text_lower or "send money" in text_lower:
            return "international money transfer with urgency indicators"
        elif "investment" in text_lower:
            return "investment opportunity with guaranteed return promises"
        elif "emergency" in text_lower:
            return "emergency financial assistance request"
        elif "help" in text_lower:
            return "request for immediate financial help"
        else:
            return f"potential {scam_type.replace('_', ' ')} scenario"
    
    def format_patterns_summary(self, patterns: Dict) -> str:
        """Format detected patterns for summary"""
        if not patterns:
            return "- No specific patterns detected by automated analysis"
        
        formatted = []
        for pattern_name, pattern_data in patterns.items():
            severity = pattern_data.get('severity', 'medium').upper()
            count = pattern_data.get('count', 1)
            matches = pattern_data.get('matches', [])
            
            formatted.append(f"- {pattern_name.replace('_', ' ').title()}: {severity} risk ({count} instances)")
            if matches:
                example = matches[0][:50] + "..." if len(matches[0]) > 50 else matches[0]
                formatted.append(f"  Example: \"{example}\"")
        
        return "\n".join(formatted)
    
    def assess_behavior_summary(self, customer_text: str, scam_type: str, risk_score: float) -> str:
        """Assess customer behavior for summary"""
        if risk_score >= 80:
            behavior = "Customer shows strong indicators of being under social engineering influence"
        elif risk_score >= 60:
            behavior = "Customer demonstrates vulnerability to manipulation tactics"
        elif risk_score >= 40:
            behavior = "Customer behavior suggests potential exposure to fraud attempts"
        else:
            behavior = "Customer interaction within normal parameters"
        
        if scam_type == "romance_scam":
            behavior += ". Emotional attachment to online relationship evident."
        elif scam_type == "investment_scam":
            behavior += ". Attracted to unrealistic investment promises."
        elif scam_type == "impersonation_scam":
            behavior += ". Responding to perceived authority without verification."
        
        return behavior
    
    def get_summary_recommendations(self, risk_score: float, scam_type: str) -> str:
        """Get recommendations for summary"""
        recommendations = []
        
        if risk_score >= 80:
            recommendations.extend([
                "- IMMEDIATE: Stop all pending transactions",
                "- IMMEDIATE: Escalate to Fraud Prevention Team",
                "- Contact customer via secure channel within 15 minutes"
            ])
        elif risk_score >= 60:
            recommendations.extend([
                "- Enhanced verification procedures required",
                "- Monitor account for 48 hours",
                "- Schedule supervisor review"
            ])
        elif risk_score >= 40:
            recommendations.extend([
                "- Document interaction thoroughly",
                "- Provide customer education materials",
                "- Standard monitoring procedures"
            ])
        
        recommendations.extend([
            f"- Educate customer on {scam_type.replace('_', ' ')} warning signs",
            "- Follow up within 24-48 hours",
            "- Update customer risk profile"
        ])
        
        return "\n".join(recommendations)
    
    def get_risk_level_description(self, risk_score: float) -> str:
        """Get descriptive risk level"""
        if risk_score >= 80:
            return "CRITICAL - Immediate intervention required"
        elif risk_score >= 60:
            return "HIGH - Enhanced monitoring and verification needed"
        elif risk_score >= 40:
            return "MEDIUM - Caution advised with standard procedures"
        else:
            return "LOW - Standard processing with documentation"
    
    def assess_financial_risk(self, risk_score: float) -> str:
        """Assess financial risk level"""
        if risk_score >= 80:
            return "HIGH - Significant financial loss potential"
        elif risk_score >= 60:
            return "MEDIUM - Moderate financial exposure"
        elif risk_score >= 40:
            return "LOW - Limited financial risk"
        else:
            return "MINIMAL - Standard transaction risk"
    
    def get_customer_info_from_session(self, session_id: str) -> Dict:
        """Get customer info for the session"""
        session_data = self.active_sessions.get(session_id, {})
        filename = session_data.get('filename', '')
        
        # Customer profiles based on audio file
        customer_profiles = {
            'romance_scam': {
                'name': 'Mrs Patricia Williams',
                'account': '****3847',
                'phone': '+44 1202 555123'
            },
            'investment_scam': {
                'name': 'Mr David Chen', 
                'account': '****5691',
                'phone': '+44 161 555456'
            },
            'impersonation_scam': {
                'name': 'Ms Sarah Thompson',
                'account': '****7234', 
                'phone': '+44 121 555789'
            }
        }
        
        for scam_type, profile in customer_profiles.items():
            if scam_type in filename:
                return profile
                
        return {
            'name': 'Unknown Customer',
            'account': '****0000',
            'phone': 'Unknown'
        }
    
    async def create_servicenow_incident(self, incident_data: Dict) -> Dict:
        """Create ServiceNow incident (mock for demo)"""
        try:
            # Mock successful incident creation
            return {
                'success': True,
                'incident_number': f"INC{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'incident_url': f"https://dev303197.service-now.com/incident/{incident_data['session_id']}"
            }
        except Exception as e:
            return {
                '
