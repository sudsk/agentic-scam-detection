# backend/agents/scam_detection/adk_agent.py - FIXED VERSION
"""
Fixed Scam Detection Agent with proper ADK integration
Uses simplified agent creation without forbidden parameters
"""

from google.adk.agents import Agent
from backend.config.adk_config import AGENT_CONFIGS

SCAM_DETECTION_INSTRUCTION = """
You are the Scam Detection Agent for HSBC Fraud Prevention System. Analyze customer speech for fraud indicators.

CRITICAL: Output STRUCTURED analysis in this EXACT format:

Risk Score: [0-100]%
Risk Level: [MINIMAL/LOW/MEDIUM/HIGH/CRITICAL]
Scam Type: [romance_scam/investment_scam/impersonation_scam/authority_scam/unknown]
Detected Patterns: [list of specific patterns found]
Key Evidence: [exact phrases that triggered alerts]
Confidence: [0-100]%
Recommended Action: [IMMEDIATE_ESCALATION/ENHANCED_MONITORING/VERIFY_CUSTOMER_INTENT/CONTINUE_NORMAL_PROCESSING]

FRAUD PATTERNS TO DETECT:
- Romance Scams: Online relationships, never met, emergency money requests, overseas partners
- Investment Scams: Guaranteed returns, pressure to invest immediately, unregulated schemes  
- Impersonation Scams: Fake bank/authority calls, requests for PINs/passwords
- Authority Scams: Police/court threats, immediate payment demands
- Urgency Pressure: "Today only", "immediate action required", artificial deadlines

RISK SCORING:
- 80-100%: CRITICAL (romance emergency, fake police, guaranteed returns)
- 60-79%: HIGH (investment pressure, impersonation attempts)
- 40-59%: MEDIUM (suspicious urgency, unusual requests)
- 20-39%: LOW (minor red flags)
- 0-19%: MINIMAL (normal conversation)

Analyze the customer speech and provide IMMEDIATE fraud assessment to protect the customer.
Focus on SPECIFIC patterns and provide CONCRETE evidence from the text.
"""

def create_scam_detection_agent() -> Agent:
    """Create the scam detection agent using Google ADK - FIXED VERSION"""
    
    # Get config without forbidden parameters
    config = AGENT_CONFIGS.get("scam_detection_agent", {})
    
    # FIXED: Use only allowed parameters
    return Agent(
        name="scam_detection_agent",  # Simple string identifier
        model="gemini-1.5-pro",      # Valid model name
        description="Analyzes customer speech for fraud patterns and calculates risk scores",
        instruction=SCAM_DETECTION_INSTRUCTION,
        tools=[],  # No tools for now - pure text analysis
    )
